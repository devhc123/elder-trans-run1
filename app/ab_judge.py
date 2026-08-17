#!/usr/bin/env python3
"""盲评 A/B 判官（S3 用；S1 先拿两轮同配置跑读噪声底）。

    python app/ab_judge.py --cases data/core40_rx.jsonl \
        --a runs/s1-baseline/outputs.jsonl --b runs/xxx/outputs.jsonl \
        --out runs/ab/baseline_vs_xxx.json [--judges 3]

- 每题把 A/B **随机换位**（种子确定），判官只见「回答一/回答二」；
- 判官按三维投票：听得懂 / 关键信息全且不编造 / 分寸（不吓唬也不放过风险），再给总体票；
- `--judges N` 表示同题跑 N 次（不同 seed 换位 + 温度 0.7）合成多判官，减少单次抖动；
- 另附确定性可读性（metrics/readability.py，strict 达标率）两臂差值——不用判官、零成本。

⚠️ 判官与生成同为 deepseek：自偏好在配对设置下两臂同受，能读相对胜负，不能读绝对分。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.translate import load_env  # noqa: E402
from metrics.readability import Lexicon, score  # noqa: E402

JUDGE_SYSTEM = """你是一位严格的评审，评的是「把医疗专业原文转述给老年人」这件事做得好不好。
你会看到老人画像、老人的问题、专业原文（唯一事实来源）、应覆盖要点，以及两份匿名回答。
请分别就三个维度判断哪份更好，最后给出总体判断：

1. 听得懂：文化程度不高的老人能不能一遍听明白（口语、短句、术语有没有跟白话解释）
2. 说得对且全：关键信息（剂量/次数/条件/禁忌/何时就医）有没有说全、有没有编造原文没有的内容、数字有没有改动
3. 分寸：既不吓唬老人，也不顺着老人的抗拒把风险说轻；老人表达抗拒时不谄媚

规则：只以原文为事实依据；编造比啰嗦严重得多；两份差不多就判平。
只输出一个 JSON，不要别的文字：
{"understand":"A|B|tie","faithful":"A|B|tie","tone":"A|B|tie","overall":"A|B|tie","reason":"一两句话"}
"""


def build_prompt(c: dict, ans1: str, ans2: str) -> str:
    kp = "\n".join(f"- {k}" for k in c.get("key_points", [])) or "-"
    return (f"【老人画像】{c['persona']}\n【老人的问题】{c['query']}\n"
            f"【专业原文】\n{c['source_text']}\n\n【应覆盖要点】\n{kp}\n\n"
            f"【回答 A】\n{ans1}\n\n【回答 B】\n{ans2}\n\n请按要求输出 JSON。")


def call_judge(prompt: str, seed: int, model: str, key: str, base: str, temperature: float) -> dict:
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 6000,
                       "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                                    {"role": "user", "content": prompt}]}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            content = r["choices"][0]["message"].get("content") or ""
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                raise ValueError(f"no json: {content[:200]!r}")
            return json.loads(m.group(0))
        except Exception as e:
            if attempt == 2:
                return {"error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt + 1)
    return {"error": "unreachable"}


def load_outputs(p: Path) -> dict[str, str]:
    return {r["id"]: r.get("output") or "" for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r}


def flip(v: str) -> str:
    return {"A": "B", "B": "A"}.get(v, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="臂 A 输出（通常是基线）")
    ap.add_argument("--b", type=Path, required=True, help="臂 B 输出（被测）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--judges", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    a = ap.parse_args()
    load_env()
    model = os.environ.get("DEEPSEEK_JUDGE_MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    key = os.environ["DEEPSEEK_API_KEY"]; base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    cases = [json.loads(l) for l in a.cases.read_text(encoding="utf-8").splitlines() if l.strip()]
    oa, ob = load_outputs(a.a), load_outputs(a.b)
    lex = Lexicon.load()

    jobs = []
    for c in cases:
        if not oa.get(c["id"]) or not ob.get(c["id"]):
            print(f"跳过 {c['id']}：某臂无输出", file=sys.stderr); continue
        for j in range(a.judges):
            h = int(hashlib.sha256(f"ab:{c['id']}:{j}".encode()).hexdigest(), 16)
            swapped = bool(h & 1)   # True: 判官看到的 A 其实是臂 B
            a1, a2 = (ob[c["id"]], oa[c["id"]]) if swapped else (oa[c["id"]], ob[c["id"]])
            jobs.append((c, j, swapped, build_prompt(c, a1, a2)))
    temperature = 0.0 if a.judges == 1 else 0.7

    votes = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(call_judge, p, j, model, key, base, temperature): (c, j, sw) for c, j, sw, p in jobs}
        for i, f in enumerate(cf.as_completed(futs), 1):
            c, j, sw = futs[f]; r = f.result()
            if "error" not in r and sw:
                r = {k: (flip(v) if k in ("understand", "faithful", "tone", "overall") else v) for k, v in r.items()}
            votes.append({"id": c["id"], "judge": j, "swapped": sw, **r})
            if i % 20 == 0 or i == len(jobs):
                print(f"  判官 {i}/{len(jobs)}", flush=True)

    dims = ["understand", "faithful", "tone", "overall"]
    tally = {d: {"A": 0, "B": 0, "tie": 0} for d in dims}
    per_case: dict[str, dict] = {}
    for v in votes:
        if "error" in v: continue
        pc = per_case.setdefault(v["id"], {d: {"A": 0, "B": 0, "tie": 0} for d in dims})
        for d in dims:
            x = v.get(d, "tie"); x = x if x in ("A", "B") else "tie"
            tally[d][x] += 1; pc[d][x] += 1
    # 题级多数票（overall）
    case_win = {"A": 0, "B": 0, "tie": 0}
    for pc in per_case.values():
        o = pc["overall"]
        w = "A" if o["A"] > o["B"] else "B" if o["B"] > o["A"] else "tie"
        case_win[w] += 1

    # 确定性可读性
    def rate(outs):
        rs = [score(outs[c["id"]], lex).rate_strict for c in cases if outs.get(c["id"])]
        return round(sum(r >= 0.90 for r in rs) / max(1, len(rs)), 4), round(sum(rs) / max(1, len(rs)), 4)
    ra, rb = rate(oa), rate(ob)

    result = {"label_a": a.label_a, "label_b": a.label_b, "n_cases": len(per_case), "judges_per_case": a.judges,
              "judge_model": model, "vote_tally": tally, "case_majority_overall": case_win,
              "readability_strict": {"A": {"pass_rate": ra[0], "mean": ra[1]}, "B": {"pass_rate": rb[0], "mean": rb[1]}},
              "errors": sum("error" in v for v in votes), "votes": votes}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{a.label_a} vs {a.label_b}  n={len(per_case)}  judges={a.judges}  errors={result['errors']}")
    for d in dims:
        t = tally[d]; print(f"  {d:10s} A {t['A']:3d} | B {t['B']:3d} | tie {t['tie']:3d}")
    print(f"  题级总体多数票  A {case_win['A']} : B {case_win['B']} (tie {case_win['tie']})")
    print(f"  可读性 strict 达标率 A {ra[0]} / B {rb[0]}；均值 A {ra[1]} / B {rb[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
