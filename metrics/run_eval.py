#!/usr/bin/env python3
"""评测 runner —— 三段式。

判官是 session 内的 Claude subagent，脚本调不了它，所以流程刻意拆成三段，
中间那段由编排者（主 agent）执行：

    --generate        调被测模型生成转译输出        -> runs/<run_id>/outputs.jsonl
    --judge-packets   产出判官分片包                -> runs/<run_id>/packets/*.json
    （编排者派 subagent 判分，写回 runs/<run_id>/judged/*.jsonl）
    --aggregate       汇总出分、写报告、回填 kpi.yaml

`--dry-run` 用确定性 mock 走完全流程，不联网、不花钱，只验管线。

**判官可审计但不可复现**：session 内的 Claude 没有可钉版本的 API model id，
换个 session 重跑就是另一批数。所以每条判官输入输出都逐条留痕，run_id 写进
kpi.yaml。可读性是确定性计算，不受此影响。
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
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.readability import Lexicon, score  # noqa: E402
from metrics.wilson import wilson  # noqa: E402

TESTSET = ROOT / "data" / "elder_translate_270.jsonl"
RUNS = ROOT / "runs"
JUDGE_PROMPT = ROOT / "metrics" / "judge_prompt.md"
KPI = ROOT / "kpi.yaml"

BATCH = 14          # 每个判官分片的题数：270 题 -> 20 个分片
MAX_WORKERS = 12
TIMEOUT = 120

SYSTEM_PROMPT = """你是一位面向老年人的健康助手，服务对象是文化程度不高的银发用户。

请把医疗专业文本转述成老人能听懂的大白话，做到：
- 用日常口语，避免专业术语；实在绕不开的术语，紧跟一句白话解释
- 把关键信息说全，尤其是剂量、条件、禁忌、什么情况必须就医
- 不编造原文没有的内容
- 语气平实，既不吓唬老人，也不轻描淡写地放过风险"""


def load_cases(only: list[str] | None = None) -> list[dict]:
    cases = [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]
    if only:
        keep = set(only)
        cases = [c for c in cases if c["id"] in keep]
    return cases


def build_user_prompt(c: dict) -> str:
    return (
        f"【老人情况】{c['persona']}\n"
        f"【老人的问题】{c['query']}\n"
        f"【需要转述的原文】\n{c['source_text']}\n\n"
        "请用老人能懂的话回答。"
    )


# ---------------------------------------------------------------- 生成

def call_model(c: dict, model: str, key: str, base: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(c)},
            ],
            "temperature": 0.3,
            # deepseek-v4-flash 是**推理模型**，max_tokens 是含推理在内的总额度。
            # 早期设 1500 的后果：11 题把预算全烧在推理上、正文一个字没输出
            # （reasoning_tokens 恰为 1500、content 为空），另有 13 题恰好撞顶
            # 被静默截断——后者更坏，截断的答案会被照常计分并压低分数。
            "max_tokens": 8000,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
            ch = r["choices"][0]
            content = ch["message"].get("content") or ""
            fr = ch.get("finish_reason")
            rec = {
                "id": c["id"],
                "output": content,
                "finish_reason": fr,
                "model": r.get("model", model),
                "usage": r.get("usage", {}),
                "latency_s": round(time.time() - t0, 2),
            }
            # 截断必须显式标出来，不能混进正常样本静默拉低分数
            if fr not in (None, "stop"):
                rec["truncated"] = True
            if not content.strip():
                rec["error"] = f"空输出（finish_reason={fr}，推理占用 " \
                    f"{r.get('usage',{}).get('completion_tokens_details',{}).get('reasoning_tokens')} tok）"
            return rec
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as e:
            if attempt == 2:
                return {"id": c["id"], "output": None, "error": f"{type(e).__name__}: {e}"}
            time.sleep(2 * (attempt + 1))
    return {"id": c["id"], "output": None, "error": "unreachable"}


def mock_output(c: dict) -> str:
    """确定性 mock：把原文按标点切开、逐句加白话前缀。

    刻意做得「像样但不够好」——它必须能跑通管线，又不能让 dry-run 的分数
    看起来像真数。dry-run 的数字永远不许填进 kpi.yaml 的 actual。
    """
    h = int(hashlib.sha256(c["id"].encode()).hexdigest()[:8], 16)
    parts = [p for p in re.split(r"[。；\n]", c["source_text"]) if p.strip()]
    keep = parts[: max(1, len(parts) - h % 3)]
    return "您好，我给您说说。" + "。".join(keep) + "。有不明白的再问我。"


def generate(run_id: str, cases: list[dict], live: bool) -> Path:
    out_dir = RUNS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "outputs.jsonl"

    if not live:
        rows = [{"id": c["id"], "output": mock_output(c), "model": "mock", "usage": {}} for c in cases]
    else:
        key = os.environ.get("DEEPSEEK_API_KEY")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        if not key:
            print("缺 DEEPSEEK_API_KEY（source .env）", file=sys.stderr)
            raise SystemExit(1)
        rows = []
        done = 0
        with cf.ThreadPoolExecutor(MAX_WORKERS) as ex:
            futs = {ex.submit(call_model, c, model, key, base): c for c in cases}
            for f in cf.as_completed(futs):
                rows.append(f.result())
                done += 1
                if done % 20 == 0 or done == len(cases):
                    print(f"  生成 {done}/{len(cases)}", flush=True)
        rows.sort(key=lambda r: r["id"])

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    errs = [r for r in rows if not r.get("output")]
    if errs:
        print(f"[warn] {len(errs)} 条生成失败：{[e['id'] for e in errs][:5]}", file=sys.stderr)
    return path


# ---------------------------------------------------------------- 判官分片

def judge_packets(run_id: str, cases: list[dict]) -> int:
    out_dir = RUNS / run_id
    outputs = {
        json.loads(l)["id"]: json.loads(l)
        for l in (out_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    pdir = out_dir / "packets"
    pdir.mkdir(exist_ok=True)
    (out_dir / "judged").mkdir(exist_ok=True)

    items = []
    for c in cases:
        o = outputs.get(c["id"], {})
        if not o.get("output"):
            continue
        items.append(
            {
                "id": c["id"],
                "scenario": c["scenario"],
                "persona": c["persona"],
                "query": c["query"],
                "source_text": c["source_text"],
                "key_points": c["key_points"],
                "negative_rubric": c["negative_rubric"],
                "answer": o["output"],
            }
        )
    n = 0
    for i in range(0, len(items), BATCH):
        chunk = items[i : i + BATCH]
        (pdir / f"batch_{i // BATCH:02d}.json").write_text(
            json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n += 1
    print(f"  {len(items)} 题 -> {n} 个分片（每片 ≤{BATCH} 题）：{pdir}")
    print(f"  判官写回：{out_dir / 'judged'}/batch_NN.jsonl")
    return n


# ---------------------------------------------------------------- 汇总

def aggregate(run_id: str, cases: list[dict], write_kpi: bool) -> int:
    out_dir = RUNS / run_id
    outputs = {
        json.loads(l)["id"]: json.loads(l)
        for l in (out_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    judged: dict[str, dict] = {}
    jdir = out_dir / "judged"
    if jdir.exists():
        for p in sorted(jdir.glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    judged[j["id"]] = j

    lex = Lexicon.load()
    by_id = {c["id"]: c for c in cases}
    rows = []
    for cid, o in sorted(outputs.items()):
        c = by_id.get(cid)
        if not c or not o.get("output"):
            continue
        r = score(o["output"], lex)
        src = c.get("source_readability")
        j = judged.get(cid, {})
        rows.append(
            {
                "id": cid,
                "scenario": c["scenario"],
                "difficulty": c["difficulty"],
                "strict": r.rate_strict,
                "glossed": r.rate_glossed,
                "lenient": r.rate_lenient,
                "jargon_density": r.jargon_density,
                "n_chars": r.n_chars,
                "src_readability": src,
                "net_gain": (r.rate_strict - src) if src is not None else None,
                "len_ratio": round(r.n_chars / max(len(c["source_text"]), 1), 3),
                "has_headroom": (src is not None and src < 0.90),
                "faithful": j.get("faithful_overall"),
                "violated": j.get("any_violated"),
                "kp_covered": j.get("key_points_covered"),
                "kp_total": j.get("key_points_total"),
            }
        )

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def pass_rate(rs, gauge="strict", target=0.90):
        ok = sum(1 for r in rs if r[gauge] >= target)
        return ok, len(rs), (ok / len(rs) if rs else 0.0)

    print(f"\n{'='*62}\nrun_id: {run_id}   题数 {len(rows)}\n{'='*62}")

    # KPI-1：词级微平均。整体 = 所有词拉通，与 KPI 定义字面一致。
    print("\n【KPI-1 可读性】")
    for gauge in ("strict", "glossed", "lenient"):
        m = mean(r[gauge] for r in rows)
        ok, n, pr = pass_rate(rows, gauge)
        lo, hi = wilson(ok, n)
        star = "  <- 主 KPI" if gauge == "strict" else ""
        print(f"  {gauge:<8} 均值 {m:.4f}   ≥0.90 的题 {ok}/{n} = {pr:.1%}  [{lo:.1%},{hi:.1%}]{star}")

    src_m = mean(r["src_readability"] for r in rows)
    gain = mean(r["net_gain"] for r in rows)
    print(f"\n  原文基线均值 {src_m:.4f}   净提升 {gain:+.4f}   <- 必须与绝对值并排看")

    head = [r for r in rows if r["has_headroom"]]
    if head:
        ok, n, pr = pass_rate(head)
        m = mean(r["strict"] for r in head)
        g = mean(r["net_gain"] for r in head)
        print(f"  原文未达标子集 {n} 题：strict 均值 {m:.4f}  ≥0.90 {ok}/{n}={pr:.1%}  净提升 {g:+.4f}")
        print("     ^ 这才是真正测转译能力的部分（其余题照抄原文即通过）")

    hard = [r for r in rows if r["difficulty"] == "难"]
    if hard:
        ok, n, pr = pass_rate(hard)
        print(f"  hard 子集 {n} 题：strict 均值 {mean(r['strict'] for r in hard):.4f}  ≥0.90 {ok}/{n}={pr:.1%}")

    print("\n  按场景（宏平均，诊断用，不可与微平均混报）：")
    scs: dict[str, list] = {}
    for r in rows:
        scs.setdefault(r["scenario"], []).append(r)
    for s in sorted(scs, key=lambda s: mean(x["strict"] for x in scs[s])):
        v = scs[s]
        print(f"    {s:<14}{mean(x['strict'] for x in v):.4f}  (n={len(v)}, 净提升 {mean(x['net_gain'] for x in v):+.4f})")

    print("\n【长度】")
    lr = sorted(r["len_ratio"] for r in rows)
    nc = sorted(r["n_chars"] for r in rows)
    print(f"  转译/原文字数比 中位 {lr[len(lr)//2]:.2f}  P90 {lr[9*len(lr)//10]:.2f}")
    print(f"  绝对字数 中位 {nc[len(nc)//2]}  P90 {nc[9*len(nc)//10]}")
    print(f"  jargon 密度均值 {mean(r['jargon_density'] for r in rows):.2f} /100词")

    print("\n【守门指标 忠实性】")
    jrows = [r for r in rows if r["faithful"] is not None]
    if not jrows:
        print("  尚无判官结果 —— 先派 subagent 判分（见 --judge-packets）")
    else:
        okf = sum(1 for r in jrows if r["faithful"] and not r["violated"])
        lo, hi = wilson(okf, len(jrows))
        print(f"  通过 {okf}/{len(jrows)} = {okf/len(jrows):.1%}  [{lo:.1%},{hi:.1%}]")
        kc = sum(r["kp_covered"] or 0 for r in jrows)
        kt = sum(r["kp_total"] or 0 for r in jrows)
        if kt:
            print(f"  要点覆盖 {kc}/{kt} = {kc/kt:.1%}")
        viol = sum(1 for r in jrows if r["violated"])
        print(f"  红线违规 {viol}/{len(jrows)} = {viol/len(jrows):.1%}")

    (out_dir / "per_case.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    print(f"\n逐题结果 -> {out_dir/'per_case.jsonl'}")

    if write_kpi:
        _write_kpi(run_id, rows, jrows)
        print(f"已回填 kpi.yaml（run_id={run_id}）")
    return 0


def _write_kpi(run_id: str, rows: list[dict], jrows: list[dict]) -> None:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 4) if xs else None

    ok = sum(1 for r in rows if r["strict"] >= 0.90)
    lo, hi = wilson(ok, len(rows))
    head = [r for r in rows if r["has_headroom"]]
    hard = [r for r in rows if r["difficulty"] == "难"]
    okf = sum(1 for r in jrows if r["faithful"] and not r["violated"]) if jrows else None

    text = KPI.read_text(encoding="utf-8")
    marker = "\n# --- filled by run_eval.py --aggregate ---\n"
    text = text.split(marker)[0].rstrip() + "\n" + marker
    text += f"last_run_id: {run_id}\n"
    text += f"kpi1_strict_mean: {mean(r['strict'] for r in rows)}\n"
    text += f"kpi1_strict_pass_rate: {round(ok/len(rows),4)}\n"
    text += f"kpi1_ci95: [{round(lo,4)}, {round(hi,4)}]\n"
    text += f"kpi1_net_gain: {mean(r['net_gain'] for r in rows)}\n"
    text += f"kpi1_on_headroom_subset: {mean(r['strict'] for r in head)}\n"
    text += f"kpi1_hard_subset: {mean(r['strict'] for r in hard)}\n"
    text += f"guard_faithfulness: {round(okf/len(jrows),4) if jrows else 'null'}\n"
    text += f"judge_note: Claude Sonnet 5 session subagent；可审计不可复现，留痕见 runs/{run_id}/\n"
    KPI.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--judge-packets", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--live", action="store_true", help="真调模型；缺省用确定性 mock")
    ap.add_argument("--run-id", help="缺省按内容+模式生成")
    ap.add_argument("--only", help="逗号分隔的 case id 子集")
    ap.add_argument("--write-kpi", action="store_true", help="把结果回填 kpi.yaml")
    args = ap.parse_args()

    only = [x.strip() for x in args.only.split(",")] if args.only else None
    cases = load_cases(only)
    if not cases:
        print("没有可跑的题", file=sys.stderr)
        return 1

    run_id = args.run_id or ("live" if args.live else "dry") + "-" + hashlib.sha256(
        (",".join(c["id"] for c in cases)).encode()
    ).hexdigest()[:8]

    if args.generate:
        print(f"生成 {len(cases)} 题（{'live' if args.live else 'mock'}）run_id={run_id}")
        generate(run_id, cases, args.live)
    if args.judge_packets:
        judge_packets(run_id, cases)
    if args.aggregate:
        return aggregate(run_id, cases, args.write_kpi)
    if not any((args.generate, args.judge_packets, args.aggregate)):
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
