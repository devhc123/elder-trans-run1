#!/usr/bin/env python3
"""用 DeepSeek 对已标注样本做独立第二意见抽查。

**这不是验收，是找分歧候选。** kpi.yaml 定的纪律是判官厂商必须与被测厂商
异厂解耦——这里刻意破例：DeepSeek 既是写 `answer` 的被测系统，又在本脚本里
被拿来当第二判官。它可能对自己写的话系统性偏松（自己的话自己觉得没问题），
也可能系统性偏盲（生成时漏看的东西，判断时一样漏看）。这个脚本的分歧率
**只能当"值得人工复核的候选清单"，不能当教师标注或 verifier 的验收依据**。

用法：
    python3 verifier/cross_check_teacher.py --n 40 --seed 20260812
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LABELS = ROOT / "verifier" / "labels"
WORK = ROOT / "verifier" / "work"

TIMEOUT = 180
RETRIES = 4

# 同 verifier/TEACHER_TASK.md 的判分纪律，改写成第二判官的 system prompt。
JUDGE_SYSTEM = """你是医疗转述的忠实性判别器。只以「原文」为事实依据，判断「回答」是否忠实。

逐条判断：每个要点是否被回答传达到；每条红线是否被触犯。
回答里说了原文没说、但医学上正确的事，仍记作触犯红线——你判的是有没有原文依据，
不是医学上对不对。拿不准时：覆盖从严（含糊即未覆盖），红线从严。

只输出 JSON，不要解释，不要写 markdown 代码块围栏。格式：
{"key_points": [{"idx": 0, "covered": true, "evidence": "..."}],
 "red_lines": [{"idx": 0, "violated": false, "evidence": ""}],
 "verdict": "pass"}"""


def build_prompt(c: dict) -> str:
    kp = "\n".join(f"{i}. {k}" for i, k in enumerate(c["key_points"]))
    rl = "\n".join(f"{i}. {r}" for i, r in enumerate(c["red_lines"]))
    return (
        f"【原文】\n{c['source_text']}\n\n"
        f"【回答】\n{c['answer']}\n\n"
        f"【要点】\n{kp}\n\n"
        f"【红线】\n{rl}"
    )


def call_judge(case_id: str, prompt: str, model: str, key: str, base: str) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 4000,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    for attempt in range(RETRIES):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=TIMEOUT))
            content = r["choices"][0]["message"].get("content") or ""
            return {"case_id": case_id, "raw": content}
        except Exception as e:  # noqa: BLE001 — 网络失败形态太多，宽泛捕获，见 run_eval.py 同款注释
            if attempt == RETRIES - 1:
                return {"case_id": case_id, "raw": "", "error": str(e)}
            time.sleep(2**attempt)
    return {"case_id": case_id, "raw": "", "error": "unreachable"}


def parse_judgment(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        j = json.loads(text.strip())
        return j if isinstance(j, dict) else None
    except Exception:
        return None


def compare(teacher: dict, second: dict | None) -> dict:
    """逐点比较教师标注与第二意见。第二意见条数缺失/对不上时按无可比信息处理，
    不崩、不强行对齐。"""
    if second is None:
        return {"parse_failed": True, "disagreements": [], "verdict_match": False}

    disagreements = []
    t_kp = {k["idx"]: k["covered"] for k in teacher["key_points"]}
    s_kp = {k.get("idx"): k.get("covered") for k in second.get("key_points") or []}
    for idx, t_val in t_kp.items():
        s_val = s_kp.get(idx)
        if s_val is not None and s_val != t_val:
            disagreements.append({"field": "key_point", "idx": idx, "teacher": t_val, "second": s_val})

    t_rl = {r["idx"]: r["violated"] for r in teacher["red_lines"]}
    s_rl = {r.get("idx"): r.get("violated") for r in second.get("red_lines") or []}
    for idx, t_val in t_rl.items():
        s_val = s_rl.get(idx)
        if s_val is not None and s_val != t_val:
            disagreements.append({"field": "red_line", "idx": idx, "teacher": t_val, "second": s_val})

    verdict_match = second.get("verdict") == teacher["verdict"]
    return {"parse_failed": False, "disagreements": disagreements, "verdict_match": verdict_match}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="抽样规模")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", type=Path, default=WORK / "cross_check.json")
    args = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not key:
        print("缺 DEEPSEEK_API_KEY（source .env）", file=sys.stderr)
        return 1

    items = {c["case_id"]: c for c in json.loads((WORK / "to_label.json").read_text(encoding="utf-8"))}
    labels: dict[str, dict] = {}
    for p in sorted(LABELS.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                labels[j["case_id"]] = j

    rng = random.Random(args.seed)
    ids = sorted(labels)
    sample_ids = rng.sample(ids, min(args.n, len(ids)))
    print(f"从 {len(ids)} 条已标注样本里抽 {len(sample_ids)} 条做交叉核对（第二判官：{model}）")

    results = []
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {
            ex.submit(call_judge, cid, build_prompt(items[cid]), model, key, base): cid
            for cid in sample_ids
        }
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            second = parse_judgment(r["raw"])
            cmp = compare(labels[r["case_id"]], second)
            results.append({
                "case_id": r["case_id"], **cmp,
                "second_verdict": (second or {}).get("verdict"),
                "second_red_lines": (second or {}).get("red_lines"),
                "second_raw": r["raw"],
            })
            if i % 10 == 0 or i == len(sample_ids):
                print(f"  {i}/{len(sample_ids)}")

    n = len(results)
    parse_fail = sum(1 for r in results if r["parse_failed"])
    ok_n = n - parse_fail
    verdict_agree = sum(1 for r in results if not r["parse_failed"] and r["verdict_match"])
    total_disagreements = sum(len(r["disagreements"]) for r in results)
    flagged = [r for r in results if r["disagreements"] or r["parse_failed"]]

    print(f"\n交叉核对完成：{n} 条")
    print(f"  DeepSeek 输出解析失败: {parse_fail}/{n}")
    print(f"  verdict 一致: {verdict_agree}/{ok_n}" + (f"（{verdict_agree/ok_n:.1%}）" if ok_n else ""))
    print(f"  逐条分歧数: {total_disagreements}")
    print(f"  需要人工复核的样本: {len(flagged)}/{n}")
    print(
        "\n⚠️ 这不是独立验证——DeepSeek 既是被测系统又在这里当第二判官，"
        "同厂自证有系统性偏松/偏盲的风险。分歧率只能当复核候选清单，"
        "不得当教师标注或 verifier 的验收依据引用。"
    )

    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详情 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
