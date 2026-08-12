#!/usr/bin/env python3
"""P1 冻结评测：段 A 候选抽取器对红线 0/2 的召回（ticket 12）。

**只评一次。** wordlist（`redline_food_wordlist.txt`）与中文数词正则的校准
全部只用了 `verifier/train.jsonl` 的漏报案例（见该文件与
`docs/RESEARCH_verifier_data_scarcity.md` 的污染纪律说明）；本脚本对
`verifier/holdout.jsonl` 的评测结果是这轮唯一一次真实评测，不许看完结果
再回去调 wordlist/正则重跑。

红线 1（把原文事实说反）不进本评测——全池仅 2 例，两段式管线不覆盖它，
教师全文判兜底。

用法：
    python3 verifier/redline_candidates_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.redline_candidates import (  # noqa: E402
    COVERED_RED_LINE_IDXS,
    extract_candidates,
    load_jargon,
    parse_case_text,
)

GATE_RECALL_MIN = 0.95  # ticket 12 P1 门槛（训练前写死）：instance 级召回 ≥95%


def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def instance_level_score(records: list[dict], jargon: set[str]) -> dict:
    """红线 0/2 逐条 instance 级 TP/FN/FP/TN——与官方 `eval_verifier.py` 同口径
    （逐条而非逐 case），只是"预测"来自规则候选（段 A 有没有抽出任何候选）
    而非训练出来的模型。"""
    tp = fn = fp = tn = 0
    fn_detail: list[tuple[str, int, str]] = []
    for r in records:
        src, ans = parse_case_text(r["input"])
        flagged = bool(extract_candidates(src, ans, jargon))
        gold = json.loads(r["output"])
        for rl in gold["red_lines"]:
            if rl["idx"] not in COVERED_RED_LINE_IDXS:
                continue
            if rl["violated"]:
                if flagged:
                    tp += 1
                else:
                    fn += 1
                    fn_detail.append((r["case_id"], rl["idx"], rl["evidence"]))
            else:
                if flagged:
                    fp += 1
                else:
                    tn += 1
    pos, neg = tp + fn, fp + tn
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "miss_rate": fn / pos if pos else 0.0,
        "false_alarm_rate": fp / neg if neg else 0.0,
        "fn_detail": fn_detail,
    }


def report(name: str, m: dict) -> bool:
    ok = m["miss_rate"] <= (1 - GATE_RECALL_MIN)
    print(f"[{name}] TP={m['tp']} FN={m['fn']} FP={m['fp']} TN={m['tn']}")
    print(f"  漏报率 {m['miss_rate']:.1%}（P1 门槛 ≤{1 - GATE_RECALL_MIN:.0%}）-> {'通过' if ok else '未通过'}")
    print(f"  误报率 {m['false_alarm_rate']:.1%}（P1 不设门槛，如实报告——精度留给段 B）")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()

    jargon = load_jargon()

    train = load_split(ROOT / "verifier" / "train.jsonl")
    report("train（诊断参考，非调参依据——wordlist/正则已冻结）", instance_level_score(train, jargon))

    print()
    holdout = load_split(ROOT / "verifier" / "holdout.jsonl")
    m_holdout = instance_level_score(holdout, jargon)
    ok = report("holdout（冻结评测，P1 门槛判定，只评这一次）", m_holdout)

    if m_holdout["fn_detail"]:
        print("\nholdout FN 明细（P1 未覆盖的违规模式，供后续迭代参考——"
              "不得据此回头改 wordlist/正则重评）：")
        for cid, idx, ev in m_holdout["fn_detail"]:
            print(f"  {cid} 红线{idx}: {ev[:50]!r}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
