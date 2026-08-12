#!/usr/bin/env python3
"""P2 一致率计分（ticket 12）：教师逐候选判定 vs 派生 gold，分层报告。

**不报单一总一致率**——同 `eval_verifier.py` 的纪律：正例/负例分开报，
防止负例占多数把正例的判定质量掩盖掉。

用法：
    python3 verifier/score_teacher_judgments.py \\
        --pool verifier/work/candidate_judge_packet.json \\
        --judged verifier/work/candidate_judge_result_0.json verifier/work/candidate_judge_result_1.json ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE_AGREEMENT_MIN = 0.90  # ticket 12 P2 门槛（训练前写死）


def score(pool: list[dict], judged: list[dict]) -> dict:
    gold = {item["id"]: item["derived_label"] for item in pool}
    pred = {item["id"]: item["violated"] for item in judged}

    missing = [i for i in gold if i not in pred]

    def agreement(ids: list[str]) -> float:
        if not ids:
            return 0.0
        # 缺判按不一致计——静默跳过会让分母缩小、一致率虚高
        return sum(1 for i in ids if pred.get(i) == gold[i]) / len(ids)

    pos_ids = [i for i in gold if gold[i] is True]
    neg_ids = [i for i in gold if gold[i] is False]

    return {
        "n_total": len(gold),
        "n_positive": len(pos_ids),
        "n_negative": len(neg_ids),
        "n_missing": len(missing),
        "missing_ids": missing,
        # 缺判一律按不一致计——静默跳过会让分母缩小、一致率虚高
        "overall_agreement": sum(1 for i in gold if pred.get(i) == gold[i]) / len(gold) if gold else 0.0,
        "positive_agreement": agreement(pos_ids),
        "negative_agreement": agreement(neg_ids),
    }


def report(m: dict) -> bool:
    ok = m["overall_agreement"] >= GATE_AGREEMENT_MIN
    print(f"候选总数 {m['n_total']}（正例 {m['n_positive']} / 负例 {m['n_negative']}）"
          + (f"，缺判 {m['n_missing']}" if m["n_missing"] else ""))
    print(f"  总一致率     {m['overall_agreement']:.1%}（P2 门槛 ≥{GATE_AGREEMENT_MIN:.0%}）"
          f" -> {'通过' if ok else '未通过'}")
    print(f"  正例一致率   {m['positive_agreement']:.1%}（候选命中 gold evidence 那一层）")
    print(f"  负例一致率   {m['negative_agreement']:.1%}（候选未命中 gold evidence 那一层）")
    if m["missing_ids"]:
        print(f"  缺判 id: {m['missing_ids']}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--judged", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    judged: list[dict] = []
    for p in args.judged:
        judged += json.loads(p.read_text(encoding="utf-8"))

    m = score(pool, judged)
    return 0 if report(m) else 1


if __name__ == "__main__":
    sys.exit(main())
