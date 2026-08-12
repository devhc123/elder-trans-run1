#!/usr/bin/env python3
"""可信候选池构造器（ticket 14）。

**核心设计决定**：`verdict=pass` 案例是教师对整案例给出的"零违规"判断——
这个判断本身没有"只记一条 evidence"的结构性缺陷，段A在这类案例上抽出的
每个候选都是真负例，不需要任何反推。`verdict=fail` 案例只有 evidence 命中
的候选才是可信正例；**没命中的候选整条丢弃**，不当负例用——这是 ticket 12
P2 探针诊断出的方法论缺陷的直接修复：案例级标注只记一条 evidence，不穷举
同一红线下的所有违规词，把"没命中 evidence"当"没违规"会把系统性漏标
烤进训练数据。

train 和 holdout 用同一个构造函数——训练负例挖矿、held-out 候选级验收集
都从这里来，保证两处用的是同一套"可信"规则，不会各自漂移出一套。

用法：
    python3 verifier/candidate_pool.py --split train
    python3 verifier/candidate_pool.py --split holdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.judge_candidates import derive_candidate_label  # noqa: E402
from verifier.redline_candidates import extract_candidates, load_jargon, parse_case_text  # noqa: E402

WORK = ROOT / "verifier" / "work"


def build_trusted_candidate_pool(records: list[dict], jargon: set[str]) -> list[dict]:
    """对每条 (train 或 holdout) 记录抽候选，按案例 verdict 分流打可信标签。

    - `verdict=pass`：段A抽出的每个候选都标 `label=False`（可信负例）。
    - `verdict=fail`：只保留 `derive_candidate_label` 命中 evidence 的候选，
      标 `label=True`；未命中的候选**不进 pool**——真实标签不确定，不能
      当负例用，也不算正例。
    """
    pool: list[dict] = []
    for r in records:
        src, ans = parse_case_text(r["input"])
        gold = json.loads(r["output"])
        is_pass = gold["verdict"] == "pass"
        for c in extract_candidates(src, ans, jargon):
            if is_pass:
                label = False
            else:
                if not derive_candidate_label(c["text"], c["kind"], gold):
                    continue
                label = True
            pool.append({
                "case_id": r["case_id"],
                "source_text": src,
                "answer": ans,
                "candidate_text": c["text"],
                "kind": c["kind"],
                "red_line_guess": c["red_line_guess"],
                "label": label,
            })
    return pool


def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "holdout"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    records = load_split(ROOT / "verifier" / f"{args.split}.jsonl")
    jargon = load_jargon()
    pool = build_trusted_candidate_pool(records, jargon)

    n_pos = sum(1 for it in pool if it["label"] is True)
    n_neg = sum(1 for it in pool if it["label"] is False)
    n_rl0 = sum(1 for it in pool if it["label"] is True and it["red_line_guess"] == 0)
    n_rl2 = sum(1 for it in pool if it["label"] is True and it["red_line_guess"] == 2)
    print(f"{args.split} {len(records)} 条 -> 可信候选池 {len(pool)} 条"
          f"（正例 {n_pos}：红线0 {n_rl0} / 红线2 {n_rl2}；负例 {n_neg}）")

    out = args.out or (WORK / f"trusted_candidate_pool_{args.split}.json")
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
