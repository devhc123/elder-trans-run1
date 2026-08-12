#!/usr/bin/env python3
"""对抗子集构造（ticket 14，L2 判读规则）：「医学正确但原文未给出」的候选级
验收样本。

CATEGORY_EXAMPLES 里的示例药名都是真实存在的同类药物、数字注入都是合理的
频次/时长，只是没出现在这条 source_text 里——这正是 L2 要的对抗模式，
不需要额外构造，直接复用 `synth_minimal_edit.synthesize_all` 的合成逻辑。

**只从 holdout 源文本生成，绝不进训练集。** 如果拿训练时用过的合成样本
（哪怕是同一模式、不同案例）做验收，测的是泛化到"这套注入词表"的能力，
不是泛化到"这条具体 source_text 没给的东西"——用 holdout 独有的源文本
生成，保证对抗子集与训练数据在 case 级别零交集（train/holdout 本身
record_id 零交集，见 ticket 10 封存纪律）。

用法：
    python3 verifier/adversarial_subset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.synth_minimal_edit import (  # noqa: E402
    synthesize_all,
    synthetic_records_to_candidates,
)

WORK = ROOT / "verifier" / "work"
MIN_SIZE = 50  # L2 判读规则的门槛


def build_adversarial_subset(records: list[dict]) -> list[dict]:
    return synthetic_records_to_candidates(synthesize_all(records))


def main() -> int:
    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = build_adversarial_subset(records)
    n_rl0 = sum(1 for it in subset if it["red_line_guess"] == 0)
    n_rl2 = sum(1 for it in subset if it["red_line_guess"] == 2)
    ok = len(subset) >= MIN_SIZE
    print(f"holdout {len(records)} 条 -> 对抗子集 {len(subset)} 条"
          f"（红线0 {n_rl0} / 红线2 {n_rl2}）（L2 门槛 ≥{MIN_SIZE} -> {'通过' if ok else '未通过'}）")

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "adversarial_subset_holdout.json"
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
