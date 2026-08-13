#!/usr/bin/env python3
"""对抗子集构造（ticket 14，L2 判读规则）：「医学正确但原文未给出」的候选级
验收样本。

CATEGORY_EXAMPLES 里的示例药名都是真实存在的同类药物、数字注入都是合理的
频次/时长，只是没出现在这条 source_text 里——这正是 L2 要的对抗模式，
不需要额外构造，直接复用 `synth_minimal_edit.synthesize_all` 的合成逻辑。

**只从 holdout 源文本生成，绝不进训练集，且不与训练用的注入词表共享
成员。** 早期实现只保证了 case 级零交集（holdout 源文本≠train 源文本），
但 `synthesize_all` 用的是同一份固定 `CATEGORY_EXAMPLES`/`NUMERIC_
INJECTIONS`——独立第二意见代码审计发现：这样构造出来的对抗子集，注入
词汇与训练正例 **100% 重叠**（holdout 里出现的每一个词，train 训练时
都见过），模型可能靠"记住这些固定词"而不是真正判断"这个词有没有原文
依据"就把对抗子集答对，L2 门槛的分数会被词表级泄漏撑高。已修：
`synthesize_all(records, holdout=True)` 只从每个类别/数字候选池里留给
验收用的那部分成员选，与 train 用的成员（`holdout=False`，默认值）
互不重叠。

**同时混入结构性负例**（第三轮独立审计问题一的修复）：只看答案有没有
含固定括注模板这一个特征，不读原文，就能在纯正例对抗子集上拿到 100%
召回/0%误报——现在混入"括注存在但内容真实无害"的负例（`candidate_pool
.build_structural_negatives`），一个只认结构的分类器在这份对抗子集上
会被拉回到接近瞎猜的水平，只有真正读懂原文依据关系的分类器才能两类
都判对。

用法：
    python3 verifier/adversarial_subset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.candidate_pool import build_structural_negatives, load_jargon  # noqa: E402
from verifier.shortcut_probes import report as report_shortcut_probes  # noqa: E402
from verifier.synth_minimal_edit import (  # noqa: E402
    synthesize_all,
    synthetic_records_to_candidates,
)

WORK = ROOT / "verifier" / "work"
MIN_SIZE = 50  # L2 判读规则的门槛


def build_adversarial_subset(records: list[dict], jargon: set[str] | None = None) -> list[dict]:
    positives = synthetic_records_to_candidates(synthesize_all(records, holdout=True))
    # 负例数量对齐正例总数（约1:1，同 train_lora.py 的口径）——不需要
    # 把可信池里所有 pass 候选都装饰一遍，只要"有没有括注"在这份验收集
    # 里不再完美区分两类就够了。
    negatives = build_structural_negatives(
        records, jargon if jargon is not None else load_jargon(), holdout=True, max_items=len(positives)
    )
    return positives + negatives


def main() -> int:
    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = build_adversarial_subset(records)
    n_pos = sum(1 for it in subset if it["label"] is True)
    n_neg = sum(1 for it in subset if it["label"] is False)
    n_rl0 = sum(1 for it in subset if it["label"] is True and it["red_line_guess"] == 0)
    n_rl2 = sum(1 for it in subset if it["label"] is True and it["red_line_guess"] == 2)
    ok = n_pos >= MIN_SIZE
    print(f"holdout {len(records)} 条 -> 对抗子集 {len(subset)} 条"
          f"（正例 {n_pos}：红线0 {n_rl0} / 红线2 {n_rl2}；结构性负例 {n_neg}）"
          f"（L2 门槛正例数 ≥{MIN_SIZE} -> {'通过' if ok else '未通过'}）")

    # 退化分类器探针（ticket 17）。这里**只打印不改退出码**——本 CLI 的退出码
    # 由 L2 的正例数门槛决定，探针的强制点在 `shortcut_probes.py --gate` 与
    # 部署预检那里，一个 CLI 不该有两套互相打架的判据。
    report_shortcut_probes(subset, "对抗子集（holdout）")

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "adversarial_subset_holdout.json"
    out.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
