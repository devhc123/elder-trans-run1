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
.build_injection_negatives`）：grounded 药名注入杀"候选在括注内⇒违规"，
等价形式注入（原文「3天」→注入「三天」）杀"在括注内 ∧ 非grounded"这个
合取。只有真正读懂原文依据关系的分类器才能两类都判对。

用法：
    python3 verifier/adversarial_subset.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.candidate_pool import build_injection_negatives, load_jargon  # noqa: E402
from verifier.shortcut_probes import report as report_shortcut_probes  # noqa: E402
from verifier.synth_minimal_edit import (  # noqa: E402
    synthesize_all,
    synthetic_records_to_candidates,
)

WORK = ROOT / "verifier" / "work"
MIN_SIZE = 50  # L2 判读规则的门槛


def build_adversarial_subset(records: list[dict], jargon: set[str] | None = None) -> list[dict]:
    """holdout 源文本合成的正例 + 两类注入负例（ticket 26）。

    **负例只放注入类，不混普通可信负例。** 两种配比的探针 J 都算过：只放注入
    负例时"候选在括注内"这条被打到 0（本轮明确要杀的那条），且合取捷径最低；
    再混普通可信负例会把两个单特征压得更匀，却把合取顶上去。见 ticket 26。

    **按 kind 分别配平**：实体负例对齐实体正例、数字负例对齐数字正例。用一个
    总数上限截断会让 kind 分布随机倾斜（探针⑦ 盯的就是这个）。

    `include_equivalent_form=False` 是 ticket 25 盲检的降级档——一致率落在
    90–95% 时等价形式负例只进训练池、不进这份冻结验收集（训练池里混几条错标
    是稀释，验收集里混几条错标是整个 L2 门槛的数字不可信）。"""
    positives = synthetic_records_to_candidates(synthesize_all(records, holdout=True))
    from collections import Counter

    from verifier.shortcut_probes import infer_kind
    n_entity = sum(1 for p in positives if p["red_line_guess"] == 0)
    # 数字类负例按**正例的 kind 分布**取配额，不是按总数截断——见
    # `build_injection_negatives` 里那段注释（红线2 切片的形状捷径）。
    numeric_kinds = Counter(infer_kind(p["candidate_text"])
                            for p in positives if p["red_line_guess"] == 2)
    negatives = build_injection_negatives(
        records, jargon if jargon is not None else load_jargon(),
        max_entity=n_entity, numeric_kind_targets=dict(numeric_kinds), holdout=True,
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
    from verifier.candidate_pool import EQUIVALENT_FORM_SUFFIX
    n_eq = sum(1 for it in subset if it["case_id"].endswith(EQUIVALENT_FORM_SUFFIX))
    n_ground = n_neg - n_eq
    ok = n_pos >= MIN_SIZE
    print(f"holdout {len(records)} 条 -> 对抗子集 {len(subset)} 条"
          f"（正例 {n_pos}：红线0 {n_rl0} / 红线2 {n_rl2}；注入负例 {n_neg}——"
          f"grounded {n_ground} + 等价形式 {n_eq}）"
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
