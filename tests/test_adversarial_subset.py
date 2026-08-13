"""对抗子集构造的测试（ticket 14，L2 判读规则的落地）。

对抗子集是「医学正确但原文未给出」的天然样本——CATEGORY_EXAMPLES/数字注入
本身就是这个模式（真实存在的同类药名/合理频次，只是没出现在这条
source_text 里），不需要额外构造。这里守的是：从合成记录展开到候选级
条目时字段对不对、正例 label 恒为 True（P3 教师抽检 50/50 已确认这些是
真违规，见 ticket 12），以及**只从 holdout 源文本生成，不进训练集**——
否则模型训练时见过这些确切样本，拿它们做验收就是在测记忆力不是泛化能力。

**第三轮独立审计（问题一）后新增**：对抗子集不再是纯正例——混入了结构性
负例（`candidate_pool.build_structural_negatives`），因为实测一个只看
"答案有没有含固定括注模板"的退化分类器能在纯正例对抗子集上拿到 100%
召回/0%误报。现在两类都有，才能真正测出"读没读懂原文依据"而不是"认不认
得出括注"。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.adversarial_subset import build_adversarial_subset  # noqa: E402
from verifier.synth_minimal_edit import synthetic_records_to_candidates  # noqa: E402


def _record(case_id, source, answer, verdict="pass"):
    """默认 verdict=pass——`build_adversarial_subset` 内部现在也调用
    `build_structural_negatives`，它需要每条记录都有真实的 `output`
    字段（模拟 holdout.jsonl 里真实标注过的记录形状），不能再是只有
    `input` 的裸记录。"""
    return {
        "case_id": case_id,
        "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
        "output": f'{{"key_points": [], "red_lines": [], "verdict": "{verdict}"}}',
    }


def test_synthetic_records_to_candidates_expands_injected_examples():
    synth_records = [{
        "case_id": "vt-0001-synth",
        "source_case_id": "vt-0001",
        "source_text": "本品与抗血小板类药物合用需注意。",
        "synthetic_answer": "用药请注意。（补充一句：像阿司匹林、氯吡格雷这类药。）",
        "injected_examples": ["阿司匹林", "氯吡格雷"],
        "red_line_idx": 0,
        "verdict": "fail",
    }]
    out = synthetic_records_to_candidates(synth_records)
    assert len(out) == 2
    texts = {item["candidate_text"] for item in out}
    assert texts == {"阿司匹林", "氯吡格雷"}
    assert all(item["label"] is True for item in out)
    assert all(item["red_line_guess"] == 0 for item in out)
    assert all(item["source_case_id"] == "vt-0001" for item in out)


def test_synthetic_records_to_candidates_carries_answer_and_source():
    synth_records = [{
        "case_id": "c-synthnum",
        "source_case_id": "c",
        "source_text": "复查频率必要时而定。",
        "synthetic_answer": "请注意。（补充一句：一般连续用药不超过7天。）",
        "injected_examples": ["一般连续用药不超过7天"],
        "red_line_idx": 2,
        "verdict": "fail",
    }]
    out = synthetic_records_to_candidates(synth_records)
    assert len(out) == 1
    item = out[0]
    assert item["source_text"] == "复查频率必要时而定。"
    assert item["answer"] == "请注意。（补充一句：一般连续用药不超过7天。）"
    assert item["candidate_text"] == "一般连续用药不超过7天"
    assert item["red_line_guess"] == 2


def test_build_adversarial_subset_reaches_minimum_positive_size():
    """L2 判读规则的门槛：**正例**≥50 条（负例是问题一修复新增的，门槛
    本身没变，仍然只看正例数）。用合成触发密度足够高的一批记录验证
    "能凑够"这件事本身，不断言具体数字（具体数字由真实 holdout 跑出）。"""
    records = [
        _record(f"h{i}", "本品与抗血小板类药物合用需注意，复查频率必要时而定。", "请遵医嘱。")
        for i in range(30)
    ]
    out = build_adversarial_subset(records)
    n_pos = sum(1 for it in out if it["label"] is True)
    assert n_pos >= 50


def test_build_adversarial_subset_contains_both_positive_and_negative_labels():
    """问题一修复后对抗子集不再是纯正例——必须两类都有，否则退化分类器
    （只认括注结构）又能在这份验收集上拿满分。"""
    records = [
        _record("h1", "本品与抗血小板类药物合用需注意。", "请遵医嘱。"),
        # 注入负例要求：药名形实体、**在 source 里**、**不在 answer 里**
        # （候选只出现在括注内，位置特征才在正负例里都出现）。
        _record("h2", "本药物为硝苯地平片，用于降压治疗。", "医生说按时吃就行。"),
    ]
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(25)]
    out = build_adversarial_subset(records, jargon={"硝苯地平"})
    labels = {item["label"] for item in out}
    assert labels == {True, False}


def test_build_adversarial_subset_respects_explicitly_empty_jargon():
    """`/code-review` 发现：`jargon or load_jargon()` 把显式传入的空集合
    当成"没传"，静默换成完整生产词表——调用方如果是故意只测数字类候选
    （传 `jargon=set()`），负例挖矿会意外重新引入词表实体候选。改成
    `is not None` 判断后，空集合必须被尊重：没有词表实体可挖，负例为空。"""
    records = [_record("h1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。")]
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(25)]
    out = build_adversarial_subset(records, jargon=set())
    negatives = [it for it in out if it["label"] is False]
    assert negatives == []


def test_build_adversarial_subset_negatives_are_structurally_decorated_but_grounded():
    """负例必须真的带上括注结构（不然没有打掉捷径的效果），但候选内容
    仍然是可信的（原答案里本来就有、判 pass 的候选）。"""
    records = [_record("h1", "本药物为硝苯地平片，用于降压治疗。", "医生说按时吃就行。")]
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(25)]
    out = build_adversarial_subset(records, jargon={"硝苯地平"})
    negatives = [it for it in out if it["label"] is False]
    assert negatives
    assert any("硝苯地平" in it["candidate_text"] for it in negatives)
    # 负例的候选必须真的落在注入的括注里——这正是要打掉的那条位置捷径
    neg = next(it for it in negatives if it["candidate_text"] == "硝苯地平")
    assert neg["answer"].splitlines()[-1].startswith("（")
    assert "硝苯地平" in neg["answer"].splitlines()[-1]
