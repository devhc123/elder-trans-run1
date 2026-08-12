"""对抗子集构造的测试（ticket 14，L2 判读规则的落地）。

对抗子集是「医学正确但原文未给出」的天然样本——CATEGORY_EXAMPLES/数字注入
本身就是这个模式（真实存在的同类药名/合理频次，只是没出现在这条
source_text 里），不需要额外构造。这里守的是：从合成记录展开到候选级
条目时字段对不对、label 恒为 True（P3 教师抽检 50/50 已确认这些是真违规，
见 ticket 12），以及**只从 holdout 源文本生成，不进训练集**——否则模型
训练时见过这些确切样本，拿它们做验收就是在测记忆力不是泛化能力。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.adversarial_subset import build_adversarial_subset  # noqa: E402
from verifier.synth_minimal_edit import synthetic_records_to_candidates  # noqa: E402


def _record(case_id, source, answer):
    return {"case_id": case_id, "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x"}


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


def test_build_adversarial_subset_reaches_minimum_size():
    """L2 判读规则的门槛：≥50 条。用合成触发密度足够高的一批记录验证
    "能凑够"这件事本身，不断言具体数字（具体数字由真实 holdout 跑出）。"""
    records = [
        _record(f"h{i}", "本品与抗血小板类药物合用需注意，复查频率必要时而定。", "请遵医嘱。")
        for i in range(30)
    ]
    out = build_adversarial_subset(records)
    assert len(out) >= 50


def test_build_adversarial_subset_all_labeled_positive():
    """池子要够大让 synthesize_all 的源文档集中度上限不把这唯一一条正例
    裁到 0（同一坑之前在 test_synth_minimal_edit.py 里踩过一次）。"""
    records = [_record("h1", "本品与抗血小板类药物合用需注意。", "请遵医嘱。")]
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(25)]
    out = build_adversarial_subset(records)
    assert out and all(item["label"] is True for item in out)
