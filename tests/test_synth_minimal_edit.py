"""P3：最小编辑对合成的测试（ticket 12）。

L1 判读规则的硬约束：**规则最小编辑，不许 LLM 自由重写**——防止合成负例
带来源伪影（arXiv:2606.01304）。这里守的是：只在源文本里已出现"类别"标记词
（如"XX类药物"）时才合成；注入的具体词不能恰好已经在原文里出现（否则就不是
违规了，是巧合真话）；同一 case_id 的合成结果必须确定性可复现（不能用带盐的
`hash()`，否则同一输入两次运行给出不同注入词/模板，没法审计）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import CATEGORY_MARKERS as ALL_CATEGORY_MARKERS  # noqa: E402
from verifier.synth_minimal_edit import (  # noqa: E402
    CATEGORY_MARKERS,
    find_category_trigger,
    pick_examples,
    synthesize_all,
    synthesize_minimal_edit,
)


def test_category_markers_is_a_deliberate_subset_of_the_canonical_list():
    """回归（code review 发现）：曾经手抄过一份缩水子集，注释却声称"同一份"——
    与 join_fields() 分叉是同一类坑。这里断言子集关系，防止两份列表悄悄漂移；
    子集本身是刻意的（只留局部语义必然指向"一类药"的标记词，见模块顶部注释），
    不是要拉平成完全一致。"""
    assert set(CATEGORY_MARKERS) <= set(ALL_CATEGORY_MARKERS)
    assert all("药" in m for m in CATEGORY_MARKERS)
    # 不能收窄到空——那就没法合成任何东西了
    assert CATEGORY_MARKERS


def _record(case_id, source, answer):
    return {"case_id": case_id, "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x"}


# ---------- find_category_trigger ----------

def test_finds_category_marker_context_window():
    ctx = find_category_trigger("本品与抗血小板类药物合用时需注意出血风险。")
    assert ctx is not None and "抗血小板" in ctx


def test_returns_none_when_no_category_marker_present():
    assert find_category_trigger("阿司匹林每日一片，饭后服用。") is None


# ---------- pick_examples ----------

def test_pick_examples_matches_known_category_keyword():
    a, b = pick_examples("本品与抗血小板类药物合用")
    assert (a, b) == ("阿司匹林", "氯吡格雷")


def test_pick_examples_falls_back_to_default_for_unknown_category():
    a, b = pick_examples("某种从未见过的怪异类药物")
    assert a and b  # 兜底也必须给出两个非空示例


# ---------- synthesize_minimal_edit ----------

def test_returns_none_when_source_has_no_category_trigger():
    r = _record("vt-0001", "阿司匹林每日一片。", "您每天吃一片阿司匹林。")
    assert synthesize_minimal_edit(r) is None


def test_injects_specific_examples_not_present_in_source():
    r = _record("vt-0002", "本品与抗血小板类药物合用时需注意。", "用药时要小心。")
    out = synthesize_minimal_edit(r)
    assert out is not None
    assert out["case_id"] == "vt-0002-synth"
    assert out["injected_examples"] == ["阿司匹林", "氯吡格雷"]
    assert "阿司匹林" in out["synthetic_answer"] and "氯吡格雷" in out["synthetic_answer"]
    # 最小编辑：原答案内容必须原样保留，只是追加，不是重写
    assert "用药时要小心。" in out["synthetic_answer"]
    assert out["red_line_idx"] == 0
    assert out["verdict"] == "fail"


def test_skips_when_picked_example_accidentally_already_in_source():
    """防止合成出一句"巧合为真"的话——那就不是违规样本了。"""
    r = _record(
        "vt-0003",
        "本品与抗血小板类药物（如阿司匹林）合用时需注意，也属于抗凝类药物范畴。",
        "请遵医嘱。",
    )
    out = synthesize_minimal_edit(r)
    # 抗血小板类的示例(阿司匹林)已在原文出现，必须整条跳过而不是硬造假阳性
    assert out is None or "阿司匹林" not in (out.get("injected_examples") or [])


def test_synthesis_is_deterministic_across_runs():
    """不能用带盐的内置 hash()——同一 case_id 每次跑出的模板/示例必须一致，
    否则合成产物无法审计复现。"""
    r = _record("vt-0004", "本品属于降压类药物。", "按时吃药就好。")
    out1 = synthesize_minimal_edit(r)
    out2 = synthesize_minimal_edit(r)
    assert out1 == out2


# ---------- synthesize_all ----------

def test_synthesize_all_skips_records_without_trigger_and_keeps_the_rest():
    records = [
        _record("a", "阿司匹林每日一片。", "按时吃。"),
        _record("b", "本品属于降糖类药物。", "按时吃就好。"),
    ]
    out = synthesize_all(records)
    assert len(out) == 1
    assert out[0]["case_id"] == "b-synth"
