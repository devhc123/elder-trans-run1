"""FAQ 归类与题数配比的测试（ticket 04）。

配比是要拿去答辩的材料，所以断言集中在**可核销性**上：总数对得上、
每类有保底、8 个父类一个不缺、剔除规则不误伤老年问题。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.classify_faq import (  # noqa: E402
    FLOOR_PER_SCENARIO,
    SCENARIOS,
    TOTAL_ITEMS,
    allocate,
    classify,
    is_elder_specific,
)

RESULT = ROOT / "docs" / "faq_classification.json"

# 项目书第六栏逐字列举的 8 个父类。这份名单不得改动——改它要走流程。
PROJECT_PARENTS = {
    "检验报告解读", "医嘱转译", "用药说明", "健康科普",
    "权益匹配", "分科医生", "复诊提醒", "用药干预",
}


# ---------- 场景体系 ----------

def test_parents_exactly_match_the_project_document():
    assert set(SCENARIOS.values()) == PROJECT_PARENTS


def test_every_parent_has_at_least_one_subscenario():
    covered = set(SCENARIOS.values())
    missing = PROJECT_PARENTS - covered
    assert not missing, f"父类无子场景，指标说明核销会缺项：{missing}"


def test_subscenario_count_meets_kpi_target():
    """KPI-2 目标 10、挑战档 ≥9。"""
    assert len(SCENARIOS) >= 10


# ---------- 分配 ----------

def test_allocation_sums_exactly_to_total():
    """余数分配不能把总数弄漂——270 是写进 kpi.yaml 的数。"""
    for freqs in (
        {},
        {"日常照护": 100},
        {s: i * 37 for i, s in enumerate(SCENARIOS)},
    ):
        assert sum(allocate(freqs).values()) == TOTAL_ITEMS


def test_every_scenario_gets_at_least_the_floor():
    """权益匹配、医嘱转译频次为 0，但项目书列了它们，不能空着。"""
    alloc = allocate({"日常照护": 9999})
    for s, n in alloc.items():
        assert n >= FLOOR_PER_SCENARIO, f"{s} 只分到 {n} 题"


def test_higher_frequency_gets_more_items():
    alloc = allocate({"日常照护": 5000, "科普辟谣": 100})
    assert alloc["日常照护"] > alloc["科普辟谣"]


# ---------- 归类规则 ----------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("小孩半夜哭闹怎么办？", "_排除_儿科"),
        ("孕妇能吃感冒药吗？", "_排除_孕产"),
        ("打开我的健康档案", "_排除_产品功能"),
        ("你好。", "_排除_无效"),
    ],
)
def test_exclusions(text, expected):
    assert classify(text)[0] == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("血糖多少算正常范围？", "检验报告解读"),
        ("忘记吃降压药怎么办？", "用药干预"),
        ("胃药饭前吃还是饭后吃？", "用药咨询"),
        ("喝醋软化血管有科学依据吗？", "科普辟谣"),
        ("异物卡喉时怎么急救？", "急症处置"),
        ("体检发现肺部结节挂什么科？", "分科导诊"),
        ("老年人失眠怎么改善？", "运动与养生"),
        ("冠心病患者饮食要注意什么？", "饮食营养"),
        ("老人关节疼痛怎么缓解？", "日常照护"),
    ],
)
def test_representative_classifications(text, expected):
    """真实日志里的代表性提问，归类不得漂。"""
    assert classify(text)[0] == expected


def test_no_elder_question_is_excluded():
    """剔除规则只该打掉儿科/孕产/产品功能，绝不能误伤老年提问。"""
    for text in (
        "老年人适合吃哪些保健品？",
        "老人发烧到多少度需要就医？",
        "老人记忆力下降吃什么好？",
        "老年人怎么预防骨质疏松？",
    ):
        assert not classify(text)[0].startswith("_排除_"), text


# ---------- 老年特异判定（敏感性检验的基础） ----------

@pytest.mark.parametrize(
    "text,elder",
    [
        ("老年人失眠怎么改善？", True),
        ("高血压患者怎么运动？", True),
        ("糖尿病患者能吃水果吗？", True),
        ("出差住酒店皮肤痒怎么办？", False),
        ("办公室空调吹得头痛怎么办？", False),
    ],
)
def test_elder_specificity(text, elder):
    assert is_elder_specific(text) is elder


# ---------- 落盘结果 ----------

@pytest.fixture(scope="module")
def result():
    if not RESULT.exists():
        pytest.skip("先跑 pipeline/classify_faq.py")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_allocation_is_consistent(result):
    assert sum(result["allocation"].values()) == TOTAL_ITEMS
    assert set(result["allocation"]) == set(SCENARIOS)


def test_result_uses_elder_subset_as_basis(result):
    """主依据必须是老年子集——全量日志人群不对，见 AUTHORING_SPEC §2.2。"""
    assert result["allocation_basis"] == "elder_subset"
    assert result["allocation"] != result["full_log_allocation_reference"], (
        "两套配比若完全相同，说明敏感性检验没起作用"
    )


def test_every_faq_item_is_accounted_for(result):
    """150 条一条不落，全部有归属（保留或剔除），可逐条核。"""
    assert len(result["kept"]) + len(result["dropped"]) == result["total_items"] == 150
    for c in result["kept"] + result["dropped"]:
        assert c["text"] and c["scenario"] and "matched" in c


def test_disclosed_elder_subset_is_small_enough_to_warrant_disclosure(result):
    """老年子集偏小是本配比最大的不确定性，文档必须写明。

    这条测试的意义是：如果将来样本变大了，提醒回去更新 AUTHORING_SPEC §5 的表述。
    """
    assert result["elder_subset_n"] < len(result["kept"]) * 0.5
    spec = (ROOT / "docs" / "AUTHORING_SPEC.md").read_text(encoding="utf-8")
    assert str(result["elder_subset_n"]) in spec, "老年子集规模未写进 AUTHORING_SPEC"
