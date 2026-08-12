"""可信候选池构造器的测试（ticket 14）。

**这是 ticket 14 最核心的设计决定的代码化**：段A规则在 `verdict=pass` 案例
上抽出的候选，天然是真负例（教师对整案例判过零违规，不需要反推）；`verdict
=fail` 案例只取 evidence 命中的候选当正例，**没命中的候选整条丢弃**——不能
当负例用（那是 ticket 12 P2 诊断出的方法论缺陷：单条 evidence 不代表案例里
只有那一个违规词，把"没命中"当"没违规"会把系统性漏标烤进训练数据）。

train 和 holdout 用同一个构造函数——这里守的是这份"可信"规则本身对不对，
不是某个切分的具体数字。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.candidate_pool import build_trusted_candidate_pool  # noqa: E402


def _record(case_id, source, answer, verdict, red_lines=None):
    """red_lines: [(idx, violated, evidence), ...]；默认三条都不违规。"""
    red_lines = red_lines or [(0, False, ""), (1, False, ""), (2, False, "")]
    return {
        "case_id": case_id,
        "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
        "output": json.dumps({
            "key_points": [],
            "red_lines": [{"idx": i, "violated": v, "evidence": e} for i, v, e in red_lines],
            "verdict": verdict,
        }, ensure_ascii=False),
    }


# ---------- pass 案例：所有候选都是可信负例 ----------

def test_pass_case_candidates_are_all_labeled_negative():
    r = _record("p1", "请遵医嘱。", "医生给您开的是阿司匹林。", "pass")
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert len(pool) == 1
    assert pool[0]["label"] is False
    assert pool[0]["case_id"] == "p1"
    assert pool[0]["candidate_text"] == "阿司匹林"


def test_pass_case_with_no_candidates_contributes_nothing():
    r = _record("p2", "阿司匹林每日一片。", "阿司匹林您要每日一片。", "pass")
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert pool == []


# ---------- fail 案例：只有 evidence 命中的候选才是可信正例 ----------

def test_fail_case_candidate_matching_evidence_is_labeled_positive():
    r = _record(
        "f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert len(pool) == 1
    assert pool[0]["label"] is True
    assert pool[0]["candidate_text"] == "阿司匹林"


def test_fail_case_candidate_not_matching_evidence_is_dropped_not_labeled_negative():
    """这是本模块存在的唯一理由：不确定的候选整条丢弃，不能既不算正例
    也硬当负例塞进训练集——那正是 ticket 12 P2 诊断出的方法论缺陷。"""
    r = _record(
        "f2", "请遵医嘱。", "医生给您开的是阿司匹林，另外还有布洛芬。", "fail",
        # evidence 只记了"阿司匹林"，"布洛芬"这个候选没有对应的 evidence
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林", "布洛芬"})
    texts = {item["candidate_text"] for item in pool}
    assert "阿司匹林" in texts  # 命中 evidence，保留且为正例
    assert "布洛芬" not in texts  # 没命中，整条丢弃——不在 pool 里，既非正例也非负例


def test_fail_case_with_no_evidence_matching_candidates_contributes_nothing():
    r = _record(
        "f3", "请遵医嘱。", "医生说了些别的事。", "fail",
        red_lines=[(0, True, "别的事"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert pool == []


# ---------- 红线1（说反）永远不产出可信正例——两段式管线不覆盖它 ----------

def test_red_line_1_evidence_never_produces_a_trusted_positive():
    r = _record(
        "f4", "医保：否。", "医保能报销。", "fail",
        red_lines=[(0, False, ""), (1, True, "能报销"), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert pool == []


# ---------- 混合切分：pass 与 fail 案例并存 ----------

def test_mixed_split_combines_pass_negatives_and_fail_positives():
    records = [
        # "硝苯地平"不在 source，但案例整体判 pass（比如触发了红线0的
        # 主体身份例外条款）——段A仍会把它抽成候选，可信池里标为负例。
        _record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass"),
        _record("f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
                red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")]),
    ]
    pool = build_trusted_candidate_pool(records, jargon={"阿司匹林", "硝苯地平"})
    labels = {item["case_id"]: item["label"] for item in pool}
    assert labels == {"p1": False, "f1": True}


# ---------- 每条候选携带溯源信息，供上下游（held-out 报告/训练格式）使用 ----------

def test_each_candidate_carries_source_context_and_kind():
    r = _record("f5", "血糖不超过112。", "血糖别超过12就好。", "fail",
                red_lines=[(0, False, ""), (1, False, ""), (2, True, "12")])
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert len(pool) == 1
    item = pool[0]
    assert item["source_text"] and item["answer"]
    assert item["kind"] == "digit"
    assert item["red_line_guess"] == 2
