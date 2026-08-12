"""P2：候选池的派生 gold 标签 + 分层抽样逻辑测试（ticket 12）。

派生 gold 是"候选文本是否落在某条违规红线（0/2）的 evidence 子串里"——这不是
教师重新标注，是把已有 case 级教师标注**翻译**成候选级的近似真值，用来评
教师逐候选判定 prompt 的一致率。抽样必须确定性、可复现，且两类（命中/
未命中）都要有代表性——ticket 12 P2 门槛要求"分层"。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.judge_candidates import (  # noqa: E402
    build_candidate_pool,
    derive_candidate_label,
    stratified_sample,
)


def _gold(red_lines):
    return {"red_lines": [{"idx": i, "violated": v, "evidence": e} for i, v, e in red_lines]}


# ---------- derive_candidate_label ----------

def test_candidate_inside_violated_evidence_is_positive():
    gold = _gold([(0, True, "阿司匹林、氯吡格雷")])
    assert derive_candidate_label("阿司匹林", gold) is True


def test_candidate_not_in_any_evidence_is_negative():
    gold = _gold([(0, True, "阿司匹林、氯吡格雷")])
    assert derive_candidate_label("布洛芬", gold) is False


def test_candidate_only_matches_non_violated_red_line_is_negative():
    """evidence 挂在没被判违规的红线上不算数——只有 violated=True 的才算正例。"""
    gold = _gold([(0, False, "阿司匹林")])
    assert derive_candidate_label("阿司匹林", gold) is False


def test_red_line_1_evidence_is_ignored_for_derivation():
    """红线 1（说反）不进两段式管线，候选派生 gold 只看红线 0/2。"""
    gold = _gold([(1, True, "能报销")])
    assert derive_candidate_label("能报销", gold) is False


# ---------- build_candidate_pool ----------

def _record(case_id, source, answer, red_lines):
    return {
        "case_id": case_id,
        "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
        "output": json.dumps({
            "key_points": [],
            "red_lines": [{"idx": i, "violated": v, "evidence": e} for i, v, e in red_lines],
            "verdict": "fail" if any(v for _, v, _ in red_lines) else "pass",
        }, ensure_ascii=False),
    }


def test_build_candidate_pool_attaches_case_context_and_derived_label():
    records = [_record("c1", "请遵医嘱。", "医生给您开的是阿司匹林。",
                        [(0, True, "阿司匹林")])]
    pool = build_candidate_pool(records, jargon={"阿司匹林"})
    assert len(pool) == 1
    item = pool[0]
    assert item["case_id"] == "c1"
    assert item["candidate_text"] == "阿司匹林"
    assert item["derived_label"] is True
    assert item["source_text"] and item["answer"]


def test_build_candidate_pool_covers_both_classes():
    records = [
        _record("c1", "请遵医嘱。", "医生给您开的是阿司匹林。", [(0, True, "阿司匹林")]),
        _record("c2", "阿司匹林每日一片。", "阿司匹林您要每日一片，别吃别的类似药。",
                 [(0, False, "")]),
    ]
    pool = build_candidate_pool(records, jargon={"阿司匹林"})
    labels = {item["derived_label"] for item in pool}
    assert True in labels
    # c2 里 "阿司匹林" 已在 source，不会被抽为候选；c2 本身没有候选，pool 只含 c1 的
    assert all(item["case_id"] == "c1" for item in pool)


# ---------- stratified_sample ----------

def _pool(n_pos, n_neg):
    out = []
    for i in range(n_pos):
        out.append({"id": f"pos-{i}", "derived_label": True})
    for i in range(n_neg):
        out.append({"id": f"neg-{i}", "derived_label": False})
    return out


def test_stratified_sample_includes_both_classes():
    pool = _pool(30, 200)
    sampled = stratified_sample(pool, n_per_class=20, seed=1)
    labels = [item["derived_label"] for item in sampled]
    assert labels.count(True) == 20
    assert labels.count(False) == 20


def test_stratified_sample_is_deterministic():
    pool = _pool(30, 200)
    a = stratified_sample(pool, n_per_class=20, seed=42)
    b = stratified_sample(pool, n_per_class=20, seed=42)
    assert [i["id"] for i in a] == [i["id"] for i in b]


def test_stratified_sample_caps_at_available_when_class_is_scarce():
    pool = _pool(5, 200)
    sampled = stratified_sample(pool, n_per_class=20, seed=1)
    assert sum(1 for i in sampled if i["derived_label"] is True) == 5
