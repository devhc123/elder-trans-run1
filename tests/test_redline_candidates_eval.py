"""P1 冻结评测脚本的计分逻辑测试（ticket 12）。

用合成 fixture，不碰真实 holdout——这里守的是 TP/FN/FP/TN 计数与门槛判定
本身对不对，真实召回数字由 `redline_candidates_eval.py` 单次跑出、写进
研究笔记，不在测试里断言具体数值（那会诱使日后为了让测试变绿去调 wordlist，
违反"只评一次"的纪律）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.redline_candidates_eval import GATE_RECALL_MIN, instance_level_score, report  # noqa: E402


def _record(case_id, source, answer, red_lines):
    """red_lines: [(idx, violated, evidence), ...]"""
    return {
        "case_id": case_id,
        "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
        "output": json.dumps({
            "key_points": [],
            "red_lines": [{"idx": i, "violated": v, "evidence": e} for i, v, e in red_lines],
            "verdict": "fail" if any(v for _, v, _ in red_lines) else "pass",
        }, ensure_ascii=False),
    }


def test_true_positive_when_gold_violation_and_candidate_found():
    records = [_record("c1", "请遵医嘱。", "医生给您开的是阿司匹林。",
                        [(0, True, "阿司匹林")])]
    m = instance_level_score(records, jargon={"阿司匹林"})
    assert m["tp"] == 1 and m["fn"] == 0


def test_false_negative_when_gold_violation_but_no_candidate_found():
    """段 A 的词表/正则没覆盖到的违规模式——如实记漏报，不能悄悄算对。"""
    records = [_record("c2", "请遵医嘱。", "老人家自己感觉不太舒服。",
                        [(0, True, "自己感觉不太舒服")])]
    m = instance_level_score(records, jargon=set())
    assert m["fn"] == 1 and m["tp"] == 0
    assert m["fn_detail"] == [("c2", 0, "自己感觉不太舒服")]


def test_false_positive_when_no_violation_but_candidate_flagged():
    records = [_record("c3", "请遵医嘱。", "医生给您开的是阿司匹林。",
                        [(0, False, "")])]
    m = instance_level_score(records, jargon={"阿司匹林"})
    assert m["fp"] == 1 and m["tn"] == 0


def test_true_negative_when_no_violation_and_no_candidate():
    records = [_record("c4", "阿司匹林每日一片。", "阿司匹林您要每日一片。",
                        [(0, False, "")])]
    m = instance_level_score(records, jargon={"阿司匹林"})
    assert m["tn"] == 1 and m["fp"] == 0


def test_red_line_1_is_ignored_entirely():
    """红线 1（说反）全池仅 2 例，两段式管线不覆盖——不该被计入 P1 的分子分母。"""
    records = [_record("c5", "医保：否。", "医保能报销。", [(1, True, "医保能报销")])]
    m = instance_level_score(records, jargon=set())
    assert m["tp"] == m["fn"] == m["fp"] == m["tn"] == 0


def test_gate_math_matches_the_frozen_95_percent_recall_threshold():
    assert GATE_RECALL_MIN == 0.95


def test_report_pass_fail_matches_miss_rate_vs_gate(capsys):
    passing = {"tp": 96, "fn": 4, "fp": 0, "tn": 100, "miss_rate": 0.04,
               "false_alarm_rate": 0.0, "fn_detail": []}
    assert report("x", passing) is True

    failing = {"tp": 80, "fn": 20, "fp": 0, "tn": 100, "miss_rate": 0.20,
               "false_alarm_rate": 0.0, "fn_detail": []}
    assert report("y", failing) is False
