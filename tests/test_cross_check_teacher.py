"""DeepSeek 第二意见抽查的测试（用户要求：除教师标注外，抽样找一个不同来源的意见）。

**这不是独立验证。** DeepSeek 既是被测系统（写 answer 的那个模型），又在这里被
拿来当第二判官——同厂自证有系统性偏松或偏盲的风险，kpi.yaml 的判官/被测异厂
解耦纪律在这里刻意破例。这里测的是纯逻辑（解析、比对），不测网络调用本身。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.cross_check_teacher import build_prompt, compare, parse_judgment  # noqa: E402


# ---------- parse_judgment：容错解析 ----------

def test_parse_judgment_accepts_plain_json():
    j = parse_judgment('{"verdict": "pass", "key_points": [], "red_lines": []}')
    assert j["verdict"] == "pass"


def test_parse_judgment_strips_markdown_fence():
    j = parse_judgment('```json\n{"verdict": "fail", "key_points": [], "red_lines": []}\n```')
    assert j["verdict"] == "fail"


def test_parse_judgment_returns_none_on_garbage():
    assert parse_judgment("这不是 JSON，我拒绝回答") is None


# ---------- compare：逐点比对教师标注与第二意见 ----------

def _teacher():
    return {
        "case_id": "vt-0001",
        "key_points": [{"idx": 0, "covered": True, "evidence": "a"},
                       {"idx": 1, "covered": False, "evidence": ""}],
        "red_lines": [{"idx": 0, "violated": True, "evidence": "阿司匹林"},
                      {"idx": 1, "violated": False, "evidence": ""}],
        "verdict": "fail",
    }


def test_compare_finds_no_disagreement_when_identical():
    second = {
        "key_points": [{"idx": 0, "covered": True}, {"idx": 1, "covered": False}],
        "red_lines": [{"idx": 0, "violated": True}, {"idx": 1, "violated": False}],
        "verdict": "fail",
    }
    r = compare(_teacher(), second)
    assert r["disagreements"] == []
    assert r["verdict_match"] is True
    assert r["parse_failed"] is False


def test_compare_flags_key_point_disagreement():
    second = {
        "key_points": [{"idx": 0, "covered": False}, {"idx": 1, "covered": False}],
        "red_lines": [{"idx": 0, "violated": True}, {"idx": 1, "violated": False}],
        "verdict": "fail",
    }
    r = compare(_teacher(), second)
    assert len(r["disagreements"]) == 1
    assert r["disagreements"][0]["field"] == "key_point"
    assert r["disagreements"][0]["idx"] == 0


def test_compare_flags_red_line_disagreement():
    second = {
        "key_points": [{"idx": 0, "covered": True}, {"idx": 1, "covered": False}],
        "red_lines": [{"idx": 0, "violated": False}, {"idx": 1, "violated": False}],
        "verdict": "pass",
    }
    r = compare(_teacher(), second)
    assert any(d["field"] == "red_line" and d["idx"] == 0 for d in r["disagreements"])
    assert r["verdict_match"] is False


def test_compare_handles_unparseable_second_opinion():
    r = compare(_teacher(), None)
    assert r["parse_failed"] is True
    assert r["disagreements"] == []
    assert r["verdict_match"] is False


def test_compare_ignores_missing_indices_rather_than_crashing():
    """第二意见条数对不上教师标注时不能崩，缺的按"没有可比信息"处理。"""
    second = {"key_points": [{"idx": 0, "covered": True}], "red_lines": [], "verdict": "fail"}
    r = compare(_teacher(), second)
    assert isinstance(r["disagreements"], list)


# ---------- build_prompt ----------

def test_build_prompt_contains_all_four_sections():
    c = {
        "source_text": "原文内容",
        "answer": "回答内容",
        "key_points": ["要点一"],
        "red_lines": ["红线一"],
    }
    p = build_prompt(c)
    for part in ("原文", "回答", "要点", "红线", "原文内容", "回答内容"):
        assert part in p
