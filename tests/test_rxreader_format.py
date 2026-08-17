"""rxreader 三处统一（训练目标 / 线上解析 / hint 注入）的格式往返测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.build_rxreader_data import format_target, parse_target  # noqa: E402
from app.rxreader_label import snap  # noqa: E402
from app.translate import build_hint_block  # noqa: E402


def test_roundtrip_both():
    t = format_target(["每次1片", "每8小时一次"], ["肠溶片", "INR"])
    assert t == "保留: 每次1片、每8小时一次\n解释: 肠溶片、INR"
    assert parse_target(t) == (["每次1片", "每8小时一次"], ["肠溶片", "INR"])


def test_roundtrip_empty_and_half():
    assert format_target([], []) == "-" and parse_target("-") == ([], [])
    t = format_target([], ["INR"])
    assert parse_target(t) == ([], ["INR"])
    assert parse_target("保留：a，b\n解释：-") == (["a", "b"], [])   # 全角标点也认


def test_snap_substring_and_fuzzy():
    text = "阿司匹林肠溶片 100mg 每日一次 早餐前空腹口服"
    assert snap("每日一次", text) == "每日一次"
    assert snap("每日一次口服", text) is None or snap("每日一次口服", text) in text
    assert snap("阿斯匹林肠溶片", text) == "阿司匹林肠溶片"   # 一字之差吸附回原文
    assert snap("华法林", text) is None


def test_hint_block_empty_means_no_injection():
    assert build_hint_block({"keep": [], "explain": []}) == ""
    assert build_hint_block(None) == ""
    b = build_hint_block({"keep": ["每次1片"], "explain": ["INR"]})
    assert "每次1片" in b and "INR" in b and "绝不能在回复中提及" in b
