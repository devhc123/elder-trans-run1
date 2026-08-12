"""两段式红线判别段 A（规则候选抽取）的测试（ticket 12 P1）。

段 A 的唯一职责是「别漏」——宁可高误报也不能漏掉 answer 里那些 source 没有的
具体名词/数字。这里守的是：候选类型（词表/数字/中文数词/分数）各自的抽取
正确性，以及「source 里已经有的词/数不算候选」这条精度地板。

**校准纪律**：本文件里用的例句全部来自 `verifier/train.jsonl` 的真实漏报
案例（vt-0138/vt-0160，见 ticket 12 研究笔记）或通用行业常识（药品计量单位、
常见食材），**不引用 holdout 里的任何具体案例文本**——holdout 只在
`verifier/redline_candidates_eval.py` 里评一次，测试文件本身不能变成
"看着 holdout 答案反推规则"的通道。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.redline_candidates import (  # noqa: E402
    FOOD_WORDLIST_PATH,
    JARGON_PATH,
    extract_candidates,
    flag_case,
    load_jargon,
    load_wordlist,
    normalize,
    parse_case_text,
)


# ---------- parse_case_text ----------

def test_parse_case_text_splits_source_and_answer():
    blob = "【原文】\n阿司匹林每日一片。\n\n【回答】\n您每天吃一片就行。\n\n【要点】\n0. xxx"
    src, ans = parse_case_text(blob)
    assert "阿司匹林每日一片" in src
    assert "您每天吃一片就行" in ans
    assert "要点" not in ans


# ---------- normalize ----------

def test_normalize_full_width_to_half_width_and_strips_whitespace():
    assert normalize("４ 公 斤") == "4公斤"


# ---------- load_wordlist / load_jargon ----------

def test_load_wordlist_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("# 注释\n\n阿司匹林\n \n氯吡格雷\n", encoding="utf-8")
    assert load_wordlist(p) == {"阿司匹林", "氯吡格雷"}


def test_load_wordlist_missing_file_returns_empty_set(tmp_path):
    assert load_wordlist(tmp_path / "nope.txt") == set()


def test_load_jargon_merges_medical_and_food_wordlists():
    jargon = load_jargon()
    # 医学词表（长条目）
    assert "阿司匹林" in jargon
    # 食物词表补充（ticket 12 P1 新增，train 切分 vt-0138 漏报驱动）——
    # 常见食材大多是 2 字词（鸡蛋/瘦肉/豆腐），length 过滤不能连它们一起砍掉
    assert "鸡蛋" in jargon and "瘦肉" in jargon and "豆腐" in jargon


def test_load_jargon_length_filter_only_applies_to_the_open_scraped_medical_list():
    """jargon.txt 是 4.3 万条开放爬取词，短条目噪声大，必须过滤；
    食物词表是几十条精选闭表，噪声风险有限，不该被同一把尺子砍。"""
    jargon = load_jargon()
    medical_only = load_wordlist(JARGON_PATH)
    food = load_wordlist(FOOD_WORDLIST_PATH)
    # 短条目里排除掉食物词表刻意重收的那几个（如「鸭肉」恰好两边都有）——
    # 这里测的是「jargon.txt 单独贡献的短条目」有没有被过滤掉，不是全集
    short_medical_only_terms = {w for w in medical_only if len(w) < 3} - food
    assert short_medical_only_terms, "jargon.txt 里应该确实存在短条目，测试前提才成立"
    assert not (short_medical_only_terms & jargon), "jargon.txt 的短条目不该漏过长度过滤"


# ---------- extract_candidates：词表越界 ----------

def test_lexicon_candidate_flagged_when_answer_has_it_and_source_does_not():
    cands = extract_candidates("这是一种抗血小板药物。", "医生给您开的是阿司匹林。",
                                jargon={"阿司匹林"})
    assert any(c["text"] == "阿司匹林" and c["kind"] == "lexicon" for c in cands)


def test_lexicon_candidate_not_flagged_when_source_already_has_it():
    cands = extract_candidates("阿司匹林每日一片。", "您吃的阿司匹林要按时吃。",
                                jargon={"阿司匹林"})
    assert not any(c["text"] == "阿司匹林" for c in cands)


# ---------- extract_candidates：阿拉伯数字越界 ----------

def test_digit_candidate_flagged_when_not_in_source():
    cands = extract_candidates("请遵医嘱服用。", "血压降到90以下就要停药。", jargon=set())
    assert any(c["text"] == "90" and c["kind"] == "digit" and c["red_line_guess"] == 2
               for c in cands)


def test_digit_candidate_not_flagged_when_already_in_source():
    cands = extract_candidates("一日2次，每次1片。", "这个药一日2次，一次1片。", jargon=set())
    assert not any(c["kind"] == "digit" for c in cands)


def test_digit_candidate_flagged_even_when_it_is_a_substring_of_an_unrelated_source_number():
    """回归：`"12" in "112"` 为真，但 12 ≠ 112——朴素子串判定会把一个真正
    编造的数字误判成"原文已有"，漏掉一个红线2违规（编造的具体数字）。"""
    cands = extract_candidates("血糖控制目标是空腹不超过112。", "医生说您血糖别超过12就好。",
                                jargon=set())
    assert any(c["text"] == "12" and c["kind"] == "digit" for c in cands)


def test_cn_numeral_candidate_flagged_even_when_it_is_a_substring_of_an_unrelated_source_numeral():
    """同一个 bug 也适用于中文数词：source"十三天"里含"三天"这个子串，
    但 answer 说的"三天"和 source 的"十三天"是两个不同的时长。"""
    cands = extract_candidates("疗程一般是十三天。", "疗程大概三天左右。", jargon=set())
    assert any(c["kind"] == "cn_numeral" and c["text"] == "三天" for c in cands)


# ---------- extract_candidates：中文数词+单位 ----------

@pytest.mark.parametrize("phrase", ["两三个礼拜", "三天", "一两个小时", "半年", "一个星期"])
def test_cn_numeral_with_unit_is_flagged_when_not_in_source(phrase):
    """例句来自 train 切分真实漏报 vt-0160「两三个礼拜」及同类时长表达。"""
    cands = extract_candidates("请遵医嘱复查。", f"大概要{phrase}才能看到效果。", jargon=set())
    assert any(c["kind"] == "cn_numeral" and phrase in c["text"] for c in cands)


def test_cn_numeral_not_flagged_when_already_in_source():
    cands = extract_candidates("疗程一般是三天。", "疗程大概三天左右。", jargon=set())
    assert not any(c["kind"] == "cn_numeral" for c in cands)


def test_bare_numeral_without_unit_is_not_flagged():
    """"一下""一定"这类常见虚词不该被当成数量候选——必须跟着单位词才算。"""
    cands = extract_candidates("请遵医嘱。", "您一定要按时吃药，我再说一下。", jargon=set())
    assert not any(c["kind"] == "cn_numeral" for c in cands)


# ---------- extract_candidates：分数/比例 ----------

def test_fraction_pattern_is_flagged_and_not_double_counted_as_bare_numeral():
    cands = extract_candidates("发生率很低。", "发生的概率大概是三千分之一。", jargon=set())
    fraction = [c for c in cands if c["kind"] == "fraction"]
    assert any(c["text"] == "三千分之一" for c in fraction)
    # 分数不该被 cn_numeral 规则再拆出一个错位的"三千分"之类的碎片
    assert not any(c["kind"] == "cn_numeral" and c["text"] == "三千分" for c in cands)


# ---------- extract_candidates：去重与多类型共存 ----------

def test_candidates_are_deduplicated_by_text():
    cands = extract_candidates("请遵医嘱。", "阿司匹林、阿司匹林都可以。", jargon={"阿司匹林"})
    assert sum(1 for c in cands if c["text"] == "阿司匹林") == 1


def test_multiple_candidate_kinds_can_coexist_in_one_case():
    cands = extract_candidates(
        "请遵医嘱服用抗血小板药物。",
        "医生说吃阿司匹林，一天90毫克，大概吃三个月。",
        jargon={"阿司匹林"},
    )
    kinds = {c["kind"] for c in cands}
    assert "lexicon" in kinds
    assert "digit" in kinds
    assert "cn_numeral" in kinds


# ---------- flag_case ----------

def test_flag_case_true_when_any_candidate_exists():
    assert flag_case("这是一种抗血小板药物。", "医生给您开的是阿司匹林。", jargon={"阿司匹林"})


def test_flag_case_false_when_answer_stays_within_source():
    assert not flag_case("阿司匹林每日一片。", "阿司匹林您要每日一片吃。", jargon={"阿司匹林"})
