"""可读性打分器测试。

分两层：
  - 单元测试用合成词表，判定逻辑可控、断言精确；
  - 集成测试用 lexicon/ 下的真实官方词表，验证端到端行为与区分度。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.readability import Lexicon, score  # noqa: E402


@pytest.fixture
def tiny() -> Lexicon:
    """合成词表：只认少数几个字，便于精确断言。"""
    return Lexicon(
        grade6_chars=set("我今天吃了两片药身体好多谢医生你的话很有用水"),
        ext_chars=set("糖尿"),
        jargon={"糖尿病", "心肌梗死", "地高辛"},
        units={"片", "毫克"},
        health_common={"医生", "身体"},
        hsk_words=set(),
    )


# ---------- 计分词的界定 ----------

def test_punctuation_digits_excluded_from_denominator(tiny):
    r = score("我今天吃了2片药。", tiny)
    # "2" 与 "。" 不计入分母
    assert "2" not in [f.word for f in r.failures]
    assert all(not w.isdigit() for w in r.scored_words)
    assert "。" not in r.scored_words


def test_empty_text_is_not_a_crash_and_not_a_pass(tiny):
    r = score("", tiny)
    assert r.n_scored == 0
    assert r.rate_strict == 0.0


def test_whitespace_and_symbols_only(tiny):
    r = score("   ，。！ 123 45% ", tiny)
    assert r.n_scored == 0


@pytest.mark.parametrize(
    "junk",
    [
        "，。！？；：、",          # 中文标点
        "“”‘’（）《》【】",        # 中文引号括号书名号
        ",.!?;:'\"()[]{}",         # 西文标点
        "—…·~|/\\+*=<>",          # 连接号省略号与符号
        "2024年3月1日",            # 日期
        "12:30",                   # 时间
        "45%",                     # 百分比
        "3.14",                    # 小数
        "1-2",                     # 数值区间
    ],
)
def test_non_word_tokens_never_enter_denominator(tiny, junk):
    """回归：早期版本用手写标点字符类，字符串被 ASCII 引号提前截断导致字符类失效。

    改为按字符构成判定（含汉字或以拉丁字母开头才计分），这里覆盖各类符号。
    """
    r = score(junk, tiny)
    assert r.n_scored == 0, f"{junk!r} 不应计入分母，实得 {r.scored_words}"


def test_module_imports_without_syntax_warning():
    """回归：正则字符串曾因引号截断产生 SyntaxWarning，属真实语法缺陷。"""
    import importlib
    import warnings

    import metrics.readability as m

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(m)
    assert not [w for w in caught if issubclass(w.category, SyntaxWarning)]


# ---------- 白名单 ----------

def test_unit_whitelist_passes_even_if_chars_unknown(tiny):
    r = score("每次吃5毫克", tiny)
    assert "毫克" not in [f.word for f in r.failures]


def test_health_common_whitelist_passes(tiny):
    r = score("医生说身体好", tiny)
    for w in ("医生", "身体"):
        assert w not in [f.word for f in r.failures]


# ---------- jargon 不豁免 ----------

def test_jargon_fails_even_when_all_chars_are_common(tiny):
    """医学实体不豁免：即使每个字都在六年级字表里，术语整体仍判不达标。"""
    lex = Lexicon(
        grade6_chars=set("糖尿病"),
        ext_chars=set(),
        jargon={"糖尿病"},
        units=set(),
        health_common=set(),
        hsk_words=set(),
    )
    r = score("糖尿病", lex)
    assert r.rate_strict == 0.0
    assert r.failures[0].reason == "jargon"


def test_unknown_char_fails_with_reason(tiny):
    r = score("心肌梗死", tiny)
    reasons = {f.reason for f in r.failures}
    assert reasons  # 必有失败原因，不允许静默失败
    assert all(f.word for f in r.failures)


# ---------- 三个口径 ----------

def test_lenient_3500_is_never_stricter_than_strict(tiny):
    r = score("我今天吃了两片药，糖尿病要控制", tiny)
    assert r.rate_lenient >= r.rate_strict


def test_glossed_rescues_a_glossed_term(tiny):
    """术语后紧跟 ≤15 字白话括注，glossed 口径记达标，strict 仍不达标。"""
    plain = score("糖尿病要控制", tiny)
    glossed = score("糖尿病（血糖高的病）要控制", tiny)
    assert glossed.rate_glossed > plain.rate_glossed
    assert glossed.rate_strict <= glossed.rate_glossed


def test_glossed_does_not_rescue_overlong_parenthetical(tiny):
    """括注过长（>15字）不算有效解释——那是又一段专业文本。"""
    short = score("糖尿病（血糖高的病）", tiny)
    long_ = score("糖尿病（一种以慢性高血糖为特征的代谢性疾病并伴随多系统损害）", tiny)
    assert short.rate_glossed > long_.rate_glossed


# ---------- 拉丁 token ----------

def test_latin_token_fails_unless_whitelisted(tiny):
    r = score("HbA1c 偏高", tiny)
    assert "HbA1c" in [f.word for f in r.failures]
    assert any(f.reason == "latin" for f in r.failures)


# ---------- 可追溯性（本指标相对 FKGL/LENS 的核心优势）----------

def test_every_failure_has_word_and_reason(tiny):
    r = score("心肌梗死伴随地高辛中毒", tiny)
    assert r.failures
    for f in r.failures:
        assert f.word and f.reason
        assert f.reason in {"jargon", "char", "latin"}


def test_pass_count_and_rate_are_consistent(tiny):
    r = score("我今天吃了两片药，医生说身体好多了", tiny)
    assert r.n_scored == len(r.scored_words)
    assert r.n_pass_strict + len(r.failures) == r.n_scored
    assert r.rate_strict == pytest.approx(r.n_pass_strict / r.n_scored)


# ---------- 辅助指标 ----------

def test_reports_length_and_sentence_stats(tiny):
    r = score("我今天吃药。医生说好。", tiny)
    assert r.n_chars > 0
    assert r.n_sentences == 2
    assert r.avg_sentence_len > 0


def test_jargon_density_counts_per_100_words(tiny):
    dense = score("糖尿病心肌梗死地高辛", tiny)
    sparse = score("我今天吃了两片药身体好多了谢谢医生你的话很有用", tiny)
    assert dense.jargon_density > sparse.jargon_density


# ---------- 确定性 ----------

def test_deterministic(tiny):
    t = "我今天吃了两片药，糖尿病（血糖高的病）要控制，HbA1c 偏高"
    a, b = score(t, tiny), score(t, tiny)
    assert a.as_dict() == b.as_dict()


# ---------- 集成：真实官方词表 ----------

@pytest.fixture(scope="module")
def real() -> Lexicon:
    try:
        return Lexicon.load()
    except FileNotFoundError as e:
        pytest.skip(f"词表未构建，先跑 pipeline/build_lexicon.py：{e}")


def test_official_grade6_table_is_exactly_2500(real):
    """课标常用字表一 = 2500 字，是本项目六年级基准的官方依据。"""
    assert len(real.grade6_chars) == 2500


def test_plain_elderly_text_scores_high(real):
    t = "这个药一天吃两次，早上一次晚上一次，饭后吃。要是忘了吃，想起来就补上。"
    r = score(t, real)
    assert r.rate_strict >= 0.85, f"日常口语文本不该低于 0.85，实得 {r.rate_strict}: {[f.word for f in r.failures]}"


def test_jargon_dense_source_text_scores_low(real):
    t = "对任何强心苷制剂中毒者禁用。室性心动过速、心室颤动、梗阻性肥厚型心肌病患者禁用。"
    r = score(t, real)
    assert r.rate_strict <= 0.75, f"术语密集原文不该高于 0.75，实得 {r.rate_strict}"


def test_discrimination_gap_is_wide(real):
    """区分度是指标有效性的前提：没有这个差距，指标就是坏的。"""
    plain = score("这个药一天吃两次，早上一次晚上一次，饭后吃。", real)
    jargon = score("室性心动过速、心室颤动、梗阻性肥厚型心肌病者禁用本品。", real)
    assert plain.rate_strict - jargon.rate_strict >= 0.20, (
        f"区分度不足：通俗 {plain.rate_strict:.3f} vs 术语 {jargon.rate_strict:.3f}"
    )


@pytest.mark.parametrize(
    "text,must_keep",
    [
        ("3个月后复查", "月"),      # 疗程，不是日期
        ("1日3次，每次2片", "日"),  # 用法用量，不是日期
        ("连续吃7天", "天"),
        ("每年体检一次", "年"),
    ],
)
def test_datetime_mask_does_not_eat_dosage_expressions(tiny, text, must_keep):
    """日期掩码只吃完整日期/钟点，不得误伤剂量与疗程表达。"""
    r = score(text, tiny)
    assert must_keep in "".join(r.scored_words), f"{text!r} 中的 {must_keep!r} 被误掩，实得 {r.scored_words}"
