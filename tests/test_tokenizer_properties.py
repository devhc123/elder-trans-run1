"""术语感知分词器的性质测试。

这里守的是**不变量**，不是具体样例。跨度重切算法（在原文上扫术语跨度，再把
jieba 的切分结果按跨度边界重切）最容易出的错是丢字、重字或错序——一旦发生，
达标率的分母就是错的，而错得静默、看分数看不出来。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.readability import DATETIME, Lexicon, _jargon_spans, _segment, score  # noqa: E402


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    try:
        return Lexicon.load()
    except FileNotFoundError as e:
        pytest.skip(f"词表未构建，先跑 pipeline/build_lexicon.py：{e}")


EDGE_CASES = [
    "",
    "心",
    "心肌梗死",                                   # 整串就是一个术语
    "aaa心肌梗死",                                # 术语不在起点
    "心肌梗死bbb",                                # 术语贴到结尾
    "预激综合征者禁用本品。",
    "血清肌酐、尿素氮及内生肌酐清除率提示肾小球滤过功能减退。",
    "糖尿病（血糖高的病）要控制，HbA1c 偏高，每次5毫克。",
    "，。、；：（）",
    " ",
]


def _random_texts(lex: Lexicon, n: int = 800, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    jar = sorted(lex.jargon)
    pool = "的了在和是我你他这那不有一二三，。、；：（）"
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(0, 8)):
            if rng.random() < 0.5:
                parts.append(rng.choice(jar))
            else:
                parts.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 4))))
        out.append("".join(parts))
    return out


def test_segmentation_is_lossless_on_edge_cases(lex):
    for t in EDGE_CASES:
        masked = DATETIME.sub(" ", t)
        toks, _ = _segment(masked, lex.jargon)
        assert "".join(toks) == masked, f"分词有损：{t!r}"


def test_segmentation_is_lossless_on_random_texts(lex):
    """拼回去必须逐字等于输入——丢字/重字/错序都会在这里暴露。"""
    for t in _random_texts(lex):
        masked = DATETIME.sub(" ", t)
        toks, _ = _segment(masked, lex.jargon)
        assert "".join(toks) == masked, f"分词有损：{t!r}"


def test_jargon_spans_are_ordered_and_non_overlapping(lex):
    for t in EDGE_CASES + _random_texts(lex, n=200, seed=11):
        spans = _jargon_spans(t, lex.jargon)
        prev_end = -1
        for a, b in spans:
            assert 0 <= a < b <= len(t)
            assert a >= prev_end, f"跨度重叠或乱序：{spans} on {t!r}"
            prev_end = b


def test_jargon_spans_match_actual_substrings(lex):
    for t in _random_texts(lex, n=200, seed=13):
        for a, b in _jargon_spans(t, lex.jargon):
            assert t[a:b] in lex.jargon


def test_longest_match_wins(lex):
    """'心肌梗死' 应整体命中，不得先命中更短的 '心肌'。"""
    if "心肌梗死" not in lex.jargon:
        pytest.skip("语料未收录该术语")
    spans = _jargon_spans("心肌梗死", lex.jargon)
    assert spans == [(0, 4)]


def test_counts_never_exceed_denominator(lex):
    """三个口径的达标数都不得超过计分词总数，failures 与 strict 必须自洽。"""
    for t in EDGE_CASES + _random_texts(lex, n=300, seed=17):
        r = score(t, lex)
        assert 0 <= r.n_pass_strict <= r.n_scored
        assert 0 <= r.n_pass_lenient <= r.n_scored
        assert 0 <= r.n_pass_glossed <= r.n_scored
        assert r.n_pass_strict + len(r.failures) == r.n_scored
        for rate in (r.rate_strict, r.rate_lenient, r.rate_glossed):
            assert 0.0 <= rate <= 1.0


def test_glossed_is_never_below_strict(lex):
    """glossed 只放宽不收紧，任何输入下都不得低于 strict。"""
    for t in EDGE_CASES + _random_texts(lex, n=300, seed=19):
        r = score(t, lex)
        assert r.rate_glossed >= r.rate_strict, f"glossed < strict on {t!r}"
