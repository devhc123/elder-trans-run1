"""code review（high，34 条核实后收敛为 10 个缺陷）的逐条回归。

每个测试对应一条 finding，用 review 给出的**原始复现输入**，避免修好后又退回去。
这些缺陷的共同特征是**静默抬高 KPI**：不报错、不崩，只是把分数悄悄打高。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.readability import (  # noqa: E402
    DATETIME,
    MAX_GLOSS_LEN,
    Lexicon,
    _gloss_anchors,
    _jargon_spans,
    _judge,
    score,
)


@pytest.fixture(scope="module")
def lex() -> Lexicon:
    try:
        return Lexicon.load()
    except FileNotFoundError as e:
        pytest.skip(f"词表未构建：{e}")


# --- F1：语料缺失时不得写出空 jargon.txt ---------------------------------

def test_f1_build_fails_loudly_when_corpus_missing(tmp_path):
    """语料不在时必须非零退出，绝不能把 jargon.txt 截断成空表。

    空表会让术语判定整体失效：未转译的术语密集原文被打到 ~0.9 并报告为达标，
    区分度从 0.313 塌到 0.114。
    """
    r = subprocess.run(
        [sys.executable, "pipeline/build_lexicon.py"],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "ELDER_CORPUS": str(tmp_path / "nope"), "HOME": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert r.returncode != 0, "语料缺失却退出码为 0"
    assert "BUILD FAILED" in r.stderr or "语料目录不存在" in r.stderr


def test_f1_loading_empty_jargon_is_rejected():
    """即便 jargon.txt 已被写坏，载入时也要拦下来。"""
    lex = Lexicon(
        grade6_chars={chr(0x4E00 + i) for i in range(2500)},
        ext_chars={chr(0x9000 + i) for i in range(1000)},
        jargon=set(),
        units=set(),
        health_common=set(),
    )
    with pytest.raises(ValueError, match="jargon"):
        lex.check_integrity()


# --- F2：括注不得给文档别处同名/同后缀的术语放行 -------------------------

def test_f2_gloss_credits_only_the_adjacent_occurrence(lex):
    """同一个术语出现两次、只有一次带括注时，只有带括注的那次该被记达标。

    原实现把括注前整段汉字及其**每一个后缀**登记为"已解释"，于是文档里任何位置
    的同名/同后缀术语都被放行——等于加一个括注就能给它从未解释过的术语放行。
    """
    if "心肌梗死" not in lex.jargon:
        pytest.skip("语料未收录该术语")
    r = score("心肌梗死（心脏病发作）很危险。他又一次心肌梗死。", lex)
    rescued = r.n_pass_glossed - r.n_pass_strict
    assert rescued == 1, f"应只救紧贴括注的那一次，实际救了 {rescued} 个"


def test_f2_gloss_does_not_credit_unrelated_suffix_terms(lex):
    """review 原始复现：前句括注不得让后句一个毫无解释的术语过关。"""
    t = "他得了慢性病（长期的病）。医院还查出高脂血症。"
    r = score(t, lex)
    rescued = r.n_pass_glossed - r.n_pass_strict
    assert rescued <= 1, f"括注放行了别处未解释的词，救了 {rescued} 个"


def test_f2_gloss_anchor_is_positional_not_lexical(lex):
    """锚点是偏移集合，不是词面集合——后缀不该被登记。"""
    anchors = _gloss_anchors("他得了慢性病（长期的病）。医院还查出性病。")
    assert all(isinstance(a, int) for a in anchors)
    assert len(anchors) == 1


# --- F3：加白话解释不得反而掉分 -------------------------------------------

def test_f3_adding_a_plain_language_gloss_does_not_lower_the_score(lex):
    """指标必须奖励它要推广的转译行为，不能惩罚。

    原缺陷：口语说法（"拍片子"）被从别名列收进 jargon，于是解释术语反而多出一个
    不达标词。
    """
    bare = score("做了核磁共振检查", lex)
    glossed = score("做了核磁共振（一种拍片子）检查", lex)
    assert glossed.rate_strict >= bare.rate_strict, (
        f"加白话解释反而掉分：{bare.rate_strict:.3f} -> {glossed.rate_strict:.3f}"
    )


def test_f3_colloquial_alias_is_not_jargon(lex):
    assert "拍片子" not in lex.jargon


def test_f3_formal_alias_is_still_jargon(lex):
    """口语过滤不得误伤正式同义词——那会丢掉真术语的覆盖。"""
    assert "核磁共振检查" in lex.jargon


# --- F4：拉丁缩写的判定不得取决于分词器是否粘连 ---------------------------

@pytest.mark.parametrize("w", ["CT", "CT检查", "PET检查", "B超", "HbA1c"])
def test_f4_latin_bearing_tokens_all_fail_consistently(lex, w):
    f = _judge(w, lex, lex.grade6_chars)
    assert f is not None and f.reason == "latin", f"{w} 应判 latin，实得 {f}"


# --- F5：日期掩码不得吞掉病程 ---------------------------------------------

@pytest.mark.parametrize("text", ["高血压20年了", "病史15年", "近10年", "糖尿病30年"])
def test_f5_datetime_mask_keeps_multi_digit_durations(text):
    assert DATETIME.sub(" ", text) == text, f"病程被当日期吃掉：{text!r}"


@pytest.mark.parametrize("text", ["2024年3月1日", "2024年", "3月1日", "12:30"])
def test_f5_datetime_mask_still_eats_real_dates(text):
    assert DATETIME.sub(" ", text).strip() == "", f"真日期未被掩掉：{text!r}"


# --- F6：括注对 char 类失败同样有效 ---------------------------------------

@pytest.mark.parametrize("text", ["眩晕（头晕）", "痉挛（抽筋）", "瘙痒（痒）"])
def test_f6_gloss_rescues_regardless_of_failure_reason(lex, text):
    """括注是否有效，不该取决于 jargon.txt 恰好收没收这个词。

    '痉挛' 不在 jargon（因 痉/挛 不在 2500 表而以 char 失败），原实现不救它，
    于是两段解释得同样好的转译报出不同的 glossed 分。
    """
    r = score(text, lex)
    assert r.rate_glossed > r.rate_strict, f"{text} 未被括注救回"


# --- F7：超长术语必须可命中 -----------------------------------------------

def test_f7_overlong_jargon_entries_are_matchable(lex):
    long_terms = sorted((t for t in lex.jargon if len(t) > 12), key=len, reverse=True)
    if not long_terms:
        pytest.skip("表中无超长条目")
    t = long_terms[0]
    spans = _jargon_spans(t, lex.jargon, lex.jargon_lens)
    assert spans and spans[0] == (0, len(t)), f"超长术语未整体命中：{t[:30]}"


def test_f7_probe_lengths_cover_the_whole_table(lex):
    assert max(lex.jargon_lens) == max(len(t) for t in lex.jargon)


# --- F8：字表规模不符必须硬失败 -------------------------------------------

@pytest.mark.parametrize("attr,bad", [("grade6_chars", 2499), ("ext_chars", 999)])
def test_f8_wrong_sized_char_table_is_rejected_at_load(attr, bad):
    kw = dict(
        grade6_chars={chr(0x4E00 + i) for i in range(2500)},
        ext_chars={chr(0x9000 + i) for i in range(1000)},
        jargon={"心肌梗死"},
        units=set(),
        health_common=set(),
    )
    kw[attr] = set(list(kw[attr])[:bad])
    with pytest.raises(ValueError, match=attr):
        Lexicon(**kw).check_integrity()


# --- F9：长度统计与打分必须同源 -------------------------------------------

def test_f9_length_stats_use_the_same_text_as_scoring(lex):
    """原缺陷：字数句数按原文算，日期时间被计入长度却未计入分母。"""
    r = score("2024年3月1日 12:30 采血。今天天气好。", lex)
    assert r.n_chars <= 12, f"日期时间被计入字数：n_chars={r.n_chars}"


# --- F10：括注长度上限必须真的被守住 --------------------------------------

def test_f10_gloss_cap_is_actually_enforced(lex):
    """直接卡边界，而不是比较两段分母不同的文本。

    原测试比较 '糖尿病（血糖高的病）' 与一段超长括注文本，两者分母不同，
    把 MAX_GLOSS_LEN 改成 999 它照样通过——等于没守住任何东西。
    """
    ok = "x" * MAX_GLOSS_LEN
    too_long = "x" * (MAX_GLOSS_LEN + 1)
    assert _gloss_anchors(f"心肌梗死（{ok}）"), "恰好等于上限的括注应有效"
    assert not _gloss_anchors(f"心肌梗死（{too_long}）"), "超过上限的括注不该生效"


def test_f10_overlong_gloss_does_not_rescue_the_term(lex):
    """同一个术语，只改括注长度，跨过上限就不该再被救。"""
    short = score("心肌梗死（心脏病发作）", lex)
    long_ = score("心肌梗死（一种因冠状动脉阻塞导致心脏肌肉坏死的急重症）", lex)
    assert short.n_pass_glossed > short.n_pass_strict
    assert long_.n_pass_glossed == long_.n_pass_strict
