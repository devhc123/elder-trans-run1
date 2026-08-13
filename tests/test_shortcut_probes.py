"""退化分类器探针的测试（ticket 17）。

探针本身是**只读测量**，不改任何数据——所以这里守的不是"数据对不对"，而是
"测量对不对"：探针的判定语义、TPR/FPR/Youden J 的算术、以及门槛只作用在
该作用的地方。测量错了比没有测量更糟——会给出"捷径已经堵上了"的假信号。

最后一条测试固化的是**修复前的基线**：当前对抗子集上"候选是否落在括注之后"
这个纯位置特征的 J≈1.0。ticket 26 改完数据构造之后，这条测试会连同基线数字
一起更新——它存在的意义就是让那次更新是**有意识的**，而不是悄悄地就变了。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.shortcut_probes import (  # noqa: E402
    PROBES,
    REPORT_ONLY_PROBES,
    STRUCTURE_GATE_J,
    STRUCTURE_PROBES,
    ProbeScore,
    format_table,
    gate_failures,
    infer_kind,
    report,
    score_pool,
)

# 正文长度要贴近真实答案（实测 prompt p90 就有 2594 字）——注入句只占答案末尾
# 一小段，"候选落在末 15%" 这条位置探针才测得出它本来要测的东西。用一条 24 字
# 的玩具正文会让注入句占掉大半篇幅，把探针测成假阴性。
_BODY = (
    "医生给您开的是硝苯地平，这是一种常用的降压药。您每天按时吃，别自己停也别自己加量。"
    "吃药期间如果觉得头晕、脸发红，或者脚踝有点肿，都是可能出现的反应，先别慌，"
    "记下来下次复诊的时候告诉医生。平时在家量血压，把数字记在本子上，复诊带过去给医生看。"
)
TEMPLATED = _BODY + "\n\n（补充一句：像华法林这类药，也是您问的这一类里常见的。）"


def _item(candidate_text, label, answer=TEMPLATED, source="本药物用于降压治疗。", kind=None):
    it = {
        "case_id": "vt-0001",
        "source_text": source,
        "answer": answer,
        "candidate_text": candidate_text,
        "label": label,
    }
    if kind is not None:
        it["kind"] = kind
    return it


def _probe(name):
    return dict(PROBES)[name]


# ---------- 探针的判定语义 ----------

def test_template_prefix_probe_only_looks_at_the_answer():
    assert _probe("template_prefix_present")(_item("华法林", True)) is True
    assert _probe("template_prefix_present")(_item("硝苯地平", False, answer="医生给您开的是硝苯地平。")) is False


def test_candidate_in_parenthetical_distinguishes_position_not_presence():
    """这是 ticket 17 要杀的那条捷径：两个候选在**同一个**答案里，一个在
    括注内、一个在括注外——只有位置不同。"""
    assert _probe("candidate_in_parenthetical")(_item("华法林", True)) is True
    assert _probe("candidate_in_parenthetical")(_item("硝苯地平", False)) is False


def test_candidate_in_parenthetical_is_false_when_there_is_no_parenthetical():
    assert _probe("candidate_in_parenthetical")(
        _item("硝苯地平", False, answer="医生给您开的是硝苯地平。")
    ) is False


def test_candidate_in_last_15pct_fires_on_trailing_injection():
    assert _probe("candidate_in_last_15pct")(_item("华法林", True)) is True
    assert _probe("candidate_in_last_15pct")(_item("硝苯地平", False)) is False


def test_answer_ends_with_paren_probe():
    assert _probe("answer_ends_with_paren")(_item("华法林", True)) is True
    assert _probe("answer_ends_with_paren")(
        _item("硝苯地平", False, answer="医生给您开的是硝苯地平。")
    ) is False


def test_candidate_not_grounded_uses_whole_span_matching_for_numbers():
    """与 `extract_candidates` 同一口径：数字必须整段匹配。"12" 是 "112" 的
    子串但 12≠112——用朴素子串判定会把一个真编造的数字判成"原文已有"。"""
    src = "每次不超过112毫克。"
    assert _probe("candidate_not_grounded")(_item("12", True, source=src, kind="digit")) is True
    assert _probe("candidate_not_grounded")(_item("112", False, source=src, kind="digit")) is False


def test_candidate_not_grounded_uses_substring_for_lexicon():
    """词表词沿用朴素子串——中文复合词的子串通常仍是同一实体（"他汀"之于
    "阿托伐他汀"），这也是 `extract_candidates` 的既有口径。"""
    src = "本品为阿托伐他汀钙片。"
    assert _probe("candidate_not_grounded")(_item("他汀", False, source=src, kind="lexicon")) is False
    assert _probe("candidate_not_grounded")(_item("华法林", True, source=src, kind="lexicon")) is True


def test_kind_is_inferred_when_the_pool_does_not_carry_it():
    """`synthetic_records_to_candidates` 产出的条目**没有** `kind` 字段
    （只有 `red_line_guess`），而对抗子集的正例全部来自那里。探针不能因此
    静默把数字候选当词表词判——那会让"非 grounded"这条探针的口径在正负例
    之间不一致，测出来的 J 是假的。"""
    assert infer_kind("112") == "digit"
    assert infer_kind("三天") == "cn_numeral"
    assert infer_kind("三分之一") == "fraction"
    assert infer_kind("阿司匹林") == "lexicon"
    # 不带 kind 字段也必须走整段匹配
    src = "每次不超过112毫克。"
    assert _probe("candidate_not_grounded")(_item("12", True, source=src)) is True


def test_conjunction_probe_is_exactly_position_and_not_grounded():
    src = "本药物用于降压治疗。"
    # 在括注内 + 原文没有 -> 合取成立
    assert _probe("conjunction_in_paren_and_not_grounded")(_item("华法林", True, source=src)) is True
    # 在括注内但原文有 -> 不成立（grounded 注入负例正是这一格）
    assert _probe("conjunction_in_paren_and_not_grounded")(
        _item("降压", False, source=src, kind="lexicon")
    ) is False
    # 原文没有但不在括注内 -> 不成立（普通可信负例正是这一格）
    assert _probe("conjunction_in_paren_and_not_grounded")(
        _item("硝苯地平", False, source=src)
    ) is False


# ---------- 算术 ----------

def test_score_pool_computes_tpr_fpr_and_youden_j():
    pool = [
        _item("华法林", True),                                        # 括注内正例 -> TP
        _item("硝苯地平", False),                                      # 括注外负例 -> TN
        _item("华法林", False),                                        # 括注内负例 -> FP
        _item("氨氯地平", True, answer=_BODY + "氨氯地平也可以。"),      # 括注外正例 -> FN
    ]
    s = score_pool(pool)["candidate_in_parenthetical"]
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 1, 1, 1)
    assert s.tpr == pytest.approx(0.5)
    assert s.fpr == pytest.approx(0.5)
    assert s.youden_j == pytest.approx(0.0)


def test_youden_j_is_absolute_so_inverted_shortcuts_still_show_up():
    """一个把两类判反的特征同样是捷径（模型学到取反即可）。J 取绝对值，
    否则"负例专属特征"会显示成 0，看起来干净。"""
    pool = [_item("硝苯地平", True, answer=_BODY), _item("华法林", False)]
    s = score_pool(pool)["candidate_in_parenthetical"]
    assert s.tpr == 0.0 and s.fpr == 1.0
    assert s.youden_j == pytest.approx(1.0)


def test_score_pool_handles_a_class_with_no_members():
    """全正例（或全负例）的池不该崩——修复前的对抗子集就一度是纯正例。"""
    s = score_pool([_item("华法林", True)])["candidate_in_parenthetical"]
    assert s.tpr == 1.0 and s.fpr == 0.0


# ---------- 门槛只作用在该作用的地方 ----------

def test_gate_applies_to_structure_probes_only():
    """探针⑤⑥⑦是内容型/分布型，**只报告不设门**——「正例 = 注入的非 grounded
    项」这个等式就是数据构造本身，用规则消除它在数学上不可能，给它设门只会
    逼出无意义的数据堆砌。"""
    assert set(STRUCTURE_PROBES).isdisjoint(REPORT_ONLY_PROBES)
    assert set(STRUCTURE_PROBES) | set(REPORT_ONLY_PROBES) == {n for n, _ in PROBES}


def test_gate_failures_flags_structure_probes_over_threshold():
    scores = {
        "candidate_in_parenthetical": ProbeScore("candidate_in_parenthetical", 10, 0, 0, 10),
        "conjunction_in_paren_and_not_grounded": ProbeScore(
            "conjunction_in_paren_and_not_grounded", 10, 0, 0, 10),
    }
    fails = gate_failures(scores)
    assert fails == ["candidate_in_parenthetical"]     # 内容型探针不进门槛


def test_gate_threshold_leaves_room_for_drift_but_is_not_slack():
    """构造上纯结构探针应当是 0；0.3 是留给数据漂移的余量，不是"差不多就行"。"""
    assert 0 < STRUCTURE_GATE_J <= 0.3


def test_format_table_marks_report_only_probes():
    scores = score_pool([_item("华法林", True), _item("硝苯地平", False)])
    table = format_table(scores, "t")
    assert "仅报告" in table
    for name, _ in PROBES:
        assert name in table


# ---------- DeepSeek 独立审计（ticket 21）发现并已实测复现的四处 ----------

def test_empty_pool_is_an_error_not_a_silent_pass():
    """**DeepSeek #1（SEVERE）**：空池让七个探针全部 J=0，`gate_failures` 返回空
    列表——"根本没有验收数据"和"验收数据很干净"给出**完全一样**的信号。这正是
    本项目反复强调的"看起来成功和真正成功长得一样"，而且它守的是花钱那一步。"""
    with pytest.raises(ValueError, match="空"):
        score_pool([])


def test_report_refuses_an_empty_pool():
    with pytest.raises(ValueError, match="空"):
        report([], "空池", gate=True)


def test_candidate_after_the_closing_paren_is_not_inside_it():
    """**DeepSeek #3（MODERATE）**：探针从第一个前缀往后找候选，只要出现在前缀
    之后就算命中——哪怕候选已经在闭合的 `）` 外面。测量仪器本身报错，量出来的
    J 就是错的。实测复现：候选"布洛芬"在括注之后，旧实现判 True。"""
    it = _item("布洛芬", False,
               answer="请遵医嘱。（补充一句：像阿司匹林这类药。）随后予布洛芬")
    assert _probe("candidate_in_parenthetical")(it) is False
    # 括注**内**的候选仍然要判 True，别把探针修瞎了
    it2 = _item("阿司匹林", True,
                answer="请遵医嘱。（补充一句：像阿司匹林这类药。）随后予布洛芬")
    assert _probe("candidate_in_parenthetical")(it2) is True


def test_tail_probe_uses_the_last_occurrence_not_the_first():
    """**DeepSeek #5（MINOR）**：候选在正文里自然出现过一次、又被注入句用了
    一次时，`find` 取首次出现会判"不在尾部"，从而**低估**尾部捷径。探针问的是
    "有没有一次出现落在尾部"，该用最后一次。"""
    body = "先用阿司匹林控制病情。" * 12
    it = _item("阿司匹林", True,
               answer=body + "\n\n（补充一句：像阿司匹林、布洛芬这类药，也是常见的。）")
    assert _probe("candidate_in_last_15pct")(it) is True


def test_template_suffix_coupling_is_asserted_not_assumed():
    """**DeepSeek #4（MODERATE）**：`probe_answer_ends_with_paren` 硬编码「。）」，
    与生成模板是隐式耦合。模板改了措辞，这个探针会**静默失效**——而探针失效的
    表现是 J=0，也就是"很干净"，方向最坏。用断言把耦合显式化（同
    `TEMPLATE_PREFIXES` 那次的做法），改坏了立刻炸而不是悄悄放行。"""
    from verifier.candidate_pool import STRUCTURAL_NEGATIVE_FILLERS
    from verifier.shortcut_probes import TEMPLATE_SUFFIX
    from verifier.synth_minimal_edit import TEMPLATES

    for t in list(TEMPLATES) + [f for v in STRUCTURAL_NEGATIVE_FILLERS.values() for f in v]:
        assert t.endswith(TEMPLATE_SUFFIX)


# ---------- 修复前的基线（ticket 26 的对照臂） ----------

def test_current_adversarial_subset_is_fully_cracked_by_the_position_probe():
    """**修复前的基线，不是期望的最终状态。**

    当前对抗子集的负例是"零候选安慰语装饰过的 pass 答案"——括注里没有任何
    候选，所以"候选是否落在括注之后"这个纯位置特征把两类完全分开。
    ticket 26 改完数据构造后这条会降到 ≤0.3，届时连同基线数字一起更新本测试。
    """
    from verifier.adversarial_subset import build_adversarial_subset
    from verifier.redline_candidates import load_jargon

    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = build_adversarial_subset(records, load_jargon())
    s = score_pool(subset)["candidate_in_parenthetical"]
    assert s.tpr == 1.0
    assert s.fpr < 0.01
    assert s.youden_j > 0.98
    assert gate_failures(score_pool(subset))          # 门槛现在就该是红的
