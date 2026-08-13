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
    from verifier.shortcut_probes import TEMPLATE_SUFFIX
    from verifier.synth_minimal_edit import NUMERIC_WRAPPER, TEMPLATES

    for phrase in list(TEMPLATES) + [NUMERIC_WRAPPER.format("x")]:
        assert phrase.endswith(TEMPLATE_SUFFIX)


# ---------- 修复前的基线（ticket 26 的对照臂） ----------

def test_adversarial_subset_position_shortcuts_are_dead():
    """**ticket 26 完成后的状态。** 这条测试原本固化的是"修复前"的基线
    （负例是零候选安慰语，括注里没有任何候选，于是"候选是否落在括注之后"
    把两类完全分开：J=1.000；"候选落在末 15%"：J=0.894）。

    换成两类注入负例（grounded 药名 + 等价形式数字）之后，括注里真的有候选了，
    四条纯结构探针全部塌到门槛以内：

    | 探针 | 修复前 | 修复后 |
    |---|---:|---:|
    | candidate_in_parenthetical | 1.000 | 0.008 |
    | candidate_in_last_15pct | 0.894 | 0.000 |
    | template_prefix_present | 0.000 | 0.000 |
    | answer_ends_with_paren | 0.000 | 0.000 |

    这条测试守的是"别退回去"：任何让纯结构探针重新抬头的数据构造改动都会
    在这里炸掉，而不是等下一轮审计去发现。
    """
    from verifier.adversarial_subset import build_adversarial_subset
    from verifier.redline_candidates import load_jargon

    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / "holdout.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = build_adversarial_subset(records, load_jargon())
    scores = score_pool(subset)
    for name in STRUCTURE_PROBES:
        assert scores[name].youden_j <= STRUCTURE_GATE_J, (
            f"{name} 的 J 回到了 {scores[name].youden_j:.3f}——位置捷径退回去了")
    assert gate_failures(scores) == []
    # 合取探针**消不掉**（正例恰好就是"注入进去的、原文没有的那个词"），
    # 但等价形式负例把它从 1.000 压到了 0.7 以下。这条只报告不设门，
    # 断言它没有回到接近满分即可。
    assert scores["conjunction_in_paren_and_not_grounded"].youden_j < 0.8


# ---------- L5（铁律15 文献门）落进代码的三条判读规则 ----------

def test_joint_view_probe_is_pinned_and_cannot_be_dropped():
    """**L5 / Feng et al. ACL 2019 (arXiv:1905.05778)**：单视角探针不够——他们实测
    hypothesis-only 模型叠加前提里的平凡模式，能解掉此前被判为 "hard" 的样本中的 15%。
    合取探针按构造就接近满分、又只报告不设门，看起来像"没用可以删"，恰恰相反：
    它是单视角探针漏掉的那部分伪影的唯一抓手。"""
    from verifier.shortcut_probes import JOINT_VIEW_PROBES
    assert JOINT_VIEW_PROBES
    assert set(JOINT_VIEW_PROBES) <= {n for n, _ in PROBES}
    # 联合视角探针必须真的同时用到答案和原文
    for name in JOINT_VIEW_PROBES:
        assert name in REPORT_ONLY_PROBES     # 不设门（规则合成集上按构造就高）


def test_passing_the_gate_is_never_reported_as_clean(capsys):
    """**L5 的核心结论：证据是单向的。** 探针超线说明可作弊；探针未超线**不能**
    说明数据集没有伪影。报告措辞必须体现这一点——写成"通过/干净"会让下一个人
    把一个无效证据当成正面证据。"""
    clean = [
        _item("硝苯地平", True, answer="医生给您开的是硝苯地平。"),
        _item("氨氯地平", False, answer="医生给您开的是氨氯地平。"),
    ]
    assert report(clean, "无注入池", gate=True) is True
    out = capsys.readouterr().out
    gate_line = [ln for ln in out.splitlines() if "纯结构探针" in ln and "超线" in ln]
    assert gate_line, "门槛结论行不见了"
    assert "没被证伪" in gate_line[0]
    assert "不构成" in gate_line[0]
    # 不许出现肯定式的通过标记——"干净"只允许出现在否定句里
    assert "✓" not in gate_line[0]


# ---------- kind 一致性断言 ----------

def test_stored_kind_must_agree_with_the_candidate_shape():
    """**这条断言的来历**：修 digit 捷径时，我给注入负例写的 `candidate_text`
    是「3天」，而段A 的 `DIGIT_RE` 从答案里抽出来的其实是裸数字「3」。
    存储的 `kind` 写着 `digit`、`infer_kind("3天")` 却推断出 `lexicon`——
    探针信字段、临时脚本信推断，两者给出不同答案，**都不报错**。

    后果不是"探针算错了一点"，是候选形状与生产分布对不上（A3 那轮修过的同类
    错误），而伪装成"探针没测出效果"。写死这条不变量：字段与形状必须一致，
    不一致就是有人把候选文本或 kind 写错了，当场炸。"""
    with pytest.raises(ValueError, match="kind"):
        score_pool([_item("3天", False, kind="digit"), _item("阿司匹林", True)])


def test_kind_consistency_error_names_the_offenders():
    """报错要能直接定位——只说"有不一致"等于让人自己再写一遍检查脚本。"""
    with pytest.raises(ValueError) as e:
        score_pool([_item("3天", False, kind="digit")])
    assert "3天" in str(e.value)


def test_items_without_a_kind_field_are_fine():
    """`synthetic_records_to_candidates` 产出的条目本来就没有 kind
    （只有 red_line_guess），那条路径靠推断，不该被这条断言误伤。"""
    scores = score_pool([_item("阿司匹林", True), _item("硝苯地平", False)])
    assert scores["candidate_kind_is_lexicon"].tp >= 0


def test_digit_shape_probe_exists_and_is_report_only():
    """**Fable 5 审计 B 的产物。** 套件原来只有"是不是词表词"，对
    digit / cn_numeral 之间的倾斜完全瞎——而那正是当时最强的活口
    （红线2 切片 J=0.722）。kind 是候选文本的形状，不读原文就能看见，
    所以必须有一条探针盯着它。"""
    names = {n for n, _ in PROBES}
    assert "candidate_is_digit" in names
    assert "candidate_is_digit" in REPORT_ONLY_PROBES   # 分布探针，看趋势不设门
    assert _probe("candidate_is_digit")(_item("3", True, kind="digit")) is True
    assert _probe("candidate_is_digit")(_item("三天", True, kind="cn_numeral")) is False
