"""P3/ticket14：最小编辑对合成的测试。

L1 判读规则的硬约束：**规则最小编辑，不许 LLM 自由重写**——防止合成负例
带来源伪影（arXiv:2606.01304）。这里守的是：只在源文本里已出现"类别"标记词
（如"XX类药物"）或"数字模糊"标记词（如"遵医嘱"）时才合成；注入的具体
词/数不能恰好已经在原文里出现（否则就不是违规了，是巧合真话）；同一
case_id 的合成结果必须确定性可复现（不能用带盐的 `hash()`）。

L4（ticket 14 扩量文献核对）新增两条纪律，本文件也守：
- 同一案例若有多处不同类别标记，各出一条合成样本（但同一类别不重复出）；
- 单案例可以注入 3-4 个类别成员而非固定 2 个；
- 新增红线2（编造数字/频次）最小编辑合成，与红线0分开统计；
- 源文档集中度上限——任一 source_case_id 贡献的合成正例数不得超过总数的
  一个封顶比例。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import CATEGORY_MARKERS as ALL_CATEGORY_MARKERS  # noqa: E402
from verifier.build_teacher_labels import CONDITIONAL_MARKERS as ALL_CONDITIONAL_MARKERS  # noqa: E402
from verifier.synth_minimal_edit import (  # noqa: E402
    CATEGORY_EXAMPLES,
    CATEGORY_MARKERS,
    DEFAULT_EXAMPLES,
    MAX_SOURCE_SHARE,
    NUMERIC_INJECTIONS,
    NUMERIC_TRIGGER_MARKERS,
    _cap_source_concentration,
    _NUMERIC_HOLDOUT_RESERVED,
    find_category_triggers,
    find_numeric_trigger,
    pick_examples,
    synthesize_all,
    synthesize_entity_edits,
    synthesize_numeric_edit,
)


def test_category_markers_is_a_deliberate_subset_of_the_canonical_list():
    """回归（code review 发现）：曾经手抄过一份缩水子集，注释却声称"同一份"——
    与 join_fields() 分叉是同一类坑。这里断言子集关系，防止两份列表悄悄漂移；
    子集本身是刻意的（只留局部语义必然指向"一类药"的标记词，见模块顶部注释），
    不是要拉平成完全一致。"""
    assert set(CATEGORY_MARKERS) <= set(ALL_CATEGORY_MARKERS)
    assert all("药" in m for m in CATEGORY_MARKERS)
    assert CATEGORY_MARKERS


def test_numeric_trigger_markers_is_a_deliberate_subset_of_conditional_markers():
    """同一纪律用在数字触发词上：只挑局部语义必然指向"这里没给具体数字"的
    标记词（"必要时""视情况""酌情""遵医嘱"），排除"如果""若""根据""以上"
    "以下"这类不蕴含"缺具体数字"的通用连接词——这些词在源文本里大量出现，
    如果不收窄会像 ticket 12 code review 那次一样合成出语义不通的样本。"""
    assert set(NUMERIC_TRIGGER_MARKERS) <= set(ALL_CONDITIONAL_MARKERS)
    assert NUMERIC_TRIGGER_MARKERS
    for weak in ("如果", "若", "根据", "以上", "以下"):
        assert weak not in NUMERIC_TRIGGER_MARKERS


def _record(case_id, source, answer):
    return {"case_id": case_id, "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x"}


# ---------- pick_examples：变长注入 ----------

def test_pick_examples_matches_known_category_keyword():
    examples = pick_examples("本品与抗血小板类药物合用")
    assert "阿司匹林" in examples and "氯吡格雷" in examples


def test_pick_examples_falls_back_to_default_for_unknown_category():
    examples = pick_examples("某种从未见过的怪异类药物")
    assert len(examples) >= 2  # 兜底也必须给出至少两个非空示例


def test_pick_examples_returns_at_least_one_and_at_most_three_for_train():
    """(a) 单案例注入最多 3 个类别成员（CATEGORY_EXAMPLES 全量最多 4 个，
    train 只用留给自己的那部分，见 train/holdout 词表隔离）——L4 决断：
    方向安全但要有上限，列表过长的枚举句读起来不自然，反而更像"合成伪影"。
    只有 2 个成员的类别拆完 train 只剩 1 个，这是隔离设计的必然代价，
    不是回归。"""
    for keyword in CATEGORY_EXAMPLES:
        examples = pick_examples(f"这属于{keyword}类药物范畴")
        assert 1 <= len(examples) <= 3, f"「{keyword}」注入数量越界: {len(examples)}"


@pytest.mark.parametrize("context,keyword", [
    ("本品与黄嘌呤类药合用时需注意。", "黄嘌呤"),
    ("有长半衰期的磺脲类药需要减量。", "磺脲"),
    ("糖尿病患者服用双胍类药物时。", "双胍"),
    ("患者可使用四环素类药治疗。", "四环素"),
    ("阳性可口服喹诺酮类药物。", "喹诺酮"),
    ("以及其他大环内酯类药合用。", "大环内酯"),
    ("青霉素与氨基糖苷类药联用。", "氨基糖苷"),
    ("酯在内的头孢菌素类药过敏。", "头孢菌素"),
    ("本品是他汀类药物的一种。", "他汀"),
    ("不宜和感冒类药同服。", "感冒"),
])
def test_pick_examples_covers_train_split_derived_category_roots(context, keyword):
    """这些关键词全部来自 train 切分兜底案例的频率统计（ticket 12 P3
    76.5% 落在兜底对的诊断），不是随手加的。"""
    examples = pick_examples(context)
    assert examples != DEFAULT_EXAMPLES, f"「{keyword}」仍然落在兜底对，词典没覆盖到"


def test_new_category_examples_never_pick_the_keyword_itself_as_the_example():
    for keyword, examples in CATEGORY_EXAMPLES.items():
        assert keyword not in examples, f"「{keyword}」的示例里出现了关键词本身：{examples}"


# ---------- train/holdout 词表隔离（独立第二意见代码审计发现：实测原本
# train 合成正例与 holdout 对抗子集注入词汇 100% 重叠） ----------

def test_pick_examples_train_and_holdout_pools_are_disjoint():
    for keyword in CATEGORY_EXAMPLES:
        ctx = f"这属于{keyword}类药物范畴"
        train_ex = set(pick_examples(ctx, holdout=False))
        holdout_ex = set(pick_examples(ctx, holdout=True))
        assert not (train_ex & holdout_ex), f"「{keyword}」train/holdout 示例有重叠"
        assert train_ex and holdout_ex, f"「{keyword}」某一侧被拆空了"


def test_pick_examples_holdout_pool_still_falls_back_to_default_for_unknown_category():
    examples = pick_examples("某种从未见过的怪异类药物", holdout=True)
    assert examples and set(examples) <= set(DEFAULT_EXAMPLES)


def test_default_examples_train_and_holdout_pools_are_disjoint():
    """兜底组本身也要遵守同一条隔离纪律，不能因为是"兜底"就漏了。"""
    train_ex = set(pick_examples("某种从未见过的怪异类药物", holdout=False))
    holdout_ex = set(pick_examples("某种从未见过的怪异类药物", holdout=True))
    assert not (train_ex & holdout_ex)


def test_numeric_injections_train_and_holdout_pools_are_disjoint():
    train_pool = set(NUMERIC_INJECTIONS[:-_NUMERIC_HOLDOUT_RESERVED])
    holdout_pool = set(NUMERIC_INJECTIONS[-_NUMERIC_HOLDOUT_RESERVED:])
    assert not (train_pool & holdout_pool)
    assert train_pool and holdout_pool


def test_word_pool_assignment_has_no_cross_category_leakage():
    """回归：第一版"每个类别自己拆最后一个成员"防不住跨类别撞车——
    同一个具体药名（如"对乙酰氨基酚"）会同时出现在多个类别的示例列表里
    （如"镇痛"和 DEFAULT_EXAMPLES 兜底组），按类别内部拆分时，这个词可能
    在 A 类别被分进 train、在 B 类别又被分进 holdout，全局汇总后 train
    词表和 holdout 词表就会有交集。这里遍历全部类别（含兜底组），汇总出
    完整的 train 侧词表与 holdout 侧词表，断言两者互不相交——这是"任何
    一个词无论出现在哪个类别，归属必须全局一致"这条不变量的直接验证。"""
    train_words: set[str] = set()
    holdout_words: set[str] = set()
    for keyword in CATEGORY_EXAMPLES:
        ctx = f"这属于{keyword}类药物范畴"
        train_words.update(pick_examples(ctx, holdout=False))
        holdout_words.update(pick_examples(ctx, holdout=True))
    train_words.update(pick_examples("从未出现过的怪异占位符类药物", holdout=False))
    holdout_words.update(pick_examples("从未出现过的怪异占位符类药物", holdout=True))

    overlap = train_words & holdout_words
    assert not overlap, f"跨类别撞车导致 train/holdout 词表有交集: {overlap}"


# ---------- find_category_triggers：多标记点 ----------

def test_finds_single_category_marker():
    ctxs = find_category_triggers("本品与抗血小板类药物合用时需注意出血风险。")
    assert len(ctxs) == 1 and "抗血小板" in ctxs[0]


def test_returns_empty_list_when_no_category_marker_present():
    assert find_category_triggers("阿司匹林每日一片，饭后服用。") == []


def test_finds_multiple_distinct_category_markers_in_one_source():
    """(b) 同一案例若有多处不同类别标记，每处各出一条——不能只用
    `.find()` 找到的第一处。"""
    src = "本品与抗血小板类药物合用需注意，也不宜与磺胺类药同服。"
    ctxs = find_category_triggers(src)
    assert len(ctxs) == 2
    assert any("抗血小板" in c for c in ctxs)
    assert any("磺胺" in c for c in ctxs)


def test_does_not_duplicate_when_same_category_mentioned_twice():
    """同一类别提两次不算"多处不同类别"，只应该出一条——否则会产出内容
    几乎相同的合成样本，凑数不凑信息量。"""
    src = "本品与抗血小板类药物合用需注意，另外抗血小板类药还会加重出血风险。"
    ctxs = find_category_triggers(src)
    assert len(ctxs) == 1


# ---------- synthesize_entity_edits：一案例可出多条 ----------

def test_returns_empty_list_when_source_has_no_category_trigger():
    r = _record("vt-0001", "阿司匹林每日一片。", "您每天吃一片阿司匹林。")
    assert synthesize_entity_edits(r) == []


def test_injects_specific_examples_not_present_in_source():
    r = _record("vt-0002", "本品与抗血小板类药物合用时需注意。", "用药时要小心。")
    out = synthesize_entity_edits(r)
    assert len(out) == 1
    item = out[0]
    assert item["case_id"] == "vt-0002-synth"
    assert set(item["injected_examples"]) >= {"阿司匹林", "氯吡格雷"}
    assert all(e in item["synthetic_answer"] for e in item["injected_examples"])
    # 最小编辑：原答案内容必须原样保留，只是追加，不是重写
    assert "用药时要小心。" in item["synthetic_answer"]
    assert item["red_line_idx"] == 0
    assert item["verdict"] == "fail"
    assert item["source_case_id"] == "vt-0002"


def test_synthesize_entity_edits_holdout_flag_uses_disjoint_examples():
    r = _record("vt-0002", "本品与抗血小板类药物合用时需注意。", "用药时要小心。")
    train_out = synthesize_entity_edits(r, holdout=False)
    holdout_out = synthesize_entity_edits(r, holdout=True)
    train_words = set(train_out[0]["injected_examples"])
    holdout_words = set(holdout_out[0]["injected_examples"])
    assert train_words and holdout_words
    assert not (train_words & holdout_words)


def test_multiple_distinct_categories_produce_multiple_synthetic_entries():
    r = _record(
        "vt-0010",
        "本品与抗血小板类药物合用需注意，也不宜与磺胺类药同服。",
        "用药请遵医嘱。",
    )
    out = synthesize_entity_edits(r)
    assert len(out) == 2
    assert {item["case_id"] for item in out} == {"vt-0010-synth", "vt-0010-synth2"}
    assert {item["source_case_id"] for item in out} == {"vt-0010"}


def test_skips_when_picked_example_accidentally_already_in_source():
    """防止合成出一句"巧合为真"的话——那就不是违规样本了。"""
    r = _record(
        "vt-0003",
        "本品与抗血小板类药物（如阿司匹林、氯吡格雷、替格瑞洛）合用时需注意。",
        "请遵医嘱。",
    )
    out = synthesize_entity_edits(r)
    assert out == [] or all(
        e not in ("阿司匹林", "氯吡格雷") for item in out for e in item["injected_examples"]
    )


def test_entity_edits_are_deterministic_across_runs():
    r = _record("vt-0004", "本品属于降压类药物。", "按时吃药就好。")
    assert synthesize_entity_edits(r) == synthesize_entity_edits(r)


# ---------- 数字类最小编辑（红线2，L4 新增） ----------

def test_finds_numeric_trigger():
    ctx = find_numeric_trigger("是否需要调整剂量，必要时请咨询医生。")
    assert ctx is not None and "必要时" in ctx


def test_returns_none_when_no_numeric_trigger_present():
    assert find_numeric_trigger("阿司匹林每日一片，饭后服用。") is None


def test_synthesize_numeric_edit_injects_a_fabricated_number_not_in_source():
    r = _record("vt-0020", "是否需要复查，视情况由医生决定。", "按时吃药，注意休息。")
    out = synthesize_numeric_edit(r)
    assert out is not None
    assert out["red_line_idx"] == 2
    assert out["verdict"] == "fail"
    assert out["case_id"] == "vt-0020-synthnum"
    assert "按时吃药，注意休息。" in out["synthetic_answer"]
    # 候选文本必须是段A真实能抽出的数字/数词 span（1-6字），不是整句编造
    # 短语（原整句9-13字）——第二轮独立审计(Fable 5)发现的候选形状不匹配。
    assert out["injected_examples"]
    for span in out["injected_examples"]:
        assert span in out["synthetic_answer"]
        assert len(span) <= 6


def test_synthesize_numeric_edit_returns_none_without_trigger():
    r = _record("vt-0021", "阿司匹林每日一片。", "您每天吃一片。")
    assert synthesize_numeric_edit(r) is None


def test_numeric_edit_is_deterministic_across_runs():
    r = _record("vt-0022", "复查频率酌情而定。", "谢谢医生。")
    assert synthesize_numeric_edit(r) == synthesize_numeric_edit(r)


def test_synthesize_numeric_edit_holdout_flag_uses_disjoint_pool():
    """train/holdout 用的是不重叠的**短语**池（NUMERIC_INJECTIONS 按索引
    切分），所以拼出的整句必须不同。候选文本现在是从短语里抽出的数字/
    数词 span（见 A3 修复），不强求 span 本身跨池不重叠——单个数字/数词
    是极低信息量的通用 token（"2""一次"这类），不像药名那样构成"记住
    这个具体事实"的捷径，段A本来就设计成对它们宁可错杀，这里不必也不该
    再叠一层跨池隔离，那是给药名这种高信息量实体准备的纪律。"""
    r = _record("vt-0020", "是否需要复查，视情况由医生决定。", "按时吃药，注意休息。")
    train_out = synthesize_numeric_edit(r, holdout=False)
    holdout_out = synthesize_numeric_edit(r, holdout=True)
    assert train_out is not None and holdout_out is not None
    assert train_out["synthetic_answer"] != holdout_out["synthetic_answer"]


# ---------- synthesize_all：合并 + 源文档集中度上限 ----------

def test_synthesize_all_skips_records_without_trigger_and_keeps_the_rest():
    """池子要够大让集中度上限的 cap ≥1（cap=int(总数*5%)）——否则单条正例
    自己就会被上限裁到 0，混淆"没触发被跳过"和"触发了但被集中度上限裁掉"
    这两件不同的事。"""
    records = [
        _record("a", "阿司匹林每日一片。", "按时吃。"),
        _record("b", "本品属于降糖类药物。", "按时吃就好。"),
    ]
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(25)]
    out = synthesize_all(records)
    ids = {item["case_id"] for item in out}
    assert "b-synth" in ids
    assert not any(item["source_case_id"] == "a" for item in out)


def test_synthesize_all_holdout_vocabulary_never_overlaps_train_vocabulary():
    """端到端回归（独立第二意见代码审计发现）：train 合成正例与 holdout
    对抗子集必须没有共同的**实体**注入词（红线0：药名这类高信息量、可能被
    当"记住这个具体事实"捷径的词）——哪怕两边用**同一份**触发内容（同一个
    案例既当 train 输入又当 holdout 输入喂进去，模拟"两个 split 里各有
    一条提到同一类别/触发词的案例"这个真实场景），产出的实体词表也不能
    重叠，否则对抗子集测的是记忆力不是泛化。

    红线2（数字/数词 span）不做这条限制——数字是极低信息量的通用 token
    （"2""一次"这类），train/holdout 两个不重叠的编造短语池仍然会在具体
    数字上偶然撞车（比如两个不同短语都用到了"2"），这不构成"记住同一个
    具体事实"，不该也不必强求数字级别互不相交（`test_synthesize_numeric_
    edit_holdout_flag_uses_disjoint_pool` 有专门说明）。"""
    records = [
        _record("shared", "本品与抗血小板类药物合用需注意，复查频率必要时而定。", "请遵医嘱。"),
    ]
    records += [
        _record(f"pad{i}", "本品属于降压类药物，复查酌情而定。", "请遵医嘱。") for i in range(25)
    ]
    train_synth = synthesize_all(records, holdout=False)
    holdout_synth = synthesize_all(records, holdout=True)

    train_words = {w for s in train_synth if s["red_line_idx"] == 0 for w in s["injected_examples"]}
    holdout_words = {w for s in holdout_synth if s["red_line_idx"] == 0 for w in s["injected_examples"]}
    assert train_words and holdout_words
    assert not (train_words & holdout_words), (
        f"train/holdout 实体注入词汇有重叠: {train_words & holdout_words}"
    )


def test_synthesize_all_combines_entity_and_numeric_red_lines():
    """总池要足够大——单条记录时源文档集中度上限会把同一案例的实体+数字
    两条边压到只剩一条，这是集中度上限该有的行为（防止小样本下单一来源
    主导），不是本测试要覆盖的东西，所以垫够背景记录避免触发那条上限。"""
    records = [_record("c", "本品属于降糖类药物，复查频率必要时而定。", "请遵医嘱。")]
    # 需要总池够大让 cap（int(总数*5%)）≥2，否则"c"自己的实体+数字两条会被
    # 集中度上限压掉一条——那是上限该有的行为，不是本测试要覆盖的东西。
    records += [_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(50)]
    out = synthesize_all(records)
    red_lines = {item["red_line_idx"] for item in out}
    assert red_lines == {0, 2}


def test_source_concentration_cap_limits_any_single_case_share():
    """L4 判读规则：任一 source_case_id 贡献的合成正例数不得超过合成正例
    总数的 `MAX_SOURCE_SHARE`——防止"来源身份"变成可学捷径
    （arXiv:2606.01304）。这里构造一个人为的极端场景验证上限真的生效。"""
    heavy = _record(
        "heavy",
        "本品与抗血小板类药物合用需注意，也不宜与磺胺类药同服，"
        "复查频率必要时而定。",
        "请遵医嘱。",
    )
    others = [
        _record(f"o{i}", "本品属于降压类药物。", "请遵医嘱。") for i in range(30)
    ]
    out = synthesize_all([heavy] + others)
    counts = Counter(item["source_case_id"] for item in out)
    max_share = max(counts.values()) / len(out)
    assert max_share <= MAX_SOURCE_SHARE + 1e-9


def test_cap_source_concentration_holds_on_small_pools_after_truncation_shrinks_total():
    """回归（code review 发现）：单次裁剪按裁剪前总数算 cap 不够——3 条 + 5×1条
    的小样本按裁剪前 8 算出 cap=1，裁完剩 6 条，主导来源份额变成
    1/6=16.7%，远超 5% 门槛却没有任何信号。这里直接测 `_cap_source_
    concentration` 本身（不经过 synthesize_all 的文本触发逻辑），逼近
    复现原始报告的确切场景：裁剪后份额必须严格不超过门槛，不能因为
    裁剪本身缩小了分母就"事后超线"。"""
    items = (
        [{"case_id": f"heavy-{i}", "source_case_id": "heavy"} for i in range(3)]
        + [{"case_id": f"o{i}-0", "source_case_id": f"o{i}"} for i in range(5)]
    )
    out = _cap_source_concentration(items, MAX_SOURCE_SHARE)
    if out:
        counts = Counter(it["source_case_id"] for it in out)
        max_share = max(counts.values()) / len(out)
        assert max_share <= MAX_SOURCE_SHARE + 1e-9


def test_cap_source_concentration_is_a_noop_on_realistic_scale_pools():
    """生产规模（千级）下每案例最多产出 3-5 条，cap 通常 ≥50，这条上限
    基本不生效——确认迭代版本没有在正常规模下引入意外裁剪。"""
    items = [{"case_id": f"c{i}-0", "source_case_id": f"c{i}"} for i in range(1200)]
    out = _cap_source_concentration(items, MAX_SOURCE_SHARE)
    assert len(out) == len(items)
