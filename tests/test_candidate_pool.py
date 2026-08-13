"""可信候选池构造器的测试（ticket 14）。

**这是 ticket 14 最核心的设计决定的代码化**：段A规则在 `verdict=pass` 案例
上抽出的候选，天然是真负例（教师对整案例判过零违规，不需要反推）；`verdict
=fail` 案例只取 evidence 命中的候选当正例，**没命中的候选整条丢弃**——不能
当负例用（那是 ticket 12 P2 诊断出的方法论缺陷：单条 evidence 不代表案例里
只有那一个违规词，把"没命中"当"没违规"会把系统性漏标烤进训练数据）。

train 和 holdout 用同一个构造函数——这里守的是这份"可信"规则本身对不对，
不是某个切分的具体数字。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.candidate_pool import (  # noqa: E402
    STRUCTURAL_NEGATIVE_FILLERS,
    apply_teacher_labels,
    build_structural_negatives,
    build_trusted_candidate_pool,
    build_uncertain_candidate_pool,
    load_teacher_candidate_labels,
    sample_uncertain_candidates,
)


def _record(case_id, source, answer, verdict, red_lines=None):
    """red_lines: [(idx, violated, evidence), ...]；默认三条都不违规。"""
    red_lines = red_lines or [(0, False, ""), (1, False, ""), (2, False, "")]
    return {
        "case_id": case_id,
        "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
        "output": json.dumps({
            "key_points": [],
            "red_lines": [{"idx": i, "violated": v, "evidence": e} for i, v, e in red_lines],
            "verdict": verdict,
        }, ensure_ascii=False),
    }


# ---------- pass 案例：所有候选都是可信负例 ----------

def test_pass_case_candidates_are_all_labeled_negative():
    r = _record("p1", "请遵医嘱。", "医生给您开的是阿司匹林。", "pass")
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert len(pool) == 1
    assert pool[0]["label"] is False
    assert pool[0]["case_id"] == "p1"
    assert pool[0]["candidate_text"] == "阿司匹林"


def test_pass_case_with_no_candidates_contributes_nothing():
    r = _record("p2", "阿司匹林每日一片。", "阿司匹林您要每日一片。", "pass")
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert pool == []


# ---------- fail 案例：只有 evidence 命中的候选才是可信正例 ----------

def test_fail_case_candidate_matching_evidence_is_labeled_positive():
    r = _record(
        "f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林"})
    assert len(pool) == 1
    assert pool[0]["label"] is True
    assert pool[0]["candidate_text"] == "阿司匹林"


def test_fail_case_candidate_not_matching_evidence_is_dropped_not_labeled_negative():
    """这是本模块存在的唯一理由：不确定的候选整条丢弃，不能既不算正例
    也硬当负例塞进训练集——那正是 ticket 12 P2 诊断出的方法论缺陷。"""
    r = _record(
        "f2", "请遵医嘱。", "医生给您开的是阿司匹林，另外还有布洛芬。", "fail",
        # evidence 只记了"阿司匹林"，"布洛芬"这个候选没有对应的 evidence
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon={"阿司匹林", "布洛芬"})
    texts = {item["candidate_text"] for item in pool}
    assert "阿司匹林" in texts  # 命中 evidence，保留且为正例
    assert "布洛芬" not in texts  # 没命中，整条丢弃——不在 pool 里，既非正例也非负例


def test_fail_case_with_no_evidence_matching_candidates_contributes_nothing():
    r = _record(
        "f3", "请遵医嘱。", "医生说了些别的事。", "fail",
        red_lines=[(0, True, "别的事"), (1, False, ""), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert pool == []


# ---------- 红线1（说反）永远不产出可信正例——两段式管线不覆盖它 ----------

def test_red_line_1_evidence_never_produces_a_trusted_positive():
    r = _record(
        "f4", "医保：否。", "医保能报销。", "fail",
        red_lines=[(0, False, ""), (1, True, "能报销"), (2, False, "")],
    )
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert pool == []


# ---------- 混合切分：pass 与 fail 案例并存 ----------

def test_mixed_split_combines_pass_negatives_and_fail_positives():
    records = [
        # "硝苯地平"不在 source，但案例整体判 pass（比如触发了红线0的
        # 主体身份例外条款）——段A仍会把它抽成候选，可信池里标为负例。
        _record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass"),
        _record("f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
                red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")]),
    ]
    pool = build_trusted_candidate_pool(records, jargon={"阿司匹林", "硝苯地平"})
    labels = {item["case_id"]: item["label"] for item in pool}
    assert labels == {"p1": False, "f1": True}


# ---------- 每条候选携带溯源信息，供上下游（held-out 报告/训练格式）使用 ----------

def test_each_candidate_carries_source_context_and_kind():
    r = _record("f5", "血糖不超过112。", "血糖别超过12就好。", "fail",
                red_lines=[(0, False, ""), (1, False, ""), (2, True, "12")])
    pool = build_trusted_candidate_pool([r], jargon=set())
    assert len(pool) == 1
    item = pool[0]
    assert item["source_text"] and item["answer"]
    assert item["kind"] == "digit"
    assert item["red_line_guess"] == 2


# ---------- 结构性负例（第三轮独立审计问题一的修复）----------
#
# 实测：只看答案有没有含固定括注模板（不读原文、不做任何语义判断）这一个
# 特征，就能在对抗子集上拿到 100% 召回 / 0% 误报——train/holdout 的合成
# 正例统一用"答案末尾追加括注"的结构，模型可能只学会认这个结构。
# `build_structural_negatives` 造"括注存在但内容真实无害"的负例：对
# verdict=pass 的真实案例，答案后面追加一句不含任何可提取候选的安慰语
# （复用同样的模板前缀，但填的是不引入新候选的通用内容），让"有没有括注"
# 这个特征在正负例里都出现，逼模型看内容不看结构。

def test_structural_negative_fillers_contain_no_extractable_candidates():
    """安慰语本身绝不能意外含数字/中文数词/词表实体——否则会造出新的、
    未经设计的候选，混淆"这是专门为打掉结构捷径而造的负例"这件事。"""
    from verifier.redline_candidates import CN_NUMERAL_RE, DIGIT_RE, FRACTION_RE, load_jargon

    jargon = load_jargon()
    for pool_name, fillers in STRUCTURAL_NEGATIVE_FILLERS.items():
        for f in fillers:
            assert not DIGIT_RE.findall(f), f"{pool_name} 安慰语含数字: {f!r}"
            assert not CN_NUMERAL_RE.findall(f), f"{pool_name} 安慰语含中文数词: {f!r}"
            assert not FRACTION_RE.findall(f), f"{pool_name} 安慰语含分数: {f!r}"
            assert not any(w in f for w in jargon), f"{pool_name} 安慰语含词表实体: {f!r}"


def test_structural_negative_fillers_share_marker_prefix_with_positive_templates():
    """安慰语必须复用跟合成正例**相同**的括注前缀（"（补充一句："等），
    不能自己另起一套不重叠的措辞——否则退化分类器只需要认"这三个具体
    前缀"，换一套新前缀完全不影响它继续 100% 命中原来的正例，等于没堵。

    前缀元组从 `synth_minimal_edit.TEMPLATE_PREFIXES` 导入，不在这里
    再手抄一份——`/code-review` 发现原来这里是第三份独立硬编码副本，
    TEMPLATES 改了措辞、这里忘了同步也不会有任何信号。"""
    from verifier.synth_minimal_edit import TEMPLATE_PREFIXES

    for fillers in STRUCTURAL_NEGATIVE_FILLERS.values():
        for f in fillers:
            assert f.startswith(TEMPLATE_PREFIXES), f"安慰语前缀跟正例模板不重合: {f!r}"


def test_structural_negatives_only_apply_to_pass_verdict_cases():
    fail_record = _record("f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
                           red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")])
    out = build_structural_negatives([fail_record], jargon={"阿司匹林"})
    assert out == []


def test_structural_negatives_skipped_when_pass_case_has_no_candidates():
    """这条 pass 答案本来就没有可信候选可以附着，造出来的负例没有意义。"""
    r = _record("p1", "阿司匹林每日一片。", "阿司匹林您要每日一片。", "pass")
    out = build_structural_negatives([r], jargon={"阿司匹林"})
    assert out == []


def test_structural_negatives_reuse_the_pass_case_original_candidates_as_false():
    r = _record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass")
    out = build_structural_negatives([r], jargon={"硝苯地平"})
    assert len(out) == 1
    item = out[0]
    assert item["candidate_text"] == "硝苯地平"
    assert item["label"] is False
    assert item["case_id"] == "p1-structneg"
    # 答案必须真的带上了括注结构（这是重点——制造"有括注但仍是负例"）
    assert any(item["answer"].endswith(f.rstrip()) or f in item["answer"]
               for pool in STRUCTURAL_NEGATIVE_FILLERS.values() for f in pool)
    # 最小编辑：原答案内容必须原样保留
    assert "医生给您开的是硝苯地平。" in item["answer"]


def test_structural_negatives_holdout_flag_uses_disjoint_fillers():
    r = _record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass")
    train_out = build_structural_negatives([r], jargon={"硝苯地平"}, holdout=False)
    holdout_out = build_structural_negatives([r], jargon={"硝苯地平"}, holdout=True)
    assert train_out[0]["answer"] != holdout_out[0]["answer"]
    assert not (set(STRUCTURAL_NEGATIVE_FILLERS["train"]) & set(STRUCTURAL_NEGATIVE_FILLERS["holdout"]))


def test_structural_negatives_are_deterministic_across_runs():
    r = _record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass")
    a = build_structural_negatives([r], jargon={"硝苯地平"})
    b = build_structural_negatives([r], jargon={"硝苯地平"})
    assert a == b


def test_structural_negatives_max_items_never_overshoots_on_multi_candidate_records():
    """`/code-review` 发现的真 bug：外层 `if len(out) >= max_items: break`
    每条记录只检查一次，但一条记录可能通过内层循环一次性追加多个候选——
    卡在刚好差 1 条时，下一条记录如果有 N 个候选会整条超发 N-1 条。
    实测过：train 全量 + max_items=1015 时返回 1016 条（该顶格记录
    vt-1536 单条就有 26 个可提取候选，最坏情况会超发到 1040）。裁剪必须
    是硬上限，不是"差不多"。"""
    records = [
        _record("p1", "本药物用于降压治疗，也用于心律失常。",
                "医生给您开的是硝苯地平和普罗帕酮。", "pass"),  # 2 candidates in one record
        _record("p2", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass"),
    ]
    out = build_structural_negatives(records, jargon={"硝苯地平", "普罗帕酮"}, max_items=1)
    assert len(out) == 1


def test_structural_negatives_max_items_caps_output_deterministically():
    """全量装饰会把正例占比再腰斩、逼近 ticket 09/11 坍缩过的区间——
    `max_items` 让调用方把结构性负例数量对齐正例总数（约1:1），不需要
    每条 pass 候选都装饰一遍。裁剪必须确定性（按 case_id 排序取前 N），
    不能是"随便丢几条"。"""
    records = [
        _record(f"p{i}", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass")
        for i in range(10)
    ]
    out_capped = build_structural_negatives(records, jargon={"硝苯地平"}, max_items=3)
    assert len(out_capped) == 3
    out_uncapped = build_structural_negatives(records, jargon={"硝苯地平"})
    assert len(out_uncapped) == 10
    # 裁剪结果必须是未裁剪结果按确定性顺序的前缀
    assert out_capped == out_uncapped[:3]


# ---------- 不确定候选池（第三轮独立审计问题二的修复第一步）----------
#
# `build_trusted_candidate_pool` 把 fail 案例里没命中 evidence 的候选整条
# 丢弃——这个决定本身仍然对（不能当负例用），但代价是候选标签变成"这个
# 候选来自哪个答案"的代理：train/holdout 里没有一个答案同时含正例候选和
# 负例候选，模型从没被逼着在同一答案内部做区分。`build_uncertain_
# candidate_pool` 把这些被丢弃的候选单独枚举出来，供教师直接候选级判定
# （不是从单条 evidence 反推）——判完之后能在同一个 fail 答案里同时出现
# 真正例和真负例。

def test_uncertain_candidate_pool_contains_exactly_the_dropped_candidates():
    r = _record(
        "f1", "请遵医嘱。", "医生给您开的是阿司匹林，另外还有布洛芬。", "fail",
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    trusted = build_trusted_candidate_pool([r], jargon={"阿司匹林", "布洛芬"})
    uncertain = build_uncertain_candidate_pool([r], jargon={"阿司匹林", "布洛芬"})
    trusted_texts = {it["candidate_text"] for it in trusted}
    uncertain_texts = {it["candidate_text"] for it in uncertain}
    assert trusted_texts == {"阿司匹林"}
    assert uncertain_texts == {"布洛芬"}
    # 两个池子对同一个 fail 案例的候选文本互不相交——一个候选不能既确定
    # 又不确定
    assert not (trusted_texts & uncertain_texts)


def test_uncertain_candidate_pool_ignores_pass_cases():
    """pass 案例的候选全部可信（进 build_trusted_candidate_pool 的负例），
    没有"不确定"这一说。"""
    r = _record("p1", "阿司匹林每日一片。", "医生给您开的是硝苯地平。", "pass")
    uncertain = build_uncertain_candidate_pool([r], jargon={"硝苯地平"})
    assert uncertain == []


def test_uncertain_candidate_pool_items_carry_full_case_context():
    """教师直接判定需要完整的 source/answer 上下文，不能只给候选文本。"""
    r = _record(
        "f1", "请遵医嘱。", "医生给您开的是阿司匹林，另外还有布洛芬。", "fail",
        red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")],
    )
    uncertain = build_uncertain_candidate_pool([r], jargon={"阿司匹林", "布洛芬"})
    assert len(uncertain) == 1
    item = uncertain[0]
    assert item["source_text"] and item["answer"]
    assert item["case_id"] == "f1"
    assert item["candidate_text"] == "布洛芬"


# ---------- 把教师直接判定的结果并回可信池（问题二修复的最后一步）----------

def test_apply_teacher_labels_merges_judgment_into_label_field():
    sampled = [
        {"id": "u1", "case_id": "f1", "source_text": "s", "answer": "a",
         "candidate_text": "布洛芬", "kind": "lexicon", "red_line_guess": 0},
    ]
    out = apply_teacher_labels(sampled, judgments={"u1": True})
    assert len(out) == 1
    assert out[0]["label"] is True
    assert out[0]["candidate_text"] == "布洛芬"
    assert "id" not in out[0]  # id 是采样/派发用的，不该混进最终候选池


def test_apply_teacher_labels_skips_items_without_a_judgment():
    """教师标注不齐全时，宁可少几条候选，也不能悄悄给个默认标签。"""
    sampled = [
        {"id": "u1", "case_id": "f1", "candidate_text": "布洛芬"},
        {"id": "u2", "case_id": "f2", "candidate_text": "阿司匹林"},
    ]
    out = apply_teacher_labels(sampled, judgments={"u1": True})
    assert len(out) == 1
    assert out[0]["candidate_text"] == "布洛芬"


def test_apply_teacher_labels_preserves_both_true_and_false_judgments():
    sampled = [
        {"id": "u1", "case_id": "f1", "candidate_text": "布洛芬"},
        {"id": "u2", "case_id": "f1", "candidate_text": "维生素"},
    ]
    out = apply_teacher_labels(sampled, judgments={"u1": True, "u2": False})
    labels = {item["candidate_text"]: item["label"] for item in out}
    assert labels == {"布洛芬": True, "维生素": False}


# ---------- 教师直接判定结果的落盘产物（问题二修复：250 条不确定候选的
# 一次性教师标注结果，提交进库，跟 train.jsonl/holdout.jsonl 同级）----------
#
# `apply_teacher_labels` 的输出（250 条，train/holdout 各一份）已跑过一次、
# 落盘为 verifier/teacher_candidates_{train,holdout}.jsonl——教师标注是
# 派发 4 个 agent 跑出来的，不是可以随时重算的纯函数，必须当成跟
# verifier/labels/*.jsonl 同等地位的一次性人工产物提交进库，用一个加载
# 函数读出来，不是每次都重新走一遍标注流程。

def test_load_teacher_candidate_labels_reads_committed_train_split(tmp_path):
    f = tmp_path / "teacher_candidates_train.jsonl"
    rows = [
        {"case_id": "f1", "source_text": "s", "answer": "a",
         "candidate_text": "布洛芬", "kind": "lexicon", "red_line_guess": 0, "label": True},
        {"case_id": "f2", "source_text": "s2", "answer": "a2",
         "candidate_text": "维生素", "kind": "lexicon", "red_line_guess": 0, "label": False},
    ]
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    out = load_teacher_candidate_labels(f)
    assert len(out) == 2
    assert out[0]["label"] is True
    assert out[1]["label"] is False
    assert "id" not in out[0]


def test_load_teacher_candidate_labels_skips_blank_lines(tmp_path):
    f = tmp_path / "teacher_candidates_train.jsonl"
    row = {"case_id": "f1", "candidate_text": "布洛芬", "label": True}
    f.write_text(json.dumps(row, ensure_ascii=False) + "\n\n", encoding="utf-8")
    out = load_teacher_candidate_labels(f)
    assert len(out) == 1


# ---------- 不确定候选的确定性抽样（`/code-review` 发现的可复现性缺口：
# 250 条不确定候选是怎么被抽出来、分配 id、派发给教师的，原本只有一次性
# 脚本，不在仓库里；补一个可重跑、有测试覆盖的入口）----------

def test_sample_uncertain_candidates_assigns_deterministic_ids():
    pool = [
        {"case_id": f"f{i}", "candidate_text": f"词{i}", "source_text": "s", "answer": "a"}
        for i in range(5)
    ]
    a = sample_uncertain_candidates(pool, n=3, prefix="utrain")
    b = sample_uncertain_candidates(pool, n=3, prefix="utrain")
    assert a == b
    assert len(a) == 3
    assert all(it["id"].startswith("utrain-") for it in a)
    assert len({it["id"] for it in a}) == 3


def test_sample_uncertain_candidates_caps_at_pool_size():
    pool = [{"case_id": "f1", "candidate_text": "词", "source_text": "s", "answer": "a"}]
    out = sample_uncertain_candidates(pool, n=10, prefix="utrain")
    assert len(out) == 1


def test_sample_uncertain_candidates_output_feeds_apply_teacher_labels():
    """抽样产物必须跟 `apply_teacher_labels` 期望的输入形状对得上——
    这是这条函数存在的理由：把 `build_uncertain_candidate_pool` 的输出
    接到教师标注派发，再接回 `apply_teacher_labels`。"""
    pool = [
        {"case_id": "f1", "candidate_text": "布洛芬", "source_text": "s", "answer": "a"},
        {"case_id": "f2", "candidate_text": "维生素", "source_text": "s2", "answer": "a2"},
    ]
    sampled = sample_uncertain_candidates(pool, n=2, prefix="utrain")
    judgments = {it["id"]: (it["candidate_text"] == "布洛芬") for it in sampled}
    out = apply_teacher_labels(sampled, judgments)
    labels = {it["candidate_text"]: it["label"] for it in out}
    assert labels == {"布洛芬": True, "维生素": False}
