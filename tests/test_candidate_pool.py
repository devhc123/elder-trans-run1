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
    apply_teacher_labels,
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


# ---------- 补标剩余不确定候选（ticket 23） ----------

def _uncertain(case_id, text):
    return {"case_id": case_id, "candidate_text": text, "source_text": "s", "answer": "a",
            "kind": "lexicon", "red_line_guess": 0}


def test_sample_excludes_already_labelled_by_content_not_by_rank():
    """**必须按 `(case_id, candidate_text)` 排除，不能按名次排除。**

    历史上那 250 条是一次性脚本抽的、脚本没入库，它的名次顺序无从复现——
    "跳过前 100 名"会跳错人：可能重复派发已经标过的，也可能永远漏掉某些条目。
    内容键是唯一稳的锚。"""
    pool = [_uncertain(f"c{i}", f"药{i}") for i in range(6)]
    done = {(pool[0]["case_id"], pool[0]["candidate_text"]),
            (pool[3]["case_id"], pool[3]["candidate_text"])}
    out = sample_uncertain_candidates(pool, n=None, prefix="u2", exclude=done)
    got = {(it["case_id"], it["candidate_text"]) for it in out}
    assert len(out) == 4
    assert got.isdisjoint(done)


def test_sample_with_n_none_takes_everything_remaining():
    """train 侧是**全量补标**（252 条），不是抽样——n=None 表示"剩下的全要"。"""
    pool = [_uncertain(f"c{i}", f"药{i}") for i in range(5)]
    assert len(sample_uncertain_candidates(pool, n=None, prefix="u2")) == 5


def test_sample_ids_do_not_collide_with_the_earlier_batch():
    """新一批的 id 前缀必须和历史那批区分开，否则 `apply_teacher_labels` 的
    judgments 字典会张冠李戴——而它是按 id 查的，撞了不会报错。"""
    pool = [_uncertain(f"c{i}", f"药{i}") for i in range(3)]
    ids = {it["id"] for it in sample_uncertain_candidates(pool, n=None, prefix="utrain2")}
    assert all(i.startswith("utrain2-") for i in ids)
    assert len(ids) == 3


def test_sampling_is_deterministic_under_exclusion():
    pool = [_uncertain(f"c{i}", f"药{i}") for i in range(20)]
    ex = {(pool[2]["case_id"], pool[2]["candidate_text"])}
    a = sample_uncertain_candidates(pool, n=5, prefix="u2", exclude=ex)
    b = sample_uncertain_candidates(pool, n=5, prefix="u2", exclude=ex)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert [x["candidate_text"] for x in a] == [x["candidate_text"] for x in b]


# ---------- 注入负例重写（ticket 26） ----------

def _pass_rec(case_id, source, answer):
    return {"case_id": case_id,
            "input": f"【原文】\n{source}\n\n【回答】\n{answer}\n\n【要点】\n0. x",
            "output": '{"key_points": [], "red_lines": [], "verdict": "pass"}'}


def test_injection_negatives_carry_distinct_suffixes_per_flavor():
    """两类负例难度差很多（grounded 是字符串查表就能做对，等价形式要真读懂），
    验收要分开报误报率——靠 case_id 后缀区分，两个后缀都是常量、不写字面量。"""
    from verifier.candidate_pool import (EQUIVALENT_FORM_SUFFIX, STRUCTURAL_NEGATIVE_SUFFIX,
                                         build_injection_negatives)
    recs = [_pass_rec("c1", "本品为甲硝唑片，连续用药不超过3天。", "请遵医嘱服药。")]
    out = build_injection_negatives(recs, {"甲硝唑"})
    suffixes = {it["case_id"].rsplit("-", 1)[-1] for it in out}
    assert suffixes <= {STRUCTURAL_NEGATIVE_SUFFIX.lstrip("-"), EQUIVALENT_FORM_SUFFIX.lstrip("-")}
    assert all(it["label"] is False for it in out)


def test_injection_negatives_respect_per_flavor_caps():
    """两类要能**分别**配平——等价形式全是数字类，实体全是词表类，用一个总数
    上限截断会让 kind 分布随机倾斜。"""
    from verifier.candidate_pool import build_injection_negatives
    recs = [_pass_rec(f"c{i}", "本品为甲硝唑片，连续用药不超过3天。", "请遵医嘱。")
            for i in range(10)]
    out = build_injection_negatives(recs, {"甲硝唑"}, max_entity=3, max_equivalent=2)
    assert sum(1 for it in out if it["kind"] == "lexicon") == 3
    assert sum(1 for it in out if it["kind"] == "cn_numeral") == 2


def test_injection_negatives_are_deterministic():
    from verifier.candidate_pool import build_injection_negatives
    recs = [_pass_rec(f"c{i}", "本品为甲硝唑片，连续用药不超过3天。", "请遵医嘱。")
            for i in range(10)]
    a = build_injection_negatives(recs, {"甲硝唑"}, max_entity=4, max_equivalent=4)
    b = build_injection_negatives(recs, {"甲硝唑"}, max_entity=4, max_equivalent=4)
    assert [x["case_id"] for x in a] == [x["case_id"] for x in b]
    assert [x["candidate_text"] for x in a] == [x["candidate_text"] for x in b]


def test_injection_negatives_only_touch_pass_cases():
    """fail 案例的答案里本来就有真违规，往上面再叠注入会让标签失去意义。"""
    from verifier.candidate_pool import build_injection_negatives
    fail = {"case_id": "f1",
            "input": "【原文】\n本品为甲硝唑片。\n\n【回答】\n请遵医嘱。\n\n【要点】\n0. x",
            "output": '{"key_points": [], "red_lines": [], "verdict": "fail"}'}
    assert build_injection_negatives([fail], {"甲硝唑"}) == []


def test_zero_candidate_fillers_are_gone():
    """零候选安慰语被两类注入负例取代。留着它只会让"括注里没有候选"重新成为
    一个可学特征——那正是 ticket 17 探针抓出来的位置捷径的来源。"""
    import verifier.candidate_pool as cp
    assert not hasattr(cp, "STRUCTURAL_NEGATIVE_FILLERS")
