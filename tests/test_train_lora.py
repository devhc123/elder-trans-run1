"""LoRA 训练脚本的测试（ticket 09）。

训练本身跑在 RunPod，本机测不了。所以这里守的是**配置约束**——那些错了
不会报错、只会让训练白跑或结果失真的地方。每一条都对应一份核实过的依据。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.train_lora import (  # noqa: E402
    BASE_MODEL,
    FALLBACK_MODEL,
    MAX_SEQ,
    RESPONSE_MARKER,
    SCALE_UP_MODEL,
    SYSTEM,
    SYSTEM_CANDIDATE,
    build_candidate_prompt,
    build_candidate_target,
    build_candidate_training_pool,
    build_prompt,
    build_target,
    infer_is_positive,
    make_candidate_dataset,
    make_dataset,
    oversample_positives,
    render_prompt,
)


# ---------- 基座选型（依据见 docs/DATASETS_AND_BENCHMARKS.md §4） ----------

def test_uses_base_not_instruct():
    """必须用 Base：判官输出是固定 JSON schema，不需要对话能力；
    且 Unsloth 官方警告 Qwen3.5-2B **指令版**更易陷入 thinking 死循环，
    批量跑时是不终止生成的灾难。"""
    for m in (BASE_MODEL, FALLBACK_MODEL, SCALE_UP_MODEL):
        assert m.endswith("-Base"), m


def test_all_candidate_models_are_apache_licensed_families():
    """Qwen2.5-3B 是 qwen-research 许可证（限制商用），不能混进来。"""
    for m in (BASE_MODEL, FALLBACK_MODEL, SCALE_UP_MODEL):
        assert "Qwen2.5" not in m, f"{m} 可能落在 qwen-research 许可证下"


def test_scale_up_path_is_same_architecture():
    """未达门槛时的升级路径必须同架构同工具链，训练脚本零改动。"""
    assert SCALE_UP_MODEL.startswith("Qwen/Qwen3.5")
    assert BASE_MODEL.startswith("Qwen/Qwen3.5")


# ---------- prompt / target ----------

def _case():
    return {
        "case_id": "vt-0001",
        "source_text": "本品与抗血小板类药物合用时需注意出血风险。",
        "answer": "您这个药和阿司匹林一起吃要小心出血。",
        "key_points": ["与抗血小板类药物合用需注意出血风险"],
        "red_lines": ["把类别替换成具体药名", "把原文事实说反"],
    }


def _label():
    return {
        "case_id": "vt-0001",
        "key_points": [{"idx": 0, "covered": True, "evidence": "一起吃要小心出血"}],
        "red_lines": [
            {"idx": 0, "violated": True, "evidence": "阿司匹林"},
            {"idx": 1, "violated": False, "evidence": ""},
        ],
        "verdict": "fail",
    }


def test_prompt_contains_all_four_inputs():
    p = build_prompt(_case())
    for part in ("原文", "回答", "要点", "红线"):
        assert part in p
    assert _case()["source_text"] in p and _case()["answer"] in p


def test_prompt_numbers_items_so_indices_line_up():
    """要点/红线必须带下标——模型输出的 idx 要能和输入对上。"""
    p = build_prompt(_case())
    assert "0. " in p


def test_target_is_single_line_json():
    t = build_target(_label())
    assert "\n" not in t
    parsed = json.loads(t)
    assert parsed["verdict"] == "fail"


def test_target_keeps_evidence():
    """evidence 是可程序化校验的防幻觉抓手，也是汇报凭据，不能省。"""
    parsed = json.loads(build_target(_label()))
    assert parsed["red_lines"][0]["evidence"] == "阿司匹林"


def test_target_preserves_item_counts():
    parsed = json.loads(build_target(_label()))
    assert len(parsed["key_points"]) == 1
    assert len(parsed["red_lines"]) == 2


# ---------- system prompt ----------

def test_system_states_the_source_only_discipline():
    """「医学上对但原文没有 = 仍算违规」是本 verifier 的定义性纪律，
    必须写进 system，否则学出来的是通用医学问答判别器。"""
    assert "只以" in SYSTEM and "原文" in SYSTEM
    assert "医学上正确" in SYSTEM or "医学上对" in SYSTEM


def test_system_forbids_free_form_explanation():
    assert "只输出 JSON" in SYSTEM or "不要解释" in SYSTEM


# ---------- 训练/推理共用的对话模板 ----------

def test_render_prompt_is_the_exact_training_prefix():
    """predict_lora.py 必须复用这个函数——训练时的完整样本文本就是
    render_prompt(...) 紧接 output 紧接 eos；推理 prompt 只是去掉了 output。"""
    system, input_, output, eos = "SYS", "IN", "OUT", "<eos>"
    training_text = render_prompt(system, input_) + output + eos
    assert training_text == f"<|system|>\nSYS\n<|user|>\nIN\n<|assistant|>\nOUT<eos>"
    assert training_text.startswith(render_prompt(system, input_))


# ---------- 数据集构造 ----------

def test_make_dataset_skips_unlabeled_items():
    ds = make_dataset([_case(), {**_case(), "case_id": "vt-9999"}],
                      {"vt-0001": _label()})
    assert len(ds) == 1


def test_make_dataset_rows_are_training_ready():
    ds = make_dataset([_case()], {"vt-0001": _label()})
    r = ds[0]
    assert set(r) == {"case_id", "system", "input", "output"}
    assert all(isinstance(v, str) and v for v in r.values())


def test_make_dataset_keeps_case_id():
    """预测产物要按 case_id 与教师标注对齐（eval_verifier.py 按它取键）——
    丢了这个字段，训完/推完都对不回 verifier/labels/ 里的 gold。"""
    ds = make_dataset([_case()], {"vt-0001": _label()})
    assert ds[0]["case_id"] == "vt-0001"


# ---------- 序列长度 ----------

def test_max_seq_covers_the_actual_data():
    """MAX_SEQ 太小会静默截断样本，训出来的模型看不到红线部分。"""
    work = ROOT / "verifier" / "work" / "to_label.json"
    if not work.exists():
        pytest.skip("待标注包未生成")
    items = json.loads(work.read_text(encoding="utf-8"))
    lens = sorted(len(build_prompt(c)) for c in items if c.get("answer"))
    if not lens:
        pytest.skip("尚无回答")
    p90 = lens[int(len(lens) * 0.9)]
    # 中文约 1 字 ≈ 1 token 的保守估计，留一倍余量给输出
    assert p90 * 2 < MAX_SEQ, f"P90 prompt {p90} 字，MAX_SEQ={MAX_SEQ} 可能不够"


# ---------- 训练数据的可学性（错了不报错，只是训出废物） ----------

from verifier.train_lora import TRAIN_RED_LINES  # noqa: E402

TRAIN = ROOT / "verifier" / "train.jsonl"
HOLD = ROOT / "verifier" / "holdout.jsonl"


def _rows(p):
    if not p.exists():
        pytest.skip(f"{p.name} 未生成")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_dropped_red_lines_are_the_zero_trigger_ones():
    """砍掉的必须是实测零触发的红线 3/4，不是随便砍的。"""
    assert TRAIN_RED_LINES == [0, 1, 2]


def test_training_has_no_constant_false_redline_slot():
    """恒为 false 的槽位让模型学成「这一位永远填 false」，是纯粹的信号稀释。"""
    rows = _rows(TRAIN)
    counts = {i: 0 for i in TRAIN_RED_LINES}
    for r in rows:
        for x in json.loads(r["output"])["red_lines"]:
            if x["violated"]:
                counts[x["idx"]] += 1
    zero = [i for i, c in counts.items() if c == 0]
    assert not zero, f"红线 {zero} 在训练集里零触发，应从训练目标里砍掉"


def test_prompt_and_target_redlines_line_up():
    """prompt 里列几条红线，target 就必须输出几条——对不上模型学不到 idx 对应。"""
    rows = _rows(TRAIN)
    for r in rows[:50]:
        n_prompt = r["input"].count("\n", r["input"].index("【红线】"))
        n_target = len(json.loads(r["output"])["red_lines"])
        assert n_target == len(TRAIN_RED_LINES)
        assert n_prompt >= n_target


def test_key_point_labels_have_both_classes():
    """covered 全 True 会让模型学成常量。"""
    rows = _rows(TRAIN)
    flags = [k["covered"] for r in rows for k in json.loads(r["output"])["key_points"]]
    assert 0 < sum(flags) < len(flags)


def test_holdout_is_disjoint_from_training():
    tr = {r["input"] for r in _rows(TRAIN)}
    ho = {r["input"] for r in _rows(HOLD)}
    assert not (tr & ho), f"训练/验收集重叠 {len(tr & ho)} 条"


def test_case_id_is_disjoint_and_present():
    """predict_lora.py 靠 case_id 把推理结果对回 eval_verifier.py 的 gold——
    每行都要有，且训练/验收两侧不能共享同一个 case_id。"""
    tr = _rows(TRAIN)
    ho = _rows(HOLD)
    tr_ids = [r["case_id"] for r in tr]
    ho_ids = [r["case_id"] for r in ho]
    assert all(tr_ids) and all(ho_ids)
    assert not (set(tr_ids) & set(ho_ids))


def test_holdout_is_enriched_for_positives():
    """正例稀缺时随机切会让验收集更没功效，必须分层。"""
    fr = lambda rows: sum(1 for r in rows if '"verdict": "fail"' in r["output"]) / len(rows)
    assert fr(_rows(HOLD)) > fr(_rows(TRAIN))


def test_rare_redlines_are_flagged_as_underpowered():
    """红线 1 实测只有个位数正例——学不出也验不出，必须有据可查地记下来，
    不能让人拿着这份数据直接去训还以为三条都能学。"""
    rows = _rows(TRAIN)
    counts = {i: 0 for i in TRAIN_RED_LINES}
    for r in rows:
        for x in json.loads(r["output"])["red_lines"]:
            if x["violated"]:
                counts[x["idx"]] += 1
    weak = [i for i, c in counts.items() if c < 20]
    doc = (ROOT / ".scratch" / "elder-trans-harness" / "issues" /
           "09-teacher-labels-pilot.md").read_text(encoding="utf-8")
    for i in weak:
        assert f"红线 {i}" in doc or f"红线{i}" in doc, (
            f"红线 {i} 正例仅 {counts[i]} 条（<20，不足以学习），但票据里没有记录这一点"
        )


# ---------- 正例过采样（ticket 11：训练集 fail 占比 4.5%、红线槽位阳性率约
# 1.6%，09 摸底在类似量级下坍缩成常量输出——只调超参数不解决，训练脚本本身
# 必须能把有效正负比例拉开） ----------

def _row(verdict, tag="x"):
    return {
        "case_id": tag, "system": "s", "input": "i",
        "output": json.dumps({"key_points": [], "red_lines": [], "verdict": verdict}),
    }


def test_oversample_positives_factor_one_is_a_no_op():
    rows = [_row("pass", "a"), _row("fail", "b")]
    assert oversample_positives(rows, 1) == rows


def test_oversample_positives_duplicates_only_fail_rows():
    rows = [_row("pass", "a"), _row("fail", "b")]
    out = oversample_positives(rows, 3)
    assert sum(1 for r in out if r["case_id"] == "a") == 1
    assert sum(1 for r in out if r["case_id"] == "b") == 3


def test_oversample_positives_is_a_noop_with_no_fail_rows():
    rows = [_row("pass", "a"), _row("pass", "b")]
    assert oversample_positives(rows, 5) == rows


def test_oversample_positives_rejects_factor_below_one():
    """factor=0 会把正例全部删光，这不是过采样能干的事，必须显式拒绝。"""
    with pytest.raises(ValueError):
        oversample_positives([_row("fail", "a")], 0)


# ---------- 候选级训练格式（ticket 14：段B判定头，联合JSON→单候选二分类）
# ----------

def _candidate_item(label=True, case_id="vt-0001", candidate_text="阿司匹林"):
    return {
        "case_id": case_id,
        "source_text": "本品与抗血小板类药物合用时需注意。",
        "answer": "医生给您开的是阿司匹林。",
        "candidate_text": candidate_text,
        "label": label,
    }


def test_candidate_system_states_the_source_only_discipline():
    assert "只以" in SYSTEM_CANDIDATE and "原文" in SYSTEM_CANDIDATE
    assert "医学上正确" in SYSTEM_CANDIDATE or "医学上对" in SYSTEM_CANDIDATE


def test_candidate_system_keeps_the_identity_exception_clause():
    """红线0"复述已知身份不算违规"的例外条款（TEACHER_TASK.md 里最容易
    判错的一条）——候选级 rubric 必须原样保留，不能训练时悄悄丢了，否则
    模型会把"这个药叫XX"这类合法复述也判成违规，重演 P2 分歧里
    "心痛定即硝苯地平"那类误判。"""
    assert "例外" in SYSTEM_CANDIDATE
    assert "本来就问的" in SYSTEM_CANDIDATE or "本身的名字" in SYSTEM_CANDIDATE


def test_candidate_system_forbids_free_form_explanation():
    assert "只输出 JSON" in SYSTEM_CANDIDATE or "不要解释" in SYSTEM_CANDIDATE


def test_candidate_prompt_contains_source_answer_and_candidate():
    item = _candidate_item()
    p = build_candidate_prompt(item)
    assert item["source_text"] in p
    assert item["answer"] in p
    assert item["candidate_text"] in p


def test_candidate_target_is_single_line_json_with_violated_key():
    t = build_candidate_target(_candidate_item(label=True))
    assert "\n" not in t
    assert json.loads(t) == {"violated": True}


def test_candidate_target_reflects_label_false():
    t = build_candidate_target(_candidate_item(label=False))
    assert json.loads(t) == {"violated": False}


def test_make_candidate_dataset_rows_are_training_ready():
    ds = make_candidate_dataset([_candidate_item()])
    r = ds[0]
    assert set(r) == {"case_id", "system", "input", "output"}
    assert all(isinstance(v, str) and v for v in r.values())


def test_make_candidate_dataset_case_id_disambiguates_multiple_candidates_per_case():
    """同一案例的多个候选必须映射到不同的训练行 case_id，否则
    predict_lora.py 推理完没法把结果一一对回来。"""
    items = [
        _candidate_item(candidate_text="阿司匹林"),
        _candidate_item(candidate_text="氯吡格雷"),
    ]
    ds = make_candidate_dataset(items)
    ids = {r["case_id"] for r in ds}
    assert len(ids) == 2


def test_make_candidate_dataset_preserves_label_in_output():
    items = [_candidate_item(label=True), _candidate_item(label=False, candidate_text="布洛芬")]
    ds = make_candidate_dataset(items)
    labels = {json.loads(r["output"])["violated"] for r in ds}
    assert labels == {True, False}


# ---------- oversample_positives 泛化：候选级 label 谓词（ticket 14）
# ----------

def _candidate_row(violated, tag="x"):
    return {
        "case_id": tag, "system": "s", "input": "i",
        "output": json.dumps({"violated": violated}),
    }


def test_infer_is_positive_detects_candidate_level_schema():
    """回归（code review 发现）：`train()` 曾经不管数据是案例级还是候选级，
    永远用案例级默认谓词——候选级数据的过采样因此静默变成 no-op（`n_fail`
    恒 0，不报错）。这里守的是自动判定本身：看到 `violated` 字段就用
    候选级谓词。"""
    rows = [_candidate_row(True, "a"), _candidate_row(False, "b")]
    is_positive = infer_is_positive(rows)
    assert is_positive(json.loads(rows[0]["output"])) is True
    assert is_positive(json.loads(rows[1]["output"])) is False


def test_infer_is_positive_detects_case_level_schema():
    rows = [_row("fail", "a"), _row("pass", "b")]
    is_positive = infer_is_positive(rows)
    assert is_positive(json.loads(rows[0]["output"])) is True
    assert is_positive(json.loads(rows[1]["output"])) is False


def test_infer_is_positive_rejects_empty_dataset():
    with pytest.raises(ValueError):
        infer_is_positive([])


def test_infer_is_positive_rejects_unknown_schema():
    rows = [{"case_id": "a", "system": "s", "input": "i",
             "output": json.dumps({"something_else": True})}]
    with pytest.raises(ValueError):
        infer_is_positive(rows)


def test_oversample_positives_works_with_candidate_level_predicate():
    rows = [_candidate_row(False, "a"), _candidate_row(True, "b")]
    out = oversample_positives(rows, 3, is_positive=lambda o: o["violated"])
    assert sum(1 for r in out if r["case_id"] == "a") == 1
    assert sum(1 for r in out if r["case_id"] == "b") == 3


def test_oversample_positives_default_predicate_still_matches_case_level_verdict():
    """回归：泛化成谓词参数后，不传参时行为必须和原来完全一致——这是
    `train()` 已有调用点、以上一整段既有测试全部沿用的默认行为，不能
    悄悄变了。"""
    rows = [_row("pass", "a"), _row("fail", "b")]
    assert oversample_positives(rows, 3) == oversample_positives(
        rows, 3, is_positive=lambda o: o.get("verdict") == "fail"
    )


# ---------- 候选级训练池组装（ticket 14：pass案例负例+fail案例可信正例+
# train切分合成正例） ----------

def _train_style_record(case_id, source, answer, verdict, red_lines=None):
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


def test_build_candidate_training_pool_combines_trusted_and_synthetic():
    records = [
        _train_style_record("p1", "本药物用于降压治疗。", "医生给您开的是硝苯地平。", "pass"),
        _train_style_record("f1", "请遵医嘱。", "医生给您开的是阿司匹林。", "fail",
                             red_lines=[(0, True, "阿司匹林"), (1, False, ""), (2, False, "")]),
        _train_style_record("s1", "本品与磺胺类药合用需注意，复查频率必要时而定。", "请遵医嘱。", "pass"),
    ]
    # 垫够背景记录，避开 synth_minimal_edit 的源文档集中度上限小样本边界行为
    records += [
        _train_style_record(f"pad{i}", "本品属于降压类药物。", "请遵医嘱。", "pass") for i in range(30)
    ]
    pool = build_candidate_training_pool(records)
    labels = {(it["case_id"], it["candidate_text"]): it["label"] for it in pool}

    # 可信池：pass 案例的候选是负例，fail 案例命中 evidence 的候选是正例
    assert labels.get(("p1", "硝苯地平")) is False
    assert labels.get(("f1", "阿司匹林")) is True
    # 合成正例：s1 触发了类别标记和数字标记，产出的合成 case_id 带 -synth 后缀
    synth_positives = [k for k in labels if k[0].startswith("s1-synth")]
    assert synth_positives, "s1 的类别/数字触发应该产出至少一条合成正例"
    assert all(labels[k] is True for k in synth_positives)


# ---------- 只在 assistant 段算 loss（否则 prompt 里的原文/回答会把本就
# 稀缺的红线信号进一步稀释——p90 prompt 2594 字，输出 JSON 只有几十字） ----------

def test_response_marker_matches_render_prompt_tail():
    """train() 靠这个常量告诉 Unsloth 的 train_on_responses_only 从哪里开始
    算 loss——必须与 render_prompt 实际吐出的 assistant 段起始逐字节一致，
    错位不报错，只会让 loss 遮罩偏移、静默训坏。"""
    p = render_prompt("SYS", "IN")
    assert p.endswith(RESPONSE_MARKER)
