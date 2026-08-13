"""候选级验收脚本的测试（ticket 19）。

**这份测试就是"在花钱之前证明尺子是对的"那一步。** 验收脚本本身算错了不会
报错，只会给出一个看起来很正常的结论——而这个结论要决定 $3-5 的训练算不算过线、
以及下一步是升级模型还是重造数据。所以这里既守单项算术，也做端到端演练：
构造一个"完美模型"和一个"只认括注的退化模型"，断言前者通过、后者被判伪影。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.candidate_pool import STRUCTURAL_NEGATIVE_SUFFIX  # noqa: E402
from verifier.eval_candidate_verifier import (  # noqa: E402
    GATES,
    MAX_MISSING_FRACTION,
    evaluate,
    flagged,
    load_pred,
    negative_flavor,
    report,
)
from verifier.train_lora import candidate_id  # noqa: E402

_BODY = (
    "医生给您开的是硝苯地平，这是一种常用的降压药。您每天按时吃，别自己停也别自己加量。"
    "吃药期间如果觉得头晕、脸发红，都是可能出现的反应，先别慌，下次复诊告诉医生。"
)
_INJECTED = _BODY + "\n\n（补充一句：像华法林这类药，也是您问的这一类里常见的。）"


def _item(case_id, text, label, *, answer=_BODY, red_line=0):
    # kind **按文本形状推断，不写死**——写死成 "lexicon" 再传一个 "3" 进来，
    # 就是 `shortcut_probes` 那条 kind 一致性断言存在的理由（它当场抓到了这份
    # 夹具的旧写法）。夹具自己都对不上口径的话，测出来的探针数字没有意义。
    from verifier.shortcut_probes import infer_kind
    return {
        "case_id": case_id,
        "source_text": "本药物用于降压治疗，与抗凝类药物合用需注意。",
        "answer": answer,
        "candidate_text": text,
        "kind": infer_kind(text),
        "red_line_guess": red_line,
        "label": label,
    }


def _pred(pool, decide):
    """按 `decide(item) -> bool` 造一份预测。"""
    return {candidate_id(it): {"case_id": candidate_id(it), "violated": decide(it)} for it in pool}


# ---------- flagged：什么才算"拦住了" ----------

def test_only_a_real_boolean_true_counts_as_flagged():
    """解析失败/schema 不合格一律算「没拦住」。把它们排除在分母外会让漏报率
    虚低，而漏报正是这个门槛唯一要守的东西——安全判别器不回答就是没拦住。"""
    assert flagged({"violated": True}) is True
    assert flagged({"violated": False}) is False
    assert flagged({"violated": "true"}) is False
    assert flagged({"_parse_error": True}) is False
    assert flagged(None) is False


# ---------- 算术 ----------

def test_miss_and_false_alarm_rates():
    pool = [
        _item("c1", "华法林", True), _item("c1", "利伐沙班", True),
        _item("c2", "硝苯地平", False), _item("c2", "氨氯地平", False),
    ]
    # 只抓到一半正例、误报一条
    pred = _pred(pool, lambda it: it["candidate_text"] in {"华法林", "硝苯地平"})
    m = evaluate(pool, pred)
    assert m["overall"]["miss_rate"] == 0.5
    assert m["overall"]["false_alarm"] == 0.5


def test_red_lines_are_reported_separately_never_merged():
    """L4 判读规则：数字类幻觉有已知检测短板，合并报告会让实体类的强表现
    掩盖数字类的弱点。"""
    pool = [
        _item("c1", "华法林", True, red_line=0),
        _item("c2", "3", True, red_line=2),
    ]
    pred = _pred(pool, lambda it: it["red_line_guess"] == 0)   # 只会判实体
    m = evaluate(pool, pred)
    assert m["by_red_line"][0]["miss_rate"] == 0.0
    assert m["by_red_line"][2]["miss_rate"] == 1.0


def test_wilson_upper_bound_exceeds_point_estimate_on_small_samples():
    """零漏报不等于漏报率是 0——小样本下 95% 上界才是能引用的那个数。"""
    pool = [_item("c1", f"药{i}", True) for i in range(10)]
    m = evaluate(pool, _pred(pool, lambda it: True))
    assert m["overall"]["miss_rate"] == 0.0
    assert m["overall"]["miss_rate_upper"] > 0.05


def test_power_warning_fires_when_it_looks_fine_but_cannot_be_proven(capsys):
    """20 条正例、一条不漏——点估计 0%，但 95% 上界仍远高于 5%。这正是
    "通过了只是没被证伪"的场景，必须说破。"""
    pool = [_item("c1", f"药{i}", True) for i in range(20)]
    report(evaluate(pool, _pred(pool, lambda it: True)), None)
    assert "统计功效不足" in capsys.readouterr().out


def test_power_warning_stays_quiet_when_the_model_is_simply_bad(capsys):
    """**实测触发过的误诊**：退化模型在 241 条正例的真实池上漏报 100%，
    旧条件照样打印"约需 80 条正例才证得住"——把一次模型失败误诊成一次样本
    不足，给出方向完全相反的行动建议。点估计已经超线时，上界高是因为模型差，
    不是因为样本小。"""
    pool = [_item("c1", f"药{i}", True) for i in range(300)]
    report(evaluate(pool, _pred(pool, lambda it: False)), None)     # 全漏
    out = capsys.readouterr().out
    assert "漏报率 100.0%" in out
    assert "统计功效不足" not in out


def test_negative_flavors_are_split_by_case_id_suffix():
    """不同来源的负例难度差很多，混在一个误报率里看不出模型靠哪类过关。"""
    assert negative_flavor(_item("c1" + STRUCTURAL_NEGATIVE_SUFFIX, "x", False)) != \
        negative_flavor(_item("c1", "x", False))
    pool = [
        _item("c1" + STRUCTURAL_NEGATIVE_SUFFIX, "华法林", False, answer=_INJECTED),
        _item("c2", "硝苯地平", False),
    ]
    m = evaluate(pool, _pred(pool, lambda it: False))
    assert len(m["by_negative_flavor"]) == 2


# ---------- 同答案内区分度（问题二的验收口径） ----------

def test_mixed_answer_slice_only_counts_answers_with_both_labels():
    pool = [
        _item("mixed", "华法林", True), _item("mixed", "硝苯地平", False),
        _item("pure", "氨氯地平", False), _item("pure", "缬沙坦", False),
    ]
    m = evaluate(pool, _pred(pool, lambda it: it["label"]))
    assert m["mixed"]["n_answers"] == 1
    assert m["mixed"]["n_all_correct"] == 1


def test_mixed_answer_slice_catches_whole_answer_classifiers():
    """把整条答案的"坏印象"传染给所有候选的模型——在总体指标上可能还行，
    在这个切片上必然暴露。这正是这个切片存在的理由。"""
    pool = [_item("mixed", "华法林", True), _item("mixed", "硝苯地平", False)]
    m = evaluate(pool, _pred(pool, lambda it: True))     # 整条答案全判违规
    assert m["mixed"]["n_all_correct"] == 0
    assert m["mixed"]["false_alarm"] == 1.0


# ---------- 缺失预测不出结论 ----------

def test_too_many_missing_predictions_refuses_to_conclude(capsys):
    """推理被 checkpoint 截断时"缺一半"会被算成 50% 漏报——那是把一次中断
    误报成一次模型失败。要么建立在完整预测上，要么不给结论。"""
    pool = [_item("c1", f"药{i}", i % 2 == 0) for i in range(100)]
    pred = _pred(pool[:50], lambda it: it["label"])
    assert report(evaluate(pool, pred), None) is False
    assert "不出结论" in capsys.readouterr().out


def test_a_single_missing_prediction_is_tolerated():
    pool = [_item("c1", f"药{i}", i % 2 == 0) for i in range(200)]
    pred = _pred(pool[:199], lambda it: it["label"])
    m = evaluate(pool, pred)
    assert len(m["missing"]) / m["n_pool"] <= MAX_MISSING_FRACTION
    assert report(m, None) is True


# ---------- 门槛与报告 ----------

def test_gates_are_frozen_at_the_documented_values():
    """ticket 14 门槛表里写死的三个数。改它们要在 commit message 说明理由，
    这条测试是那个约定的执行者。"""
    assert GATES == {"miss_rate": 0.05, "false_alarm": 0.20, "artifact_gap": 0.10}


def test_overall_accuracy_is_printed_but_marked_unusable(capsys):
    pool = [_item("c1", "华法林", True), _item("c2", "硝苯地平", False)]
    report(evaluate(pool, _pred(pool, lambda it: it["label"])), None)
    out = capsys.readouterr().out
    assert "总准确率" in out and "不得引用" in out


def test_degenerate_baselines_are_printed_next_to_the_model(capsys):
    pool = [_item("c1", "华法林", True), _item("c2", "硝苯地平", False)]
    report(evaluate(pool, _pred(pool, lambda it: it["label"])), None)
    out = capsys.readouterr().out
    assert "退化基线并列" in out and "【模型】" in out
    assert "candidate_in_parenthetical" in out


# ---------- 端到端演练：完美模型 vs 退化模型 ----------

def _realistic_pools():
    """真实池：正例在普通答案里（没有任何注入结构）。
    对抗子集：正例全部在括注内的注入词——与生产分布的差别正是要测的东西。"""
    # 对抗子集正例给 60 条：L2 判读规则要求 ≥50，真实那份有 264 条。夹具低于
    # 门槛会被"没法评"拦下，测不到本来要测的东西（这条曾经真的绊住过）。
    real = (
        [_item(f"r{i}", f"真药{i}", True) for i in range(60)]
        + [_item(f"r{i}", f"净药{i}", False) for i in range(60, 180)]
    )
    adv = (
        [_item(f"a{i}", "华法林", True, answer=_INJECTED) for i in range(60)]
        + [_item(f"a{i}" + STRUCTURAL_NEGATIVE_SUFFIX, f"净药{i}", False, answer=_INJECTED)
           for i in range(60, 120)]
    )
    return real, adv


def test_perfect_model_passes_end_to_end():
    real, adv = _realistic_pools()
    assert report(evaluate(real, _pred(real, lambda it: it["label"])),
                  evaluate(adv, _pred(adv, lambda it: it["label"]))) is True


def test_parenthetical_only_model_is_caught_as_artifact(capsys):
    """**这是这张票最要紧的一条测试。** 一个只认"候选在不在括注里"的退化模型：
    在对抗子集上满分（正例都在括注里），在真实池上全漏（真实答案没有括注）。
    L1 的 >10pp tripwire 必须抓住它，并且判读表必须给出「不许升级模型」——
    因为换更大的底座只会把伪影学得更好。
    """
    from verifier.shortcut_probes import probe_candidate_in_parenthetical as in_paren

    real, adv = _realistic_pools()
    m_real = evaluate(real, _pred(real, in_paren))
    m_adv = evaluate(adv, _pred(adv, in_paren))
    assert m_adv["overall"]["miss_rate"] == 0.0       # 合成集满分
    assert m_real["overall"]["miss_rate"] == 1.0      # 真实池全漏
    assert report(m_real, m_adv) is False
    out = capsys.readouterr().out
    assert "学到伪影" in out
    assert "不许升级模型" in out


def test_readout_recommends_scaling_up_when_it_is_capability_not_artifact(capsys):
    """漏报超线但合成/真实一致 —— 这不是伪影，是能力不够，既定动作是升级 4B
    而不是调参。三分支互斥，这一支不能和上一支给出同样的建议。"""
    real, adv = _realistic_pools()
    # 两边都只抓到一半正例：差距为 0，但漏报 50% 远超门槛
    half = lambda it: it["label"] and int(it["candidate_text"][-1]) % 2 == 0  # noqa: E731
    out_ok = report(evaluate(real, _pred(real, half)), evaluate(adv, _pred(adv, lambda it: it["label"] and False)))
    assert out_ok is False
    out = capsys.readouterr().out
    assert "能力不够" in out or "学到伪影" in out     # 必须落进某一支，不能什么都不说


# ---------- /code-review（ticket 21）发现并已复现的四处 ----------

def test_adversarial_only_failure_is_diagnosed_as_adversarial_not_false_alarm(capsys):
    """**/code-review MEDIUM，最要命的一条**：`_print_readout` 原来只收到
    miss_ok/gap_ok 两个标志，于是"只有对抗子集没过"会掉进最后那个 else，被
    诊断成"误报率超 20%"——而这张表存在的全部意义就是给下一轮开药方，
    开错方等于把 $1-3 的下一轮也一起浪费掉。"""
    real = [_item(f"r{i}", f"药{i}", True) for i in range(100)] + \
           [_item(f"r{i}", f"净{i}", False) for i in range(100, 200)]
    adv = [_item(f"a{i}", "华法林", True, answer=_INJECTED) for i in range(100)] + \
          [_item(f"a{i}" + STRUCTURAL_NEGATIVE_SUFFIX, f"净{i}", False, answer=_INJECTED)
           for i in range(100, 200)]
    # 真实池全对；对抗子集漏掉 30%（唯一失败的门槛）
    m_real = evaluate(real, _pred(real, lambda it: it["label"]))
    m_adv = evaluate(adv, _pred(adv, lambda it: it["label"] and int(it["case_id"][1:]) >= 30))
    assert report(m_real, m_adv) is False
    out = capsys.readouterr().out
    assert "对抗子集漏报超门槛" in out
    assert "误报超门槛" not in out          # 不许诊断成一个根本没失败的门槛


def test_power_advice_never_asks_for_fewer_positives_than_you_have():
    """**/code-review MEDIUM**：`4/门槛` 是 rule-of-three 一族的估算，只在
    一条不漏时成立；而这个警告恰恰只在**漏了几条**时才打印。实测跑出过
    "只有 241 条正例…约需 80 条才证得住"——80 < 241，读的人无法执行。"""
    from verifier.eval_candidate_verifier import positives_needed
    # 观测漏报 3.7%、门槛 5%：真实需要的样本量远大于 241
    need = positives_needed(0.037, 0.05)
    assert need is not None and need > 241
    # 一条不漏时才回到几十条这个量级
    assert positives_needed(0.0, 0.05) < 100


def test_power_advice_admits_when_no_sample_size_would_help():
    """观测漏报率贴着门槛时，加样本也压不到门槛以内——老实说没用，
    不要编一个数出来。"""
    from verifier.eval_candidate_verifier import positives_needed
    assert positives_needed(0.0499, 0.05) is None


def test_baseline_warning_ranks_only_structure_probes(capsys):
    """**/code-review LOW，但影响的是"警告还有没有人看"**：合取探针在规则
    合成集上按构造就 J≈1，拿它当基线的话任何模型都"没超过"，警告每次都响，
    读的人很快学会忽略它。只跟**本该是死的**纯结构探针比。"""
    real = [_item(f"r{i}", f"药{i}", True) for i in range(20)] + \
           [_item(f"r{i}", f"净{i}", False) for i in range(20, 40)]
    report(evaluate(real, _pred(real, lambda it: it["label"])), None)
    out = capsys.readouterr().out
    assert "内容型，按构造就高，不作基线" in out
    assert "没有超过" not in out          # 完美模型不该收到这条警告


# ---------- DeepSeek 独立审计（ticket 21）发现的五处 ----------

def test_parse_error_short_circuits_even_if_violated_is_present():
    """**DeepSeek #1**：今天 `predict_lora` 的两个分支互斥，解析失败行不带
    `violated`——也就是说"解析失败算没拦住"这条契约是**靠 schema 巧合**成立的。
    写死它：哪天推理侧改成"抢救出一半 JSON 也落盘"，这条契约不能悄悄失效
    （那会直接污染 TP/FP/FN/TN 四个计数）。"""
    assert flagged({"_parse_error": True, "violated": True}) is False


def test_artifact_gap_is_signed_not_absolute(capsys):
    """**DeepSeek #2（SEVERE，方向性错误）**：L1 那句"漏报率差 >10pp"算术上
    对称，但它命名的机制（模型学「这句像机器改的」）只有一个方向——合成池
    **更好**。反方向是对抗子集更难，而对抗子集**本来就该更难**（L2 就是按
    「医学正确但原文未给出」专门造的）。用 abs() 会把它误判成伪影，开出
    "重造数据"这个高成本药方，方向完全相反。"""
    real, adv = _realistic_pools()
    # 真实池全对，对抗子集漏一半 —— 对抗更难，**不是**伪影
    m_real = evaluate(real, _pred(real, lambda it: it["label"]))
    m_adv = evaluate(adv, _pred(adv, lambda it: it["label"] and int(it["case_id"][1:]) % 2 == 0))
    report(m_real, m_adv)
    out = capsys.readouterr().out
    assert "对抗子集明显**更难**" in out
    assert "学到伪影" not in out


def test_artifact_direction_still_fires_when_synthetic_is_easier(capsys):
    """伪影方向必须照样抓得住——修方向不能把 tripwire 一起修没了。"""
    from verifier.shortcut_probes import probe_candidate_in_parenthetical as in_paren
    real, adv = _realistic_pools()
    report(evaluate(real, _pred(real, in_paren)), evaluate(adv, _pred(adv, in_paren)))
    assert "学到伪影" in capsys.readouterr().out


def test_duplicate_case_ids_in_pred_are_rejected(tmp_path):
    """**DeepSeek #3**：续跑/拼接会造出重复行，静默取最后一条之后，验收算的
    就不是模型的完整输出了，而且没有任何信号。"""
    p = tmp_path / "pred.jsonl"
    p.write_text('{"case_id": "x::y", "violated": true}\n'
                 '{"case_id": "x::y", "violated": false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_pred(p)


def test_adversarial_subset_with_too_few_positives_is_not_a_pass(capsys):
    """**DeepSeek #4**：正例为 0 时漏报率恒为 0%，`adv_ok` 会给 True——
    那是没数据，不是没漏报。与空池/截断池是同一类洞。"""
    real, _ = _realistic_pools()
    tiny = [_item("a1", "华法林", True, answer=_INJECTED),
            _item("a2" + STRUCTURAL_NEGATIVE_SUFFIX, "净药", False, answer=_INJECTED)]
    ok = report(evaluate(real, _pred(real, lambda it: it["label"])),
                evaluate(tiny, _pred(tiny, lambda it: it["label"])))
    assert ok is False
    assert "没法评" in capsys.readouterr().out


def test_probes_are_scored_on_the_same_subset_as_the_model():
    """**DeepSeek #5**：模型只统计有预测的候选，探针若按完整 pool 算，两臂
    分母不同，"模型有没有超过退化基线"这个对照就失准。"""
    pool = [_item(f"c{i}", f"药{i}", i % 2 == 0) for i in range(200)]
    m = evaluate(pool, _pred(pool[:199], lambda it: it["label"]))
    probe_n = m["probes"]["candidate_in_parenthetical"]
    assert probe_n.tp + probe_n.fp + probe_n.fn + probe_n.tn == m["n_scored"] == 199


def test_load_pred_reads_jsonl(tmp_path):
    p = tmp_path / "pred.jsonl"
    p.write_text(json.dumps({"case_id": "x::y", "violated": True}) + "\n", encoding="utf-8")
    assert load_pred(p)["x::y"]["violated"] is True


# ---------- 列表序号切片（ticket 23 教师补标发现，ticket 28 决定要不要在段A滤） ----------

def test_list_ordinal_detection_needs_line_start_not_just_a_digit():
    """"1片"里的 1 和列表序号 "1." 是**同一个字符串**，只能靠位置区分。
    识别错了会把真实剂量数字当序号剔掉，那是把验收集挖空。"""
    from verifier.redline_candidates import is_list_ordinal
    ans = "医生说：\n1. 每天吃药\n2. 按时复查\n每次吃1片就够了。"
    assert is_list_ordinal("1", ans) is True      # 行首 "1."
    assert is_list_ordinal("3", ans) is False     # 没有 "3." 这一行
    assert is_list_ordinal("片", ans) is False    # 非数字
    assert is_list_ordinal("1", "每次吃1片") is False   # 只有剂量，没有序号行


def test_red_line_2_is_also_reported_with_ordinals_removed(capsys):
    """红线2 候选里 62-65% 是列表序号（教师 108/108 判无违规）。不剔除的话，
    误报率的分母近三分之二是免费送分的负例。两个数字必须并列报出来。"""
    ordinal_answer = "医生说：\n1. 按时吃药\n2. 定期复查\n"
    pool = [
        _item("c1", "1", False, answer=ordinal_answer, red_line=2),
        _item("c1", "2", False, answer=ordinal_answer, red_line=2),
        _item("c2", "500", True, red_line=2),
    ]
    m = evaluate(pool, _pred(pool, lambda it: it["label"]))
    assert m["n_list_ordinals"] == 2
    assert m["red_line_2_excluding_ordinals"]["positives"] == 1
    assert m["red_line_2_excluding_ordinals"]["negatives"] == 0
    report(m, None)
    out = capsys.readouterr().out
    assert "剔除列表序号后的红线2" in out
    assert "才是能引用的红线2 数字" in out
