"""verifier 验收器的测试（ticket 09）。

验收器的唯一职责是**拒绝不合格的模型**。所以测试的重点不是"算得对不对"，
而是"该拒的能不能拒得掉"——一个永远放行的验收器比没有验收器更糟，因为它
制造了已经把关过的错觉。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import (  # noqa: E402
    SAMPLE_MIX,
    UNIVERSAL_RED_LINES,
    auto_key_points,
    bucket,
)
from verifier.eval_verifier import GATES, evaluate, kappa  # noqa: E402

LABELS = ROOT / "verifier" / "labels"


# ---------- κ ----------

def test_kappa_edges():
    assert kappa([], []) == 0.0
    assert kappa([True] * 10, [True] * 10) == 1.0
    assert kappa([True, False] * 10, [False, True] * 10) < 0


def test_kappa_is_stricter_than_agreement_when_one_class_dominates():
    """一类占绝对多数时，一致率虚高而 κ 不会——这正是用 κ 的理由。

    红线违规是少数类，若只看一致率，一个「全判没违规」的模型也能拿高分。
    """
    gold = [False] * 95 + [True] * 5
    pred = [False] * 100          # 全判没违规
    agree = sum(1 for a, b in zip(gold, pred) if a == b) / len(gold)
    assert agree == 0.95          # 一致率好看
    assert kappa(gold, pred) == 0.0   # κ 诚实地给 0


# ---------- 门槛不对称 ----------

def test_gates_are_deliberately_asymmetric():
    """漏报门槛必须远严于误报——这是本项目最重要的一条设计决定。

    红线漏报是放过一个用药风险，误报只是多拦一条。
    """
    assert GATES["redline_miss_rate"] < GATES["redline_false_alarm"]
    assert GATES["redline_miss_rate"] <= 0.05


def _pack(cid, cov_gold, cov_pred, rl_gold, rl_pred):
    g = {"case_id": cid,
         "key_points": [{"idx": i, "covered": c, "evidence": ""} for i, c in enumerate(cov_gold)],
         "red_lines": [{"idx": i, "violated": v, "evidence": ""} for i, v in enumerate(rl_gold)],
         "verdict": "fail" if any(rl_gold) else "pass"}
    p = {"case_id": cid,
         "key_points": [{"idx": i, "covered": c, "evidence": ""} for i, c in enumerate(cov_pred)],
         "red_lines": [{"idx": i, "violated": v, "evidence": ""} for i, v in enumerate(rl_pred)],
         "verdict": "fail" if any(rl_pred) else "pass"}
    return g, p


def test_a_model_that_never_flags_violations_is_rejected():
    """「全判没违规」的退化模型必须被拒——它的总准确率很高。"""
    gold, pred = {}, {}
    for i in range(40):
        has = i < 8   # 20% 有违规
        g, p = _pack(f"c{i}", [True] * 3, [True] * 3, [has, False], [False, False])
        gold[g["case_id"]] = g
        pred[p["case_id"]] = p
    m = evaluate(pred, gold)
    assert m["redline_miss_rate"] == 1.0
    assert m["redline_miss_rate"] > GATES["redline_miss_rate"]
    # 而它的总准确率很高——正是不能用总准确率的理由
    acc = (m["redline_tp"] + m["redline_tn"]) / m["redline_n"]
    assert acc > 0.85


def test_a_model_that_flags_everything_is_rejected_too():
    """「全判违规」也不行——漏报 0 但误报爆表。"""
    gold, pred = {}, {}
    for i in range(40):
        has = i < 8
        g, p = _pack(f"c{i}", [True] * 3, [True] * 3, [has, False], [True, True])
        gold[g["case_id"]] = g
        pred[p["case_id"]] = p
    m = evaluate(pred, gold)
    assert m["redline_miss_rate"] == 0.0
    assert m["redline_false_alarm"] > GATES["redline_false_alarm"]


def test_a_good_model_passes():
    gold, pred = {}, {}
    for i in range(60):
        has = i < 12
        # 漏报 0，偶尔误报
        pv = [has, i % 12 == 0]
        g, p = _pack(f"c{i}", [True] * 4, [True] * 4, [has, False], pv)
        gold[g["case_id"]] = g
        pred[p["case_id"]] = p
    m = evaluate(pred, gold)
    assert m["redline_miss_rate"] <= GATES["redline_miss_rate"]
    assert m["redline_false_alarm"] <= GATES["redline_false_alarm"]
    assert m["coverage_agreement"] >= GATES["coverage_agreement"]


def test_missing_predictions_are_reported_not_silently_skipped():
    """预测缺条必须显式报出来——静默跳过会让分母变小、分数虚高。"""
    gold, pred = {}, {}
    for i in range(10):
        g, p = _pack(f"c{i}", [True], [True], [False], [False])
        gold[g["case_id"]] = g
        if i < 5:
            pred[p["case_id"]] = p
    m = evaluate(pred, gold)
    assert len(m["missing"]) == 5
    assert m["n_cases"] == 5


# ---------- 难例分桶 ----------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("一次10毫升，一日2次。", "thin_source"),
        ("本品与抗血小板类药物合用时需注意。" + "填充" * 80, "category"),
        ("是否医保：否。" + "补充说明" * 40, "negation"),
        ("必要时可加量，" + "其他说明" * 40, "conditional"),
    ],
)
def test_bucketing(text, expected):
    assert bucket(text) == expected


def test_sample_mix_sums_to_one():
    assert abs(sum(SAMPLE_MIX.values()) - 1.0) < 1e-9


def test_plain_bucket_exists_as_a_control_arm():
    """没有对照臂就读不出难例采样是否真的更难。"""
    assert SAMPLE_MIX.get("plain", 0) > 0


# ---------- 自动要点 ----------

def test_auto_key_points_drops_uninformative_fragments():
    kps = auto_key_points("好。" + "这是一条足够长的、承载事实的说明句子。" * 3)
    assert all(len(k) >= 12 for k in kps)


def test_auto_key_points_caps_and_keeps_the_tail():
    """尾部要点正是「长原文截断压力」要考的地方，不能只取开头。"""
    text = "。".join(f"这是第{i}条足够长的事实说明句子内容" for i in range(30))
    kps = auto_key_points(text)
    assert len(kps) <= 6
    assert "29" in kps[-1] or "28" in kps[-1]


# ---------- 通用红线 ----------

def test_universal_red_lines_cover_the_measured_failure_modes():
    """五条红线逐条对应 ticket 08 实测，不是设想的。"""
    blob = " ".join(UNIVERSAL_RED_LINES)
    for kw in ("类别", "说反", "编造", "立即就医", "顺着"):
        assert kw in blob


def test_sycophancy_red_line_is_kept_as_a_control():
    """谄媚实测几乎没被踩，但必须保留——没有它就无法证明「模型不谄媚」
    是事实而非我们没测。"""
    assert any("顺着" in r or "附和" in r for r in UNIVERSAL_RED_LINES)


# ---------- 已落盘的教师标注 ----------

@pytest.fixture(scope="module")
def labels():
    if not LABELS.exists() or not list(LABELS.glob("*.jsonl")):
        pytest.skip("教师标注尚未产出")
    out = []
    for p in sorted(LABELS.glob("*.jsonl")):
        out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def test_teacher_labels_have_consistent_verdict(labels):
    for j in labels:
        any_v = any(r["violated"] for r in j["red_lines"])
        assert j["verdict"] == ("fail" if any_v else "pass"), j["case_id"]


def test_teacher_labels_have_no_duplicates(labels):
    ids = [j["case_id"] for j in labels]
    assert len(ids) == len(set(ids))


# ---------- 统计功效（验收集太小时必须出声） ----------

def test_small_holdout_cannot_prove_the_miss_rate_gate(capsys):
    """即便一条不漏，20 条正例也证不了「漏报 ≤5%」——CI 上界仍有 16%。

    不把这点说破，会造成"验收通过了"的错觉。这条测试守的是**诚实**，
    不是正确性。
    """
    from verifier.eval_verifier import report

    gold, pred = {}, {}
    for i in range(100):
        has = i < 20                       # 20 条正例
        g, p = _pack(f"c{i}", [True] * 3, [True] * 3, [has, False], [has, False])
        gold[g["case_id"]] = g
        pred[p["case_id"]] = p
    m = evaluate(pred, gold)
    assert m["redline_miss_rate"] == 0.0   # 完美：一条不漏
    report(m)
    out = capsys.readouterr().out
    assert "统计功效不足" in out, "小样本下没有发出功效警告"
    assert "没被证伪" in out


def test_large_holdout_does_not_warn(capsys):
    from verifier.eval_verifier import report

    gold, pred = {}, {}
    for i in range(1000):
        has = i < 300
        g, p = _pack(f"c{i}", [True] * 3, [True] * 3, [has, False], [has, False])
        gold[g["case_id"]] = g
        pred[p["case_id"]] = p
    report(evaluate(pred, gold))
    assert "统计功效不足" not in capsys.readouterr().out
