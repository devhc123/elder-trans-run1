"""测试集校验与冻结的测试（ticket 07）。

冻结机制的全部意义是：**评审能确认你报的分数和你给的数据是同一份**。
所以断言集中在 hash 的两个性质上——同内容必同 hash、改一个字必变 hash。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.validate_testset import (  # noqa: E402
    PROJECT_PARENTS,
    check_coverage,
    check_flags,
    check_leakage,
    dataset_hash,
    write_kpi2_actual,
)

TESTSET = ROOT / "data" / "elder_translate_270.jsonl"
HARD = ROOT / "data" / "hard_subset.jsonl"
KPI = ROOT / "kpi.yaml"


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    if not TESTSET.exists():
        pytest.skip("测试集未合并")
    return [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------- hash 的两个性质 ----------

def test_hash_is_stable_across_runs(cases):
    assert dataset_hash(cases) == dataset_hash(cases)


def test_hash_ignores_row_order(cases):
    """按 id 排序后计算，文件行序变化不该让 hash 漂。"""
    assert dataset_hash(cases) == dataset_hash(list(reversed(cases)))


def test_hash_changes_when_any_field_changes(cases):
    h0 = dataset_hash(cases)
    for mutate in (
        lambda c: c.update(query=c["query"] + "。"),
        lambda c: c.update(key_points=c["key_points"] + ["新增"]),
        lambda c: c.update(difficulty="难"),
        lambda c: c.update(negative_rubric=c["negative_rubric"][:-1]),
    ):
        mod = json.loads(json.dumps(cases))
        mutate(mod[0])
        assert dataset_hash(mod) != h0, "改动未被 hash 检出"


# ---------- 冻结区 ----------

def test_frozen_block_matches_the_actual_dataset(cases):
    import yaml

    if not KPI.exists():
        pytest.skip("kpi.yaml 不存在")
    data = yaml.safe_load(KPI.read_text(encoding="utf-8")) or {}
    if "dataset_sha256" not in data:
        pytest.skip("尚未冻结")
    assert data["dataset_sha256"] == dataset_hash(cases), "kpi.yaml 的 hash 与数据不符——数据被改过却没重新冻结"
    assert data["dataset_n_cases"] == len(cases)
    assert data["dataset_n_scenarios"] == len({c["scenario"] for c in cases})
    assert data["dataset_n_hard"] == sum(1 for c in cases if c["difficulty"] == "难")


def test_hard_subset_is_exactly_the_hard_cases(cases):
    if not HARD.exists():
        pytest.skip("hard 子集未生成")
    hard = [json.loads(l) for l in HARD.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert {c["id"] for c in hard} == {c["id"] for c in cases if c["difficulty"] == "难"}
    assert all(c["difficulty"] == "难" for c in hard)


def test_hard_subset_is_large_enough_to_read(cases):
    """hard 子集是判别力观测口，太小就读不出信号。"""
    n = sum(1 for c in cases if c["difficulty"] == "难")
    assert n >= 30, f"hard 仅 {n} 题"


# ---------- 各校验环节 ----------

def test_no_leakage(cases):
    problems = check_leakage(cases)
    real = [p for p in problems if "跳过" not in p]
    assert not real, "；".join(real)


def test_all_project_parents_are_covered(cases):
    assert not check_coverage(cases), check_coverage(cases)
    covered = {c["parent_class"] for c in cases}
    for p in PROJECT_PARENTS:
        assert p in covered, f"项目书父类 {p} 无题目，指标说明核销会缺项"


def test_data_gap_flags_are_intact(cases):
    assert not check_flags(cases), "；".join(check_flags(cases))


def test_simulated_cases_are_all_medical_orders(cases):
    """simulated 标记只该出现在医嘱转译——它是对外披露「无真实医嘱语料」的依据。"""
    for c in cases:
        if c.get("simulated"):
            assert c["scenario"] == "医嘱转译", f"{c['id']} 不该带 simulated"


# ---------- 检出能力（校验器不能是摆设） ----------

def test_leakage_check_catches_a_planted_leak(cases):
    import sqlite3

    db = ROOT / "index" / "corpus.sqlite"
    if not db.exists():
        pytest.skip("索引不存在")
    con = sqlite3.connect(db)
    row = con.execute("SELECT record_id FROM splits WHERE pool='verifier_train' LIMIT 1").fetchone()
    con.close()
    if not row:
        pytest.skip("无 verifier_train 记录")
    planted = json.loads(json.dumps(cases[:1]))
    planted[0]["provenance"]["record_id"] = row[0]
    assert any("泄漏" in p for p in check_leakage(planted)), "植入的泄漏未被检出"


def test_flag_check_catches_a_missing_marker(cases):
    bad = json.loads(json.dumps([c for c in cases if c["scenario"] == "医嘱转译"][:1]))
    bad[0].pop("simulated", None)
    assert any("simulated" in p for p in check_flags(bad))


def test_flag_check_catches_a_forbidden_word_in_query(cases):
    bad = json.loads(json.dumps([c for c in cases if c["scenario"] == "医嘱转译"][:1]))
    bad[0]["query"] = "我的医嘱单上写着什么？"
    assert any("医嘱单" in p for p in check_flags(bad))


# ---------- kpi2_scenario_coverage.actual 回填 ----------
# kpi2 的 command 就是本脚本本身（不需要 --freeze），所以每次校验通过都该
# 把 actual 填上——它此前一直是 null，从没有脚本真正写过它，是个遗留缺口。

_KPI2_FIXTURE = """\
metrics:

  - id: kpi1_readability
    actual: null
    command: python3 metrics/run_eval.py --live

  - id: kpi2_scenario_coverage
    target: 10
    actual_subscenarios: 12
    actual: null
    command: python3 pipeline/validate_testset.py

  - id: guard_faithfulness
    actual: null
"""


def test_kpi2_actual_is_written_from_distinct_scenario_count(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI2_FIXTURE, encoding="utf-8")
    fake_cases = [{"scenario": s} for s in ["A", "B", "C", "A", "B"]]  # 3 个不同场景

    write_kpi2_actual(fake_cases, kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    kpi2 = next(m for m in data["metrics"] if m["id"] == "kpi2_scenario_coverage")
    assert kpi2["actual"] == 3


def test_kpi2_actual_write_does_not_disturb_other_actual_null_fields(tmp_path):
    """kpi1 和 guard_faithfulness 也有 `actual: null`——朴素的全局字符串替换
    会把它们一起改掉，必须只改 kpi2 那一处。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI2_FIXTURE, encoding="utf-8")
    fake_cases = [{"scenario": s} for s in ["A", "B", "C"]]

    write_kpi2_actual(fake_cases, kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    assert data["metrics"][0]["actual"] is None, "kpi1 的 actual 被误改了"
    assert data["metrics"][2]["actual"] is None, "guard_faithfulness 的 actual 被误改了"


def test_kpi2_actual_write_is_idempotent(tmp_path):
    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI2_FIXTURE, encoding="utf-8")
    fake_cases = [{"scenario": s} for s in ["A", "B", "C"]]

    write_kpi2_actual(fake_cases, kpi_path=kpi)
    once = kpi.read_text(encoding="utf-8")
    write_kpi2_actual(fake_cases, kpi_path=kpi)
    twice = kpi.read_text(encoding="utf-8")
    assert once == twice, "重复写入不该产生累积变化"
