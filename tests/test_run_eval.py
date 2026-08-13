"""评测 runner 的测试（ticket 08）。

runner 的职责是「把数算对、把脏样本挑出来」。所以断言集中在两件事：
统计口径不出错，以及**空输出/截断这类脏样本不能被静默计分**——后者是本轮
实测踩到的坑：max_tokens 设小了，11 题正文为空、13 题被截断，而截断的样本
会被当正常样本压低分数且不报错。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.run_eval import (  # noqa: E402
    BATCH,
    SYSTEM_PROMPT,
    build_user_prompt,
    load_cases,
    mock_output,
    write_guard_faithfulness_actual,
    write_kpi1_actual,
    write_kpi1_macro_by_scenario,
)
from metrics.wilson import wilson  # noqa: E402

RUNS = ROOT / "runs"


# ---------- Wilson 区间 ----------

def test_wilson_handles_edges():
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and 0 < hi < 0.2
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and 0.8 < lo < 1.0


def test_wilson_is_within_unit_interval():
    for k in range(0, 51):
        lo, hi = wilson(k, 50)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_narrows_with_more_samples():
    """小样本区间必须明显更宽——场景子集只有 15–47 题，正落在这个区间。"""
    w_small = wilson(9, 10)[1] - wilson(9, 10)[0]
    w_big = wilson(900, 1000)[1] - wilson(900, 1000)[0]
    assert w_small > w_big * 3


def test_wilson_beats_normal_approx_at_the_edge():
    """正态近似在 p 贴近 1 时会越界；Wilson 不会。"""
    lo, hi = wilson(20, 20)
    assert hi <= 1.0


# ---------- prompt ----------

def test_user_prompt_contains_all_three_inputs():
    c = load_cases()[0]
    p = build_user_prompt(c)
    for part in (c["persona"], c["query"], c["source_text"]):
        assert part in p


def test_system_prompt_states_the_four_requirements():
    """被测系统的 prompt 是评测条件的一部分，改它等于换了被测对象。"""
    for kw in ("口语", "术语", "不编造", "就医"):
        assert kw in SYSTEM_PROMPT


# ---------- mock ----------

def test_mock_is_deterministic():
    c = load_cases()[0]
    assert mock_output(c) == mock_output(c)


def test_mock_is_not_a_perfect_answer():
    """mock 必须「像样但不够好」：它要能跑通管线，又不能让 dry-run 的分数
    看起来像真数。"""
    c = load_cases()[0]
    out = mock_output(c)
    assert out and out != c["source_text"]


# ---------- 分片 ----------

def test_batch_size_yields_a_workable_number_of_shards():
    n = len(load_cases())
    shards = (n + BATCH - 1) // BATCH
    assert 15 <= shards <= 25, f"{n} 题分成 {shards} 片，并发规模不合适"


# ---------- 脏样本必须被挑出来（本轮实测踩的坑） ----------

@pytest.fixture(scope="module")
def live_outputs() -> list[dict]:
    p = RUNS / "live-002" / "outputs.jsonl"
    if not p.exists():
        pytest.skip("尚无 live 跑数")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_no_empty_output_in_the_reported_run(live_outputs):
    """回归：max_tokens=1500 时 11 题把额度全烧在推理上、正文为空。

    deepseek-v4-flash 是推理模型，max_tokens 是**含推理在内**的总额度。
    """
    empty = [r["id"] for r in live_outputs if not (r.get("output") or "").strip()]
    assert not empty, f"{len(empty)} 条空输出：{empty[:5]}"


def test_no_truncated_output_in_the_reported_run(live_outputs):
    """回归：截断比空输出更危险——它会被当正常样本计分并压低分数，且不报错。"""
    trunc = [r["id"] for r in live_outputs if r.get("truncated")]
    assert not trunc, f"{len(trunc)} 条被截断：{trunc[:5]}"


def test_every_output_finished_normally(live_outputs):
    bad = [r["id"] for r in live_outputs if r.get("finish_reason") not in (None, "stop")]
    assert not bad, f"finish_reason 异常：{bad[:5]}"


def test_run_covers_the_whole_testset(live_outputs):
    assert len(live_outputs) == len(load_cases())


# ---------- 判官产物 ----------

@pytest.fixture(scope="module")
def judged() -> list[dict]:
    d = RUNS / "live-002" / "judged"
    if not d.exists() or not list(d.glob("*.jsonl")):
        pytest.skip("判官尚未判完")
    out = []
    for p in sorted(d.glob("*.jsonl")):
        out += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def test_judged_schema(judged):
    for j in judged:
        assert isinstance(j.get("faithful_overall"), bool)
        assert isinstance(j.get("any_violated"), bool)
        assert isinstance(j.get("key_points_total"), int)
        assert 0 <= j["key_points_covered"] <= j["key_points_total"]


def test_key_point_flags_line_up(judged):
    """flags 长度必须等于要点数——对不上说明判官漏判或串行了。"""
    for j in judged:
        flags = j.get("key_point_flags")
        assert flags is not None and len(flags) == j["key_points_total"], j["id"]
        assert sum(1 for f in flags if f) == j["key_points_covered"], j["id"]


def test_violated_indices_are_consistent(judged):
    for j in judged:
        vi = j.get("violated_indices", [])
        assert bool(vi) == j["any_violated"], f"{j['id']}: 违规标记与下标不一致"


def test_no_duplicate_judgements(judged):
    ids = [j["id"] for j in judged]
    assert len(ids) == len(set(ids))


# ---------- write_kpi1_actual：把 kpi1_readability 的 actual/ci95/子集达标率
# 从"没脚本真正写过"的历史缺口里补上（ticket 08 的验收清单曾把这件事跟
# 扁平区块的填充误当成一件事打了勾，实际 metrics[0] 这段一直没被动过）。

_KPI1_FIXTURE = """\
metrics:

  - id: kpi1_readability
    name: 适老化转译可读性达标率
    definition: >
      转译后文本中符合小学六年级可读水平的词汇比例（词级微平均）。
    primary_gauge: strict
    target: 0.90
    also_report: [glossed, lenient]
    determinism: 确定性计算（metrics/readability.py），无 LLM 调用
    actual: null
    actual_hard_subset: null
    source_baseline_median: 0.864
    source_already_passing: 83
    actual_net_gain: 0.0903          # 转译后 − 原文，共同主诊断（口径唯一，无歧义）
    actual_on_headroom_subset: null  # 原文未达标的 187 题上的达标率——达标率本身还没算过
    actual_macro_by_scenario: null   # 12 场景表，还没写成结构化数据（目前只印到 stdout）
    ci95: null
    verdict: 达标  # 0.9518（均值）与 0.9741（达标率）两种读法都 ≥ target 0.90
    command: python3 metrics/run_eval.py --live

  - id: kpi2_scenario_coverage
    target: 10
    actual: null
    command: python3 pipeline/validate_testset.py

  - id: guard_faithfulness
    actual: null
    ci95: null
"""


def _fake_rows(n_pass=9, n_fail=1, headroom_pass=3, headroom_fail=1, hard_pass=2, hard_fail=1):
    rows = []
    for i in range(n_pass):
        rows.append({"strict": 0.95, "difficulty": "易", "has_headroom": False})
    for i in range(n_fail):
        rows.append({"strict": 0.50, "difficulty": "易", "has_headroom": False})
    for i in range(headroom_pass):
        rows.append({"strict": 0.95, "difficulty": "中", "has_headroom": True})
    for i in range(headroom_fail):
        rows.append({"strict": 0.50, "difficulty": "中", "has_headroom": True})
    for i in range(hard_pass):
        rows.append({"strict": 0.95, "difficulty": "难", "has_headroom": False})
    for i in range(hard_fail):
        rows.append({"strict": 0.50, "difficulty": "难", "has_headroom": False})
    return rows


def test_kpi1_actual_is_the_pass_rate_not_the_mean(tmp_path):
    """actual 必须是达标率（KPI 名字本身是"达标率"，且 ci95 只对比例有意义），
    不是均值——这是本函数存在的全部意义，写错了等于白写。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")
    rows = _fake_rows()  # 17 条（9+1+3+1+2+1），14 条 strict>=0.90 => 14/17≈0.8235

    write_kpi1_actual(rows, kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1 = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")
    assert k1["actual"] == round(14 / 17, 4)
    assert k1["verdict"] == "未达标"  # 0.82 < target 0.90


def test_kpi1_subset_fields_are_pass_rates(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")
    rows = _fake_rows()

    write_kpi1_actual(rows, kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1 = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")
    assert k1["actual_hard_subset"] == round(2 / 3, 4)
    assert k1["actual_on_headroom_subset"] == round(3 / 4, 4)


def test_kpi1_ci95_is_written_as_a_pair(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_actual(_fake_rows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    ci = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")["ci95"]
    assert isinstance(ci, list) and len(ci) == 2
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_kpi1_write_does_not_disturb_kpi2_or_guard_faithfulness(tmp_path):
    """kpi2 和 guard_faithfulness 也各有 `actual: null`/`ci95: null`——
    朴素的全局替换会把它们一起改掉，必须只改 kpi1 那一段。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_actual(_fake_rows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    kpi2 = next(m for m in data["metrics"] if m["id"] == "kpi2_scenario_coverage")
    gf = next(m for m in data["metrics"] if m["id"] == "guard_faithfulness")
    assert kpi2["actual"] is None
    assert gf["actual"] is None
    assert gf["ci95"] is None


def test_kpi1_actual_net_gain_is_untouched():
    """已经填过的、无歧义的字段不该被这次写入覆盖成别的值——回归防线，
    防止以后有人改这个函数时手滑把已确定的口径也重新计算一遍。"""
    import inspect

    src = inspect.getsource(write_kpi1_actual)
    assert "actual_net_gain" not in src, "write_kpi1_actual 不该碰 actual_net_gain"


def test_kpi1_actual_write_is_idempotent_on_rerun(tmp_path):
    """回归：与 write_guard_faithfulness_actual 同源的 bug——原实现字面匹配
    `actual: null`，第一次跑之后这些字段就不再是 null，重跑会静默 no-op。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_actual(_fake_rows(n_pass=9, n_fail=1), kpi_path=kpi)  # 17 题, 14/17 通过
    first = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1_first = next(m for m in first["metrics"] if m["id"] == "kpi1_readability")
    assert k1_first["actual"] == round(14 / 17, 4)

    # 数据变了：这次全部通过
    write_kpi1_actual(_fake_rows(n_pass=17, n_fail=0, headroom_pass=0, headroom_fail=0, hard_pass=0, hard_fail=0), kpi_path=kpi)
    second = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1_second = next(m for m in second["metrics"] if m["id"] == "kpi1_readability")
    assert k1_second["actual"] == 1.0, "重跑没能覆盖旧数字——静默 no-op 的回归"
    assert k1_second["verdict"] == "达标"


# ---------- write_guard_faithfulness_actual：同样的历史缺口——_write_kpi
# 只把这个数字写进文末扁平区块（`guard_faithfulness: 0.7259`），从没写进
# metrics[] 里 schema 真正指向的 actual/ci95。

def _fake_jrows(n_pass=7, n_fail=3):
    rows = []
    for i in range(n_pass):
        rows.append({"faithful": True, "violated": False})
    for i in range(n_fail):
        rows.append({"faithful": False, "violated": True})
    return rows


def test_guard_faithfulness_actual_is_the_pass_rate(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_guard_faithfulness_actual(_fake_jrows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    gf = next(m for m in data["metrics"] if m["id"] == "guard_faithfulness")
    assert gf["actual"] == round(7 / 10, 4)


def test_guard_faithfulness_ci95_is_written_as_a_pair(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_guard_faithfulness_actual(_fake_jrows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    gf = next(m for m in data["metrics"] if m["id"] == "guard_faithfulness")
    ci = gf["ci95"]
    assert isinstance(ci, list) and len(ci) == 2
    assert 0.0 <= ci[0] <= ci[1] <= 1.0


def test_guard_faithfulness_write_does_not_disturb_kpi1_or_kpi2(tmp_path):
    """全局字符串替换会把 kpi1/kpi2 同名的 `actual: null` 一并改掉——必须
    只改 guard_faithfulness 那一段。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_guard_faithfulness_actual(_fake_jrows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1 = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")
    kpi2 = next(m for m in data["metrics"] if m["id"] == "kpi2_scenario_coverage")
    assert k1["actual"] is None
    assert kpi2["actual"] is None


def test_guard_faithfulness_write_is_a_noop_when_no_judged_rows(tmp_path):
    """尚未判官过的跑数不该往 kpi.yaml 里写假数字。"""
    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")
    before = kpi.read_text(encoding="utf-8")

    write_guard_faithfulness_actual([], kpi_path=kpi)

    assert kpi.read_text(encoding="utf-8") == before


def test_guard_faithfulness_write_is_idempotent_on_rerun(tmp_path):
    """回归：判官数据改变后（比如重判翻正/翻负）必须能重跑覆盖旧数字，
    不能因为 actual 已经不是 null 就静默不写——这是 live-003 复核实测踩的坑：
    第一次跑填了 0.7259，后来 45 题被复核翻负，重跑却因为字面匹配
    `actual: null` 失败而完全没更新，kpi.yaml 里留着过期数字。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_guard_faithfulness_actual(_fake_jrows(n_pass=7, n_fail=3), kpi_path=kpi)
    first = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    gf1 = next(m for m in first["metrics"] if m["id"] == "guard_faithfulness")
    assert gf1["actual"] == round(7 / 10, 4)

    # 判官数据变了：这次只有 2/10 通过
    write_guard_faithfulness_actual(_fake_jrows(n_pass=2, n_fail=8), kpi_path=kpi)
    second = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    gf2 = next(m for m in second["metrics"] if m["id"] == "guard_faithfulness")
    assert gf2["actual"] == round(2 / 10, 4), "重跑没能覆盖旧数字——静默 no-op 的回归"


# ---------- write_kpi1_macro_by_scenario：12 场景可读性宏平均，此前只印到
# aggregate() 的 stdout，从没写成 kpi.yaml 里的结构化数据。

def _fake_macro_rows():
    rows = []
    for i in range(3):
        rows.append({"scenario": "用药咨询", "strict": 0.95, "net_gain": 0.08})
    for i in range(2):
        rows.append({"scenario": "日常照护", "strict": 0.90, "net_gain": 0.05})
    return rows


def test_kpi1_macro_by_scenario_is_written_as_a_list_of_per_scenario_stats(tmp_path):
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_macro_by_scenario(_fake_macro_rows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1 = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")
    macro = k1["actual_macro_by_scenario"]
    assert isinstance(macro, list) and len(macro) == 2
    by_scenario = {row["scenario"]: row for row in macro}
    assert by_scenario["用药咨询"]["n"] == 3
    assert by_scenario["用药咨询"]["strict_mean"] == 0.95
    assert by_scenario["用药咨询"]["net_gain"] == 0.08
    assert by_scenario["日常照护"]["n"] == 2


def test_kpi1_macro_by_scenario_does_not_disturb_other_fields(tmp_path):
    """局部替换必须只碰 actual_macro_by_scenario 这一行——同一段里 actual/ci95/
    kpi2/guard_faithfulness 的 null 不该被牵连。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_macro_by_scenario(_fake_macro_rows(), kpi_path=kpi)

    data = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    k1 = next(m for m in data["metrics"] if m["id"] == "kpi1_readability")
    kpi2 = next(m for m in data["metrics"] if m["id"] == "kpi2_scenario_coverage")
    gf = next(m for m in data["metrics"] if m["id"] == "guard_faithfulness")
    assert k1["actual"] is None
    assert k1["ci95"] is None
    assert kpi2["actual"] is None
    assert gf["actual"] is None


def test_kpi1_macro_by_scenario_is_a_noop_when_no_rows(tmp_path):
    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")
    before = kpi.read_text(encoding="utf-8")

    write_kpi1_macro_by_scenario([], kpi_path=kpi)

    assert kpi.read_text(encoding="utf-8") == before


def test_kpi1_macro_by_scenario_write_is_idempotent_on_rerun(tmp_path):
    """回归：与另外两个 write_* 同源的 bug——第一次写完是多行 list，不再是
    字面 `null` 单行，重跑必须仍能替换掉，而不是静默 no-op。"""
    import yaml

    kpi = tmp_path / "kpi.yaml"
    kpi.write_text(_KPI1_FIXTURE, encoding="utf-8")

    write_kpi1_macro_by_scenario(_fake_macro_rows(), kpi_path=kpi)
    first = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    macro_first = next(m for m in first["metrics"] if m["id"] == "kpi1_readability")["actual_macro_by_scenario"]
    assert {row["scenario"] for row in macro_first} == {"用药咨询", "日常照护"}

    # 场景数据变了：换成一个新场景
    new_rows = [{"scenario": "运动与养生", "strict": 0.88, "net_gain": 0.03}]
    write_kpi1_macro_by_scenario(new_rows, kpi_path=kpi)
    second = yaml.safe_load(kpi.read_text(encoding="utf-8"))
    macro_second = next(m for m in second["metrics"] if m["id"] == "kpi1_readability")["actual_macro_by_scenario"]
    assert {row["scenario"] for row in macro_second} == {"运动与养生"}, "重跑没能覆盖旧数据——静默 no-op 的回归"
