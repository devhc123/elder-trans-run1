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

from metrics.run_eval import BATCH, SYSTEM_PROMPT, build_user_prompt, load_cases, mock_output  # noqa: E402
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
