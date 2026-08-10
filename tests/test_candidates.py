"""候选抽取与抽样的测试（ticket 05）。

三条硬约束各有对应断言：只来自 test 池、provenance 可回溯、筛选体现老年导向。
最要紧的是第一条——候选一旦漏进 verifier_train 的记录，后面训出来的 verifier
就在自己判自己，而这种泄漏不会报错，只会让分数好看。
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.extract_candidates import (  # noqa: E402
    CORPUS,
    MAX_LEN,
    MIN_LEN,
    NAME_MATCH_NOT_MEANINGFUL,
    SCENARIO_SOURCES,
    SIMULATED_SCENARIOS,
    THIN_SCENARIOS,
    clean,
    is_elder_relevant,
    sample,
)

CAND = ROOT / "candidates"
DB = ROOT / "index" / "corpus.sqlite"
ALLOC = ROOT / "docs" / "faq_classification.json"


@pytest.fixture(scope="module")
def sampled() -> dict[str, list[dict]]:
    if not CAND.exists() or not list(CAND.glob("sampled_*.jsonl")):
        pytest.skip("先跑 pipeline/extract_candidates.py")
    out = {}
    for p in CAND.glob("sampled_*.jsonl"):
        s = p.stem[len("sampled_"):]
        out[s] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


@pytest.fixture(scope="module")
def pools() -> dict[str, set[str]]:
    if not DB.exists():
        pytest.skip("索引不存在")
    con = sqlite3.connect(DB)
    if not con.execute("SELECT name FROM sqlite_master WHERE name='splits'").fetchone():
        pytest.skip("尚未切分")
    out = {}
    for p in ("test", "verifier_train", "reserve"):
        out[p] = {r for (r,) in con.execute("SELECT record_id FROM splits WHERE pool=?", (p,))}
    con.close()
    return out


# ---------- 泄漏（本文件最重要的部分） ----------

def test_all_candidates_come_from_test_pool(sampled, pools):
    for scenario, rows in sampled.items():
        for r in rows:
            assert r["record_id"] in pools["test"], (
                f"{scenario} 的 {r['record_id']} 不在 test 池"
            )


@pytest.mark.parametrize("other", ["verifier_train", "reserve"])
def test_zero_intersection_with_other_pools(sampled, pools, other):
    used = {r["record_id"] for rows in sampled.values() for r in rows}
    assert not (used & pools[other]), f"与 {other} 池有交集"


def test_no_record_is_reused_across_scenarios(sampled):
    """同一条源记录不应同时出现在两个场景，否则两题共享原文。"""
    seen: dict[str, str] = {}
    for scenario, rows in sampled.items():
        for r in rows:
            rid = r["record_id"]
            assert rid not in seen, f"{rid} 同时出现在 {seen[rid]} 与 {scenario}"
            seen[rid] = scenario


# ---------- 配比 ----------

def test_sampled_counts_match_allocation(sampled):
    alloc = json.loads(ALLOC.read_text(encoding="utf-8"))["allocation"]
    for scenario, n in alloc.items():
        assert len(sampled[scenario]) == n, f"{scenario} 抽了 {len(sampled[scenario])}，应为 {n}"
    assert sum(len(v) for v in sampled.values()) == 270


def test_all_twelve_scenarios_present(sampled):
    assert set(sampled) == set(SCENARIO_SOURCES)


# ---------- provenance 可回溯 ----------

def test_provenance_is_complete(sampled):
    for rows in sampled.values():
        for r in rows:
            p = r["provenance"]
            assert p["lib"] and p["logical"] and p["fields"]
            assert isinstance(p["row_idx"], int)
            assert p["record_id"] == r["record_id"]


def test_source_text_can_be_traced_back_to_the_original_record(sampled):
    """随机抽几条真的翻回原始 csv 逐字核对——provenance 不能只是好看。"""
    con = sqlite3.connect(DB)
    checked = 0
    for scenario, rows in sorted(sampled.items()):
        r = rows[0]
        p = r["provenance"]
        row = con.execute(
            "SELECT source_file, encoding FROM files WHERE lib=? AND logical=? LIMIT 1",
            (p["lib"], p["logical"]),
        ).fetchone()
        assert row, f"{p['lib']}/{p['logical']} 不在 files 表"
        src, enc = row
        with open(CORPUS / src, encoding=enc, newline="") as f:
            original = list(csv.DictReader(f))[p["row_idx"]]
        for field in p["fields"]:
            frag = clean(original[field])
            assert frag and frag in r["source_text"], (
                f"{scenario} {r['record_id']} 的字段 {field} 对不上原始记录"
            )
        checked += 1
    con.close()
    assert checked == len(sampled)


# ---------- 老年导向 ----------

def test_every_candidate_records_why_it_is_elder_relevant(sampled):
    for rows in sampled.values():
        for r in rows:
            assert r["elder_match"], f"{r['record_id']} 没记录老年相关性依据"


def test_most_candidates_match_on_the_record_name(sampled):
    """记录名命中的质量高于正文偶然提及，抽样应优先取前者。

    例外见 NAME_MATCH_NOT_MEANINGFUL：穴位叫「三阳络穴」，名字里不可能出现老年词，
    真正的信号在功效正文。对这类场景强求记录名命中是错误的口径。
    """
    for scenario, rows in sampled.items():
        if scenario in NAME_MATCH_NOT_MEANINGFUL:
            continue
        by_name = sum(1 for r in rows if r["elder_match"].startswith("name:"))
        assert by_name / len(rows) >= 0.7, (
            f"{scenario} 仅 {by_name}/{len(rows)} 条按记录名命中"
        )


def test_exempt_scenarios_still_record_a_reason(sampled):
    """免于记录名命中的场景，仍必须有正文命中的依据，不能无条件放行。"""
    for scenario in NAME_MATCH_NOT_MEANINGFUL:
        for r in sampled[scenario]:
            assert r["elder_match"].startswith(("name:", "text:")), r["record_id"]


def test_single_char_terms_do_not_cause_false_matches():
    """回归：「钙」曾命中降钙素、「钠」曾命中头孢曲松钠。"""
    for name in ("降钙素", "注射用头孢曲松钠", "碳酸氢钠片"):
        ok, why = is_elder_relevant(name, "")
        assert not ok, f"{name} 被误判为老年相关（{why}）"


# ---------- 数据缺口标记 ----------

def test_simulated_scenarios_are_flagged(sampled):
    """医嘱转译无真实语料，每条都必须带标记，防止下游当成真实医嘱。"""
    for scenario in SIMULATED_SCENARIOS:
        for r in sampled[scenario]:
            assert r.get("simulated") is True
            assert "治疗方案转译" in r.get("simulated_note", "")


def test_thin_source_scenarios_are_flagged(sampled):
    for scenario in THIN_SCENARIOS:
        for r in sampled[scenario]:
            assert r.get("thin_source") is True


def test_non_simulated_scenarios_are_not_flagged(sampled):
    for scenario, rows in sampled.items():
        if scenario in SIMULATED_SCENARIOS:
            continue
        assert all("simulated" not in r for r in rows), f"{scenario} 被误标 simulated"


# ---------- 文本清洗与长度 ----------

def test_source_text_length_within_bounds(sampled):
    for scenario, rows in sampled.items():
        for r in rows:
            assert MIN_LEN <= r["n_chars"] <= MAX_LEN
            assert len(r["source_text"]) == r["n_chars"]


def test_no_python_list_literals_leak_into_source_text(sampled):
    """回归：部分库把多值字段 str(list) 存进正文，原样出现 ['药物治疗', '支持性治疗']。"""
    for scenario, rows in sampled.items():
        for r in rows:
            assert "['" not in r["source_text"], (
                f"{scenario} {r['record_id']} 残留列表字面量"
            )


def test_clean_converts_list_literals():
    assert clean("['药物治疗', '支持性治疗']") == "药物治疗、支持性治疗"
    assert clean("['手术治疗']") == "手术治疗"


def test_clean_strips_html():
    assert clean("<p>血压偏高</p>") == "血压偏高"


# ---------- 抽样确定性 ----------

def test_sampling_is_deterministic():
    cands = [
        {"record_id": f"lib_t#row{i}", "elder_match": "name:高血压" if i % 3 else "text:血糖"}
        for i in range(200)
    ]
    a = [c["record_id"] for c in sample(cands, 20, "用药咨询")]
    b = [c["record_id"] for c in sample(cands, 20, "用药咨询")]
    assert a == b


def test_sampling_prefers_name_matches():
    cands = [{"record_id": f"x#row{i}", "elder_match": "text:血糖"} for i in range(50)]
    cands += [{"record_id": f"y#row{i}", "elder_match": "name:高血压"} for i in range(10)]
    picked = sample(cands, 10, "用药咨询")
    assert all(c["elder_match"].startswith("name:") for c in picked)
