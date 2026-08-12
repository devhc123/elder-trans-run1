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
    FIELD_LABELS,
    MAX_LEN,
    MIN_LEN,
    NAME_MATCH_NOT_MEANINGFUL,
    SCENARIO_SOURCES,
    SIMULATED_SCENARIOS,
    THIN_SCENARIOS,
    clean,
    is_elder_relevant,
    is_excluded_emergency,
    is_excluded_population,
    join_fields,
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


# 候选池「富裕」的门槛：池子至少是所需题数的 3 倍，才谈得上挑挑拣拣。
# 池子薄的场景（急救表全库仅 276 行）只能连正文命中一起收，此时用记录名命中率
# 衡量质量是错误的口径——该管的是「有没有混进不相关的类型」，那由排除表负责。
RICH_POOL_FACTOR = 3


def test_rich_pool_scenarios_mostly_match_on_the_record_name(sampled):
    """池子富裕时，抽样应优先取记录名命中的候选。

    两类例外：
    - NAME_MATCH_NOT_MEANINGFUL：穴位叫「三阳络穴」，名字里不可能有老年词，
      真正的信号在功效正文；
    - 池薄场景：可选的就那么多，挑不动。
    """
    alloc = json.loads(ALLOC.read_text(encoding="utf-8"))["allocation"]
    for scenario, rows in sampled.items():
        if scenario in NAME_MATCH_NOT_MEANINGFUL:
            continue
        pool = CAND / f"{scenario}.jsonl"
        n_pool = len(pool.read_text(encoding="utf-8").splitlines()) if pool.exists() else 0
        if n_pool < alloc[scenario] * RICH_POOL_FACTOR:
            continue
        by_name = sum(1 for r in rows if r["elder_match"].startswith("name:"))
        assert by_name / len(rows) >= 0.7, (
            f"{scenario} 池 {n_pool} 条却仅 {by_name}/{len(rows)} 按记录名命中"
        )


def test_no_non_elder_population_slips_through(sampled):
    """回归：老年词表匹配病名不区分人群，曾抽进「小儿α-地中海贫血」
    「早产儿贫血」「妊娠期糖尿病」「胎盘早剥」等 15 条（5.6%）。"""
    for scenario, rows in sampled.items():
        for r in rows:
            assert not is_excluded_population(r["name"]), f"{scenario}: «{r['name']}»"


def test_no_non_elder_emergency_type_slips_through(sampled):
    """回归：急救库混着野外/工伤类急症，正文提到「昏迷」「出血」就被捞进来，
    曾抽进「断指急救」「海蛇咬伤急救法」「砷中毒急救法」「高原反应处理方法」。"""
    for r in sampled.get("急症处置", []):
        assert not is_excluded_emergency(r["name"]), f"«{r['name']}»"


@pytest.mark.parametrize(
    "name", ["小儿缺铁性贫血", "早产儿贫血", "妊娠期糖尿病", "胎盘早剥", "空腹血糖（产检）"]
)
def test_population_exclusion_catches_known_cases(name):
    assert is_excluded_population(name)


@pytest.mark.parametrize("name", ["老年人前列腺癌", "先天性甲状腺功能减退症", "慢性支气管炎"])
def test_population_exclusion_does_not_overreach(name):
    """先天性/遗传性疾病老人一样带着，排除它们是另一种错误。"""
    assert not is_excluded_population(name)


@pytest.mark.parametrize(
    "name", ["断指急救", "海蛇咬伤急救法", "砷中毒急救法", "高原反应处理方法", "蝎子蛰伤急救"]
)
def test_emergency_exclusion_catches_known_cases(name):
    """「蜇/蛰」是异体字，记录里两种都用，必须都能拦住。"""
    assert is_excluded_emergency(name)


@pytest.mark.parametrize("name", ["老人跌倒急救", "胸痛急救", "室颤急救", "脑梗急救", "晕厥处理方法"])
def test_emergency_exclusion_does_not_overreach(name):
    assert not is_excluded_emergency(name)


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


# ---------- join_fields：多字段拼接必须带标签 ----------
# 回归见 ticket 10——verifier/build_teacher_labels.py 曾经自己重写了一份不带
# 标签的拼接逻辑，权益匹配的裸「否」缺陷原样重现。两个调用方现在共用这一个
# 函数，不允许再各写一份。

def test_join_fields_labels_multi_field_rows():
    text = join_fields(["yibao_status", "cure_way"], ["否", "药物治疗"])
    assert text == "是否医保报销：否\n治疗方式：药物治疗"


def test_join_fields_does_not_label_single_field_rows():
    """单字段时原文本身就是完整段落，不该被硬套一个标签。"""
    text = join_fields(["用法用量"], ["口服，一日一次"])
    assert text == "口服，一日一次"


def test_join_fields_never_emits_a_bare_yes_no():
    """这就是 ticket 08 的那个缺陷本身：裸「否」不知道指什么。"""
    text = join_fields(["yibao_status"] + list(FIELD_LABELS)[:2],
                        ["否"] + ["x"] * 2)
    assert not text.splitlines()[0] == "否"


def test_benefit_matching_never_pulls_the_insurance_field():
    """人工决定（不是补标签就能解决）：live-002 实测显示 13/15 题模型本就
    正确识别「否」在问医保，却仍主动断言/暗示部分能报——更像国内医保覆盖面广
    的先验压过一个孤立布尔位，不是没看懂字段。加标签能不能修好没有实证，
    与其把未验证的假设印进测试集，不如直接不测这一位。见
    pipeline/extract_candidates.py 里 SCENARIO_SOURCES["权益匹配"] 的注释。"""
    insurance_fields = {"yibao_status", "medicalInsurance", "是否医保：", "是否医保"}
    for _lib, _logical, fields in SCENARIO_SOURCES["权益匹配"]:
        assert not (insurance_fields & set(fields)), fields


def test_join_fields_passes_through_unknown_field_names():
    """字段不在 FIELD_LABELS 表里时不硬造标签，原样透传——不能因为标不出来
    就报错或丢内容。"""
    text = join_fields(["some_unmapped_field", "cure_way"], ["原样内容", "药物治疗"])
    assert "原样内容" in text
    assert "治疗方式：药物治疗" in text


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
