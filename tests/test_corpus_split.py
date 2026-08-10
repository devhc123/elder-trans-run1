"""语料索引与 record_id 三分切的测试（ticket 03）。

切分的唯一职责是**防泄漏**，所以断言集中在互斥性与确定性上。
索引部分则守两件事：文件名乱码必须显式认定不许猜，编码嗅探不许静默失败。
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.split_records import SPLITS, assign, read_seed, verify  # noqa: E402
from pipeline.source_index import IndexError_, logical_name  # noqa: E402

DB = ROOT / "index" / "corpus.sqlite"


# ---------- 切分：纯函数部分（无需索引） ----------

def test_assign_is_deterministic():
    for rid in ("bdyd_疾病#row0", "others_drug_info#row17709", "ylys_指标#row42"):
        assert assign(rid, 20260810) == assign(rid, 20260810)


def test_assign_changes_with_seed():
    """种子换了切分必须换，否则种子就是摆设。"""
    rids = [f"bdyd_疾病#row{i}" for i in range(500)]
    a = [assign(r, 1) for r in rids]
    b = [assign(r, 2) for r in rids]
    assert a != b


def test_assign_only_returns_declared_pools():
    pools = {n for n, _ in SPLITS}
    for i in range(2000):
        assert assign(f"x#row{i}", 7) in pools


def test_assign_roughly_matches_declared_ratios():
    n = 20000
    counts: dict[str, int] = {}
    for i in range(n):
        counts[assign(f"lib_t#row{i}", 20260810)] = counts.get(assign(f"lib_t#row{i}", 20260810), 0) + 1
    for name, pct in SPLITS:
        got = 100 * counts.get(name, 0) / n
        assert abs(got - pct) < 2.0, f"{name} 实际 {got:.1f}% vs 目标 {pct}%"


def test_seed_is_read_from_kpi_yaml():
    """kpi.yaml 是单一真相源；早期手写解析被行内注释噎住过。"""
    assert isinstance(read_seed(), int)


# ---------- 索引：逻辑表名解析 ----------

def test_logical_name_passes_through_readable_names():
    assert logical_name("bdyd", "疾病") == "疾病"
    assert logical_name("others", "drug_info") == "drug_info"


def test_logical_name_resolves_known_mojibake():
    """乱码名走显式映射，映射内容由实际记录内容认定。"""
    assert logical_name("bdyd", "ҩƷ") == "药品"
    assert logical_name("bdyd", "Ѩλ") == "穴位"


def test_logical_name_refuses_to_guess():
    """未认定的乱码名必须报错，不许静默生成一个看着像中文的垃圾名。

    实测反解 'žąūČ' 会得到 '啪膮奴膶' —— 全是汉字、看着像模像样，实际是错的。
    这类"貌似成功"的降级比直接失败危险得多。
    """
    with pytest.raises(IndexError_, match="LOGICAL_NAMES"):
        logical_name("bdyd", "Ω≈ß†")


# ---------- 切分：落库后的完整性（需索引） ----------

@pytest.fixture(scope="module")
def db():
    if not DB.exists():
        pytest.skip("索引未构建，先跑 pipeline/source_index.py 与 split_records.py")
    con = sqlite3.connect(DB)
    if not con.execute("SELECT name FROM sqlite_master WHERE name='splits'").fetchone():
        pytest.skip("尚未切分，先跑 pipeline/split_records.py")
    yield con
    con.close()


def test_pools_are_mutually_exclusive_and_complete(db):
    problems = verify(db)
    assert not problems, "；".join(problems)


def test_every_record_has_exactly_one_pool(db):
    n_rec = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    n_spl = db.execute("SELECT COUNT(DISTINCT record_id) FROM splits").fetchone()[0]
    assert n_rec == n_spl > 0


def test_no_test_record_leaks_into_verifier_train(db):
    """本文件存在的全部理由。"""
    n = db.execute(
        "SELECT COUNT(*) FROM (SELECT record_id FROM splits WHERE pool='test' "
        "INTERSECT SELECT record_id FROM splits WHERE pool='verifier_train')"
    ).fetchone()[0]
    assert n == 0


def test_record_ids_are_unique_and_well_formed(db):
    total, uniq = db.execute("SELECT COUNT(*), COUNT(DISTINCT record_id) FROM records").fetchone()
    assert total == uniq
    for (rid,) in db.execute("SELECT record_id FROM records LIMIT 500"):
        assert "#row" in rid and rid.split("#row")[1].isdigit()


def test_all_logical_names_are_readable(db):
    """索引里不该留下任何乱码表名——AC 要求可读中文。"""
    for (name,) in db.execute("SELECT DISTINCT logical FROM records"):
        assert all(ord(c) < 128 or "一" <= c <= "鿿" for c in name), f"表名不可读：{name!r}"


def test_record_count_is_logical_not_physical(db):
    """记录数必须是 csv 逻辑记录数，不是物理行数。

    多个文件字段内含换行：bdyd/疾病 物理 651924 行、实为 4273 条记录。
    早先的盘点报告把行数当记录数，规模高估了两个数量级。
    """
    n = db.execute("SELECT COUNT(*) FROM records WHERE lib='bdyd' AND logical='疾病'").fetchone()[0]
    assert 4000 < n < 5000, f"bdyd/疾病 记录数异常：{n}"
