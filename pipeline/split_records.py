#!/usr/bin/env python3
"""按 record_id 把语料切成三份互不相交的池。

    test           测试集候选池（ticket 05 从这里抽 270 题）
    verifier_train verifier 教师标注池（ticket 09/10 从这里抽）
    reserve        备用扩样池

**为什么必须在 record_id 层切，而不是在题目层去重：** 同一条源记录可以派生多道
题（同一份药品说明书既能出"用药咨询"也能出"用药干预"）。按题去重的话，测试集里
的题和训练集里的题可能来自同一条记录、共享同一段原文——那是事实上的泄漏。
同类项目的实测教训：它 7387 条候选池里有 166 条与测试集 record_id 重合。

切分是**确定性**的：record_id 与种子一起哈希，同种子必得同切分，不依赖顺序、
不依赖索引是否重建。种子写进 kpi.yaml。

用法：
    python3 pipeline/split_records.py            # 执行切分并写回索引
    python3 pipeline/split_records.py --verify   # 只校验：三池互斥、覆盖完整
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "index" / "corpus.sqlite"
KPI_PATH = ROOT / "kpi.yaml"

# 切分比例。test 池只需支撑 270 题的抽样，但要留出足够的场景内余量
# （某些场景的合格候选很稀疏，比如检验指标类）。
SPLITS = [("test", 20), ("verifier_train", 60), ("reserve", 20)]

DEFAULT_SEED = 20260810


def assign(record_id: str, seed: int) -> str:
    """确定性分配。同一 record_id + 同一种子恒得同一池。"""
    h = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    acc = 0
    for name, pct in SPLITS:
        acc += pct
        if bucket < acc:
            return name
    return SPLITS[-1][0]


def read_seed() -> int:
    """从 kpi.yaml 读种子。

    用 yaml 正经解析，不手写。早期版本按行 split(':') 取值，被行内注释噎住
    （`split_seed: 20260810  # 说明` 会被解析成整行字符串）。kpi.yaml 是单一
    真相源，解析它不该将就。
    """
    if not KPI_PATH.exists():
        return DEFAULT_SEED
    import yaml

    data = yaml.safe_load(KPI_PATH.read_text(encoding="utf-8")) or {}
    seed = data.get("split_seed", DEFAULT_SEED)
    if not isinstance(seed, int):
        raise ValueError(f"kpi.yaml 的 split_seed 必须是整数，实得 {seed!r}")
    return seed


def do_split(db: sqlite3.Connection, seed: int) -> dict[str, int]:
    db.execute("DROP TABLE IF EXISTS splits")
    db.execute(
        "CREATE TABLE splits (record_id TEXT PRIMARY KEY, pool TEXT NOT NULL, seed INTEGER)"
    )
    rows = [
        (rid, assign(rid, seed), seed)
        for (rid,) in db.execute("SELECT record_id FROM records")
    ]
    db.executemany("INSERT INTO splits VALUES (?,?,?)", rows)
    db.execute("CREATE INDEX idx_pool ON splits(pool)")
    db.commit()
    return dict(db.execute("SELECT pool, COUNT(*) FROM splits GROUP BY pool"))


def verify(db: sqlite3.Connection) -> list[str]:
    """三池必须互斥且完整覆盖。任何一条不满足都是泄漏隐患。"""
    problems = []

    n_rec = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    n_spl = db.execute("SELECT COUNT(*) FROM splits").fetchone()[0]
    if n_rec != n_spl:
        problems.append(f"覆盖不全：records {n_rec} 条，splits 只有 {n_spl} 条")

    dup = db.execute(
        "SELECT COUNT(*) FROM (SELECT record_id FROM splits GROUP BY record_id HAVING COUNT(*)>1)"
    ).fetchone()[0]
    if dup:
        problems.append(f"{dup} 条 record_id 被分到多个池")

    orphan = db.execute(
        "SELECT COUNT(*) FROM splits s LEFT JOIN records r USING(record_id) WHERE r.record_id IS NULL"
    ).fetchone()[0]
    if orphan:
        problems.append(f"{orphan} 条切分记录在 records 里不存在")

    pools = {p for (p,) in db.execute("SELECT DISTINCT pool FROM splits")}
    expected = {n for n, _ in SPLITS}
    if pools != expected:
        problems.append(f"池名不符：{sorted(pools)} != {sorted(expected)}")

    # 两两交集必须为空——这是本脚本存在的全部意义
    names = sorted(expected)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n = db.execute(
                "SELECT COUNT(*) FROM (SELECT record_id FROM splits WHERE pool=? "
                "INTERSECT SELECT record_id FROM splits WHERE pool=?)",
                (a, b),
            ).fetchone()[0]
            if n:
                problems.append(f"{a} 与 {b} 交集 {n} 条")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只校验，不重新切分")
    ap.add_argument("--seed", type=int, help="覆盖 kpi.yaml 里的种子")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"索引不存在：{DB_PATH}\n先跑 pipeline/source_index.py", file=sys.stderr)
        return 1

    with sqlite3.connect(DB_PATH) as db:
        if not args.verify:
            seed = args.seed if args.seed is not None else read_seed()
            counts = do_split(db, seed)
            total = sum(counts.values())
            print(f"切分完成（seed={seed}，共 {total} 条）：")
            for name, pct in SPLITS:
                n = counts.get(name, 0)
                print(f"  {name:16} {n:>7}  ({100*n/total:.1f}%，目标 {pct}%)")

        problems = verify(db)
        if problems:
            print("\n[校验失败]", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            return 1
        print("\n[OK] 三池互斥、覆盖完整")
    return 0


if __name__ == "__main__":
    sys.exit(main())
