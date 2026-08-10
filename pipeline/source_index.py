#!/usr/bin/env python3
"""为七个爬虫库建可查询索引。

产出 SQLite（默认 index/corpus.sqlite），每条源记录一个稳定的 `record_id`，
形如 `bdyd_疾病#row1234`。下游的抽样（ticket 05）与切分（split_records.py）
都以 record_id 为唯一键。

两个必须处理的现实问题：

1. **文件名乱码。** 七库里有 10 个文件名是 GBK 字节被当 UTF-8 解出来的结果
   （`ҩƷ.csv` = 药品）。其中一部分可以反解，另一部分是二次乱码或别的编码，
   反解出来是"啪膮奴膶"这种看着像中文的垃圾。**猜文件名不可靠**，所以逻辑表名
   走 LOGICAL_NAMES 显式映射——每一条都是打开文件看实际记录内容认定的，
   证据记在该表的注释里。

2. **内容编码混杂。** utf-8-sig / utf-8 / gb18030 三种都有，逐文件嗅探。

用法：
    python3 pipeline/source_index.py                 # 建索引
    python3 pipeline/source_index.py --stats         # 只看统计
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sqlite3
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = Path(
    "/Users/chenhao/DATA/03_国内医疗语料与数据源/2025国内医疗爬虫结果/2025国内医疗爬虫结果"
)
CORPUS = Path(os.environ.get("ELDER_CORPUS") or DEFAULT_CORPUS)
INDEX_DIR = ROOT / "index"
DB_PATH = INDEX_DIR / "corpus.sqlite"

# 乱码文件名 -> 逻辑表名。**每一条都由打开文件读实际记录内容认定**，不是猜的：
#   bdyd/žąūČ      样本「百草枯中毒急救 / 老人跌倒急救 / 蜜蜂蜇伤急救」→ 急救
#   bdyd/впУч      样本「A群C群脑膜炎球菌多糖疫苗 / 流感病毒裂解疫苗」→ 疫苗
#   bdyd/Ѩλ        样本「安眠穴 / 昆仑穴 / 阑尾穴」→ 穴位
#   bdyd/ҩƷ        样本「阿胶 / 安宫牛黄丸 / 安络化纤丸」→ 药品
#   bdyd/֢״        样本「盗汗 / 手心发热 / 乳房疼痛」→ 症状
#   bdyd/ֶק׀µ      样本「拔罐器 / 绷带 / 三棱针 / 手术床」→ 医疗器械
#   bdyd/าฝังร๛ดส  样本「补钙 / 补肝 / 补气血 / 补肾」→ 养生保健
#   mkss/ҩƷ        样本「氨己烯酸 / 奥卡西平 / 阿普唑仑」→ 药品
#   ylys/žąūČ      表头含 emergency_id / 症状识别 / 120急救 → 急救
#   ylys/Ѩλ        表头含 acupoint_id / 定位取穴 / 穴位配伍 → 穴位
LOGICAL_NAMES = {  # 键统一按 NFC 归一后比较
    ("bdyd", "žąūČ"): "急救",
    ("bdyd", "впУч"): "疫苗",
    ("bdyd", "Ѩλ"): "穴位",
    ("bdyd", "ҩƷ"): "药品",
    ("bdyd", "֢״"): "症状",
    ("bdyd", "ֶק׀µ"): "医疗器械",
    ("bdyd", "าฝังร๛ดส"): "养生保健",
    ("mkss", "ҩƷ"): "药品",
    ("ylys", "žąūČ"): "急救",
    ("ylys", "Ѩλ"): "穴位",
}

# 各库里承载「这条记录叫什么」的列，按优先级取第一个存在的。
NAME_COLUMNS = ["名称", "name", "title", "disease_name", "common_name", "part_name"]

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030")


class IndexError_(RuntimeError):
    pass


def sniff_open(path: str):
    """逐个尝试编码。返回 (文件对象, 编码名)；都失败返回 (None, None)。"""
    for enc in ENCODINGS:
        try:
            f = open(path, encoding=enc, newline="")
            f.read(8192)
            f.seek(0)
            return f, enc
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception:
            return None, None
    return None, None


def logical_name(lib: str, stem: str) -> str:
    """解析逻辑表名。显式映射优先；文件名本身可读则直接用。

    macOS 上文件名以 NFD（分解形）存储，源码里的字面量是 NFC，直接比会不相等。
    两边都归一到 NFC 再查表。
    """
    stem = unicodedata.normalize("NFC", stem)
    if (lib, stem) in LOGICAL_NAMES:
        return LOGICAL_NAMES[(lib, stem)]
    if all(ord(c) < 128 or "一" <= c <= "鿿" for c in stem):
        return stem
    raise IndexError_(
        f"文件名 {stem!r}（库 {lib}）既不可读、也不在 LOGICAL_NAMES 映射里。\n"
        "请打开该文件查看实际记录内容，认定逻辑表名后补进 LOGICAL_NAMES，"
        "并把认定依据写进注释——不要靠反解文件名猜。"
    )


def build(db: sqlite3.Connection) -> dict:
    db.executescript(
        """
        DROP TABLE IF EXISTS records;
        DROP TABLE IF EXISTS files;
        CREATE TABLE files (
            lib TEXT, logical TEXT, source_file TEXT, encoding TEXT,
            n_rows INTEGER, columns TEXT
        );
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            lib TEXT NOT NULL,
            logical TEXT NOT NULL,
            row_idx INTEGER NOT NULL,
            name TEXT,
            filled_fields TEXT
        );
        """
    )

    stats = {"files": 0, "rows": 0, "skipped": []}
    for path in sorted(glob.glob(str(CORPUS / "**" / "*.csv"), recursive=True)):
        rel = os.path.relpath(path, CORPUS)
        parts = rel.split(os.sep)
        lib, stem = parts[0], os.path.basename(path)[:-4]
        # url/ 子目录是抓取用的种子链接表，不是内容记录
        if "url" in parts[:-1]:
            continue

        f, enc = sniff_open(path)
        if f is None:
            stats["skipped"].append((rel, "编码识别失败"))
            continue

        name = logical_name(lib, stem)
        try:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            name_col = next((c for c in NAME_COLUMNS if c in cols), None)
            rows = []
            n = 0
            for i, row in enumerate(reader):
                rid = f"{lib}_{name}#row{i}"
                val = (row.get(name_col) or "").strip() if name_col else ""
                filled = "|".join(c for c in cols if (row.get(c) or "").strip())
                rows.append((rid, lib, name, i, val, filled))
                n += 1
                if len(rows) >= 5000:
                    db.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)", rows)
                    rows = []
            if rows:
                db.executemany("INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)", rows)
        finally:
            f.close()

        db.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?)",
            (lib, name, rel, enc, n, "|".join(cols)),
        )
        stats["files"] += 1
        stats["rows"] += n
        print(f"  {lib}/{name:8} {n:>7} 行  enc={enc:10} ({rel})")

    db.execute("CREATE INDEX idx_lib_logical ON records(lib, logical)")
    db.commit()
    return stats


def show_stats(db: sqlite3.Connection) -> None:
    print("\n按库统计：")
    for lib, n in db.execute(
        "SELECT lib, COUNT(*) FROM records GROUP BY lib ORDER BY COUNT(*) DESC"
    ):
        print(f"  {lib:8} {n:>8}")
    total = db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    named = db.execute("SELECT COUNT(*) FROM records WHERE name != ''").fetchone()[0]
    print(f"  {'合计':8} {total:>8}   （有名称字段 {named}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="只看已建索引的统计")
    args = ap.parse_args()

    INDEX_DIR.mkdir(exist_ok=True)
    if args.stats:
        if not DB_PATH.exists():
            print(f"索引不存在：{DB_PATH}", file=sys.stderr)
            return 1
        with sqlite3.connect(DB_PATH) as db:
            show_stats(db)
        return 0

    if not CORPUS.exists():
        print(f"语料目录不存在：{CORPUS}\n用 ELDER_CORPUS=/path 指定。", file=sys.stderr)
        return 1

    print(f"建索引：{CORPUS}")
    with sqlite3.connect(DB_PATH) as db:
        stats = build(db)
        show_stats(db)
    if stats["skipped"]:
        print("\n跳过的文件：", file=sys.stderr)
        for rel, why in stats["skipped"]:
            print(f"  {rel}: {why}", file=sys.stderr)
    print(f"\n索引写入 {DB_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except IndexError_ as e:
        print(f"\n[INDEX FAILED] {e}", file=sys.stderr)
        sys.exit(1)
