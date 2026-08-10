#!/usr/bin/env python3
"""定点修复被污染的候选槽位，尽量不动其余抽样。

背景：老年筛选只匹配病名、不区分人群，把「小儿α-地中海贫血」「妊娠期糖尿病」
这类记录抽了进来——病名对得上，人群完全不对（270 条里 13 条 / 4.8%）。
筛选器已在 extract_candidates.py 里修好（EXCLUDE_POPULATION）。

本脚本只做替换，不重抽全量：重抽会让 270 个槽位全部换掉，已完成的撰写全部作废。
替换后打印需要重新撰写的题号。

用法：
    python3 pipeline/repair_candidates.py --dry-run   # 只看要换哪些
    python3 pipeline/repair_candidates.py             # 执行替换
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.extract_candidates import CAND_DIR, DB_PATH, extract, sample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with sqlite3.connect(DB_PATH) as db:
        clean_pools = extract(db)

    # 全局已占用的 record_id：替换品不能撞上任何一个仍然有效的槽位
    current: dict[str, list[dict]] = {}
    for p in sorted(CAND_DIR.glob("sampled_*.jsonl")):
        s = p.stem[len("sampled_"):]
        current[s] = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    to_reauthor: list[str] = []
    for scenario, rows in current.items():
        valid = {c["record_id"]: c for c in clean_pools[scenario]}
        keep_ids = {c["record_id"] for c in rows if c["record_id"] in valid}
        taken = {c["record_id"] for rs in current.values() for c in rs}

        pool = [c for c in clean_pools[scenario] if c["record_id"] not in taken]
        spare = sample(pool, len(pool), scenario)  # 按同一确定性顺序排好备选
        si = 0

        new_rows = []
        for i, c in enumerate(rows):
            if c["record_id"] in valid:
                new_rows.append(valid[c["record_id"]])  # 用干净池里的同一条（字段可能已更新）
                continue
            while si < len(spare) and spare[si]["record_id"] in keep_ids:
                si += 1
            if si >= len(spare):
                print(f"[FAIL] {scenario} 没有可用替换品", file=sys.stderr)
                return 1
            rep = spare[si]
            si += 1
            keep_ids.add(rep["record_id"])
            new_rows.append(rep)
            cid = f"elder-{scenario}-{i + 1:03d}"
            to_reauthor.append(cid)
            print(f"  {cid}: {c['name'][:20]} -> {rep['name'][:20]}")

        if not args.dry_run:
            with open(CAND_DIR / f"sampled_{scenario}.jsonl", "w", encoding="utf-8") as f:
                for c in new_rows:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n需重新撰写 {len(to_reauthor)} 题：")
    for cid in to_reauthor:
        print(f"  {cid}")
    (ROOT / "work" / "reauthor.json").write_text(
        json.dumps(to_reauthor, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print("\n(dry-run，未写回)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
