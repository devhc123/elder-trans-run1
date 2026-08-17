#!/usr/bin/env python3
"""把 rxreader silver 标签拼成 0.8B 的 SFT 数据（messages 格式，train/val 切分）。

    python app/build_rxreader_data.py --pool runs/s2-train/pool.jsonl --labels runs/s2-train/labels.jsonl \
        --out-dir deploy/rxreader/data

输入形态（与线上 app/rxreader.py 一致，三处统一）：
    【医嘱读析】
    【背景】<老人档案，没有写 ->
    【医嘱】
    <原文>
输出形态（单行两栏，空栏写 -）：
    保留: 每次1片、每8小时一次
    解释: 肠溶片、INR
两栏皆空整体输出 `-`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

USER_TMPL = "【医嘱读析】\n【背景】{persona}\n【医嘱】\n{text}"


def format_target(keep: list[str], explain: list[str]) -> str:
    if not keep and not explain:
        return "-"
    return f"保留: {'、'.join(keep) if keep else '-'}\n解释: {'、'.join(explain) if explain else '-'}"


def parse_target(s: str) -> tuple[list[str], list[str]]:
    """线上解析同一份格式（app/rxreader.py import 这个函数）。"""
    s = (s or "").strip()
    if not s or s == "-":
        return [], []
    keep, explain = [], []
    for line in s.splitlines():
        line = line.strip()
        for tag, dst in (("保留:", keep), ("保留：", keep), ("解释:", explain), ("解释：", explain)):
            if line.startswith(tag):
                body = line[len(tag):].strip()
                if body and body != "-":
                    dst.extend(p.strip() for p in body.replace("，", "、").split("、") if p.strip())
                break
    return keep, explain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-chars", type=int, default=1800, help="原文超长的丢弃（3072 token 预算）")
    a = ap.parse_args()
    pool = {r["id"]: r for r in map(json.loads, a.pool.read_text(encoding="utf-8").splitlines()) if r}
    labels = [json.loads(l) for l in a.labels.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, dropped = [], {"error": 0, "toolong": 0, "missing": 0}
    for lb in labels:
        if lb.get("error"): dropped["error"] += 1; continue
        c = pool.get(lb["id"])
        if not c: dropped["missing"] += 1; continue
        if len(c["source_text"]) > a.max_chars: dropped["toolong"] += 1; continue
        # 再做一次逐字子串守门（标注器已做，但训练目标绝不能含原文没有的词）
        keep = [k for k in lb["keep"] if k in c["source_text"]]
        explain = [k for k in lb["explain"] if k in c["source_text"] and k not in keep]
        rows.append({"id": lb["id"], "scenario": c.get("scenario"),
                     "messages": [{"role": "user", "content": USER_TMPL.format(persona="-", text=c["source_text"])},
                                  {"role": "assistant", "content": format_target(keep, explain)}]})
    # 自检：目标能被 parse_target 无损还原
    for r in rows[:200]:
        k, e = parse_target(r["messages"][1]["content"])
        assert format_target(k, e) == r["messages"][1]["content"], r["id"]
    rows.sort(key=lambda r: hashlib.sha256(f"rx:{r['id']}".encode()).hexdigest())
    n_val = max(1, int(len(rows) * a.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        (a.out_dir / f"{name}_rxreader.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
    empty = sum(1 for r in rows if r["messages"][1]["content"] == "-")
    ulen = sorted(len(r["messages"][0]["content"]) for r in rows)
    print(f"train {len(train)} / val {len(val)}；丢弃 {dropped}；目标为 - 的 {empty}（{empty/len(rows):.1%}）")
    print(f"user 字数 p50 {ulen[len(ulen)//2]} p95 {ulen[int(len(ulen)*.95)]} max {ulen[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
