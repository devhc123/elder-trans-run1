#!/usr/bin/env python3
"""医嘱翻译器 —— 一体化 CLI：rxreader（本地 0.8B）读析 → hint 注入 → deepseek 转译。

    python app/rxtrans.py --persona "80岁，女，识字不多，高血压、房颤" --text "阿司匹林肠溶片 100mg 每日一次……"
    cat 医嘱.txt | python app/rxtrans.py --persona "..."
    python app/rxtrans.py --case elder-医嘱转译-003             # 测试集取题
    python app/rxtrans.py --text ... --no-hint                  # 对照臂：裸大模型
    python app/rxtrans.py --text ... --json                     # 结构化输出（hint、译文、耗时、token）
    python app/rxtrans.py --batch data/core40_rx.jsonl --out runs/x/outputs.jsonl [--hints-out runs/x/hints.jsonl]

前置：ollama 里有 `rxreader-v1`（见 deploy/rxreader/Modelfile 或 HF chenhaodev/rxreader-qwen3.5-0.8b），
.env 里有 DEEPSEEK_API_KEY。ollama 不可用时退化为无 hint 并在 stderr 警告（不静默）。
编排规则同 eqreader：逐字子串校验 → 空则不注入 → 背景槽只放档案事实。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.rxreader import DEFAULT_MODEL as READER_MODEL  # noqa: E402
from app.rxreader import read  # noqa: E402
from app.translate import DEFAULT_QUERY, translate  # noqa: E402


def run_one(case: dict, *, use_hint: bool = True, reader_model: str = READER_MODEL) -> dict:
    """返回 {id, hint, hinted, output, reader_latency_s, llm_latency_s, usage, warnings}."""
    warnings: list[str] = []
    hint = None
    reader = {}
    if use_hint:
        reader = read(case["source_text"], case.get("persona") or "-", reader_model)
        if reader.get("error"):
            warnings.append(f"rxreader 不可用，退化为无 hint：{reader['error']}")
        elif reader["keep"] or reader["explain"]:
            hint = {"keep": reader["keep"], "explain": reader["explain"]}
        else:
            warnings.append("rxreader 输出为空，不注入")
    r = translate(case, hint if hint else {"keep": [], "explain": []})
    return {"id": case.get("id", "adhoc"), "hint": hint, "hinted": bool(hint), "output": r.get("output"),
            "error": r.get("error"), "truncated": r.get("truncated", False),
            "reader_latency_s": reader.get("latency_s"), "reader_raw": reader.get("raw"),
            "reader_dropped": reader.get("dropped", []),
            "llm_latency_s": r.get("latency_s"), "usage": r.get("usage", {}), "model": r.get("model"),
            "warnings": warnings}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="医嘱原文；不给则读 stdin")
    ap.add_argument("--persona", default="老年用户，文化程度不高", help="老人档案：年龄/病史/识字程度（只放事实）")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--case", help="从 data/elder_translate_270.jsonl 取一题")
    ap.add_argument("--no-hint", action="store_true", help="对照臂：不跑 rxreader，直接裸大模型")
    ap.add_argument("--reader-model", default=READER_MODEL, help="ollama 里的读析器名（默认 rxreader-v1）")
    ap.add_argument("--json", action="store_true", help="结构化输出")
    ap.add_argument("--batch", type=Path); ap.add_argument("--out", type=Path); ap.add_argument("--hints-out", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    if a.batch:
        cases = [json.loads(l) for l in a.batch.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = a.out or Path("runs/adhoc/rxtrans_outputs.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        # rxreader 本地串行更稳（ollama 单卡），大模型并发
        hints = {}
        for i, c in enumerate(cases, 1):
            if a.no_hint: break
            r = read(c["source_text"], c.get("persona") or "-", a.reader_model)
            hints[c["id"]] = r
            if i % 10 == 0 or i == len(cases): print(f"  读析 {i}/{len(cases)}", file=sys.stderr, flush=True)
        if a.hints_out:
            a.hints_out.parent.mkdir(parents=True, exist_ok=True)
            a.hints_out.write_text("".join(json.dumps({"id": k, **v}, ensure_ascii=False) + "\n" for k, v in hints.items()), encoding="utf-8")

        def one(c):
            h = hints.get(c["id"], {})
            hint = {"keep": h.get("keep", []), "explain": h.get("explain", [])} if not a.no_hint else {"keep": [], "explain": []}
            r = translate(c, hint)
            return {**r, "hint": hint if (hint["keep"] or hint["explain"]) else None}
        rs = {}
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for i, r in enumerate(ex.map(one, cases), 1):
                rs[r["id"]] = r
                if i % 10 == 0 or i == len(cases): print(f"  转译 {i}/{len(cases)}", file=sys.stderr, flush=True)
        out.write_text("".join(json.dumps(rs[c["id"]], ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
        bad = [r["id"] for r in rs.values() if r.get("error") or r.get("truncated")]
        ct = sum(r.get("usage", {}).get("completion_tokens", 0) for r in rs.values())
        print(f"写入 {out}：{len(rs)} 条，异常 {len(bad)}{'：' + ','.join(bad) if bad else ''}；completion {ct} tok", file=sys.stderr)
        return 1 if bad else 0

    if a.case:
        cases = {c["id"]: c for c in map(json.loads, (ROOT / "data/elder_translate_270.jsonl").read_text(encoding="utf-8").splitlines())}
        case = cases[a.case]
    else:
        text = a.text if a.text is not None else sys.stdin.read()
        if not text.strip():
            print("没有医嘱原文（--text 或 stdin）", file=sys.stderr); return 2
        case = {"id": "adhoc", "persona": a.persona, "query": a.query, "source_text": text.strip()}

    t0 = time.time()
    r = run_one(case, use_hint=not a.no_hint, reader_model=a.reader_model)
    r["total_s"] = round(time.time() - t0, 2)
    for w in r["warnings"]:
        print("警告：" + w, file=sys.stderr)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2)); return 0 if not r.get("error") else 1
    if r.get("error"):
        print("ERROR:", r["error"], file=sys.stderr); return 1
    if r["hint"]:
        print("【读析】保留: " + ("、".join(r["hint"]["keep"]) or "-") + "\n【读析】解释: " + ("、".join(r["hint"]["explain"]) or "-") + "\n", file=sys.stderr)
    print(r["output"])
    print(f"\n--- 读析 {r['reader_latency_s']}s + 转译 {r['llm_latency_s']}s = {r['total_s']}s；hint={'有' if r['hinted'] else '无'}；{r['model']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
