#!/usr/bin/env python3
"""rxreader 线上推理端（ollama）：医嘱原文 -> {keep, explain} hint。

    python app/rxreader.py --text "阿司匹林肠溶片 100mg 每日一次……" [--persona "80岁，房颤"]
    python app/rxreader.py --batch data/core40_rx.jsonl --out runs/s3/hints.jsonl   # 给 translate.py --hints 用
    python app/rxreader.py --batch data/core40_rx.jsonl --model qwen3.5:0.8b --out runs/s3/hints_zeroshot.jsonl

编排三件事（同 eqreader）：逐字子串校验（0.75 吸附）→ 空则不注入 → 背景槽只放档案事实。
`think=False` 必带：Qwen3.5 默认进 thinking 通道，num_predict 被吃光后 content 为空（ticket 29 坑）。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.build_rxreader_data import USER_TMPL, parse_target  # noqa: E402
from app.rxreader_label import snap  # noqa: E402

DEFAULT_MODEL = "rxreader-v1"
OLLAMA = "http://localhost:11434/api/chat"


def read(text: str, persona: str = "-", model: str = DEFAULT_MODEL, temperature: float = 0.0,
         num_predict: int = 256, timeout: int = 120) -> dict:
    prompt = USER_TMPL.format(persona=persona or "-", text=text.strip())
    body = json.dumps({"model": model, "stream": False, "think": False,
                       "options": {"temperature": temperature, "num_predict": num_predict, "num_ctx": 8192},
                       "messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        raw = r.get("message", {}).get("content") or ""
    except Exception as e:
        return {"keep": [], "explain": [], "raw": "", "error": f"{type(e).__name__}: {e}", "latency_s": round(time.time() - t0, 2)}
    keep_r, exp_r = parse_target(raw)
    out = {"keep": [], "explain": [], "dropped": [], "raw": raw, "latency_s": round(time.time() - t0, 2)}
    for col, items in (("keep", keep_r), ("explain", exp_r)):
        seen = set(out["keep"])
        for p in items:
            s = snap(p, text)
            if s and s not in seen:
                out[col].append(s); seen.add(s)
            else:
                out["dropped"].append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text"); ap.add_argument("--persona", default="-")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.batch:
        cases = [json.loads(l) for l in a.batch.read_text(encoding="utf-8").splitlines() if l.strip()]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            rs = list(ex.map(lambda c: {"id": c["id"], **read(c["source_text"], c.get("persona", "-"), a.model)}, cases))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        nk = sum(len(r["keep"]) for r in rs); ne = sum(len(r["explain"]) for r in rs)
        nd = sum(len(r.get("dropped", [])) for r in rs); err = sum(bool(r.get("error")) for r in rs)
        empty = sum(1 for r in rs if not r["keep"] and not r["explain"])
        lat = sum(r["latency_s"] for r in rs) / max(1, len(rs))
        print(f"{len(rs)} 条 keep {nk} explain {ne} 丢弃 {nd} 空 {empty} 错误 {err} 均耗时 {lat:.2f}s -> {a.out}")
        return 0
    text = a.text if a.text is not None else sys.stdin.read()
    r = read(text, a.persona, a.model)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
