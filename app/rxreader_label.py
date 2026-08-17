#!/usr/bin/env python3
"""rxreader 标注器：用 deepseek 从医嘱原文抽两栏（silver 标签 / oracle hint）。

    python app/rxreader_label.py --in data/core40_rx.jsonl --out runs/s2-oracle/hints.jsonl

输出每行 {id, keep:[...], explain:[...], raw, dropped:[...]}：
- keep    必须原样保留的数字/剂量/次数/时间/禁忌短语（原文逐字子串）
- explain 老人听不懂、必须白话解释的术语（原文逐字子串）
子串校验：对不上的先做相似度 ≥0.75 的原文吸附，仍对不上就丢进 dropped（同 eqreader 编排规则）。
这份输出既是 S2 的训练标签，也是 S3 之前「oracle hint 天花板」POC 的输入。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import difflib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.translate import load_env  # noqa: E402

SYSTEM = """你是「医嘱读析器」。读一段医疗专业原文，抽出两类**原文逐字子串**，交给下游大模型做适老化转述。

1. keep：转述时必须一字不改保留的硬信息——剂量、次数、频率、时长、时间点、数值区间、禁忌/不可做的条件、必须就医的条件。只抽承载数字或硬性条件的**最短短语**（如「每次1片」「每8小时一次」「3~6个月」「INR 2.0-3.0」「不能与布洛芬同服」），不抽整句。
2. explain：文化程度不高的老人听不懂、转述时必须跟一句白话解释的专业术语（如「肠溶片」「INR」「窦性心律」「血肌酐」）。常见词（血压、感冒、发烧）不算。

规则：每个短语必须是原文里连续出现的逐字子串；宁缺毋滥；每栏最多 8 个；没有就给空数组。
只输出一个 JSON：{"keep":[...],"explain":[...]}"""


def snap(phrase: str, text: str, thr: float = 0.75) -> str | None:
    """逐字子串校验；不中则在原文里找相似度 ≥thr 的同长窗口吸附。"""
    if phrase and phrase in text:
        return phrase
    n = len(phrase)
    if n < 2:
        return None
    best, best_r = None, thr
    for L in (n, n - 1, n + 1):
        if L < 2: continue
        for i in range(0, max(1, len(text) - L + 1)):
            w = text[i:i + L]
            r = difflib.SequenceMatcher(None, phrase, w).ratio()
            if r > best_r:
                best, best_r = w, r
    return best


def label_one(c: dict, model: str, key: str, base: str) -> dict:
    body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 12000,   # 推理模型：thinking 计入额度，4000 时 13/40 被吃光、content 空
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": "【原文】\n" + c["source_text"]}]},
                      ensure_ascii=False).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            content = r["choices"][0]["message"].get("content") or ""
            if not content.strip():
                fr = r["choices"][0].get("finish_reason")
                rt = r.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens")
                raise RuntimeError(f"空输出 finish={fr} reasoning_tokens={rt}")
            m = re.search(r"\{.*\}", content, re.S)
            obj = json.loads(m.group(0)) if m else {}
            out = {"id": c["id"], "keep": [], "explain": [], "raw": content, "dropped": []}
            for col in ("keep", "explain"):
                seen = set()
                for p in obj.get(col, []) or []:
                    p = str(p).strip()
                    s = snap(p, c["source_text"])
                    if s and s not in seen:
                        out[col].append(s); seen.add(s)
                    elif p:
                        out["dropped"].append(p)
            return out
        except Exception as e:
            if attempt == 2:
                return {"id": c["id"], "keep": [], "explain": [], "error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt + 1)
    return {"id": c["id"], "keep": [], "explain": [], "error": "unreachable"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    load_env()
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    key = os.environ["DEEPSEEK_API_KEY"]; base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    cases = [json.loads(l) for l in a.inp.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = {}
    if a.out.exists():
        for l in a.out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if not r.get("error"): done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} 条，已有 {len(done)}，待标 {len(todo)}", flush=True)
    # 逐条追加落盘（append 模式）：两小时的批量不能只在结束时写一次
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a", encoding="utf-8") as fh, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(label_one, c, model, key, base): c for c in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result(); done[r["id"]] = r
            fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
            if i % 20 == 0 or i == len(todo): print(f"  {i}/{len(todo)}", flush=True)
    # 收尾按 cases 顺序重写一份干净的（去掉续跑产生的重复/错误行）
    with a.out.open("w", encoding="utf-8") as fh:
        for c in cases:
            if c["id"] in done: fh.write(json.dumps(done[c["id"]], ensure_ascii=False) + "\n")
    rs = [done[c["id"]] for c in cases if c["id"] in done]
    err = sum(bool(r.get("error")) for r in rs)
    nk = sum(len(r["keep"]) for r in rs); ne = sum(len(r["explain"]) for r in rs); nd = sum(len(r.get("dropped", [])) for r in rs)
    empty = sum(1 for r in rs if not r["keep"] and not r["explain"])
    print(f"写入 {a.out}: {len(rs)} 条, 错误 {err}, keep 共 {nk}, explain 共 {ne}, 丢弃 {nd}, 两栏皆空 {empty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
