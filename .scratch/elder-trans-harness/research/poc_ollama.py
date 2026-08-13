#!/usr/bin/env python3
"""POC：现成的 ollama 小模型能不能当级联第一段（不训练，先试）。

跑的是**与 deepseek 那轮完全相同的 80 条**，所以两边的逐条判定可以离线拼成级联，
不用再调一次 deepseek。

**关键坑（今天栽了两次）**：qwen3.5 / minicpm5 这类模型默认开 thinking，
token 预算会被思考吃光、`content` 返回空串、`done_reason=length`——看起来像
"模型判不出来"，其实是没给它说话的机会。这里显式 `think=False`。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path("/Users/chenhao/ClaudeCode/elder-trans-run1")
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

from verifier.predict_lora import parse_prediction  # noqa: E402
from verifier.redline_candidates import is_list_ordinal  # noqa: E402
from verifier.train_lora import (  # noqa: E402
    SYSTEM_CANDIDATE,
    build_candidate_prompt,
    candidate_id,
)

MODELS = [
    "qwen3.5:0.8b",
    "openbmb/minicpm5:latest",
    "minicpm-v4.6:1b",
    "medgemma:4b",
    "chenhaodev/med-guard:latest",
]
N_POS, N_NEG = 40, 40


def sample(pool: list[dict]) -> list[dict]:
    """必须与 poc_deepseek.py 的抽样**逐条一致**，否则拼不出级联。"""
    def rank(items):
        return sorted(items, key=lambda it: hashlib.sha256(
            f"poc:{it['case_id']}:{it['candidate_text']}".encode()).hexdigest())
    pos = [it for it in pool if it["label"]]
    neg = [it for it in pool if not it["label"]]
    p0 = [it for it in pos if it["red_line_guess"] == 0]
    p2 = [it for it in pos if it["red_line_guess"] == 2]
    n2 = min(len(p2), max(1, round(N_POS * len(p2) / len(pos))))
    picked_pos = rank(p2)[:n2] + rank(p0)[: N_POS - n2]
    ordinal = [it for it in neg if is_list_ordinal(it["candidate_text"], it["answer"])]
    plain = [it for it in neg if not is_list_ordinal(it["candidate_text"], it["answer"])]
    n_ord = round(N_NEG * len(ordinal) / len(neg))
    return picked_pos + rank(ordinal)[:n_ord] + rank(plain)[: N_NEG - n_ord]


def ask(model: str, item: dict) -> dict:
    body = json.dumps({
        "model": model, "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 64},
        "messages": [
            {"role": "system", "content": SYSTEM_CANDIDATE},
            {"role": "user", "content": build_candidate_prompt(item)},
        ],
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        raw = d["message"].get("content") or ""
        parsed = parse_prediction(raw)
        return {"id": candidate_id(item), "label": item["label"],
                "red_line_guess": item["red_line_guess"],
                "is_ordinal": is_list_ordinal(item["candidate_text"], item["answer"]),
                "violated": parsed.get("violated") if isinstance(parsed, dict) else None,
                "raw": raw[:100], "done": d.get("done_reason"),
                "seconds": time.time() - t0}
    except Exception as e:                                   # noqa: BLE001
        return {"id": candidate_id(item), "label": item["label"],
                "red_line_guess": item["red_line_guess"],
                "is_ordinal": is_list_ordinal(item["candidate_text"], item["answer"]),
                "violated": None, "raw": f"ERROR {e}", "done": "error",
                "seconds": time.time() - t0}


def main() -> int:
    pool = json.loads((ROOT / "verifier/work/trusted_candidate_pool_holdout.json")
                      .read_text(encoding="utf-8"))
    items = sample(pool)
    out = {}
    for model in MODELS:
        t0 = time.time()
        rows = [ask(model, it) for it in items]          # 串行：本地 ollama 一次一个
        out[model] = rows
        bad = sum(1 for r in rows if r["violated"] is None)
        print(f"{model:30s} {time.time()-t0:6.1f}s  单条中位 "
              f"{sorted(r['seconds'] for r in rows)[len(rows)//2]:5.2f}s  解析失败 {bad}/{len(rows)}")
        (OUT / "poc_ollama_result.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
