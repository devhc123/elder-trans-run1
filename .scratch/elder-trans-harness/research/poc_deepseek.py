#!/usr/bin/env python3
"""POC：deepseek-v4-flash 单独在 silver-ref 上判候选，准不准、快不快。

**这测的是两段架构的第二段，不是端到端。** 小模型还没训出来，所以这一步回答的是
链上最弱的一环：如果 deepseek 本身判不准，两段架构无论第一段多好都不成立。

对照的是**当前 harness**：silver 标签本身就是教师（Claude）逐条判出来的，
所以「与 silver 一致率」= 「与现行 harness 一致率」，「每条耗时」直接可比。

prompt 用的是**小模型将来要用的那一份**（`SYSTEM_CANDIDATE` + `build_candidate_prompt`），
否则测出来的数换不到那条链上。
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.error
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

N_POS, N_NEG = 40, 40


def env(name: str, default: str = "") -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return os.environ.get(name, default)


def sample(pool: list[dict]) -> list[dict]:
    """确定性分层抽样：正例按红线比例、负例按「是否列表序号」比例。
    量很小（80 条）——POC 要的是大到不用统计就能看见的信号，不是把条数堆上去。"""
    def rank(items):
        return sorted(items, key=lambda it: hashlib.sha256(
            f"poc:{it['case_id']}:{it['candidate_text']}".encode()).hexdigest())

    pos = [it for it in pool if it["label"]]
    neg = [it for it in pool if not it["label"]]
    p0 = [it for it in pos if it["red_line_guess"] == 0]
    p2 = [it for it in pos if it["red_line_guess"] == 2]
    # 红线2 正例总共才 24 条，全要；其余用红线0 补足
    n2 = min(len(p2), max(1, round(N_POS * len(p2) / len(pos))))
    picked_pos = rank(p2)[:n2] + rank(p0)[: N_POS - n2]

    ordinal = [it for it in neg if is_list_ordinal(it["candidate_text"], it["answer"])]
    plain = [it for it in neg if not is_list_ordinal(it["candidate_text"], it["answer"])]
    n_ord = round(N_NEG * len(ordinal) / len(neg))
    picked_neg = rank(ordinal)[:n_ord] + rank(plain)[: N_NEG - n_ord]
    return picked_pos + picked_neg


def ask(item: dict) -> dict:
    key = env("DEEPSEEK_API_KEY")
    body = json.dumps({
        "model": env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "messages": [
            {"role": "system", "content": SYSTEM_CANDIDATE},
            {"role": "user", "content": build_candidate_prompt(item)},
        ],
        "temperature": 0.0,
        # **不能按小模型的口径给 32**：deepseek-v4-flash 是推理模型，32 个 token 全被
        # reasoning 吃掉、content 返回空串，finish_reason=length。第一次跑就是这么
        # 栽的——80/80 "解析失败"，看起来像"模型判不出来"，其实是我把它的嘴堵上了。
        "max_tokens": 2048,
    }).encode()
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read())
            msg = payload["choices"][0]["message"]
            raw = msg.get("content") or ""
            finish = payload["choices"][0].get("finish_reason")
            parsed = parse_prediction(raw)
            return {
                "id": candidate_id(item), "label": item["label"],
                "red_line_guess": item["red_line_guess"],
                "is_ordinal": is_list_ordinal(item["candidate_text"], item["answer"]),
                "violated": parsed.get("violated") if isinstance(parsed, dict) else None,
                "raw": raw[:120], "finish": finish, "seconds": time.time() - t0,
                "reasoning_tokens": payload.get("usage", {}).get(
                    "completion_tokens_details", {}).get("reasoning_tokens", 0),
                "usage": payload.get("usage", {}),
            }
        except (urllib.error.URLError, TimeoutError, KeyError,
                http.client.IncompleteRead, ConnectionError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    return {"id": candidate_id(item), "label": item["label"],
            "red_line_guess": item["red_line_guess"],
            "is_ordinal": is_list_ordinal(item["candidate_text"], item["answer"]),
            "violated": None, "raw": f"ERROR {last}", "seconds": time.time() - t0, "usage": {}}


def main() -> int:
    pool = json.loads((ROOT / "verifier/work/trusted_candidate_pool_holdout.json")
                      .read_text(encoding="utf-8"))
    items = sample(pool)
    print(f"抽样 {len(items)} 条（正 {sum(1 for i in items if i['label'])} / "
          f"负 {sum(1 for i in items if not i['label'])}）")

    wall0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(ask, items))
    wall = time.time() - wall0

    (OUT / "poc_deepseek_result.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"墙钟 {wall:.1f}s（8 并发），单条中位 "
          f"{sorted(r['seconds'] for r in rows)[len(rows)//2]:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
