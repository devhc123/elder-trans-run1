#!/usr/bin/env python3
"""给待标注的源文本生成回答 —— verifier 的判别对象。

复用 run_eval 的调用逻辑与 system prompt：verifier 将来要判的就是这个系统的
输出，训练数据的分布必须和它一致。换一套 prompt 造数据，训出来的 verifier
判的就是另一个东西。
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.run_eval import SYSTEM_PROMPT, call_model  # noqa: E402

WORK = ROOT / "verifier" / "work"


def main() -> int:
    items = json.loads((WORK / "to_label.json").read_text(encoding="utf-8"))
    key = os.environ.get("DEEPSEEK_API_KEY")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not key:
        print("缺 DEEPSEEK_API_KEY", file=sys.stderr)
        return 1

    # 训练数据没有 persona/query（那是测试集才有的人工撰写物），
    # 用一个统一的转述指令替代，保持与线上"拿到一段原文要转述"的形态一致。
    def as_case(c):
        return {
            "id": c["case_id"],
            "persona": "老年用户，文化程度不高",
            "query": f"医生给我看了一段关于「{c['name']}」的说明，我看不懂，您给我说说？",
            "source_text": c["source_text"],
        }

    out, done = [], 0
    with cf.ThreadPoolExecutor(12) as ex:
        futs = {ex.submit(call_model, as_case(c), model, key, base): c for c in items}
        for f in cf.as_completed(futs):
            out.append(f.result()); done += 1
            if done % 50 == 0 or done == len(items):
                print(f"  {done}/{len(items)}", flush=True)
    by = {r["id"]: r for r in out}
    bad = 0
    for c in items:
        r = by.get(c["case_id"], {})
        c["answer"] = r.get("output") or ""
        c["finish_reason"] = r.get("finish_reason")
        if not c["answer"].strip() or r.get("truncated"):
            bad += 1
    items = [c for c in items if c["answer"].strip() and not by.get(c["case_id"], {}).get("truncated")]
    print(f"剔除空/截断 {bad} 条，剩 {len(items)}")
    (WORK / "to_label.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
