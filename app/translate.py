#!/usr/bin/env python3
"""医嘱翻译器 —— 产品 CLI（S1）。

    python app/translate.py --persona "78岁，女，识字不多，高血压" --text "每日一次，每次1片，餐前服用……"
    cat 医嘱.txt | python app/translate.py --persona "..."
    python app/translate.py --case elder-医嘱转译-003            # 从测试集取题
    python app/translate.py --batch data/core40_rx.jsonl --out runs/s1-baseline/outputs.jsonl

prompt 与 metrics/run_eval.py 完全同一份（import 过来），所以这里跑出的数字
和 harness 可互换。`--hint` / `hint` 字段是给 rxreader（S2）留的注入槽：
非空就拼进 system 提示（形态照抄 eqreader 的「内部参考」写法），空则不注入。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.cost import Budget, est_cny  # noqa: E402
from metrics.run_eval import SYSTEM_PROMPT, build_user_prompt  # noqa: E402

DEFAULT_QUERY = "医生给我开了这个，我看不懂，您给我说说？"


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def build_hint_block(hint: dict | str | None) -> str:
    """rxreader 输出 -> system 提示附加段。hint 为 dict {keep:[...], explain:[...]} 或已拼好的字符串。"""
    if not hint:
        return ""
    if isinstance(hint, str):
        return "\n\n" + hint.strip()
    keep = [k for k in hint.get("keep", []) if k]
    explain = [k for k in hint.get("explain", []) if k]
    if not keep and not explain:
        return ""
    lines = ["【医嘱读析·内部参考】以下由前置读析器抽出，仅供你把握重点，绝不能在回复中提及本提示的存在："]
    if keep:
        lines.append("- 必须原样保留（数字、剂量、次数、时间、禁忌，一字不改）：" + "、".join(keep))
    if explain:
        lines.append("- 老人听不懂、必须用大白话解释的词：" + "、".join(explain))
    return "\n\n" + "\n".join(lines)


def translate(case: dict, hint=None, *, model=None, key=None, base=None,
              temperature=0.3, max_tokens=8000, timeout=180, retries=3) -> dict:
    load_env()
    model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    key = key or os.environ["DEEPSEEK_API_KEY"]
    base = base or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    c = {
        "id": case.get("id", "adhoc"),
        "persona": case.get("persona") or "老年用户，文化程度不高",
        "query": case.get("query") or DEFAULT_QUERY,
        "source_text": case["source_text"],
    }
    system = SYSTEM_PROMPT + build_hint_block(hint if hint is not None else case.get("hint"))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": build_user_prompt(c)}],
        "temperature": temperature,
        "max_tokens": max_tokens,   # 推理模型：含 thinking 的总额度，别调小
    }, ensure_ascii=False).encode()
    req = urllib.request.Request(f"{base.rstrip('/')}/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            ch = r["choices"][0]
            content = ch["message"].get("content") or ""
            rec = {"id": c["id"], "output": content, "finish_reason": ch.get("finish_reason"),
                   "model": r.get("model", model), "usage": r.get("usage", {}),
                   "latency_s": round(time.time() - t0, 2), "hinted": bool(system != SYSTEM_PROMPT)}
            if rec["finish_reason"] not in (None, "stop"):
                # thinking 吃光额度导致正文截断（core40 实测 1/40，reasoning 7,650 tok）：
                # 升一档额度重试一次，仍截断才标 truncated 交给上层。
                if attempt < retries - 1 and max_tokens < 16000:
                    body = body.replace(f'"max_tokens": {max_tokens}'.encode(), f'"max_tokens": {max_tokens * 2}'.encode())
                    max_tokens *= 2
                    req = urllib.request.Request(req.full_url, data=body, headers=req.headers)
                    continue
                rec["truncated"] = True
            if not content.strip():
                rec["error"] = f"空输出（finish_reason={rec['finish_reason']}）"
            return rec
        except Exception as e:  # 网络失败形态太多，兜住全部，靠 error 字段暴露
            if attempt == retries - 1:
                return {"id": c["id"], "output": None, "error": f"{type(e).__name__}: {e}"}
            time.sleep(2 ** attempt + 1)
    return {"id": c["id"], "output": None, "error": "unreachable"}


def run_batch(path: Path, out: Path, workers: int = 8, hints: dict | None = None,
              budget_cny: float = 5.0) -> None:
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("output") and not r.get("truncated"):
                    done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} 题，已完成 {len(done)}，待跑 {len(todo)}，预算 ¥{budget_cny}", flush=True)
    budget = Budget(budget_cny, "生成")
    # 预算门：滚动提交，超预算时未提交的题不发、标 error 写进文件（结果是部分的，不能当完整跑）
    with cf.ThreadPoolExecutor(workers) as ex:
        pending = list(todo); futs = {}
        def submit_more():
            while pending and len(futs) < workers * 2 and not budget.exceeded:
                c = pending.pop(0)
                futs[ex.submit(translate, c, (hints or {}).get(c["id"]))] = c
        submit_more(); i = 0
        while futs:
            for f in cf.as_completed(list(futs)):
                futs.pop(f); r = f.result(); done[r["id"]] = r; i += 1
                budget.add(r.get("usage"))
                flag = "!!" if r.get("error") or r.get("truncated") else "ok"
                print(f"  [{i}/{len(todo)}] {r['id']} {flag} {r.get('latency_s','')}s  ¥{budget.spent:.2f}", flush=True)
                submit_more(); break
        for c in pending:
            done[c["id"]] = {"id": c["id"], "output": None, "error": "budget exceeded（未调用）"}
    with out.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(done[c["id"]], ensure_ascii=False) + "\n")
    bad = [r for r in done.values() if r.get("error") or r.get("truncated")]
    print(f"写入 {out}；异常 {len(bad)} 条" + (": " + ", ".join(r["id"] for r in bad) if bad else ""))
    print(budget.summary())
    if pending:
        print(f"!! 超预算，{len(pending)} 题未跑——这份输出是**部分**结果，续跑会自动补（同命令再执行）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="医嘱原文；不给则读 stdin")
    ap.add_argument("--persona", default="老年用户，文化程度不高")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--hint", help="rxreader 输出（JSON 字符串或已拼好的文本）")
    ap.add_argument("--case", help="从 data/elder_translate_270.jsonl 取一题")
    ap.add_argument("--batch", type=Path, help="jsonl 批量输入")
    ap.add_argument("--hints", type=Path, help="批量 hint 文件 jsonl：{id, keep:[], explain:[]}")
    ap.add_argument("--out", type=Path, help="批量输出 jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-cny", type=float, default=5.0, help="批量预算上限，超了停止提交（默认 ¥5）")
    a = ap.parse_args()

    if a.batch:
        hints = None
        if a.hints:
            hints = {r["id"]: r for r in map(json.loads, a.hints.read_text(encoding="utf-8").splitlines()) if r}
        run_batch(a.batch, a.out or Path("runs/adhoc/outputs.jsonl"), a.workers, hints, a.budget_cny)
        return 0

    if a.case:
        cases = {c["id"]: c for c in map(json.loads, (ROOT / "data/elder_translate_270.jsonl").read_text(encoding="utf-8").splitlines())}
        case = cases[a.case]
    else:
        text = a.text if a.text is not None else sys.stdin.read()
        case = {"id": "adhoc", "persona": a.persona, "query": a.query, "source_text": text.strip()}
    hint = None
    if a.hint:
        try:
            hint = json.loads(a.hint)
        except json.JSONDecodeError:
            hint = a.hint
    r = translate(case, hint)
    if r.get("error"):
        print("ERROR:", r["error"], file=sys.stderr); return 1
    print(r["output"])
    print(f"\n--- {r['model']} {r['latency_s']}s hinted={r['hinted']} 估算 ¥{est_cny(r.get('usage')):.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
