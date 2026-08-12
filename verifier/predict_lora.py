#!/usr/bin/env python3
"""verifier 推理 —— 加载训好的 LoRA 权重，对 holdout 集生成预测（ticket 09）。

本脚本**不在本机跑**（需要 unsloth + GPU），和 `train_lora.py` 同样的道理：
被 rsync 到 RunPod、训练跑完后紧接着在同一台机器上执行。产出 `pred.jsonl`
后 rsync 回本机，喂给 `verifier/eval_verifier.py --pred`。

推理 prompt 必须和训练时**字面一致**——复用 `train_lora.render_prompt` /
`train_lora.build_prompt`，不要在这里重写模板。两份字面量模板不同步的偏差
不会报错，只会让生成质量莫名下降（模型没见过这个格式）。

用法（RunPod 上，训练跑完之后）：
    python3 predict_lora.py --ckpt ./ckpt --case-ids-from holdout.jsonl --out pred.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.train_lora import MAX_SEQ, SYSTEM, TRAIN_RED_LINES, build_prompt, render_prompt  # noqa: E402

WORK = ROOT / "verifier" / "work"

# 生成的新 token 上限。训练目标本身很短（几个布尔 + 几句 evidence 的紧凑
# JSON），440 条标注实测 output 字数 p99 远小于此，留够余量防止真实产出被切。
DEFAULT_MAX_NEW_TOKENS = 768

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def load_items() -> dict[str, dict]:
    """`to_label.json` 是候选案例的唯一来源——训练集/验收集的 prompt 都是从
    这里用 `build_prompt` 现场重建的，不从 holdout.jsonl 里已渲染好的 `input`
    反解，那样脆弱（要反着猜【原文】【回答】的分段边界）。"""
    return {c["case_id"]: c for c in json.loads((WORK / "to_label.json").read_text(encoding="utf-8"))}


def load_case_ids(path: Path) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.append(json.loads(line)["case_id"])
    return ids


def parse_prediction(raw: str) -> dict | None:
    """从模型原始输出里抠出 JSON。训练目标是纯 JSON，但摸底阶段的 checkpoint
    不保证守规矩——可能带 markdown 围栏、可能前后夹解释、可能被截断。

    截断的半截 JSON 必须解析失败而不是"凑巧解析出一部分"：`json.loads` 对半截
    输入本来就会报错，这里不做任何"尽量抢救"的宽松解析，抢救出来的是噪声。
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    candidate = text[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def evidence_is_grounded(evidence: str, item: dict) -> bool:
    """与 `build_teacher_labels.py check()` 的教师标注校验规则完全一致：
    空 evidence 不算编造（多是 covered=false / violated=false 时的合规留空），
    非空则必须是原文或回答的原样子串。规则不一致，摸底阶段测出的"合规率"
    就和教师标注的口径对不上，两个数字没法比。"""
    if not evidence:
        return True
    return evidence in item["source_text"] or evidence in item["answer"]


def audit_prediction(pred: dict, gold_item: dict) -> dict:
    """结构诊断：schema 条数对不对齐、evidence 合规率多少。

    这是 ticket 09 明确要拿到的两个数字（"schema 稳不稳、evidence 子串合规率
    多少"），必须在这一步算，不能等汇总时才发现某条预测条数就没对上。
    """
    n_kp_expected = len(gold_item["key_points"])
    n_rl_expected = len(TRAIN_RED_LINES)

    kp = pred.get("key_points")
    rl = pred.get("red_lines")
    schema_ok = (
        pred.get("verdict") in ("pass", "fail")
        and isinstance(kp, list) and len(kp) == n_kp_expected
        and isinstance(rl, list) and len(rl) == n_rl_expected
        and all(isinstance(k.get("covered"), bool) for k in kp)
        and all(isinstance(r.get("violated"), bool) for r in rl)
    )

    evidence_total = 0
    evidence_grounded = 0
    for it in (kp or []) + (rl or []):
        ev = it.get("evidence", "") if isinstance(it, dict) else ""
        if ev:
            evidence_total += 1
            if evidence_is_grounded(ev, gold_item):
                evidence_grounded += 1

    return {
        "schema_ok": schema_ok,
        "evidence_total": evidence_total,
        "evidence_grounded": evidence_grounded,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    # Qwen3.5 是统一视觉-语言模型，`tok` 实为 Unsloth patch 过的 Processor——
    # 位置参数会被当成 `images` 而不是 `text`（实测报 "Incorrect image source.
    # Got <|system|>..."，把整段 prompt 当图片路径解析），必须显式传关键字。
    inputs = tok(text=prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # 摸底测的是能力上限，不要采样噪声混进"学不学得动"这个问题
        pad_token_id=tok.eos_token_id,
    )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def run(ckpt: Path, case_ids: list[str], items: dict[str, dict],
        out_path: Path, max_new_tokens: int, checkpoint_every: int) -> dict:
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("需要 unsloth（只在 RunPod 上跑）：pip install unsloth", file=sys.stderr)
        raise SystemExit(1)

    model, tok = FastLanguageModel.from_pretrained(
        model_name=str(ckpt), max_seq_length=MAX_SEQ, load_in_4bit=False, dtype=None,
    )
    FastLanguageModel.for_inference(model)

    preds = []
    stats = {"n": 0, "parse_fail": 0, "schema_fail": 0, "evidence_total": 0, "evidence_grounded": 0}
    for i, cid in enumerate(case_ids, 1):
        item = items[cid]
        prompt = render_prompt(SYSTEM, build_prompt(item))
        raw = generate(model, tok, prompt, max_new_tokens)
        parsed = parse_prediction(raw)
        stats["n"] += 1
        if parsed is None:
            stats["parse_fail"] += 1
            preds.append({"case_id": cid, "_raw": raw, "_parse_error": True})
        else:
            diag = audit_prediction(parsed, item)
            stats["evidence_total"] += diag["evidence_total"]
            stats["evidence_grounded"] += diag["evidence_grounded"]
            if not diag["schema_ok"]:
                stats["schema_fail"] += 1
            parsed["case_id"] = cid
            preds.append(parsed)

        # **每 N 条落盘一次**——这个纪律在本项目撞过至少两次代价惨重的教训
        # （session 限额中断、agent 只在最后才写盘丢光整轮产出）。长跑推理
        # 同样会中断，checkpoint 让中断只损失最后一批，不是全部。
        if i % checkpoint_every == 0 or i == len(case_ids):
            write_jsonl(out_path, preds)
            print(f"  [{i}/{len(case_ids)}] 已落盘 -> {out_path}")

    return stats


def report(stats: dict) -> None:
    n = stats["n"]
    print(f"\n推理 {n} 条")
    print(f"  解析失败（非法/截断 JSON） {stats['parse_fail']}/{n} = {stats['parse_fail'] / n:.1%}")
    print(f"  schema 条数不符           {stats['schema_fail']}/{n} = {stats['schema_fail'] / n:.1%}")
    et = stats["evidence_total"]
    if et:
        print(f"  evidence 子串合规率       {stats['evidence_grounded']}/{et} = "
              f"{stats['evidence_grounded'] / et:.1%}")
    else:
        print("  evidence 子串合规率       无非空 evidence，无法计算")
    print("\n这是摸底结果，不是验收结果——不满足门槛不代表最终失败，见 ticket 09/11。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="train_lora.py 存的 LoRA 权重目录")
    ap.add_argument("--case-ids-from", type=Path, default=Path("holdout.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("pred.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    items = load_items()
    case_ids = load_case_ids(args.case_ids_from)
    missing = [c for c in case_ids if c not in items]
    if missing:
        print(f"{len(missing)} 个 case_id 在 to_label.json 里找不到，例如 {missing[:3]}", file=sys.stderr)
        return 1

    stats = run(args.ckpt, case_ids, items, args.out, args.max_new_tokens, args.checkpoint_every)
    report(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
