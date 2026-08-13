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

from verifier.train_lora import (  # noqa: E402
    MAX_SEQ,
    SYSTEM,
    SYSTEM_CANDIDATE,
    TRAIN_RED_LINES,
    build_candidate_prompt,
    build_prompt,
    candidate_id,
    render_prompt,
)

WORK = ROOT / "verifier" / "work"

# 生成的新 token 上限。训练目标本身很短（几个布尔 + 几句 evidence 的紧凑
# JSON），440 条标注实测 output 字数 p99 远小于此，留够余量防止真实产出被切。
DEFAULT_MAX_NEW_TOKENS = 768

# 候选级（ticket 18）单独一个上限：目标就是 `{"violated": true}`，十几个 token
# 就够。候选级一轮要跑 2,000+ 次串行生成（真实 holdout 池 + 对抗子集），沿用
# 768 会把没生成完就该停的那段时间全部计进 RunPod 机时——按 ticket 09 实测的
# 速度这是真金白银，不是纸面优化。
CANDIDATE_MAX_NEW_TOKENS = 32

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
        and all(isinstance(k, dict) and isinstance(k.get("covered"), bool) for k in kp)
        and all(isinstance(r, dict) and isinstance(r.get("violated"), bool) for r in rl)
    )

    # **形状异常必须记成 schema 违规，不能让它把推理循环炸掉**（DeepSeek 审计 #1）。
    # 生成式模型完全可能吐 `{"key_points": {...}}`（dict 不是 list）或
    # `{"key_points": ["x"]}`（元素不是 dict）——旧写法会在 `(kp or []) + (rl or [])`
    # 抛 TypeError、或在 `k.get` 抛 AttributeError，**整轮推理当场中断**，
    # 最后一个 checkpoint 之后的结果全丢，而这是在按秒计费的 GPU 上。
    evidence_total = 0
    evidence_grounded = 0
    items = [x for x in (kp if isinstance(kp, list) else []) +
             (rl if isinstance(rl, list) else []) if isinstance(x, dict)]
    for it in items:      # items 已经过滤过，全是 dict
        ev = it.get("evidence", "")
        if isinstance(ev, str) and ev:
            evidence_total += 1
            if evidence_is_grounded(ev, gold_item):
                evidence_grounded += 1

    return {
        "schema_ok": schema_ok,
        "evidence_total": evidence_total,
        "evidence_grounded": evidence_grounded,
    }


# ---------- 候选级（ticket 18：段B判定头，一候选→单个布尔判定） ----------

def load_candidate_pool(path: Path) -> list[dict]:
    """读候选池 JSON（`trusted_candidate_pool_*.json` / `adversarial_subset_
    holdout.json`）。候选级模式**不读 `to_label.json`**——候选 track 的 scp
    清单里根本没有那个文件，池文件自带 source_text/answer/candidate_text，
    prompt 直接从池里重建，不需要回头去案例表里查。"""
    pool = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(pool, list):
        # 顶层不是 list 时，`build_candidate_tasks` 会去遍历 dict 的键（字符串），
        # 产出一堆无意义的 prompt 或在 candidate_id 里炸——都是在 GPU 上才发现
        # （DeepSeek 审计 #7）。在读文件这一步就说清楚。
        raise ValueError(f"{path} 的顶层不是 JSON 数组（拿到 {type(pool).__name__}）")
    return pool


def build_candidate_tasks(pool: list[dict]) -> list[tuple[str, str, dict]]:
    """候选池 -> `(row_id, prompt, item)` 任务列表。

    `row_id` 必须由 `train_lora.candidate_id` 产出、prompt 必须由
    `render_prompt(SYSTEM_CANDIDATE, build_candidate_prompt(...))` 产出——
    两者都是训练时用的那一份，不在这里另写。id 格式漂移会让验收时 join 不上
    （表现是"所有条目都缺预测"），prompt 格式漂移不报错、只让生成质量莫名
    变差，两种都是查起来很贵的静默故障。"""
    return [(candidate_id(item), render_prompt(SYSTEM_CANDIDATE, build_candidate_prompt(item)), item)
            for item in pool]


def audit_candidate_prediction(pred: dict, item: dict) -> dict:
    """候选级的结构诊断。**只认真正的布尔**——`"true"` 这个字符串、缺字段、
    或者吐回一份案例级 schema，都算 schema 违规，不做"宽松解释"。

    候选级 schema 没有 evidence 字段，evidence 两项恒为 0：统计累加器与案例级
    共用一份（`run_tasks`），返回缺键会在累加时 KeyError。"""
    return {
        "schema_ok": isinstance(pred.get("violated"), bool),
        "evidence_total": 0,
        "evidence_grounded": 0,
    }


def build_candidate_row(row_id: str, pred: dict) -> dict:
    """落盘的预测行：**只有 id 与判定，不含 gold**。验收脚本从池文件取 gold，
    预测文件自带答案等于给自己留一条"验收时不小心读到答案"的后门。

    schema 不合格时也照样落盘（否则这条会从验收集里凭空消失，看起来像模型
    从没被问过），但原样保留那个非法值，让验收侧看得出它不是一个合法的 True。"""
    return {"case_id": row_id, "violated": pred.get("violated")}


def build_case_row(row_id: str, pred: dict) -> dict:
    """案例级的落盘行：整份预测 JSON 加上 case_id（ticket 09 以来的原样行为）。"""
    pred["case_id"] = row_id
    return pred


def build_case_tasks(case_ids: list[str], items: dict[str, dict]) -> list[tuple[str, str, dict]]:
    return [(cid, render_prompt(SYSTEM, build_prompt(items[cid])), items[cid]) for cid in case_ids]


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


def run_tasks(ckpt: Path, tasks: list[tuple[str, str, dict]], out_path: Path,
              max_new_tokens: int, checkpoint_every: int, audit, make_row) -> dict:
    """案例级与候选级**共用**的推理循环。

    两种模式的差别只有三处：prompt 怎么建（`tasks` 已经建好）、诊断怎么做
    （`audit`）、落盘行长什么样（`make_row`）。模型加载、生成、解析、统计累加、
    checkpoint 落盘全部共用一份——复制第二份循环意味着 checkpoint 纪律、
    解析失败的记法、统计口径都要各维护一遍，而这些恰恰是错了不报错的地方。
    """
    # **先检查任务非空、参数合法，再加载模型**（DeepSeek 审计 #2/#5）：空任务列表
    # 会加载完模型跑一个空循环、不写输出文件、还以 0 退出——白烧一次建机+装依赖+
    # 载模型的机时，下游验收却找不到 pred 文件，且没有任何错误提示。
    if not tasks:
        raise SystemExit("任务列表是空的——检查 --case-ids-from / --candidates-from 指向的文件")
    if checkpoint_every < 1:
        raise SystemExit(f"--checkpoint-every 必须 ≥1，拿到 {checkpoint_every}")

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
    for i, (row_id, prompt, aux) in enumerate(tasks, 1):
        raw = generate(model, tok, prompt, max_new_tokens)
        parsed = parse_prediction(raw)
        stats["n"] += 1
        if parsed is None:
            stats["parse_fail"] += 1
            preds.append({"case_id": row_id, "_raw": raw, "_parse_error": True})
        else:
            diag = audit(parsed, aux)
            stats["evidence_total"] += diag["evidence_total"]
            stats["evidence_grounded"] += diag["evidence_grounded"]
            if not diag["schema_ok"]:
                stats["schema_fail"] += 1
            preds.append(make_row(row_id, parsed))

        # **每 N 条落盘一次**——这个纪律在本项目撞过至少两次代价惨重的教训
        # （session 限额中断、agent 只在最后才写盘丢光整轮产出）。长跑推理
        # 同样会中断，checkpoint 让中断只损失最后一批，不是全部。
        if i % checkpoint_every == 0 or i == len(tasks):
            write_jsonl(out_path, preds)
            print(f"  [{i}/{len(tasks)}] 已落盘 -> {out_path}")

    return stats


def report(stats: dict, *, candidate_mode: bool = False) -> None:
    n = stats["n"]
    if not n:
        print("\n推理 0 条（没有任务）")
        return
    print(f"\n推理 {n} 条")
    print(f"  解析失败（非法/截断 JSON） {stats['parse_fail']}/{n} = {stats['parse_fail'] / n:.1%}")
    print(f"  schema 不符               {stats['schema_fail']}/{n} = {stats['schema_fail'] / n:.1%}")
    if candidate_mode:
        # 候选级 schema 本来就没有 evidence 字段——照案例级那样打印"无非空
        # evidence，无法计算"会让人以为模型没吐 evidence，其实是根本没这一项。
        print("  evidence 子串合规率       不适用（候选级 schema 只有 violated）")
    elif stats["evidence_total"]:
        et = stats["evidence_total"]
        print(f"  evidence 子串合规率       {stats['evidence_grounded']}/{et} = "
              f"{stats['evidence_grounded'] / et:.1%}")
    else:
        print("  evidence 子串合规率       无非空 evidence，无法计算")
    if candidate_mode:
        print("\n这是原始预测，不是验收结果——门槛判读全部在 "
              "verifier/eval_candidate_verifier.py，不要拿这里的数字下结论。")
    else:
        print("\n这是摸底结果，不是验收结果——不满足门槛不代表最终失败，见 ticket 09/11。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True, help="train_lora.py 存的 LoRA 权重目录")
    # **互斥关系交给 argparse 强制**（DeepSeek 审计 #3）：原来 --case-ids-from 带
    # 默认值，两个都传时候选级模式会静默忽略案例级那个，跑完才发现只出了一半结果。
    # 默认值挪到解析之后再补，这样 argparse 才分得清"没传"和"传了默认值"。
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--case-ids-from", type=Path, default=None,
                      help="案例级模式的 case_id 来源，默认 holdout.jsonl")
    mode.add_argument("--candidates-from", type=Path, default=None,
                      help="候选池 JSON（ticket 14 段B判定头）。给了它就走候选级模式："
                           "prompt 从池里重建，不读 to_label.json，输出 "
                           "{case_id, violated}。与 --case-ids-from 互斥")
    ap.add_argument("--out", type=Path, default=Path("pred.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help=f"默认按模式取：案例级 {DEFAULT_MAX_NEW_TOKENS}、"
                         f"候选级 {CANDIDATE_MAX_NEW_TOKENS}")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    args = ap.parse_args()

    candidate_mode = args.candidates_from is not None
    if candidate_mode:
        # **不碰 load_items()**：候选 track 的 scp 清单里没有 to_label.json，
        # 远程那个文件根本不存在，读一下就是一上机就 FileNotFoundError。
        pool = load_candidate_pool(args.candidates_from)
        if not pool:
            print(f"候选池是空的：{args.candidates_from}", file=sys.stderr)
            return 1
        tasks = build_candidate_tasks(pool)
        audit, make_row = audit_candidate_prediction, build_candidate_row
        max_new_tokens = args.max_new_tokens or CANDIDATE_MAX_NEW_TOKENS
    else:
        items = load_items()
        case_ids = load_case_ids(args.case_ids_from or Path("holdout.jsonl"))
        missing = [c for c in case_ids if c not in items]
        if missing:
            print(f"{len(missing)} 个 case_id 在 to_label.json 里找不到，例如 {missing[:3]}",
                  file=sys.stderr)
            return 1
        tasks = build_case_tasks(case_ids, items)
        audit, make_row = audit_prediction, build_case_row
        max_new_tokens = args.max_new_tokens or DEFAULT_MAX_NEW_TOKENS

    stats = run_tasks(args.ckpt, tasks, args.out, max_new_tokens,
                      args.checkpoint_every, audit, make_row)
    report(stats, candidate_mode=candidate_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
