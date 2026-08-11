#!/usr/bin/env python3
"""verifier 的 LoRA 微调 —— Qwen3.5-2B-Base + Unsloth，跑在 RunPod。

**基座与训练配置的每一条都有核实依据**（见 docs/DATASETS_AND_BENCHMARKS.md §4）：

  Qwen/Qwen3.5-2B-Base   Apache-2.0；262K 上下文；Unsloth 有官方 notebook
  选 Base 不选指令版      判官输出是固定 JSON schema，不需要对话能力；中文医疗
                         内容容易触发指令版拒答与免责说教；且 Unsloth 官方警告
                         Qwen3.5-2B **指令版**比同系列更易陷入 thinking 死循环，
                         批量跑时是不终止生成的灾难
  bf16 LoRA 而非 4bit    Unsloth 明确不建议对 Qwen3.5 做 4-bit QLoRA（量化误差
                         偏大）。2B 的 bf16 LoRA 只要约 5GB，不构成成本问题
  需装 vision 依赖        Qwen3.5 是统一视觉-语言模型，纯文本也要装；视觉编码器
                         对本任务是死重，推理时不挂 mmproj

本脚本**不在本机跑**。它被 rsync 到 RunPod 后执行；本机只负责产数据与验收。

用法（RunPod 上）：
    pip install unsloth
    python3 train_lora.py --data train.jsonl --out ./ckpt --epochs 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
FALLBACK_MODEL = "Qwen/Qwen3-1.7B-Base"   # 低风险备选：标准 dense、GGUF 零风险
SCALE_UP_MODEL = "Qwen/Qwen3.5-4B-Base"   # 未达门槛时的既定升级路径

# 实测 P90 的 prompt 就有 2594 字（原文 + 回答 + 要点 + 红线），加上输出 JSON
# 还要一份。早期设 4096 会**静默截断**——被截掉的正是排在最后的红线部分和
# 输出目标，训出来的模型等于没见过红线。Qwen3.5 原生 262K，8192 毫无压力。
MAX_SEQ = 8192

# 训练用的对话模板。**用 Base 模型就必须自己定模板**——这既是选 Base 的代价，
# 也是选 Base 的理由：不与指令版的 chat template 和 thinking 机制打架。
SYSTEM = """你是医疗转述的忠实性判别器。只以「原文」为事实依据，判断「回答」是否忠实。

逐条判断：每个要点是否被回答传达到；每条红线是否被触犯。
回答里说了原文没说、但医学上正确的事，仍记作触犯红线——你判的是有没有原文依据，
不是医学上对不对。

只输出 JSON，不要解释。evidence 必须是原文或回答里的原样子串。"""


def build_prompt(c: dict) -> str:
    kp = "\n".join(f"{i}. {k}" for i, k in enumerate(c["key_points"]))
    rl = "\n".join(f"{i}. {r}" for i, r in enumerate(c["red_lines"]))
    return (
        f"【原文】\n{c['source_text']}\n\n"
        f"【回答】\n{c['answer']}\n\n"
        f"【要点】\n{kp}\n\n"
        f"【红线】\n{rl}"
    )


def build_target(label: dict) -> str:
    """训练目标即教师标注本身，压成单行 JSON。

    刻意保留 evidence：它可程序化校验（必须是原样子串），既是防幻觉的抓手，
    也是汇报时能指着说「小模型自己判出这里越界了」的凭据。
    """
    return json.dumps(
        {
            "key_points": [
                {"idx": k["idx"], "covered": k["covered"], "evidence": k.get("evidence", "")}
                for k in label["key_points"]
            ],
            "red_lines": [
                {"idx": r["idx"], "violated": r["violated"], "evidence": r.get("evidence", "")}
                for r in label["red_lines"]
            ],
            "verdict": label["verdict"],
        },
        ensure_ascii=False,
    )


def make_dataset(items: list[dict], labels: dict[str, dict]) -> list[dict]:
    out = []
    for c in items:
        lab = labels.get(c["case_id"])
        if not lab:
            continue
        out.append({"system": SYSTEM, "input": build_prompt(c), "output": build_target(lab)})
    return out


def train(data_path: Path, out_dir: Path, model: str, epochs: int, bsz: int) -> int:
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("需要 unsloth（只在 RunPod 上跑）：pip install unsloth", file=sys.stderr)
        return 1
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"训练样本 {len(rows)}")

    m, tok = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=MAX_SEQ,
        # **不要开 load_in_4bit**：Unsloth 明确不建议对 Qwen3.5 做 4-bit QLoRA
        load_in_4bit=False,
        dtype=None,   # 让 Unsloth 自选 bf16
    )
    m = FastLanguageModel.get_peft_model(
        m,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=20260811,
    )

    def fmt(ex):
        return {
            "text": f"<|system|>\n{ex['system']}\n<|user|>\n{ex['input']}\n"
                    f"<|assistant|>\n{ex['output']}{tok.eos_token}"
        }

    ds = Dataset.from_list(rows).map(fmt)
    trainer = SFTTrainer(
        model=m,
        tokenizer=tok,
        train_dataset=ds,
        args=SFTConfig(
            per_device_train_batch_size=bsz,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            logging_steps=5,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            seed=20260811,
            output_dir=str(out_dir),
            report_to="none",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
        ),
    )
    trainer.train()
    m.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"已保存到 {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-data", action="store_true", help="从标注产出训练集（本机跑）")
    ap.add_argument("--data", type=Path, default=Path("train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("./ckpt"))
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--holdout", type=int, default=0,
                    help="留出多少条做验收（必须在训练前切出并封存）")
    args = ap.parse_args()

    if args.make_data:
        root = Path(__file__).resolve().parent.parent
        items = json.loads((root / "verifier" / "work" / "to_label.json").read_text(encoding="utf-8"))
        labels = {}
        for p in sorted((root / "verifier" / "labels").glob("*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    labels[j["case_id"]] = j
        ds = make_dataset(items, labels)
        print(f"可用样本 {len(ds)}（标注 {len(labels)} 条）")

        if args.holdout:
            # **验收集必须在训练前切出并封存**，不能等训完再从剩余数据里挑——
            # 那等于自己给自己划及格线。切分按 case_id 哈希，确定性。
            import hashlib

            def h(x):
                return int(hashlib.sha256(x["input"][:64].encode()).hexdigest()[:8], 16)

            ds.sort(key=h)
            hold, ds = ds[: args.holdout], ds[args.holdout:]
            (root / "verifier" / "holdout.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in hold), encoding="utf-8"
            )
            print(f"验收集 {len(hold)} 条已封存 -> verifier/holdout.jsonl")

        args.data.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in ds), encoding="utf-8"
        )
        print(f"训练集 {len(ds)} 条 -> {args.data}")
        return 0

    return train(args.data, args.out, args.model, args.epochs, args.bsz)


if __name__ == "__main__":
    sys.exit(main())
