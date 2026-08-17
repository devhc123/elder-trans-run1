#!/usr/bin/env python3
"""rxreader（医嘱读析器）LoRA SFT —— Qwen3.5-0.8B + Unsloth，**在 RunPod pod 上跑**，本地没有 GPU。

整份脚本抄自 eqbench-run2/deploy/23_train_lora.py（eqreader v4 就是它训出来的），只改了路径与默认变体：
- 数据：deploy/rxreader/data/{train,val}_rxreader.jsonl（app/build_rxreader_data.py 产出，messages 格式）
- 模板锚点：deploy/rxreader/template_probe.json（同一基座同一 probe；pod 上 transformers 若换大版本
  需重跑 eqbench-run2/scripts/03_template_probe.py 核对 response_part 未变）
- LoRA r=32/α=64、lr 2e-4、cosine、只在 assistant 段算 loss、训前 mask 自测、训后导 q8_0 + q4_k_m GGUF。

用法（pod 上）：
    python3 deploy/rxreader/train_rxreader.py --epochs 3 --eval-steps 25
"""
from __future__ import annotations

import argparse
import json
import os

BASE_MODEL = "Qwen/Qwen3.5-0.8B"   # 与 eqreader 同款，Modelfile/probe 直接复用
MAX_SEQ_LENGTH = 3072   # 铁律16：按 22_prepare_training_data.py 实测 p99(withsys)=2322/max=2539 定，留余量
LORA_R = 32
LORA_ALPHA = 64


def load_response_part(probe_path: str = "deploy/rxreader/template_probe.json") -> str:
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    return probe["response_part"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="rxreader")
    ap.add_argument("--train", default=None, help="默认 data/splits/train_{variant}.jsonl")
    ap.add_argument("--val", default=None, help="默认 data/splits/val_{variant}.jsonl")
    ap.add_argument("--out-dir", default=None, help="默认 outputs/lora_{variant}")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-steps", type=int, default=50,
                    help="小数据集（如 reader 163 条）必须调小，否则一次 eval 都不触发")
    args = ap.parse_args()

    train_path = args.train or f"deploy/rxreader/data/train_{args.variant}.jsonl"
    val_path = args.val or f"deploy/rxreader/data/val_{args.variant}.jsonl"
    out_dir = args.out_dir or f"outputs/lora_{args.variant}"

    # 只在 pod 上才有 unsloth——本地跑这个脚本到这里应该已经因为 import 失败而停下，
    # 那是正确行为，不要 try/except 把它悄悄兜住。
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None, load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
    )

    response_part = load_response_part()
    # instruction_part 只用于 mask 的定位锚点；probe 里已经存了，直接复用，
    # 不在这里重新拼——三处统一（铁律1）就是靠这份文件当唯一来源。
    with open("deploy/rxreader/template_probe.json", encoding="utf-8") as fh:
        instruction_part = json.load(fh)["instruction_part"]

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    train_ds = load_dataset("json", data_files=train_path, split="train").map(to_text)
    val_ds = load_dataset("json", data_files=val_path, split="train").map(to_text)
    print(f"variant={args.variant}  train={len(train_ds)}  val={len(val_ds)}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=val_ds,
        dataset_text_field="text", max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.05, lr_scheduler_type="cosine",
            logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
            save_strategy="steps", save_steps=args.eval_steps, save_total_limit=3,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            output_dir=out_dir, optim="adamw_8bit", seed=3407,
            report_to="none",
        ),
    )

    # **只在 assistant 段算 loss**——不然模型会学着预测用户轮和候选回复本身，
    # 那些是输入不是训练目标。probe 出来的两个字符串就是给这个用的。
    trainer = train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part,
    )

    # 训前自测：随便取一条样本，确认 mask 之后 assistant 段真的有非 -100 的 label，
    # 且非 assistant 段全是 -100——这是铁律1 那个"空 think 块吃进训练目标"事故的
    # 复现检测，训练开始前挂了立刻能看见，不用等一轮训完才发现输出全是空 think。
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    n_supervised = sum(1 for x in labels if x != -100)
    print(f"训前自测：样本0 supervised token 数 = {n_supervised}"
          f"（应 >0 且远小于总长度 {len(labels)}，否则 mask 没生效）")
    if n_supervised == 0:
        raise RuntimeError("mask 后没有任何 supervised token——response_part 定位失败，别开训")

    trainer.train()

    model.save_pretrained(f"{out_dir}/lora_adapter")
    tokenizer.save_pretrained(f"{out_dir}/lora_adapter")
    print(f"LoRA adapter 存到 {out_dir}/lora_adapter")

    # GGUF 导出。**目录名带 _gguf 后缀是 unsloth 的行为，不是我们起的名**
    # （铁律：save_pretrained_gguf 输出目录是 <name>_gguf，命名对不上会在票 5
    # 回传时找不到文件）。
    for quant in ("q8_0", "q4_k_m"):
        gguf_dir = f"{out_dir}/gguf_{quant}"
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=quant)
        print(f"GGUF({quant}) 存到 {gguf_dir}_gguf")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
