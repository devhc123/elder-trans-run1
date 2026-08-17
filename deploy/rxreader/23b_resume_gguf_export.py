#!/usr/bin/env python3
"""断点续跑：从已保存的 LoRA adapter 重新加载模型，只补跑失败的那一档量化导出。

背景：`23_train_lora.py --variant withsys` 训练本身成功了（LoRA adapter 已存盘），
q8_0 那档 GGUF 也实际写出成功（文件都在），但 unsloth 内部
`shutil.move(...)` 在这台 pod 极慢的磁盘 I/O 上撞上了一次竞态
（文件被检测到"不存在"时其实还没写完/还没 flush），抛了个假警报式的
FileNotFoundError 把整个训练进程带崩——不重训，只补导出。

用法：
    python3 deploy/23b_resume_gguf_export.py --variant withsys --quant q4_k_m
"""
from __future__ import annotations

import argparse
import json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--quant", required=True, choices=["q8_0", "q4_k_m"])
    ap.add_argument("--max-seq-length", type=int, default=3072)
    args = ap.parse_args()

    adapter_dir = f"outputs/lora_{args.variant}/lora_adapter"
    out_dir = f"outputs/lora_{args.variant}/gguf_{args.quant}"

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=args.max_seq_length,
        dtype=None, load_in_4bit=False,
    )
    print(f"已从 {adapter_dir} 加载模型（含 LoRA），开始导出 {args.quant}")
    model.save_pretrained_gguf(out_dir, tokenizer, quantization_method=args.quant)
    print(f"GGUF({args.quant}) 存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
