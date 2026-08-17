#!/usr/bin/env python3
"""跑在 pod 上：把训练产物推到 HF 私仓，删机前的最后一步（铁律6）。

上传范围：rxreader 的 Q8_0/Q4_K_M GGUF + LoRA adapter（仓库需先建好，私有）。
"""
from __future__ import annotations

import os

from huggingface_hub import HfApi

REPO_ID = "chenhaodev/rxreader-qwen3.5-0.8b"


def main() -> int:
    api = HfApi(token=os.environ["HF_TOKEN"])

    uploads = [
        ("outputs/lora_rxreader/gguf_q8_0_gguf/Qwen3.5-0.8B.Q8_0.gguf", "v1/Qwen3.5-0.8B.Q8_0.gguf"),
        ("outputs/lora_rxreader/gguf_q4_k_m_gguf/Qwen3.5-0.8B.Q4_K_M.gguf", "v1/Qwen3.5-0.8B.Q4_K_M.gguf"),
    ]
    for local, remote in uploads:
        print(f"uploading {local} -> {remote}")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID)
    api.upload_folder(folder_path="outputs/lora_rxreader/lora_adapter", path_in_repo="v1/lora_adapter", repo_id=REPO_ID)
    print("ALL_UPLOADS_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
