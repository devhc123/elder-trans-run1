#!/bin/bash
# rxreader 训练——pod 侧一键脚本（抄 eqbench-run2/deploy/pod_run_reader_v4.sh）。
# 本机：rsync 仓库的 deploy/rxreader/ 到 pod:/workspace/elder/deploy/rxreader/，然后
#   nohup bash deploy/rxreader/pod_run_rxreader.sh > /workspace/elder/run.log 2>&1 &
# 镜像 runpod/pytorch:1.1.0-*-torch280-*：自带 torch2.8。python3.10/3.12 并存，全程用探测到的解释器。
set -x
cd /workspace/elder || exit 1
PY="python$(python3 -m pip --version | grep -oE '\(python 3\.[0-9]+\)' | grep -oE '3\.[0-9]+')"
echo "USING $PY"
$PY -m pip install --upgrade unsloth unsloth_zoo || exit 2
# unsloth 升级会把 torch 拖到 cu130；驱动 550 节点带不动。最后一步顶回镜像同代 cu128。
$PY -m pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128 || exit 2
$PY - <<'CHK' || exit 3
import torch, unsloth, trl, transformers
print("TORCH", torch.__version__, torch.cuda.get_device_name(0))
print("UNSLOTH", unsloth.__version__, "TRL", trl.__version__, "TF", transformers.__version__)
print("SETUP_DONE")
CHK
# ~1850 训练样本 / 有效 batch 32 ≈ 58 步/epoch，3 轮 ≈ 175 步；eval 每 25 步
$PY deploy/rxreader/train_rxreader.py --epochs 3 --eval-steps 25 || exit 4
# 撞 GGUF 导出 FileNotFoundError 就用 23b_resume_gguf_export.py --variant rxreader --quant q4_k_m 补跑
echo "ALL_DONE"
