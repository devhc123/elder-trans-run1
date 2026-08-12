#!/usr/bin/env bash
# ticket 09 摸底训练：把 verifier LoRA 训练 + 推理跑在 RunPod 上。
#
# 两段式，同 eqbench-run2 项目验证过的模式（避免 run1 白烧 100 分钟 + $0.45
# 那个教训——`runpodctl create pod` 不会自动注入账户密钥，缺
# `--env PUBLIC_KEY=...` 就必然 Connection refused，长得和坏节点一模一样）：
#
#   第一段（默认）—— 全部**零成本**检查，不建机器
#   第二段（--create）—— 真建 pod、跑训练+推理、取回 pred.jsonl、删机（**会计费**）
#
# 用法：
#   bash deploy/runpod_pilot.sh              # 只跑零成本检查
#   bash deploy/runpod_pilot.sh --create     # 建 pod 跑完整摸底（**会计费，需人工确认**）
set -uo pipefail

KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_automation}"
GPU_TYPE="${RUNPOD_GPU:-NVIDIA A40}"
# 09 摸底实际跑的 pod 用了 runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204-cluster
# （A5000，A40 缺货时的自动替代）——两个镜像的 torch 都已跟 unsloth 同期，任一个都行。
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04}"
POD_NAME="${RUNPOD_POD_NAME:-eldertrans-verifier-pilot}"
EPOCHS="${RUNPOD_EPOCHS:-3}"
REMOTE_DIR="/workspace/eldertrans"

fail=0
ok()   { printf '  ✓ %s\n' "$*"; }
bad()  { printf '  ✗ %s\n' "$*"; fail=1; }
warn() { printf '  ! %s\n' "$*"; }

echo "═══ 第一段：零成本检查（不建机器）═══"
echo

echo "[1] runpodctl"
if ! command -v runpodctl >/dev/null; then
  bad "runpodctl 没装：brew install runpod/runpodctl/runpodctl"
else
  ver=$(runpodctl version 2>&1 | head -1)
  ok "$ver"
  # 上游文档说新形式是 `runpodctl pod create`，但装在本机的版本未必支持——
  # 照文档写而不核一次，会得到 unknown command（eqbench-run2 项目已实测过）。
  if runpodctl pod create --help >/dev/null 2>&1; then
    ok "支持新形式 runpodctl pod create（flag 名：--gpu-id/--image/--cloud-type）"
    CREATE_FORM=new
  elif runpodctl create pod --help >/dev/null 2>&1; then
    ok "只支持旧形式 runpodctl create pod（flag 名：--gpuType/--imageName/--secureCloud）"
    CREATE_FORM=old
  else
    bad "两种形式都不认，先看 runpodctl --help"
    CREATE_FORM=none
  fi
fi
echo

echo "[2] API key"
ENV_KEY=$(grep '^RUNPOD_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d ' \r')
CFG_KEY=$(grep '^apikey' "$HOME/.runpod/config.toml" 2>/dev/null | cut -d= -f2- | tr -d ' "'"'"'\r')
if [ -z "$ENV_KEY" ]; then
  bad ".env 里没有 RUNPOD_API_KEY"
else
  ok "RUNPOD_API_KEY 在 .env 里（长度 ${#ENV_KEY}）"
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 \
         -H "Authorization: Bearer $ENV_KEY" https://rest.runpod.io/v1/pods)
  if [ "$code" = 200 ]; then
    ok ".env 的 key 通过 API 验证（HTTP 200）"
  else
    bad ".env 的 key 被拒（HTTP $code）—— 去 runpod.io 换一把"
  fi
fi
if [ -n "$CFG_KEY" ] && [ "$CFG_KEY" != "$ENV_KEY" ]; then
  bad "runpodctl 配置里的 key 与 .env 不一致（config 长度 ${#CFG_KEY} vs env ${#ENV_KEY}）。
       runpodctl 用的是 config.toml 那把，不是 .env。同步：
       runpodctl config --apiKey \"\$(grep '^RUNPOD_API_KEY=' .env | cut -d= -f2-)\""
elif [ -n "$CFG_KEY" ]; then
  ok "runpodctl 配置里的 key 与 .env 一致"
fi
if timeout 60 runpodctl get pod >/dev/null 2>&1; then
  ok "runpodctl 已认证（get pod 通）"
else
  bad "runpodctl 未认证：runpodctl config --apiKey <key>"
fi
running=$(timeout 60 runpodctl get pod 2>/dev/null | tail -n +2 | grep -c . || true)
if [ "${running:-0}" -gt 0 ]; then
  warn "当前有 $running 台 pod 在跑 —— 确认是不是忘了删（在计费）"
else
  ok "当前没有 pod 在跑（没有意外计费）"
fi
echo

echo "[3] SSH 密钥：$KEY"
if [ ! -f "$KEY" ]; then
  bad "私钥不存在。生成：ssh-keygen -t ed25519 -f $KEY -N ''"
elif [ ! -f "$KEY.pub" ]; then
  bad "公钥 $KEY.pub 不存在"
else
  ok "私钥与公钥都在"
  probe=$(ssh-keygen -y -f "$KEY" </dev/null 2>&1 | head -1)
  if [[ "$probe" == ssh-* ]]; then
    ok "无口令，可无人值守使用"
  else
    bad "有口令（$probe）——无人值守流程会卡在交互输入上"
  fi
  ok "指纹 $(ssh-keygen -lf "$KEY.pub" | awk '{print $2}')"
fi
echo

echo "[4] 训练/推理前置产物"
[ -f verifier/train.jsonl ] && ok "verifier/train.jsonl 存在（$(wc -l < verifier/train.jsonl | tr -d ' ') 条）" || \
  bad "缺 verifier/train.jsonl —— 先跑 python3 verifier/train_lora.py --make-data --holdout 100"
[ -f verifier/holdout.jsonl ] && ok "verifier/holdout.jsonl 存在（$(wc -l < verifier/holdout.jsonl | tr -d ' ') 条，已封存）" || \
  bad "缺 verifier/holdout.jsonl"
[ -f verifier/work/to_label.json ] && ok "verifier/work/to_label.json 存在（predict_lora.py 现场重建 prompt 要用）" || \
  bad "缺 verifier/work/to_label.json"
[ -f verifier/train_lora.py ] && [ -f verifier/predict_lora.py ] && ok "训练/推理脚本都在" || \
  bad "缺 verifier/train_lora.py 或 verifier/predict_lora.py"
if command -v python3 >/dev/null; then
  python3 -m pytest tests/test_train_lora.py tests/test_predict_lora.py -q >/tmp/eldertrans_pilot_pytest.log 2>&1 \
    && ok "本机可测部分（配置约束 + 解析/判分逻辑）全绿" \
    || bad "本机测试未过，见 /tmp/eldertrans_pilot_pytest.log —— 先修，别带着已知问题上 RunPod 烧钱"
fi
echo

echo "[5] 建机命令（**必须显式传 PUBLIC_KEY**）"
if [ -f "$KEY.pub" ] && [ "${CREATE_FORM:-none}" = old ]; then
  cat <<CMD
  runpodctl create pod --name $POD_NAME \\
    --gpuType "$GPU_TYPE" \\
    --imageName "$IMAGE" \\
    --containerDiskSize 40 --volumeSize 30 --volumePath /workspace \\
    --ports '22/tcp' --secureCloud \\
    --env "PUBLIC_KEY=\$(cat $KEY.pub)"
CMD
  ok "已按本机 runpodctl 的实际 flag 名生成（不是照抄文档）"
  warn "缺 --env PUBLIC_KEY 时 sshd 没有 authorized_keys，表现为持续 Connection refused。"
else
  warn "跳过（缺公钥或建机形式未知）"
fi
echo

if [ "$fail" -ne 0 ]; then
  echo "═══ 零成本检查未全过 —— 不要建机器 ═══"
  exit 1
fi
echo "═══ 零成本检查全过 ═══"

if [ "${1:-}" != "--create" ]; then
  echo
  echo "第二段（建 pod → 训练 → 推理 → 取回 pred.jsonl → 删机）需要显式加 --create，**会计费**。"
  echo "A40 约 \$0.39/hr；2B LoRA 340 条 3 epoch + 100 条推理，预计 30-60 分钟（首次含装依赖）。"
  echo "**执行前请先与人确认**——这是本脚本的既定纪律，不要因为看到这行提示就自动补 --create。"
  exit 0
fi

echo
echo "═══ 第二段：真建机器（计费中）═══"

echo "[建 pod]"
if [ "$CREATE_FORM" != old ]; then
  echo "只实现了旧形式 runpodctl create pod 的建机流程，本机是 $CREATE_FORM，先手动核实 flag 名再继续。"
  exit 1
fi
POD_ID=$(runpodctl create pod --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
  --containerDiskSize 40 --volumeSize 30 --volumePath /workspace \
  --ports '22/tcp' --secureCloud \
  --env "PUBLIC_KEY=$(cat "$KEY.pub")" 2>&1 | tee /tmp/eldertrans_pilot_create.log \
  | grep -oE 'pod "[a-z0-9]+"' | grep -oE '[a-z0-9]+' | tail -1)
if [ -z "$POD_ID" ]; then
  echo "建机失败，见 /tmp/eldertrans_pilot_create.log"
  exit 1
fi
echo "  pod id: $POD_ID"

cleanup() {
  echo "[删机] runpodctl remove pod $POD_ID"
  runpodctl remove pod "$POD_ID" >/dev/null 2>&1 || warn "删机失败，去控制台手动删：$POD_ID"
}
trap cleanup EXIT

echo "[等 SSH 就绪]"
HOST=""
for i in $(seq 1 30); do
  INFO=$(runpodctl get pod "$POD_ID" 2>/dev/null)
  HOST=$(echo "$INFO" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | head -1)
  PORT=$(echo "$INFO" | grep -oE '"22/tcp":\s*[0-9]+' | grep -oE '[0-9]+$')
  [ -n "$HOST" ] && [ -n "${PORT:-}" ] && break
  sleep 10
done
if [ -z "$HOST" ] || [ -z "${PORT:-}" ]; then
  echo "30 次轮询后仍拿不到 SSH host:port，去控制台看 pod 状态：$POD_ID"
  exit 1
fi
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$HOST"
for i in $(seq 1 20); do
  $SSH -o ConnectTimeout=10 'echo ok' >/dev/null 2>&1 && break
  sleep 10
done
if ! $SSH -o ConnectTimeout=10 'echo ok' >/dev/null 2>&1; then
  echo "SSH 一直连不上（$HOST:$PORT）。若非 --env PUBLIC_KEY 缺失（[4] 已检查过），去控制台看日志。"
  exit 1
fi
ok "SSH 通：$HOST:$PORT"

echo "[装依赖]"
$SSH "pip install -q unsloth trl datasets peft accelerate bitsandbytes 2>&1 | tail -5" \
  || { echo "装依赖失败，中止（避免带着装不全的环境继续训练/推理）"; exit 1; }

echo "[传数据与脚本]"
$SSH "mkdir -p $REMOTE_DIR/verifier $REMOTE_DIR/verifier/work"
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
  verifier/train_lora.py verifier/predict_lora.py verifier/train.jsonl verifier/holdout.jsonl \
  "root@$HOST:$REMOTE_DIR/verifier/" \
  || { echo "传数据/脚本失败，中止"; exit 1; }
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
  verifier/work/to_label.json "root@$HOST:$REMOTE_DIR/verifier/work/" \
  || { echo "传 to_label.json 失败，中止"; exit 1; }

# **remote 端也要 set -o pipefail**：不然 `python3 ... | tail -60` 这条流水线的
# 退出码是 tail 的（几乎总是 0），训练/推理真正失败时本地这边看到的仍是"成功"，
# 会带着一个陈旧/半截的 pred.jsonl 往下跑 eval_verifier.py，把它当成摸底结果汇报
# ——这正是 code review 抓出来的那类"看起来成功和真正成功长得一样"的坑。
echo "[训练]"
$SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/train_lora.py --data verifier/train.jsonl --out ./ckpt --epochs $EPOCHS 2>&1 | tail -60" \
  || { echo "远程训练失败（退出码非 0），中止——不要拿这次的 ckpt/pred 当结果"; exit 1; }

echo "[推理（holdout）]"
$SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/predict_lora.py --ckpt ./ckpt --case-ids-from verifier/holdout.jsonl --out pred.jsonl --checkpoint-every 10 2>&1 | tail -60" \
  || { echo "远程推理失败（退出码非 0），中止——不要拿这次的 pred.jsonl 当结果"; exit 1; }

echo "[取回结果]"
mkdir -p runs/verifier
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" "root@$HOST:$REMOTE_DIR/pred.jsonl" runs/verifier/pred.jsonl \
  || { echo "取回 pred.jsonl 失败，中止（别拿本地残留的旧文件当新结果评）"; exit 1; }

echo "[本地验收]"
python3 verifier/eval_verifier.py --pred runs/verifier/pred.jsonl

echo
echo "═══ 摸底跑完。这是 ticket 09 的摸底结果，不是 ticket 11 的验收结果 ═══"
echo "（验收集只有 31 条红线正例，即便零漏报，真实漏报率 95% 上界仍有 11.0%——"
echo " 门槛 5% 在这个规模下数学上证不了，需 ticket 10 扩量。见 verifier/eval_verifier.py 的功效警告。）"
