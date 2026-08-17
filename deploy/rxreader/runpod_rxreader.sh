#!/usr/bin/env bash
# rxreader（医嘱读析器，Qwen3.5-0.8B LoRA）在 RunPod 上一键训练 + 取回 GGUF。
# 抄 deploy/runpod_pilot.sh 的两段式骨架（零成本预检 / --create 真建机），训练部分抄 eqreader v4。
#
#   bash deploy/rxreader/runpod_rxreader.sh            # 只跑零成本检查
#   bash deploy/rxreader/runpod_rxreader.sh --create   # 建 pod → 训练 → 导 GGUF → 取回 → 删机（**计费，需人工确认**）
#
# 取回后本地：
#   cp runs/rxreader/Qwen3.5-0.8B.Q4_K_M.gguf deploy/rxreader/ && (cd deploy/rxreader && ollama create rxreader-v1 -f Modelfile)
#   python app/rxreader.py --batch data/core40_rx.jsonl --out runs/s3/hints.jsonl
#   python app/translate.py --batch data/core40_rx.jsonl --hints runs/s3/hints.jsonl --out runs/s3/outputs.jsonl
#   python app/ab_judge.py --cases data/core40_rx.jsonl --a runs/s1-baseline/outputs.jsonl --b runs/s3/outputs.jsonl \
#       --out runs/ab/baseline_vs_rxreader.json --judges 3 --label-a baseline --label-b rxreader-v1
set -uo pipefail
CREATE=no; [ "${1:-}" = "--create" ] && CREATE=yes

KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/runpod_automation}"
GPU_TYPE="${RUNPOD_GPU:-NVIDIA A40}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
POD_NAME="${RUNPOD_POD_NAME:-eldertrans-rxreader}"
REMOTE_DIR="/workspace/elder"
TRAIN="deploy/rxreader/data/train_rxreader.jsonl"; VAL="deploy/rxreader/data/val_rxreader.jsonl"
fail=0
ok()   { printf '  ✓ %s\n' "$*"; }
bad()  { printf '  ✗ %s\n' "$*"; fail=1; }
warn() { printf '  ! %s\n' "$*"; }

echo "═══ 第一段：零成本检查 ═══"
echo "[1] runpodctl"
if ! command -v runpodctl >/dev/null; then bad "runpodctl 没装：brew install runpod/runpodctl/runpodctl"; CREATE_FORM=none
else
  ok "$(runpodctl version 2>&1 | head -1)"
  if runpodctl create pod --help >/dev/null 2>&1; then CREATE_FORM=old; ok "旧形式 runpodctl create pod"
  elif runpodctl pod create --help >/dev/null 2>&1; then CREATE_FORM=new; warn "只有新形式，本脚本建机流程按旧形式写，需手动核 flag"
  else CREATE_FORM=none; bad "两种形式都不认"; fi
fi
echo "[2] API key / 认证"
ENV_KEY=$(grep '^RUNPOD_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d ' \r')
[ -n "$ENV_KEY" ] && ok ".env 有 RUNPOD_API_KEY" || bad ".env 缺 RUNPOD_API_KEY"
if timeout 60 runpodctl get pod >/dev/null 2>&1; then ok "runpodctl 已认证"; else bad "runpodctl 未认证：runpodctl config --apiKey <key>"; fi
running=$(timeout 60 runpodctl get pod 2>/dev/null | tail -n +2 | grep -c . || true)
[ "${running:-0}" -gt 0 ] && warn "当前有 $running 台 pod 在跑（在计费）" || ok "没有 pod 在跑"
echo "[3] SSH 密钥 $KEY"
[ -f "$KEY" ] && [ -f "$KEY.pub" ] && ok "私钥公钥都在" || bad "缺 SSH 密钥：ssh-keygen -t ed25519 -f $KEY -N ''"
echo "[4] 训练产物"
for f in deploy/rxreader/train_rxreader.py deploy/rxreader/template_probe.json deploy/rxreader/pod_run_rxreader.sh deploy/rxreader/23b_resume_gguf_export.py; do
  [ -f "$f" ] && ok "$f" || bad "缺 $f"; done
if [ -f "$TRAIN" ] && [ -f "$VAL" ]; then
  out=$(python3 - "$TRAIN" "$VAL" <<'PY' 2>&1
import json, sys
for p in sys.argv[1:]:
    rows=[json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
    assert rows and all(r["messages"][0]["role"]=="user" and r["messages"][1]["role"]=="assistant" for r in rows), p
    assert all(r["messages"][0]["content"].startswith("【医嘱读析】") for r in rows), "user 模板不对"
    empty=sum(1 for r in rows if r["messages"][1]["content"]=="-")
    print(f"{p}: {len(rows)} 条，目标为 - 的 {empty}（{empty/len(rows):.1%}）")
PY
) && { echo "$out" | sed 's/^/  ✓ /'; } || bad "训练数据校验失败：$out"
else bad "缺 $TRAIN / $VAL —— python3 app/build_rxreader_data.py --pool runs/s2-train/pool.jsonl --labels runs/s2-train/labels.jsonl --out-dir deploy/rxreader/data"; fi
python3 -m pytest tests/test_rxreader_format.py -q >/tmp/rxreader_pytest.log 2>&1 && ok "格式往返测试绿" || bad "tests/test_rxreader_format.py 未过，见 /tmp/rxreader_pytest.log"
echo
[ "$fail" -ne 0 ] && { echo "═══ 预检未全过，不建机 ═══"; exit 1; }
echo "═══ 预检全过 ═══"
if [ "$CREATE" != yes ]; then
  echo "第二段需显式 --create（**计费**）。A40 ≈ \$0.39/hr；0.8B LoRA ~1,850 条 3 epoch ≈ 175 步，"
  echo "eqreader v4 同配置实测建机+装依赖+训练+GGUF ≈ 40 分钟 ≈ \$0.3。**执行前先与人确认。**"
  exit 0
fi

echo "═══ 第二段：建机（计费中）═══"
[ "$CREATE_FORM" = old ] || { echo "只实现了旧形式 runpodctl create pod"; exit 1; }
POD_ID=$(runpodctl create pod --name "$POD_NAME" --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
  --containerDiskSize 40 --volumeSize 30 --volumePath /workspace --ports '22/tcp' --secureCloud \
  --env "PUBLIC_KEY=$(cat "$KEY.pub")" 2>&1 | tee /tmp/rxreader_create.log \
  | grep -oE 'pod "[a-z0-9]+"' | grep -oE '[a-z0-9]+' | tail -1)
[ -n "$POD_ID" ] || { echo "建机失败，见 /tmp/rxreader_create.log；核对 runpodctl get pod 有没有漏下计费的 pod"; exit 1; }
echo "  pod id: $POD_ID"
cleanup() { echo "[删机] $POD_ID"; runpodctl remove pod "$POD_ID" >/dev/null 2>&1 || warn "删机失败，去控制台手动删：$POD_ID"; }
trap cleanup EXIT
# 第二层保险：脱离会话的定时删机（90 分钟硬线）。本地脚本被杀 / 终端断了，pod 也不会一直计费。
nohup bash -c "sleep 5400; runpodctl remove pod $POD_ID >/dev/null 2>&1" >/dev/null 2>&1 &
echo "  已挂 90 分钟硬线删机（pid $!）"

echo "[等 SSH]"
# host:port 优先走 REST API（publicIp + portMappings["22"]），后备 runpodctl get pod -a。
# 之前用 `get pod <id>` + grep '"22/tcp"' 从没真跑通过（本项目 pilot 脚本那段是未验证的），第一次建机
# 就在这里空等 5 分钟后删机。原始输出留在 runs/rxreader/podinfo.log 供下次核对格式。
HOST=""; PORT=""
for i in $(seq 1 60); do
  J=$(curl -s -m 20 -H "Authorization: Bearer $ENV_KEY" "https://rest.runpod.io/v1/pods/$POD_ID")
  read -r HOST PORT < <(python3 - "$J" <<'PY'
import json, sys
try:
    d = json.loads(sys.argv[1])
    ip = d.get("publicIp") or ""
    pm = d.get("portMappings") or {}
    port = pm.get("22") or pm.get(22) or ""
    print(ip, port)
except Exception:
    print("", "")
PY
)
  if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    INFO=$(runpodctl get pod -a 2>/dev/null | grep "$POD_ID")
    echo "$INFO" >> runs/rxreader/podinfo.log
    HP=$(echo "$INFO" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+->22' | head -1)
    HOST=${HP%%:*}; PORT=$(echo "$HP" | grep -oE ':[0-9]+' | tr -d : | head -1)
  fi
  [ -n "$HOST" ] && [ -n "$PORT" ] && break; sleep 10
done
[ -n "$HOST" ] && [ -n "$PORT" ] || { echo "10 分钟拿不到 SSH host:port，pod=$POD_ID；REST 最后返回：$J"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $KEY -p $PORT root@$HOST"
for i in $(seq 1 20); do $SSH -o ConnectTimeout=10 'echo ok' >/dev/null 2>&1 && break; sleep 10; done
$SSH -o ConnectTimeout=10 'echo ok' >/dev/null 2>&1 || { echo "SSH 连不上 $HOST:$PORT"; exit 1; }
ok "SSH 通 $HOST:$PORT"

echo "[传脚本与数据]"
$SSH "mkdir -p $REMOTE_DIR/deploy/rxreader/data"
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" -r deploy/rxreader/* "root@$HOST:$REMOTE_DIR/deploy/rxreader/" || { echo "传文件失败"; exit 1; }

echo "[训练（pod_run_rxreader.sh：装 unsloth → 训 → 导 q8_0/q4_k_m）]"
$SSH "set -o pipefail; cd $REMOTE_DIR && bash deploy/rxreader/pod_run_rxreader.sh 2>&1 | tee run.log | grep -E 'USING|SETUP_DONE|train=|supervised|loss|GGUF|存到|ALL_DONE|Error|Traceback' " \
  || { echo "远程训练失败，先看 run.log："; $SSH "tail -60 $REMOTE_DIR/run.log"; exit 1; }
$SSH "grep -q ALL_DONE $REMOTE_DIR/run.log" || {
  echo "没看到 ALL_DONE——多半 GGUF 导出撞 FileNotFoundError，补跑 23b："
  $SSH "cd $REMOTE_DIR && python3 deploy/rxreader/23b_resume_gguf_export.py --variant rxreader --quant q4_k_m 2>&1 | tail -5" || { echo "补导失败"; exit 1; }
}

echo "[取回 GGUF + adapter]"
mkdir -p runs/rxreader
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
  "root@$HOST:$REMOTE_DIR/outputs/lora_rxreader/gguf_q4_k_m_gguf/Qwen3.5-0.8B.Q4_K_M.gguf" runs/rxreader/ || { echo "取回 Q4 失败"; exit 1; }
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
  "root@$HOST:$REMOTE_DIR/outputs/lora_rxreader/gguf_q8_0_gguf/Qwen3.5-0.8B.Q8_0.gguf" runs/rxreader/ || warn "Q8 取回失败（非致命）"
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" -r "root@$HOST:$REMOTE_DIR/outputs/lora_rxreader/lora_adapter" runs/rxreader/ || warn "adapter 取回失败（非致命）"
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" "root@$HOST:$REMOTE_DIR/run.log" runs/rxreader/train_run.log || true
ls -la runs/rxreader/
echo "═══ 训练完成，pod 将在退出时删除。接下来按脚本头部注释做本地 ollama create + S3 盲评 ═══"
