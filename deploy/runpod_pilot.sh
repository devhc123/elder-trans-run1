#!/usr/bin/env bash
# verifier LoRA 训练 + 推理跑在 RunPod 上。
#
# **两条 track**（ticket 20）：
#   candidate（默认）—— ticket 14 段B判定头，候选级训练 + 候选级验收
#   case            —— ticket 09/11 的案例级联合 JSON 实验（已跑完，保留可复现）
#
# 为什么默认 candidate：本脚本长期指着案例级文件，而零成本预检**照样全绿**——
# 跑 --create 会在毫无提示的情况下花 $3-5 训一个跟当前工作无关的旧实验。默认值
# 本身也是一种"悄悄替你选了实验"，所以另加两道显式提示：预检第一行就打印当前
# track 与将要训练的文件名，--create 在建机之前再确认一次。
#
# 两段式，同 eqbench-run2 项目验证过的模式（避免 run1 白烧 100 分钟 + $0.45
# 那个教训——`runpodctl create pod` 不会自动注入账户密钥，缺
# `--env PUBLIC_KEY=...` 就必然 Connection refused，长得和坏节点一模一样）：
#
#   第一段（默认）—— 全部**零成本**检查，不建机器
#   第二段（--create）—— 真建 pod、跑训练+推理、取回 pred.jsonl、删机（**会计费**）
#
# 用法：
#   bash deploy/runpod_pilot.sh                       # 只跑零成本检查（candidate track）
#   bash deploy/runpod_pilot.sh --track case          # 换成 ticket 09/11 的旧实验
#   bash deploy/runpod_pilot.sh --create              # 建 pod 真跑（**会计费，需人工确认**）
set -uo pipefail

TRACK=candidate
CREATE=no
while [ $# -gt 0 ]; do
  case "$1" in
    --create) CREATE=yes; shift ;;
    # `shift 2` 在只剩一个参数时**会失败但不改变位置参数**，于是 while 循环
    # 原地重新匹配 --track，无限转（/code-review 实测：timeout 5 退出 124）。
    # 先卡住参数个数再 shift。
    --track)  [ $# -ge 2 ] || { echo "--track 后面要跟一个值（candidate 或 case）"; exit 2; }
              TRACK="$2"; shift 2 ;;
    --track=*) TRACK="${1#*=}"; shift ;;
    *) echo "未知参数：$1（认识的是 --create / --track candidate|case）"; exit 2 ;;
  esac
done
if [ "$TRACK" != candidate ] && [ "$TRACK" != case ]; then
  echo "--track 只能是 candidate 或 case，拿到：$TRACK"; exit 2
fi

# track 决定训什么、推什么、验收什么。**这三样必须一起切换**——历史上就是
# 训练指向 A、验收指向 B 而预检不检查，才会全绿着训错东西。
if [ "$TRACK" = candidate ]; then
  TRAIN_DATA="verifier/work/candidate_train.jsonl"
  TRAIN_SCHEMA_KEY="violated"
  # 候选级重构的**目的之一**就是把正例占比从案例级的极端稀疏拉起来（实测
  # 13.8%）。掉回 5% 以下说明组装出了问题，硬拦。
  MIN_POS_RATIO=0.05
  POS_RATIO_HARD=yes
  POOL_REAL="verifier/work/trusted_candidate_pool_holdout.json"
  POOL_ADV="verifier/work/adversarial_subset_holdout.json"
  REBUILD_HINT="python3 verifier/candidate_pool.py --split holdout
       python3 verifier/adversarial_subset.py
       python3 verifier/train_lora.py --make-candidate-data --data verifier/train.jsonl"
else
  TRAIN_DATA="verifier/train.jsonl"
  TRAIN_SCHEMA_KEY="verdict"
  # 案例级实测 4.5%（按 verdict=fail 算；按单条红线槽位算更低，约 1.6%）。
  # **这个稀疏度是 ticket 09/11 记录在案的实验性质，不是组装 bug**——正是它
  # 导致了那两次坍缩。所以这里只报不拦，拦了等于禁止复现历史实验。
  MIN_POS_RATIO=0.05
  POS_RATIO_HARD=no
  REBUILD_HINT="python3 verifier/train_lora.py --make-data --holdout 100"
fi

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
# **第一行就把"这次到底要训什么"摊开。** 默认值替人做了选择，那就必须让人
# 一眼看见默认选的是什么，而不是等花完钱看日志才发现。
printf '  track = %s   将要训练的文件 = %s\n' "$TRACK" "$TRAIN_DATA"
[ "$TRACK" = case ] && printf '  （这是 ticket 09/11 的旧实验。ticket 14 段B判定头用默认的 --track candidate）\n'
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
    bad ".env 的 key 被拒（HTTP ${code}）—— 去 runpod.io 换一把"
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
    bad "有口令（${probe}）——无人值守流程会卡在交互输入上"
  fi
  ok "指纹 $(ssh-keygen -lf "$KEY.pub" | awk '{print $2}')"
fi
echo

echo "[4] 训练/推理前置产物（track = ${TRACK}）"
[ -f verifier/train_lora.py ] && [ -f verifier/predict_lora.py ] && ok "训练/推理脚本都在" || \
  bad "缺 verifier/train_lora.py 或 verifier/predict_lora.py"
[ -f verifier/holdout.jsonl ] && ok "verifier/holdout.jsonl 存在（$(wc -l < verifier/holdout.jsonl | tr -d ' ') 条，已封存）" || \
  bad "缺 verifier/holdout.jsonl"

if [ -f "$TRAIN_DATA" ]; then
  # 这里**不报条数**：`wc -l` 数的是换行符，文件末尾没有换行就比真实条数少 1，
  # 与下面 schema 校验里 Python 数出来的条数对不上。在一个专门用来抓不一致的
  # 预检里放两个互相矛盾的数字，只会制造噪声。条数由下面那条统一报。
  ok "$TRAIN_DATA 存在"
else
  bad "缺 $TRAIN_DATA —— 重建：
       $REBUILD_HINT"
fi

# **文件与 track 匹配的硬校验。** 光检查"文件在不在"不够——真正会烧钱的情形是
# 文件在、而且是**另一个实验的**。读第一行看输出 schema：候选级是
# {"violated": bool}，案例级是 {"verdict": ...}。顺手报正例占比，占比异常
# （比如又退回到 ticket 09/11 那个 1.6% 的坍缩区间）同样拦下来。
if [ -f "$TRAIN_DATA" ] && command -v python3 >/dev/null; then
  schema_out=$(python3 - "$TRAIN_DATA" "$TRAIN_SCHEMA_KEY" "$MIN_POS_RATIO" <<'PY' 2>&1
import json, sys
path, key, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
if not rows:
    print("EMPTY"); raise SystemExit
sample = json.loads(rows[0]["output"])
if key not in sample:
    print(f"MISMATCH 输出 schema 里没有 {key}，实际字段 {sorted(sample)}")
    raise SystemExit
if key == "violated":
    n_pos = sum(1 for r in rows if json.loads(r["output"])["violated"])
else:
    n_pos = sum(1 for r in rows if json.loads(r["output"]).get("verdict") == "fail")
# 占比的格式化与"够不够 5%"的判定都在 Python 里做完，bash 只认字符串——
# 在 shell 里嵌套引号做浮点比较是纯粹给自己找 bug（shellcheck SC2027 抓到过一次）。
ratio = n_pos / len(rows)
# 上下界都要卡（DeepSeek 审计 #5）：只判下限的话，"负例全丢了、100% 都是正例"
# 这种组装事故会走 FINE 分支直接放行。0 条正例更是任何 track 都不能接受。
if n_pos == 0:
    verdict = "ZERO"
elif ratio > 0.5:
    verdict = "HIGH"
elif ratio < floor:
    verdict = "LOW"
else:
    verdict = "FINE"
print(f"OK {n_pos} {len(rows)} {ratio * 100:.1f} {verdict}")
PY
)
  case "$schema_out" in
    OK*)
      set -- $schema_out
      if [ "$5" = FINE ]; then
        ok "训练集 schema 与 track 匹配（含 ${TRAIN_SCHEMA_KEY}），正例 $2/$3 = $4%"
      elif [ "$5" = ZERO ]; then
        bad "训练集里**一条正例都没有**（$3 条全是负例）—— 组装坏了，任何 track 都不能这样训"
      elif [ "$5" = HIGH ]; then
        bad "正例占比 $4%（>50%）—— 多半是负例整批丢了。候选级正常应为 13.8%"
      elif [ "$POS_RATIO_HARD" = yes ]; then
        bad "schema 匹配，但正例占比只有 $4%（门槛 ≥5%）—— 候选级重构的目的之一
       就是把占比从案例级的极端稀疏拉起来（实测应为 13.8%），掉回来说明组装出了问题"
      else
        warn "正例占比 $4%（案例级的已知稀疏度，ticket 09/11 两次坍缩的根因）。
       这是那次实验记录在案的性质，不是组装 bug，所以只报不拦。"
      fi ;;
    MISMATCH*)
      bad "**训练集与 track 不匹配**：$TRAIN_DATA ${schema_out#MISMATCH }
       track=$TRACK 期望输出 schema 含 ${TRAIN_SCHEMA_KEY}。
       这正是「预检全绿但训的是另一个实验」那个坑——重建对应产物，或改 --track。" ;;
    EMPTY) bad "$TRAIN_DATA 是空的" ;;
    *)     bad "读 $TRAIN_DATA 失败：$schema_out" ;;
  esac
fi

if [ "$TRACK" = candidate ]; then
  for pool in "$POOL_REAL" "$POOL_ADV"; do
    [ -f "$pool" ] && ok "$pool 存在" || bad "缺 $pool —— 重建：
       $REBUILD_HINT"
  done
  # 对抗子集的结构探针门槛（ticket 17）。**本批先 warn 不 bad**：ticket 26
  # 之前它必然是红的（位置捷径尚未修复），此时 bad 会让整个预检永远过不去、
  # 反而逼人去忽略它。ticket 26 的验收清单里有一条就是把这里翻成 bad。
  if [ -f "$POOL_ADV" ] && command -v python3 >/dev/null; then
    # --min-positives 50 = adversarial_subset.MIN_SIZE（L2 判读规则）。光检查
    # 文件在不在不够：一份被截断到 3 条的池同样能让所有探针 J=0 而"通过"
    # （/code-review #5）。文件存在 ≠ 文件够用。
    if python3 verifier/shortcut_probes.py --pool "$POOL_ADV" --gate --min-positives 50 \
         >/tmp/eldertrans_probe.log 2>&1; then
      ok "对抗子集通过纯结构探针门槛"
    else
      warn "对抗子集**未通过**纯结构探针门槛（见 /tmp/eldertrans_probe.log）。
       ticket 26 完成前这是预期状态；完成后这一条要翻成硬失败。
       含义：这份验收集能被不读原文的退化分类器分开，它的分数不能单独当结论。"
    fi
  fi
else
  [ -f verifier/work/to_label.json ] && ok "verifier/work/to_label.json 存在（案例级 predict 现场重建 prompt 要用）" || \
    bad "缺 verifier/work/to_label.json"
fi

if ! command -v python3 >/dev/null; then
  # 没有 python3 时，上面的 schema 校验和下面的本机测试都会被静默跳过，`fail`
  # 不增加，预检"全过"——但第二段结束时的本地验收又必须用 python3，那时钱已经
  # 花完了（DeepSeek 审计 #4）。缺 python3 直接判失败。
  bad "本机没有 python3 —— schema 校验、本机测试、以及第二段末尾的本地验收全都要用它"
else
  PILOT_TESTS="tests/test_train_lora.py tests/test_predict_lora.py"
  [ "$TRACK" = candidate ] && PILOT_TESTS="$PILOT_TESTS tests/test_candidate_pool.py \
tests/test_adversarial_subset.py tests/test_eval_candidate_verifier.py tests/test_shortcut_probes.py"
  # shellcheck disable=SC2086
  python3 -m pytest $PILOT_TESTS -q >/tmp/eldertrans_pilot_pytest.log 2>&1 \
    && ok "本机可测部分全绿（$(echo $PILOT_TESTS | wc -w | tr -d ' ') 份测试）" \
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

if [ "$CREATE" != yes ]; then
  echo
  echo "第二段（建 pod → 训练 → 推理 → 取回预测 → 删机）需要显式加 --create，**会计费**。"
  if [ "$TRACK" = candidate ]; then
    # 按 ticket 09 实测标定：340 条 3 epoch = 129 步 / 8分25秒（约 3.1 s/step）。
    echo "A40 约 \$0.39/hr。候选级训练集 3 epoch"
    echo "≈ 2,800+ 步 ≈ 2.4 小时，加建机装依赖约 20 分、加两轮共 2,100+ 次推理约 35 分"
    echo "≈ 3.3 小时 ≈ \$1.3。**预算只够 2-3 个网格点**——第一次固定 --oversample-fail 1"
    echo "单点跑，这趟买的信息是「有没有学到伪影」；伪影若成立，网格搜索的钱全是白花的。"
  else
    echo "A40 约 \$0.39/hr；2B LoRA 340 条 3 epoch + 100 条推理，预计 30-60 分钟（首次含装依赖）。"
  fi
  echo "**执行前请先与人确认**——这是本脚本的既定纪律，不要因为看到这行提示就自动补 --create。"
  exit 0
fi

echo
# **探针门槛在这里硬拦**（DeepSeek 审计 #2）。
#
# 零成本段里它只是 warn——那一段回答的是"你本机的环境和产物齐不齐"，
# ticket 26 完成前对抗子集必然是红的，把整段判失败只会训练出"忽略这一条"的习惯。
# 但**花钱**是另一回事：明知验收集能被不读原文的退化分类器打穿，还去花 $1-3
# 训一轮、再把分数当结果，正是这一批要防的那个形态。所以门槛拦在计费的入口，
# 不拦在检查清单上。
if [ "$TRACK" = candidate ] && [ -f "$POOL_ADV" ]; then
  if ! python3 verifier/shortcut_probes.py --pool "$POOL_ADV" --gate --min-positives 50 \
       >/tmp/eldertrans_probe.log 2>&1; then
    echo "═══ 拒绝建机：对抗子集没通过纯结构探针门槛 ═══"
    echo "  见 /tmp/eldertrans_probe.log。这份验收集能被一个不读原文的退化分类器分开，"
    echo "  花钱训出来的分数没法证明模型学会了判断（ticket 26 未完成前这是预期状态）。"
    echo "  真要在这个状态下跑，先自己确认清楚再临时改这段——不要顺手删掉。"
    exit 1
  fi
fi

echo "═══ 第二段：真建机器（计费中）═══"
# 建机之前把 track 再摊一次。默认值 + 一屏预检输出之后，人很容易已经忘了
# 第一行说的是哪条 track——而这一步之后每一分钟都在计费。
echo "  track = $TRACK"
echo "  训练  = $TRAIN_DATA"
if [ "$TRACK" = candidate ]; then
  echo "  验收  = $POOL_REAL + $POOL_ADV"
else
  echo "  验收  = verifier/holdout.jsonl（案例级）"
fi

echo "[建 pod]"
if [ "$CREATE_FORM" != old ]; then
  echo "只实现了旧形式 runpodctl create pod 的建机流程，本机是 ${CREATE_FORM}，先手动核实 flag 名再继续。"
  exit 1
fi
POD_ID=$(runpodctl create pod --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" --imageName "$IMAGE" \
  --containerDiskSize 40 --volumeSize 30 --volumePath /workspace \
  --ports '22/tcp' --secureCloud \
  --env "PUBLIC_KEY=$(cat "$KEY.pub")" 2>&1 | tee /tmp/eldertrans_pilot_create.log \
  | grep -oE 'pod "[a-z0-9]+"' | grep -oE '[a-z0-9]+' | tail -1)
if [ -z "$POD_ID" ]; then
  # **pod 可能已经建出来了，只是 id 没解析到**（DeepSeek 审计 #8）——`trap cleanup`
  # 还没设，直接 exit 会把一台在计费的机器留在那儿。先去问一遍 RunPod 有没有
  # 活着的 pod，把它打出来，而不是只说"建机失败"。
  echo "建机失败或 pod id 没解析出来，见 /tmp/eldertrans_pilot_create.log"
  echo "[核对是否有漏下的 pod 在计费]"
  timeout 60 runpodctl get pod 2>/dev/null | tail -n +2 | grep . && {
    echo "  ⚠️ **上面这些 pod 还活着，正在计费** —— 确认是不是这次建出来的，"
    echo "     是就手动删：runpodctl remove pod <id>"
  } || echo "  ✓ 没有 pod 在跑，这次没漏下计费资源"
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
  echo "SSH 一直连不上（$HOST:${PORT}）。若非 --env PUBLIC_KEY 缺失（[4] 已检查过），去控制台看日志。"
  exit 1
fi
ok "SSH 通：$HOST:$PORT"

echo "[装依赖]"
# 锁 ticket 09 那次 pilot 日志里**实际记录过**的三个版本（Unsloth 2026.8.15 /
# Transformers 5.5.0 / Torch 2.11.0+cu130）。**TRL 没有记录**，锁不了——
# `SFTConfig` 的序列长度参数名在版本间变过（max_seq_length -> max_length），
# 那条风险靠 train_lora.sft_config_kwargs() 运行时探测兜，不靠这里。
# **remote 也要 set -o pipefail**（DeepSeek 审计 #3）：本地的 `set -uo pipefail`
# 传不到远程 shell，`pip install ... | tail -5` 的退出码是 tail 的（几乎总是 0），
# 装依赖失败会被当成成功，然后带着装不全的环境继续训练——白烧一整个 pod 周期。
# 训练/推理那两条命令早就加了这个前缀，装依赖这条一直漏着。
$SSH "set -o pipefail; pip install -q 'unsloth==2026.8.15' 'transformers==5.5.0' trl datasets peft accelerate bitsandbytes 2>&1 | tail -5" \
  || { echo "装依赖失败，中止（避免带着装不全的环境继续训练/推理）"; exit 1; }

echo "[传数据与脚本]"
$SSH "mkdir -p $REMOTE_DIR/verifier $REMOTE_DIR/verifier/work"
scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
  verifier/train_lora.py verifier/predict_lora.py verifier/holdout.jsonl \
  "root@$HOST:$REMOTE_DIR/verifier/" \
  || { echo "传脚本失败，中止"; exit 1; }
if [ "$TRACK" = candidate ]; then
  # 候选级：训练集 + 两个池。**不传 to_label.json**——候选级 predict 从池里
  # 重建 prompt，根本不读它（predict_lora 有测试守着这一点）。
  scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
    "$TRAIN_DATA" "$POOL_REAL" "$POOL_ADV" "root@$HOST:$REMOTE_DIR/verifier/work/" \
    || { echo "传候选级产物失败，中止"; exit 1; }
else
  scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
    verifier/train.jsonl "root@$HOST:$REMOTE_DIR/verifier/" \
    || { echo "传 train.jsonl 失败，中止"; exit 1; }
  scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
    verifier/work/to_label.json "root@$HOST:$REMOTE_DIR/verifier/work/" \
    || { echo "传 to_label.json 失败，中止"; exit 1; }
fi

# **remote 端也要 set -o pipefail**：不然 `python3 ... | tail -60` 这条流水线的
# 退出码是 tail 的（几乎总是 0），训练/推理真正失败时本地这边看到的仍是"成功"，
# 会带着一个陈旧/半截的 pred.jsonl 往下跑 eval_verifier.py，把它当成摸底结果汇报
# ——这正是 code review 抓出来的那类"看起来成功和真正成功长得一样"的坑。
echo "[训练]"
# --oversample-fail 1（单点）是 ticket 20 写死的：预算只够 2-3 个网格点，而
# 这一趟要买的信息是"有没有学到伪影"。3 epoch 不动，为的是与 ticket 09/11 可比。
$SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/train_lora.py --data $TRAIN_DATA --out ./ckpt --epochs $EPOCHS --oversample-fail 1 2>&1 | tail -60" \
  || { echo "远程训练失败（退出码非 0），中止——不要拿这次的 ckpt/pred 当结果"; exit 1; }

mkdir -p runs/verifier
if [ "$TRACK" = candidate ]; then
  echo "[推理 1/2：真实 holdout 候选池]"
  $SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/predict_lora.py --ckpt ./ckpt --candidates-from $POOL_REAL --out pred_real.jsonl --checkpoint-every 50 2>&1 | tail -20" \
    || { echo "真实池推理失败（退出码非 0），中止"; exit 1; }
  echo "[推理 2/2：对抗子集]"
  $SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/predict_lora.py --ckpt ./ckpt --candidates-from $POOL_ADV --out pred_adv.jsonl --checkpoint-every 50 2>&1 | tail -20" \
    || { echo "对抗子集推理失败（退出码非 0），中止"; exit 1; }

  echo "[取回结果]"
  scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" \
    "root@$HOST:$REMOTE_DIR/pred_real.jsonl" "root@$HOST:$REMOTE_DIR/pred_adv.jsonl" runs/verifier/ \
    || { echo "取回预测失败，中止（别拿本地残留的旧文件当新结果评）"; exit 1; }

  echo "[本地验收]"
  python3 verifier/eval_candidate_verifier.py \
    --pred-real runs/verifier/pred_real.jsonl --pool-real "$POOL_REAL" \
    --pred-adv  runs/verifier/pred_adv.jsonl  --pool-adv  "$POOL_ADV"
  echo
  echo "═══ 候选级一轮跑完（ticket 14 段B判定头）═══"
  echo "判读照验收报告末尾那张三分支预注册表走，不要另外解释数字。"
else
  echo "[推理（holdout，案例级）]"
  $SSH "set -o pipefail; cd $REMOTE_DIR && python3 verifier/predict_lora.py --ckpt ./ckpt --case-ids-from verifier/holdout.jsonl --out pred.jsonl --checkpoint-every 10 2>&1 | tail -60" \
    || { echo "远程推理失败（退出码非 0），中止——不要拿这次的 pred.jsonl 当结果"; exit 1; }

  echo "[取回结果]"
  scp -o StrictHostKeyChecking=no -i "$KEY" -P "$PORT" "root@$HOST:$REMOTE_DIR/pred.jsonl" runs/verifier/pred.jsonl \
    || { echo "取回 pred.jsonl 失败，中止（别拿本地残留的旧文件当新结果评）"; exit 1; }

  echo "[本地验收]"
  python3 verifier/eval_verifier.py --pred runs/verifier/pred.jsonl

  echo
  echo "═══ 摸底跑完。这是 ticket 09 的摸底结果，不是 ticket 11 的验收结果 ═══"
  echo "（验收集只有 31 条红线正例，即便零漏报，真实漏报率 95% 上界仍有 11.0%——"
  echo " 门槛 5% 在这个规模下数学上证不了，需 ticket 10 扩量。见 verifier/eval_verifier.py 的功效警告。）"
fi
