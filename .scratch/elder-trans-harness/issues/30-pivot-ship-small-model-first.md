# 30 — 转向：先发布一个能用的小模型，再谈 harness（对标 eqreader）

**用户判断（2026-08-17）**：「我就是想做一个医嘱转化、给老年人看得懂的翻译器。
小模型+大模型应该可以达到很好的效果。怎么搞得这么复杂？先发布小模型再说，
学习 https://huggingface.co/chenhaodev/eqreader-qwen3.5-0.8b 。」

## 为什么会复杂（复盘，一段话）

前 29 张票把小模型放在了**最难的岗位**上：段 B 红线判定器（漏报硬门 ≤5%）。
判定器需要 gold 标签、正负例分布配平、捷径探针、教师交叉复核……每一步都是对的，
但它们是**验收一个安全门**所需的纪律，不是**发布一个翻译器**所需的步骤。
结果：64 个 commit、0 个可下载的模型、0 个能跑的翻译器。

## eqreader 的可复制之处（不是它的领域，是它的形态）

1. 小模型只干一件**可校验的窄活**：抽原文逐字子串。错了能被编排层丢弃，不伤安全。
2. 大模型干生成；小模型的输出只是**前置提示**（DSP），空则不注入。
3. 验收只有一个数：~40 题盲评 A/B 胜负（21:13）。没有 CI、没有 Wilson。
4. 模型卡 = 一个端到端实拍案例 + 安装三行 + 编排规则三条。

## 本项目对应的小模型：`rxreader`（医嘱读析器）

输入：【背景】老人档案（年龄/病史/文化程度，没有写 -）+【医嘱】原文
输出（单行、逐字子串、无则 -）：
- `必须原样保留: <剂量/频次/时间/禁忌数字>`   ← 对应忠实性红线
- `需要解释: <老人听不懂的术语>`               ← 对应 KPI-1 可读性

编排层：子串校验 → 拼进 system 提示 → deepseek 生成适老化转译。
两个字段都可用规则/lexicon 兜底，所以 0.8B 训不好也有下限。

## 分步（每步都产出能用的东西）

- **S1（$0，今天）**：纯 deepseek 翻译器 CLI + prompt，用 `candidates/sampled_医嘱转译.jsonl`
  抽 40 题做 core40 式盲评基线。这就是 ticket 29 里「全量 ds」那条臂。
- **S2**：用 deepseek 给 corpus 医嘱标 rxreader 输出（silver 就够——这是提示不是闸门，
  DISCIPLINE D1.2b 的 silver 顾虑在这里不成立），复用 `verifier/train_lora.py`
  的 unsloth+RunPod 链，底座 Qwen3.5-0.8B（与 eqreader 同款，Modelfile 直接抄）。
- **S3**：40 题 A/B 盲评「带 hint vs 裸 deepseek」，胜即发布 HF 卡（结构照抄 eqreader）。
- **S4（以后）**：ticket 14–29 的红线判定器作为**可选后置复检**接回来。

**Blocked by:** 无。前置：确认 S1 的 prompt 与 40 题抽样口径。

## S1 完成（2026-08-17）

交付物：
- `app/translate.py` —— 产品 CLI/库：单条（`--text/--persona`、stdin、`--case`）与批量（`--batch/--out`，断点续跑）；
  prompt 直接 import 自 `metrics/run_eval.py`，数字与 harness 互换；`--hint/--hints` 是 rxreader 注入槽
  （形态照抄 eqreader「内部参考」段），空则不注入。
- `data/core40_rx.jsonl` —— 40 题（医嘱转译 15 / 用药咨询 5 / 用药干预 5 / 检验报告解读 5 / 复诊随访 5 /
  日常照护 3 / 饮食营养 2；难度 易20 中15 难5），确定性抽样。
- `runs/s1-baseline/outputs.jsonl` —— 纯 deepseek-v4-flash 基线，40/40 无截断无空输出，单条 11–18s。
- `app/ab_judge.py` —— 盲评 A/B：题内随机换位、三维（听得懂/说得对且全/分寸）+ 总体投票、`--judges N`
  合成多判官、附确定性可读性两臂对照。判官=deepseek（与生成同源，配对下两臂同受自偏好，只读相对胜负）。

**噪声底（必须先读，D1.2 对照臂）**：`runs/ab/noise_baseline_vs_live003.json`
同一 prompt、同一模型的两次生成互判，3 判官 × 40 题：

| 维度 | 臂A(s1) | 臂B(live-003) | 平 |
|---|---:|---:|---:|
| 听得懂 | 30 | 54 | 34 |
| 说得对且全 | 50 | 57 | 11 |
| 分寸 | 24 | 28 | 66 |
| 总体票 | 51 | 63 | 4 |
| **题级多数票** | **17** | **22** | 1 |

可读性 strict 达标率两臂均 1.0（均值 0.9585 / 0.9588）——KPI-1 在 deepseek 基线上已饱和，
S3 的读析 hint 不该拿可读性当主证据，主战场是「说得对且全」。

**推论**：题级 17:22 是纯噪声，因此 S3 的判据定为——40 题、3 判官，**题级多数票 ≥ 27:13**
（噪声底 22 再往上 5 题）且「说得对且全」维度票数净胜 ≥ 20，两条同时满足才算 hint 有效；
达不到就加到 80 题再看，不加判官。eqreader 的 21:13 幅度在本任务上不构成证据。

下一步 S2：写 rxreader 的 silver 标注脚本（deepseek 抽「必须原样保留 / 需要解释」两栏、子串校验），
底座 Qwen3.5-0.8B，复用 `verifier/train_lora.py` 的 unsloth+RunPod 链。

## S2-0：oracle hint 天花板 POC（$0 级，2026-08-17）——**成立**

训 0.8B 之前先问：hint 这条路本身有没有天花板？让 deepseek 自己当 rxreader
（`app/rxreader_label.py`，抽 keep/explain 两栏、逐字子串校验+0.75 吸附），
把 hint 注回 `app/translate.py`，与 S1 基线盲评（同判官、同 3×40）：

| 维度 | 基线 | oracle-hint | 平 |
|---|---:|---:|---:|
| 听得懂 | 53 | 54 | 12 |
| 说得对且全 | 24 | **83** | 12 |
| 分寸 | 20 | 49 | 50 |
| 总体票 | 30 | 84 | 5 |
| **题级多数票** | **11** | **29** | 0 |

对照噪声底 17:22 → **29:11 且忠实维度净胜 +59，两条判据都过**。增益几乎全部来自
「说得对且全」，与预期一致（可读性在基线上已饱和）。

副作用：hint 臂平均长 +12%（709→796 字）；耗时持平（15.6s vs 16.0s）；可读性 strict
达标率 1.0→0.95，掉的两题（医嘱转译-011、用药干预-002）是药名/术语被反复念
（「奥美拉唑」出现 5 次）——explain 栏的措辞可以加一句「解释一次后用白话代称」，S3 时调。

标注器踩坑：deepseek-v4-flash `max_tokens=4000` 时 13/40 条 thinking 吃光、content 空
（reasoning_tokens 3789–4000）；已改 12000 并把空输出记为 error 触发重标。**第三次栽同一个坑**，
已在 `app/rxreader_label.py` 注释里钉住。

**结论**：hint 岗位值得训。目标：0.8B 复现 deepseek 标注的 keep/explain（silver 即可），
验收仍是 S3 盲评（带 0.8B hint vs 裸），不是标注一致率。

产物：`runs/s2-oracle/{hints,outputs}.jsonl`、`runs/ab/baseline_vs_oraclehint.json`。

## S2：训练数据 + RunPod（2026-08-17 晚）

- 训练池：`verifier/work/to_label.json` 的 1,955 条 verifier_train 原文（record_id 与测试集零重叠；
  另按原文前 60 字查重剔掉 13 条跨库同文）→ 1,942 条。
- 标签：`app/rxreader_label.py`（deepseek-v4-flash）→ `runs/s2-train/labels.jsonl`，1 条错误；
  `app/build_rxreader_data.py` → **train 1,844 / val 97**，目标为 `-` 的 0.4%，user 段 p95 1,101 字。
- 零样本 qwen3.5:0.8b 当 rxreader：core40 两栏全空（写满篇解读）→ 微调必要性成立。
- RunPod：`deploy/rxreader/runpod_rxreader.sh --create`（用户 17:5x 授权），A40，Qwen3.5-0.8B LoRA
  r32/α64、3 epoch，配置抄 eqreader v4。日志 `runs/rxreader/runpod.log`。

### 💸 事故：deepseek 一个 batch 烧了 ~¥48

- 标注器沿用 harness 的默认（thinking 开、`max_tokens` 12000），12 并发起跑没先算账。
- 实测每条 completion ~3.9k token，**reasoning 占 95%**；deepseek 当天 16:00 UTC 起改峰谷计价，
  峰时（06–10 UTC，正是跑的时段）输出 $1.32/M → 1,311 条 ≈ ¥48，账户余额剩 ¥43。
- 修复：`thinking: {type: disabled}` 后同一条 **63 tok / 3s vs 3,844 tok / 40s，输出等价**；
  剩余 632 条 ¥0.99 跑完。标注器现默认关 thinking、逐条记 usage、`--budget-cny` 默认 5 超了自停。
- 已写入 `~/.claude/skills/finetune-gguf/SKILL.md` 北极星「推论二：每 batch ≤ ¥5（硬门）」，
  含三条规矩（抽取任务先关 thinking / 先跑 20 条读 usage / 记账+自停+挪谷时）。
- 未修的：`app/translate.py`（生成，thinking 可能有用，保留）与 `app/ab_judge.py`（判官）还没接预算门，
  下一次批量前接上。

## S3：rxreader-v1 盲评——**过线**（2026-08-17 19:xx）

训练：RunPod A40（pod hmm5kjx3cwby85，第一台 09tn7169dhsio5 因 SSH host:port 解析失败空转 5 分钟后自动删机），
Qwen3.5-0.8B LoRA r32/α64、3 epoch/174 步、13 分钟；train_loss 0.40，eval_loss 0.437→0.442→0.441（第 2 轮起收敛）。
GGUF q4_k_m 542MB / q8_0 834MB 已取回 `runs/rxreader/`，pod 已删。GPU 花费约 $0.3。

本地 ollama `rxreader-v1`：demo 医嘱 1.9s，六个硬信息 + 四个术语全为原文子串；core40 均耗时 1.17s，
两栏皆空 0，非子串丢弃 30/371（8%）。零样本 0.8B 两栏全空 → 微调把格式与倾向都学进去了。

| 集合 | 题级多数票 (基线:rxreader-v1) | 忠实 A:B | 听得懂 A:B | 分寸 A:B |
|---|---:|---:|---:|---:|
| core40 | 11 : **26**（平 3） | 30 : **81** | 53 : 47 | 20 : 35 |
| core40b（预注册扩样） | 10 : **29**（平 1） | 37 : **78** | 43 : 52 | 25 : 39 |
| **core80** | **21 : 55**（平 4） | **67 : 159（+92）** | 96 : 99 | 45 : 74 |

对照：噪声底 17:22；oracle hint（deepseek 自己当读析器）29:11 / 忠实 +59。**rxreader-v1 拿到了 oracle 增益的大部分**
（同 40 题 26:11 / +51 vs 29:11 / +59），且 1s 级本地跑。

判读过程如实记录：
- core40 首轮 26:12（平 2）。其中医嘱转译-007 的 hint 臂被 thinking 吃光 8000 额度、正文半句截断
  （`finish_reason=length`），判官 3:0 判输。按 harness 既有规则（截断不得混进正常样本）重生成一次
  （加了「截断自动升额度重试」）、只重判该题 3 票拼回 → 26:11（平 3）；原始文件 `*_orig.json` 保留。
- 字面判据「≥27:13」差 1 题（净差 +15 其实已超 27:13 的 +14），**不动判据**，按票据预注册的后备
  「加到 80 题再看」执行 → core40b 29:10，core80 55:21。
- 可读性 strict 达标率两臂 0.975–1.0 持平（基线上已饱和，非主战场）。

费用：hint 臂生成 + 判官 core40 ¥1.9（峰）+ core40b ¥1.0（谷）；训练 $0.3。

产物：`runs/s3/{hints,outputs}[_b].jsonl`、`runs/ab/baseline_vs_rxreader_v1{,_b,_core80,_orig}.json`、
`runs/s3/demo/demo.json`（模型卡实拍）、`deploy/rxreader/hf_model_card.md`（已填数）。
