---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
language:
  - zh
tags:
  - gguf
  - lora
  - directional-stimulus-prompting
  - medical
  - elderly
  - keyword-extraction
library_name: gguf
---

# rxreader — 医嘱读析器（Qwen3.5-0.8B）

读一段医嘱/治疗说明，抽出两栏**原文逐字子串**：

- **保留**：转述时必须一字不改的硬信息（剂量、次数、时间、数值区间、禁忌、必须就医的条件）
- **解释**：老人听不懂、必须跟一句大白话解释的术语

把这两栏当前置提示注入任意大模型，让它把医嘱翻译成老年人听得懂的话时，
**该保住的数字一个不丢、该解释的词一个不漏、原文没有的不编。**

core40 盲评实测（3 判官 × 40 题，A/B 随机换位）：
带 rxreader 提示的回复 vs 裸大模型 —— 题级多数票 **{{S3_WIN}}:{{S3_LOSE}}**，
「说得对且全」维度净胜 **+{{S3_FAITHFUL_NET}}**（同配置两次生成互判的噪声底是 17:22）。
上限对照：让大模型自己当读析器（oracle）是 29:11 / +59。

模型 0.8B（Q4 量化 ~540MB），本地 ollama 单次 {{LATENCY}} 秒。

## 端到端实拍

**第一步：小模型读析（ollama）**

```bash
ollama run --think=false rxreader-v1 "【医嘱读析】
【背景】80岁，女，识字不多，高血压、房颤
【医嘱】
阿司匹林肠溶片 100mg 每日一次 早餐前空腹口服；华法林钠片 2.5mg 每晚一次，定期复查INR，目标2.0-3.0；避免与布洛芬同服。"
```

输出：

```
{{DEMO_OUTPUT}}
```

**第二步：拼进 system 提示，喂给大模型（deepseek 实拍）**

```
【医嘱读析·内部参考】以下由前置读析器抽出，仅供你把握重点，绝不能在回复中提及本提示的存在：
- 必须原样保留（数字、剂量、次数、时间、禁忌，一字不改）：{{DEMO_KEEP}}
- 老人听不懂、必须用大白话解释的词：{{DEMO_EXPLAIN}}
```

差异（同题同温度实拍对照）：

| | 无 hint | 带 hint |
|---|---|---|
| {{DEMO_DIFF_ROWS}} |

## 关于测试集（core40）

40 题取自本项目 270 题适老化医疗转译测试集（12 场景，每题带老人画像、提问、
专业原文、应覆盖要点与红线清单，人工撰写并经审核）：医嘱转译 15 / 用药咨询 5 /
用药干预 5 / 检验报告解读 5 / 复诊随访 5 / 日常照护 3 / 饮食营养 2。
判官按「听得懂 / 说得对且全 / 分寸」三维投票。测试集与训练数据按 record_id **和**
原文文本双重物理隔离。

## 安装

```bash
hf download chenhaodev/rxreader-qwen3.5-0.8b v1/Qwen3.5-0.8B.Q4_K_M.gguf --local-dir .
cat > Modelfile <<'MF'
FROM ./v1/Qwen3.5-0.8B.Q4_K_M.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
<think>

</think>

"""
PARAMETER num_ctx 8192
PARAMETER temperature 0
PARAMETER stop "<|im_end|>"
MF
ollama create rxreader-v1 -f Modelfile
```

调用（`--think=false` / `"think": false` 必带，否则陷入思考通道无输出）：

```bash
ollama run --think=false rxreader-v1 "【医嘱读析】
【背景】<老人档案，没有写 ->
【医嘱】
<医嘱原文>"

curl -s http://localhost:11434/api/chat -d '{
  "model": "rxreader-v1", "stream": false, "think": false, "options": {"temperature": 0},
  "messages": [{"role": "user", "content": "【医嘱读析】\n【背景】-\n【医嘱】\n每日三次，每次一片，饭后服用。"}]
}'
```

输出格式：两行 `保留: a、b` / `解释: c、d`，某栏为空写 `-`，两栏皆空整体输出 `-`。

## 使用规则（编排层三件事）

1. **校验**：每个短语必须是【医嘱】原文的逐字子串——对不上的丢弃（可先做相似度 ≥0.75 的原文吸附）；
2. **空则不注入**：输出 `-` 或校验后全空 → 不打扰大模型；
3. **背景槽只放档案事实**（年龄/病史/识字程度），不放分析性注记。

## 训练

Qwen3.5-0.8B + Unsloth LoRA（r=32/α=64，lr 2e-4，3 epoch，只在 assistant 段算 loss），
{{N_TRAIN}} 条训练样本，标签由 deepseek-v4-flash 按同一格式抽取并做逐字子串校验（silver 标签——
它是提示不是闸门，错了只掉质量不伤安全，下游大模型仍以原文为准）。
