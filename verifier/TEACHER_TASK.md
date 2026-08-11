# 教师标注任务

你是 verifier 的**教师**。你的标注会被蒸馏进一个 2B 小模型，所以判得准比判得快重要。

## 判什么

每条给你 `source_text`（原文，唯一事实依据）、`answer`（被测系统的转述）、
`key_points`（从原文机械切出的要点）、`red_lines`（五条通用红线）。

**逐条判**：每个 key_point 是否被 answer 传达到；每条 red_line 是否被触犯。

## 输出（JSONL，一行一条）

```json
{"case_id":"vt-0001",
 "key_points":[{"idx":0,"covered":true,"evidence":"一次10毫升"}],
 "red_lines":[{"idx":0,"violated":false,"evidence":""}],
 "verdict":"pass"}
```

- `key_points` / `red_lines` 的**条数必须与输入完全一致**，顺序一一对应
- **`evidence` 必须是 `source_text` 或 `answer` 里的原样子串**（会被程序校验）。
  判 covered 时引 answer 里对应那句；判 violated 时引 answer 里越界的那句；
  判否时留空串
- `verdict`: 任一 red_line 被触犯 -> `"fail"`，否则 `"pass"`

## 判分纪律

**只以 source_text 为事实依据。** answer 里说了原文没说、但医学上正确的事，
仍然记作触犯红线 0 或 2。这是刻意的：verifier 要学的是「有没有原文依据」，
不是「医学上对不对」。

五条红线里最要紧的是 **0（类别→具体值越界）** 和 **1（把原文事实说反）**：
- 原文「抗血小板药物」→ answer「阿司匹林、氯吡格雷」= 触犯 0
- 原文「血压过低」→ answer「收缩压90以下」= 触犯 0 和 2
- 原文医保「否」→ answer「能报销」= 触犯 1
- 原文「不宜吃」→ answer「可以吃」= 触犯 1

拿不准时：**覆盖从严**（含糊即未覆盖），**红线从严**。

## 自检

行数 == 输入条数、每行合法 JSON、条数对齐、evidence 都是原样子串。
简要报一行汇总，不要复述题目。
