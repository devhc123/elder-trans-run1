# 09 — 教师标注管线 + 500 条摸底训练与验收脚本

**What to build:** 把「Claude 判官」的判别能力开始往小模型里搬，先花最小代价摸清 2B 能不能胜任——**别一次把标注预算花完**。

三件事一起交付，因为它们只有连起来才回答得了一个问题：2B 离教师差多少？

**一、教师标注管线。** 用 Claude subagent 在 verifier 训练池（03 切出的那份）上打标，产出 500 条摸底集。输出 schema 现在就写死——**标完 3000 条再想加字段就得重标**：

```json
{"case_id": "...",
 "key_points": [{"idx": 0, "covered": false, "evidence": ""}],
 "red_lines":  [{"idx": 0, "violated": true, "evidence": "每天吃一片"}],
 "verdict": "fail"}
```

`evidence` 必须是**原文子串**。这是刻意选的：可以程序化校验它没编造（不是子串即判格式违规），不会像自由理由那样退化成幻觉；同时汇报时能直接指着 evidence 说「小模型自己判出这里漏了禁忌」——纯布尔输出给不了这个，那 verifier 就和一个分类头没区别。

**二、摸底训练。** 基座 `Qwen/Qwen3.5-2B-Base`（Apache-2.0）。选 Base 不选指令版有三个实在理由：判官输出是固定 schema 不需要对话能力；中文医疗内容容易触发指令版的拒答与免责说教；Unsloth 官方警告 Qwen3.5-2B 指令版比同系列更容易陷入 thinking 死循环，批量跑时是不终止生成的灾难。走 Unsloth **bf16 LoRA**（2B 约 5GB）——**Unsloth 明确不建议对 Qwen3.5 做 4-bit QLoRA**，量化误差偏大。注意 Qwen3.5 是统一视觉-语言模型，需装 vision 依赖，视觉编码器对本任务是死重。

**三、验收脚本。** 算 verifier 与教师的一致性，两项分开算：保真（key_point covered/not）的一致率与 Cohen's κ；用药安全红线（violated/not）的**漏报率与误报率分列**。

**Blocked by:** 08（判官 prompt 需先在真跑中验证过）、03（verifier 训练池）

**Status:** ready-for-agent

- [ ] 500 条教师标注产出，全部符合 schema
- [ ] `evidence` 原文子串校验脚本可用，标注中不合规的能被检出并重标
- [ ] Qwen3.5-2B-Base bf16 LoRA 训练跑通，未使用 4-bit QLoRA
- [ ] 验收脚本输出：保真一致率、Cohen's κ、红线漏报率、红线误报率，四项分列
- [ ] **拿到「2B 离教师差多少」的数字**，据此给出结论：扩量到 3–5k，还是换 4B
- [ ] 低风险备选路径记录在案：`Qwen/Qwen3-1.7B-Base`（标准 dense、GGUF 零风险，代价是 32K 上下文与落后一代半）
