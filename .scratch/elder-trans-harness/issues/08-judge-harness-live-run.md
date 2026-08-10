# 08 — 判官 harness 与第一次真跑

**What to build:** 让 270 题真的跑一遍，产出两个 KPI 的第一个真实数字。在这之前，所有关于「怎么从 88% 提到 90%」的讨论都是空谈。

被测系统：`deepseek-v4-flash`（已核实存在；注意 `deepseek-chat` 已不在该 key 的模型列表里，别照抄旧项目配置）。

判官：**Claude Sonnet 5，走 session subagent**。这满足医疗高风险域的异厂解耦要求（判官厂商 ≠ 被测厂商），也省掉再申请一个 API key。

吞吐：**270 题拆约 20 个 subagent 并发**，每个批判 13–15 题，返回结构化 JSON 数组。串行不可接受。

判官 prompt **固化成文件**，不许在 subagent 里临时改——rubric 骨架取自 `/Users/chenhao/ClaudeCode/eqbench-run1/SOURCE/rubric_core3.yaml`（同理心 / 校准顺从 / 专业信任三维，0–20 分锚点，「专业信任」已含 `elder_communication` 子项）。

**判官不可复现，只能可审计——这个降级必须落实成证据留存。** session 内的 Claude 没有可钉版本的 API model id，换个 session 重跑就是另一批数。所以每次判官调用都要逐条落盘 `{case_id, 判官输入全文, 判官输出全文, 时间戳, 判官标识}`，run_id 写进 kpi.yaml。对外表述从「可复现」改为「可审计」——数字重跑会漂，但每一条判定都能被人翻出来核。可读性 KPI 是确定性计算，不受此影响。

产出：strict / glossed / strict_2500 三口径达标率（词级微平均，主报 strict）+ Wilson 95% CI + 场景宏平均 + **hard 子集分** + 忠实性通过率 + 逐题审计。

**不设长度护栏**——该能力独立调用，扩长是允许的方向。但字数与转译/原文字数比要一并输出，供观察。

**Blocked by:** 07（测试集冻结）、02（可读性打分器）

**Status:** ready-for-agent

- [ ] 判官 prompt 固化成文件，rubric 骨架继承自 EQbench 三维
- [ ] 270 题分片并发跑通，单次全量跑完在可接受时长内
- [ ] 每条判官调用留痕落盘，run_id 写进 kpi.yaml
- [ ] 输出三口径达标率 + Wilson CI + 场景宏平均 + hard 子集分 + 忠实性通过率 + 字数统计
- [ ] **kpi.yaml 的 `actual` 从 null 变成真实数字**
- [ ] caveat 写明：忠实性判官可审计但不可复现，重跑会漂；可读性为确定性计算不受影响
- [ ] 支持 `--only` 子集重跑，便于抽查
