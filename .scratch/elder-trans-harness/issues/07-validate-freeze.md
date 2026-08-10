# 07 — 测试集校验与冻结

**What to build:** 一个校验器，把 270 题从「一堆文件」变成「一个可以被引用的、有版本的评测集」。跑通它，冻结版本。

校验内容：字段完整性（每题必备 persona/query/source_text/key_points/negative_rubric/difficulty/provenance）、场景数 = 10、每类题数符合配比、难度分布落在 易40/中45/难15、`record_id` 全部落在测试集池内且与 verifier 训练池零交集、provenance 可回溯。

同时切出 **`hard` 子集**（难度=难的那 15%）作为固定的判别力观测口。此后所有跑数**必须同时报全集分和 hard 子集分**——只报全集分的 harness，三个月后没人知道自己在优化什么。

冻结：把测试集内容 hash 写进 kpi.yaml 的 frozen 区，连同题数、场景数、切分种子。此后任何改动都要重新冻结，改动可被发现。

**Blocked by:** 06（撰写完成）

**Status:** ready-for-agent

- [ ] 校验器覆盖上述全部检查项，任一不通过即报错并指出具体题目
- [ ] 270 题全部通过校验
- [ ] `hard` 子集切出并固化，规模约 40 题
- [ ] 内容 hash、题数、场景数、切分种子写进 kpi.yaml frozen 区
- [ ] 改动任意一题后重跑校验，hash 变化能被检出
