# 01 — 调研落盘：数据集与榜单文档

**What to build:** 把本轮已完成的调研写成一份可交付的文档，让任何人（包括评审）能凭它判断「哪些公开数据集/榜单能用、哪些不能用、为什么」，不必重做一遍检索。

内容分四块：中文医疗评测榜单（MedBench v4 是唯一同时覆盖四个理解域 + Agent 追问闭环的中文榜单，但数据不公开下载、季度轮转题池，只能评测不能训练）；医学通俗化与可读性数据集；中文可读性词表与工具；verifier / 奖励模型相关榜单。

两份清单是这份文档的核心价值，必须单独成节：

- **许可证红线**：cMedQA2、MedDG（GPL-3.0 且明示 non-commercial）、C-Eval / CMMLU / ChatMed（CC BY-NC）禁用；DISC-Med-SFT 虽标 Apache-2.0 但衍生自 MedDialog/cMedQA，需上游穿透审查；`Qwen2.5-3B` 是 `qwen-research` 许可证限制商用。可商用干净底座：Huatuo-26M、CMB（Apache-2.0）、CliMedBench、Toyhom、CMtMedQA（MIT）。
- **未验证清单**：诚实标注哪些结论只来自搜索摘要、哪些页面打不开。宁可写「未核实」也不要含糊带过。

**Blocked by:** None — can start immediately.

**Status:** done — 交付于 `docs/DATASETS_AND_BENCHMARKS.md`

- [x] 四个内容块齐备，每条结论带一手来源 URL（官网 / GitHub / arXiv / HF dataset card），非二手博客
- [x] 许可证红线清单成节，明确区分「禁用」「需穿透审查」「可商用」三档（第 5 节）
- [x] 未验证清单成节，逐条说明为何未核实（第 6 节）
- [x] 随机抽 10 条链接可打开，且页面内容与文档结论一致 —— 抽查 12 条全部 HTTP 200，另对两条吃重结论做内容级核对（BioLaySumm 2026 未举办、2022 课标字表机读版存在），记录见文档「链接抽查记录」节
- [x] 明确写出结论：检验报告解读与医嘱转译两个域**没有公开中文数据集**，必须自建（第 0 节结论一、第 1.4 节、第 7 节）

**交付备注：** 第 4 节（verifier / 判官 / 事实一致性）的核实质量显著低于前三节——调研时 huggingface / arxiv / github 的抓取大量被网络策略拦截，多数条目停留在搜索摘要层。该节已在节首显著标注，且全部条目在第 6 节未验证清单中登记。**ticket 09 采用 AlignScore / MiniCheck 作 warm start 前，必须自行核实其许可证。**
