# 适老化医嘱解读：公开数据集与评测榜单调研

> 调研日期：2026-08-10
> 用途：为「面向银发群体的适老化医嘱解读与个性化健康服务智能推荐」选型——哪些公开数据集/榜单能用、哪些不能用、为什么。
> 关联票据：`.scratch/elder-trans-harness/issues/01-research-landing.md`

## 核实标记约定

本文档全部结论带来源。标记含义严格区分，**不要把 🔎 当成已核实**：

| 标记 | 含义 |
|---|---|
| ✅ | 调研时实际打开该页面并读取了内容 |
| 🔎 | 仅在搜索结果摘要中看到，**未打开正文确认** |
| ❌ | 尝试查找但未找到一手来源 |

第 6 节是完整的「未验证清单」。**引用本文档做决策前，先看该节确认你依赖的那条是不是 🔎。**

### 链接抽查记录（2026-08-10）

抽查 12 条引用链接，**全部返回 HTTP 200**：arXiv:2511.14439、CMB、CliMedBench、IMCS-21、biolaysumm.org、ACL Anthology 2026.bionlp-2、elkmovie/hsk30、blcuicall/mcts、Liyan06/MiniCheck、MedReadMe(EMNLP 2024)、Qwen/Qwen3.5-2B-Base、zispace/hanzi-chars。

另对两条**吃重的结论**做了内容级核对，而非仅测可达性：

1. **「BioLaySumm 2026 未举办」** —— 抓取 ACL Anthology `2026.bionlp-2` 卷的全部论文标题，**无任何 BioLaySumm 条目**；实际 shared task 为 CRF Filling、ClinicalSkillQA、PsyDefDetect、MedExACT。结论成立。
2. **「2022 课标常用字表有机读版」** —— `zispace/hanzi-chars` 页面内容确认含《义务教育语文课程》（2022年版）常用字表，及 3500 / 2500 / 1000 分档与《通用规范汉字表》《现代汉语常用字表》。结论成立（首次用 GitHub API 查询返回空是撞上限流，非内容缺失）。

---

## 0. 三条决定性结论

**一、没有任何现成的中文公开数据集是为"适老化医嘱解读"设计的。** 逐个核实 40+ 个资源后，四个理解域里只有「健康科普」和「用药说明」有较直接的公开对应任务（且都藏在 MedBench 的子任务里，不单独开放下载）。**「检验报告解读」与「医嘱转译」在公开中文数据集里是零**——两轮定向检索无果。**中文「医学术语→通俗表达」平行语料同样不存在。** 结论：本项目的评测集必须自建，这不是选择而是前提。

**二、MedBench v4 是当前唯一同时具备「与本项目场景直接对齐的子任务 + 活跃榜单」的中文医疗评测平台**，但其数据不公开下载（云端轮转题池、ground truth 永不下发），只能用于评测、不能用于训练。它的 LLM 赛道明确含 Science Popularization Generation（科普生成）、Drugs Consulting（用药咨询）、Patient Guidance（患者指导）、Prescription Review（处方点评）；Agent 赛道含 MedIntentID、MedRoleAdapt、MedLongConv，正对应「追问、确认、反馈」闭环。

**三、许可证风险集中在训练数据侧，且有几个常用集是明确禁用的。** cMedQA2、MedDG（GPL-3.0 + 明示 non-commercial）、C-Eval / CMMLU（CC BY-NC-SA 4.0）、ChatMed（CC BY-NC 4.0）、ASSET（CC BY-NC 4.0）、Newsela（NDA）都不能进商业产品。可商用的干净底座见第 5 节。

---

## 1. 中文医疗评测榜单与数据集

### 1.1 MedBench v4 —— 与本项目最相关

✅ 论文全文已读：[arXiv:2511.14439](https://arxiv.org/abs/2511.14439)。平台 https://medbench.opencompass.org.cn/ 为 SPA，WebFetch 未能打开（🔎 搜索结果确认 `/leaderboard`、`/dataset`、`/docs` 三个子页存在）。

| 项 | 内容 |
|---|---|
| 机构 | 上海人工智能实验室 + 复旦大学上海感染与生物安全研究院 + 上海市卫生发展研究中心 + Imperial College London |
| 规模 | **70 万+ 专家策展任务**，24 个一级 / 91 个二级临床专科，500+ 家成员机构临床医生多轮审核 |
| 三赛道 | LLM（36 数据集 / 43 专科 / 5 维度：MLU 语言理解、MLG 语言生成、MKQA 知识问答、CMR 复杂推理、**HSE 医疗安全与伦理**）；多模态（10 数据集）；**Agent（14 数据集 / 6 维度）** |
| 评测方法 | Qwen2.5-72B-Instruct 作 LLM-as-a-judge，每任务族专用 meta-prompt；rubric 拆多维度，**每维五分制带锚点描述**（从 "dangerously incorrect" 到 "fully correct and safely actionable"），可指定维度权重。1,000 名执业医师对约 20% 样本人工校准，关键任务 Cohen's κ > 0.82 |
| 提交 | 两种模式：API 模式（注册托管端点，平台加密推送随机题）/ 答案上传模式（下载随机化切分本地推理后上传）。**ground truth 始终留在服务端** |
| 题池 | 36 数据集组成轮转池，**约每季度重新生成子集**。分数不可跨周期直接比较，也无法背题刷分 |
| 数据可得 | ❌ 不公开下载，**不能用于训练** |

**与本项目直接对齐的子任务**（✅ 从论文 Figure 3 读到）：

- 医学知识问答：**Drugs Consulting（用药咨询）**、**Patient Guidance（患者指导）**、Healthcare consulting、Specialist Q&A
- 医学语言生成：**Science Popularization Generation（科普生成）** ← 所有中文公开评测中唯一明确的「科普生成」任务
- 医学语言理解：**Prescription Review（处方点评）**、Document Structure、Medical Record Quality Control
- Agent 赛道意图识别与角色适配：**MedIntentID、MedRoleAdapt**（能否识别用户目标并按角色调整表达风格）
- Agent 赛道长上下文：**MedLongConv、MedLongQA**（多轮交互中保持记忆）

**2025-11 论文基线**：Base LLM 均分 54.1/100（最好 Claude Sonnet 4.5 = 62.5），**安全伦理维度仅 18.4/100**；多模态均分 47.5（GPT-5 = 54.9）；**Agent 系统均分 79.8**（Claude Sonnet 4.5 agent 达 85.3 总分 / 88.9 安全分）。

MedBench 3.0 阶段（✅ [上海AI实验室新闻](https://www.shlab.org.cn/news/5444068)）：累计 4,204 次模型评测、近 80 家机构共建；新增 MedLitQA、**CriID（临床危急值识别）**、CMB-Clin-extended；新增眼科多模态、医学影像质控、**影像报告分析**、中医药四个专项赛道。

> ⚠️ 本仓库环境中已有 `medbench-agent-v5-judge` skill（11 任务 × 30 题），与论文 v4 的 Agent 赛道（14 数据集）任务名能对上但集合不完全一致。**以平台实际下发的任务清单为准，论文只反映设计意图。**

### 1.2 综合榜单与考试题库

| 名称 | 规模 | 许可证 | 榜单 | 核实 |
|---|---|---|---|---|
| [CBLUE](https://github.com/CBLUEbenchmark/CBLUE) 1.0 | 8 任务：CMeEE 15,000/5,000/3,000、CMeIE 14,339/3,585/4,482、CHIP-CDN 6,000/2,000/10,192、CHIP-CTC 22,962/7,682/10,000、CHIP-STS 16,000/4,000/10,000、KUAKE-QIC/QTR/QQR | **代码 Apache-2.0**；**数据本体许可未核实**（天池托管，需登录同意协议） | 天池长期开放（2026 现状未验证） | ✅ GitHub；❌ 天池页打不开 |
| CBLUE 2.0 / 3.0 | 2.0 增 CMedCausal、IMCS-V2、Text2DT、MedDG、CHIP-MDCFNPC；3.0 为 5 大类 18 子任务，含医疗 OCR 要素识别 | 同上 | 同上 | 🔎 |
| [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE) | 16 任务生成式改写，train 68,900 / dev 10,270 / testA 10,270 / testB 10,270，**94 个指令模板** | **README 未声明许可证** | CCKS-2023 评测，会后天池长期开放 | ✅ |
| [CMB](https://github.com/FreedomIntelligence/CMB) | CMB-Exam **280,839** 题（6 大类 28 子类）+ **CMB-Clin 74 例真实复杂病历**（每例含患者描述 + 多个开放式问答） | **Apache-2.0** | 有榜单，邮件提交人工审核；CMB-Clin 按流畅性/相关性/完整性/专业性四维打分 | ✅ |
| [CMExam](https://github.com/williamliujl/CMExam) | **68,119** 题（54,497/6,811/6,811），带五类题级标注：疾病组 27 值、临床科室 36 值、医学学科 7 值、能力领域 4 值、难度 5 级 | **Apache-2.0**（🔎 有二手来源称"仅学术"，与仓库 LICENSE 冲突，**以 LICENSE 为准，商用前建议邮件确认**） | 无 | ✅ |
| [CliMedBench](https://github.com/Optifine-TAT/CliMedBench) | **33,735** 题 / **14 场景**，数据来自三甲医院真实医疗报告 | **MIT** | 无活跃榜单 | ✅ |
| [LLMEval-Med](https://github.com/llmeval/LLMEval-Med) | 2,996 题（**公开 test 667 题**），来自真实 EHR + 专家设计场景 | 未明示，要求引用 | 无 | ✅ |
| [MLEC-QA](https://github.com/Judenpech/MLEC-QA) | **136,236** 题 / 5 子集（临床、口腔、公卫、中医、中西医结合） | **MIT** | 无 | ✅ |
| [CNMLEQA](https://github.com/zonghui0228/CNMLEQA) | 9,890 / 2,949 题，**按 5 个临床维度标注：疾病诊断、手术、用药、实验室检查、症状体征** | **未核实**（Zenodo 页需自行确认） | 无 | ✅ GitHub |
| [C-Eval](https://github.com/hkust-nlp/ceval) | 13,948 题 / 52 学科；**医学相关 4 科**：basic_medicine、clinical_medicine、physician、veterinary_medicine（兽医，建议剔除） | **CC BY-NC-SA 4.0 ❌ 禁商用** | test 标签不公开需提交 | ✅ |
| [CMMLU](https://github.com/haonan-li/CMMLU) | 67 科目；**具体医学科目名单 README 未列出，未核实** | **CC BY-NC-SA 4.0 ❌ 禁商用** | — | ✅ |
| [MedXpertQA](https://medxpertqa.github.io/) | 4,460 题 / 17 专科（**英文为主**） | **MIT** | 活跃，最新条目 2025-01 | ✅ |
| [HealthBench](https://github.com/openai/simple-evals) | 5,000 对话 / **48,562 rubric 条目** / 262 名医生 / 26 专科；每对话一份医生手写 rubric，平均 11–12 条行为标准 | simple-evals 仓库 **MIT** | 无公开榜单 | ✅ 仓库；**是否含中文对话未验证**（openai.com 返回 403） |
| MedS-Bench / MedS-Ins | MedS-Bench 覆盖 11 类临床任务（含**临床报告摘要、医学概念解释**）；MedS-Ins 58 语料 / 500 万实例 / 1.9 万指令 | 声明"完全可获取" | — | 🔎 |
| [MedRealMM](https://arxiv.org/abs/2607.09142) | **5,620 真实多模态案例 / 64 科室**，来自全国性互联网医院真实患者-医生交互，**每例带医生细化 rubric** | 待 HF 公开 | 无 | ✅ abs 页 |
| [CARE-Bench](https://arxiv.org/abs/2608.03731) | 500 case / 1,059 患者披露前缀（**英文**），4 标签逐轮"当前动作"任务 | **CC BY 4.0** | 无 | ✅ abs 页 |

**CliMedBench 14 场景**（✅ README）：In-hospital Diagnosis（4 阶段）、Basic Knowledge Test、False Info Test、False Treatment Test、Wrong Treatment Detection、**Discharge Summary（出院小结）**、**Medicine Consultation（用药咨询）**、Surgical Steps Organization、Clinical Pathway Reasoning、Keyword Extraction、Case Summary。
→ **Discharge Summary 与 Medicine Consultation 是公开可下载数据里离「医嘱转译」和「用药说明」最近的两个场景**，且 MIT 许可。

### 1.3 医患对话 / 追问闭环

| 名称 | 规模 | 许可证 | 与「追问、确认、反馈」的关系 | 核实 |
|---|---|---|---|---|
| [IMCS-21](https://github.com/lemuria-wchen/imcs21) | **4,116** 条细粒度标注问诊对话（10 种儿科疾病） | **README 未声明 license** ⚠️ | **最高**。五个任务里三个直接对应：**DAC 对话行为分类**（每句话意图，= 追问的显式标注）、**SLI 症状标签推断**（阳性/阴性/不确定，= 确认的显式标注）、**DDP 面向诊断的对话策略**（下一步该问什么）。中文里唯一同时标注这三层的数据集 | ✅ |
| [MedDG](https://github.com/lwgkzl/MedDG) | 17,000+ 对话 / 380,000+ utterance，12 种消化系统疾病，160 类医学实体 | **GPL-3.0 ❌** | 中高。topic prediction 本质是"下一步该聊什么" | ✅ |
| ReMeDi | 96,965 组对话，其中 **1,557 组带 intents/actions/slots/values 四层细粒度标注** | ❌ 未核实 | 高（标注最完整），但带标注的量小 | 🔎 |
| MidMed | 混合类型医疗咨询对话，动机就是"患者一开始说不清要什么，需在闲聊/知识问答/推荐/问诊间切换" | ❌ 未核实 | 高（对应老人表述模糊需主动澄清），但可得性未验证 | 🔎 |

**CARE-Bench 虽是英文，其方法论对本项目有直接价值**（✅ 论文摘要）：逐轮判断"当前该采取什么动作"的 4 标签评测协议，以及论文指出的关键错误模式——**「模型在该问澄清问题时过早给出就医建议」**。

### 1.4 四个理解域的具体资源情况

| 域 | 公开中文资源 |
|---|---|
| **检验报告解读** | **❌ 零。** 两轮定向检索（检验报告/体检报告/化验单/lab test interpretation Chinese）均无果。仅有：MedBench 内部 CriID 危急值识别（不可下载）、CBLUE 3.0 医疗 OCR 要素识别（识别版面要素非解读）、CNMLEQA 的"实验室检查"维度标注。**MedRealMM 是最值得跟进的**——真实互联网医院患者上传图片，若如期上 HF，将是最接近"老人拍化验单问 AI"的真实分布 |
| **医嘱转译** | **❌ 零。** 无任何公开的中文医嘱/处方/出院小结原文（PHI 最敏感）。可拼装的替代物：CMeEE/CMeIE/CHIP-CDN 做结构化前置、**Text2DT**（从临床诊疗文本抽取**诊疗决策树**，"条件→决策"的树结构 = 医嘱背后的逻辑，schema 有参考价值）、CliMedBench 的 Discharge Summary 场景、CMB-Clin 74 例可作改写种子 |
| **用药说明** | 西药说明书 **❌ 零**（无公开、可下载、带许可证的 NMPA 说明书结构化数据集）。中药有一个：**CHIP2020 中药说明书实体识别**，1,997 份（1,000/500/497），13 类实体，59,803 个标注实体，🔎 CC BY-SA 4.0。知识侧可用 Huatuo-26M 百科/图谱 QA（Apache-2.0） |
| **健康科普** | 语料充足，**但没有一个是适老化口吻的**——所有答案都是医生/客服语域，不是给 70 岁老人讲话的口吻。见下表 |

**健康科普可用语料**：

| 名称 | 规模 | 许可证 | 核实 |
|---|---|---|---|
| [Toyhom 中文医疗对话](https://github.com/Toyhom/Chinese-medical-dialogue-data) | **792,099 条**（内科 220,606 / 妇产 183,751 / 外科 115,991 / 儿科 101,602 / 男科 94,596 / 肿瘤 75,553） | **MIT** ✅ | ✅ |
| [Huatuo-26M](https://github.com/FreedomIntelligence/Huatuo-26M) | 2600 万+ QA 对，4 子集；Huatuo26M-Lite 经 ChatGPT 重写并带科室/疾病标签 | **Apache-2.0** | ✅ |
| [CMtMedQA](https://huggingface.co/datasets/Suprit/CMtMedQA) | **68,023** 条多轮真实医患对话 | **MIT** | ✅ |
| [DISC-Med-SFT](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) | 465,000 条 | Apache-2.0 ⚠️ **但衍生自 MedDialog/cMedQA，需上游穿透审查** | ✅ |
| [cMedQA2](https://github.com/zhangsheng93/cMedQA2) | 108,000 问 / 203,569 答 | **GPL-3.0，README 明示 non-commercial ❌** | ✅ |
| [ChatMed](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | 110,113 条（GPT-3.5 生成答案） | **CC BY-NC 4.0 ❌** | 🔎 |
| MedDialog-CN | 中文 340 万对话 / 1,130 万 utterance / 172 专科（论文数值；另有"110 万对话"的版本说法，**版本不一致**） | ❌ 未核实 | 🔎 |

---

## 2. 医学通俗化（plain-language）数据集与 shared task

**关键背景**：本项目的 KPI「转译后文本中符合小学六年级可读水平的词汇比例 ≥ 90%」**不是自创指标**。它是 **Dale–Chall 可读性公式**核心自变量（"不在 3000 词熟词表内的难词百分比"）的中文迁移版，而 **DCRS（Dale–Chall）正是 BioLaySumm 2023–2025 shared task 的官方可读性指标之一**。见 [Dale–Chall](https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula)、[BioLaySumm](https://biolaysumm.org/)。

另：**AMA 与 NIH 均建议患者教育材料不超过六年级阅读水平**（[PMC 综述](https://pmc.ncbi.nlm.nih.gov/articles/PMC11262515/)）——这直接回答"为什么是六年级"。

### 2.1 英文 shared task（方法论与 rubric 来源）

| 任务 | 状态 | 关键事实 |
|---|---|---|
| **[PLABA](https://bionlp.nlm.nih.gov/plaba2024/)** | 2023、2024 两届，已收官 | 主办是 **TREC/NIST**（不是 TAC），NLM/NIH 承办。公开训练数据 **750 篇摘要 / 7,643 个句对**，**CC BY 4.0**（[Sci Data 论文](https://www.nature.com/articles/s41597-022-01920-3)），数据在 [OSF](https://osf.io/rnpmf/)。**官方人工 rubric 四维：simplicity / accuracy / completeness / brevity** |
| **[BioLaySumm](https://biolaysumm.org/)** | 2023 / 2024 / 2025 三届，**2026 未举办** | ✅ 已核实 [BioNLP 2026 shared tasks 论文集](https://aclanthology.org/volumes/2026.bionlp-2/)，五个任务为 CRF Filling、ClinicalSkillQA、PsyDefDetect、MedExACT、MedGenVidQA，**无 BioLaySumm**。2025 届数据：PLOS 24,773 / eLife 4,346 训练样本 |
| **[CLEF 2026 SimpleText](http://simpletext-project.com/2026/tasks)** | **在办** | Task 1 句级/篇章级科技文本简化（Cochrane-auto 生物医学语料），指标 **SARI / BLEU / LENS / BERTScore + 人工评价**；Task 2 Controlled Creativity（幻觉检测/失真分类/有据生成）。**任务页未提及中文** |
| [TSAR-2022](https://taln.upf.edu/pages/tsar2022-st/) | 已结束 | 多语词汇简化，英/葡/西，**无中文** |

**BioLaySumm 官方三维评测框架**（值得整体照抄）：

- **Relevance**：ROUGE-1/2/L、BLEU、METEOR、BERTScore
- **Readability**：**FKGL、DCRS（Dale–Chall）、CLI（Coleman-Liau）、LENS**
- **Factuality**：**AlignScore、SummaC**
- 打分：各指标先 min-max 归一化，再在维度内平均

### 2.2 平行语料

| 数据集 | 规模 | 许可证 | 含中文？ | 核实 |
|---|---|---|---|---|
| **[MultiMSD](https://aclanthology.org/2025.findings-acl.481/)** ⭐ | **9 语种（含中文），每语种 1 万句对**，共 9 万对；由在线医学参考资料的**专业版 vs 消费者版自动句对齐**构建（MSD Manuals 模式）。ACL Findings 2025 | 论文 CC BY 4.0；**数据本体 license 未给出，Anthology 页无仓库链接，需联系作者** | ✅ **是** | ✅ |
| [Cochrane PLS (GEM)](https://huggingface.co/datasets/GEM/cochrane-simplification) | train 3,568 / val 411 / test 480 | **CC BY 4.0** | 否（英文） | ✅ |
| [Cochrane-auto](https://aclanthology.org/2024.tsar-1.5/) | 9,160 句 + 666 篇摘要；被 CLEF 2025/2026 采用 | 论文称 freely available（**具体 license 未验证**） | 否 | ✅ 论文页 |
| [MultiCochrane](https://arxiv.org/pdf/2305.12532) | 7,755 对 abstract-PLS | 未验证 | **否**——只有 En/Es/Fr/Farsi | ✅ |
| Cochrane 官方简体中文 PLS | Cochrane Library 已翻译 20 语种含简体中文 | — | **是** | 🔎（[Cochrane HK](https://hongkong.cochrane.org/publications/cochrane-plain-language-summary-simplified-chinese)；Cochrane Library 翻译页 403 未抓取） |
| [Med-EASi](https://arxiv.org/abs/2302.09155) | 1,979 句对，**细粒度标注 4 种变换**：elaboration / replacement / deletion / insertion | 未核实 | 否 | ✅ |
| [MCTS](https://github.com/blcuicall/mcts) | **723 复杂句 × 5 条人工简化参考**，另附 **691,474 条伪平行训练句对**。中文简化任务最大评测集 | 未核实 | **是**（但非医学域） | ✅ |
| CSS | 383 句 × 2 参考 | 未核实 | 是（非医学） | 🔎 |
| **中文「医学术语→通俗表达」平行语料** | — | — | **❌ 未找到任何公开数据集** | 定向检索无果 |

### 2.3 2024–2026 新工作（对指标设计有直接影响）

| 名称 | 关键结论 |
|---|---|
| **[MedReadMe](https://aclanthology.org/2024.emnlp-main.958/)**（EMNLP 2024） | 4,520 句人工可读性打分 + **细粒度"复杂片段"跨度标注（7 类，含医学术语 jargon）**。**核心发现：在传统可读性公式中只加入一个特征——jargon 片段数量——就能显著提升与人工判断的相关性与稳定性。** → 本项目应把 jargon 密度作为第二指标 |
| **[Lessons from TREC PLABA](https://arxiv.org/abs/2507.14096)** | 12 支队伍；最好的模型**在事实准确与完整性上已接近人类，但简洁性与简短性没有**；且 **"自动的、基于参考的指标普遍与人工判断相关性差"** |
| **[Evaluating the Evaluators](https://arxiv.org/abs/2508.19221)** | 评了 8 个可读性指标与人工判断的相关性：**传统公式相关性差；LM 作判官最好也仅 Pearson 0.56** |
| **[PlainMedScale](https://arxiv.org/abs/2608.01158)**（KONVENS 2026） | 德/英四级可理解度医学语料，CC BY-SA 4.0。发现**现有可读性指标无法跨全部难度层级泛化**，且 **LLM 即便被要求用通俗语言也会部分保留输入复杂度** |
| [RephQA](https://arxiv.org/abs/2509.16360) | 533 条专家审核公共卫生 QA；用 FKGL + professional score 两个可读性指标；**评测 25 个 LLM，多数达不到可读性标准** |

> **这三条负面结论恰恰支持本项目的指标选型**：词表驱动的达标率**可解释、可审计、可逐词追溯**（每个不达标词都能列出来），这是 FKGL / LENS / LM 判官做不到的。这一点应写进指标规格文档作为方法论优势。

---

## 3. 中文可读性词表与工具

### 3.1 官方分级标准（本项目 KPI 的裁决基准）

| 标准 | 内容 | 机读版本 | 核实 |
|---|---|---|---|
| **《义务教育语文课程标准》（2022 版）附录《义务教育语文课程常用字表》** | **3500 字 = 字表一 2500 + 字表二 1000**；**字表一可作为第三学段（5–6 年级）识字写字能力评价依据** | ✅ [zispace/hanzi-chars](https://github.com/zispace/hanzi-chars)（含「义务教育语文课程（2022）：基础字表 300 + 常用字表 3500」，UTF-8 纯文本）⚠️ **该仓库无明确开源许可声明** | ✅ 仓库；🔎 字表一用于第三学段评价的表述（知乎页 403 未抓正文） |
| **《通用规范汉字表》**（国务院 2013-06-05 颁布） | 共 **8,105 字**：**一级 3,500（常用字集）**、二级 3,000、三级 1,605 | ✅ [jaywcjlove/...](https://github.com/jaywcjlove/table-of-general-standard-chinese-characters)（JSON + 拼音）、[shengdoushi/...](https://github.com/shengdoushi/common-standard-chinese-characters-table)、[cdtym/...](https://github.com/cdtym/digital-table-of-general-standard-chinese-characters) | ✅ |
| **《国际中文教育中文水平等级标准》GF 0025-2021** | **11,092 词** + 1,110 音节 + 3,000 汉字 + 572 语法点，**三等九级**。国家语委 2021 标准 | ✅ [elkmovie/hsk30](https://github.com/elkmovie/hsk30)（**MIT**，从教育部官方 PDF 提取）、[drkameleon/complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary)（**MIT**，JSON，含频率/词性/繁简/注音）。官方 PDF：[moe.gov.cn](http://www.moe.gov.cn/jyb_sjzl/ziliao/A19/202111/W020211118507389477190.pdf) | ✅ |
| **《义务教育常用词表（草案）》** ⭐ | 教育部语信司组织、**厦门大学苏新春**团队研制、商务印书馆 2019-05。音序表 **15,114 词**（含读音、**词级**、词性），义类表 17,092 词。**四级分级：一级=1–2 年级、二级=3–4 年级、三级=5–6 年级、四级=初中**。定位是"填补空白，中国词汇教学首次有了量化标准" | **❌ 未找到公开机读版/在线查询库**（GitHub / 商务印书馆 / 教育部多轮检索均无，仅纸质书） | ✅ 多个一手报道 |
| 《汉语水平词汇与汉字等级大纲》（旧 HSK） | **8,822 词**：甲 1,033 / 乙 2,018 / 丙 2,202 / 丁 3,569 | ⚠️ [zispace/hanzi-words](https://github.com/zispace/hanzi-words)（TSV，**无明确 license**） | ✅ |
| 《现代汉语常用词表（草案）》 | **56,008 词**，基于 2.5 亿字语料 | **❌ 只按频次排序，不分年级**。[liangqi/chinese-frequency-word-list](https://github.com/liangqi/chinese-frequency-word-list) | 🔎（教育部页 WebFetch 失败） |

> **《义务教育常用词表（草案）》是国内唯一官方「词→年级」标准，也是本项目 KPI 的金标准**，但无公开机读版。获取路径：采购商务印书馆纸书内部数字化（**词表本体不可对外分发**，只公布达标率数值与方法说明），或联系厦大苏新春团队 / 教育部语信司询问授权。
>
> **好消息是字级层不需要妥协**：2022 课标字表一 2500 字被官方直接指定为第三学段评价依据，词级层可用 MIT 许可的 GF 0025-2021。**因此本项目不必使用词频代理表，也不必依赖小学课文校准**——校准是代理表才需要的补丁。

### 3.2 可读性工具（结论：没有一个能直接回答"某词属于几年级"）

| 工具 | 说明 | 可用性 |
|---|---|---|
| **CRIE 3.0**（宋曜廷等，台师大） | 82 个多层次语言特征 + 机器学习分级，支持简繁 | 官网 [chinesereadability.net](http://www.chinesereadability.net/CRIE/) **WebFetch 失败，未验证是否仍可下载** |
| [AlphaReadabilityChinese](https://github.com/leileibama/AlphaReadabilityChinese) | GUI，计算词汇/句法/语义三层指标，批量 txt→csv | ✅ "free to use" 但**无正式 license**；有引用要求 |
| [python-readability-cn](https://github.com/chenryn/python-readability-cn) | 实现多篇论文公式，默认词表用复旦语料，依赖 LTP | ✅ **无明确 license** |
| [cntext](https://github.com/hiDaDeng/cntext) | **MIT**，`ct.readability(text, lang='chinese')`，实现 Gunning Fog / SMOG / Coleman-Liau / ARI / RIX | ✅ ⚠️ **"复杂词"= 字数 ≥ 阈值的词，不内置任何官方字/词表，没有年级语义，不能支撑本项目 KPI** |
| [textstat](https://github.com/textstat/textstat) | 官方支持 22 语种 | ✅ **不支持中文** |

**结论：没有任何现成开源库能直接判定"某个词属于几年级"。** 必须自建判定逻辑，词表用第 3.1 节的官方来源。

---

## 4. verifier / 判官 / 事实一致性

> ⚠️ **本节核实质量显著低于前三节。** 调研时 `huggingface.co` / `arxiv.org` / `github.com` 的抓取大量被网络策略拦截，多数条目停留在搜索摘要层（🔎）。**采用任何一条前请自行核实许可证。**

### 4.1 奖励模型 / 判官榜单

| 名称 | 规模 | 许可证 | 相关性 | 核实 |
|---|---|---|---|---|
| [RewardBench](https://github.com/allenai/reward-bench) | 🔎 约 2,985 条 | 🔎 数据 ODC-BY / 代码 Apache-2.0 | **中** — 域为 chat/code/math/safety，**✅ 已确认无医疗子集**，只能作范式参考 | ✅ README |
| RewardBench 2 | 🔎 1,865 条 | ❌ 未核实 | **高** — 六域含 **Factuality**（检测事实错误/幻觉）与 Precise Instruction Following，对应"忠实覆盖要点"判别；仍无医疗子集 | 🔎（[arXiv:2506.01937](https://arxiv.org/html/2506.01937v2)） |
| [JudgeBench](https://github.com/ScalerLab/JudgeBench) | 🔎 gpt split 350 对 + claude split 270 对 | 🔎 Apache-2.0（仅第三方汇总，非一手） | **中** — 判官"客观正确性"成对判别，格式贴近二分类 verifier，但无医疗 | 🔎 |
| [RM-Bench](https://github.com/THU-KEG/RM-Bench) | 🔎 1.33k 测试实例 | ❌ 未核实 | **中** — 专测"对细微内容差异敏感 / 抗风格偏见"，正是转译忠实性判别最易被文风带偏的失效模式，**适合当判别力体检集** | 🔎 |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) / [HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | 🔎 HelpSteer2 约 10k 响应对；HelpSteer3-Preference > 40,000 样本 | 🔎 **两者均 CC-BY-4.0**（多处摘要一致，未一手确认） | **中** — 许可最宽松可商用的人工偏好数据，适合通用能力预热；非医疗、非红线判别 | 🔎 |
| [CompassJudger](https://github.com/open-compass/CompassJudger) | 🔎 CompassJudger-1 发布 4 个尺寸 | 🔎 CompassJudger-**2**-32B 为 Apache-2.0；**CompassJudger-1 许可证与是否有 ≤2B 档位均未核实** | **中** — 唯一成体系的中文判官模型族，可作教师或蒸馏起点 | 🔎 |

### 4.2 事实一致性 / 幻觉检测（小模型权重 —— 本项目最相关）

| 名称 | 规模 | 许可证 | 相关性 | 核实 |
|---|---|---|---|---|
| **[MiniCheck](https://github.com/Liyan06/MiniCheck)** ⭐ | ✅ 变体含 MiniCheck-RoBERTa-Large、MiniCheck-DeBERTa-v3-Large、**MiniCheck-Flan-T5-Large 770M**、Bespoke-MiniCheck-7B | ✅ **Bespoke-MiniCheck-7B 需联系 company@bespokelabs.ai 取得商用授权**（非自由商用）；🔎 Flan-T5-Large 变体称 MIT，**未一手确认** | **最高** — 770M 的「文档 → 声明是否被支持」二分类器，尺寸与任务形态几乎就是本项目要的窄判别 verifier，可作 warm start | ✅ README |
| MiniCheck 延迟参考 | ✅ 单张 A6000 + Automatic Prefix Caching，**29K 测试样本 30 分钟**（不开为 55 分钟）——这是 7B 版数据 | — | 可作 2B 目标的延迟上界参考 | ✅ |
| [AlignScore](https://github.com/yuh-zha/AlignScore) | 🔎 base / large 两档，RoBERTa backbone，large 约 **355M** | ❌ 未核实（LICENSE 页被拦截） | **高** — 355M 统一 alignment 打分函数，比 MiniCheck 更小，可作轻量基线/对照臂。**许可证必须自己去 repo 确认** | 🔎 |
| [MedHallu](https://github.com/MedHallu/MedHallu) | 🔎 **10,000** 医学 QA 对，源自 PubMedQA | 🔎 MIT（未一手确认） | **高** — 任务形态就是"这条医学回答是否幻觉"的二分类，与本项目同构。🔎 基线：最佳模型在 hard 类幻觉上 **F1 仅 0.625**；加 "not sure" 类别可相对提升至多 38%。⚠️ **英文、PubMed 文献域，非中文用药医嘱场景** | 🔎 |
| HaluEval / HalluQA / [UHGEval](https://github.com/IAAR-Shanghai/UHGEval) | 🔎 HaluEval 35,000；HalluQA **450** 条中文对抗题；UHGEval 全量 5,141 / 精简 1,000 | ❌ 三者均未核实 | **中** — 少数中文幻觉基准，可做域外泛化测试；规模小且非医疗，不足以当训练集 | 🔎 |

### 4.3 中文医疗判别可复用资产

| 名称 | 规模 | 许可证 | 相关性 | 核实 |
|---|---|---|---|---|
| **MedFact** ⭐ **（注意有两个同名不同工作，引用必须带 arXiv 编号）** | ① [arXiv:2509.12440](https://arxiv.org/abs/2509.12440)「Benchmarking Fact-Checking on Chinese Medical Texts」，🔎 **2,116** 条专家标注，13 科室 / 8 种错误类型 / 4 种文体 / 5 个难度；② [arXiv:2509.17436](https://arxiv.org/abs/2509.17436)「Evidence-based Medical Fact-checking of **LLM Responses**」（EMNLP 2025），规模未核实 | ❌ 两者均未核实 | **最高** — ①做 veracity 分类 + 错误定位；②直接针对核查 LLM 生成的医疗回答。**🔎 论文报告的关键失效模式：模型能判断有错但难以精确定位，且存在 "over-criticism"（把正确信息误判为错误），多智能体协作反而加剧** | 🔎 |
| [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) | 🔎 4 个 subset：en 19.7k / en_mix 24.9k / **zh 20.2k** / zh_mix 25.4k | 🔎 apache-2.0（摘要级证据） | **中** — 中文医疗 CoT SFT 数据，但是"生成"数据不是"判别"数据，需自行用教师造正/负对。**字段结构未核实** | 🔎 |
| [PRM800K](https://github.com/openai/prm800k) / [Math-Shepherd](https://arxiv.org/pdf/2312.08935) | 🔎 规模均未一手核实 | 🔎 PRM800K MIT；Math-Shepherd 无明确许可证 | **中** — 作为"教师打标 → 训小判别器"范式参考成立。**Math-Shepherd 用 Monte Carlo 估计自动产生步级标签且无需人工，正是本项目"大模型教师自动打标"路线的直接先例** | 🔎 |

### 4.4 本项目的三条落地判断

1. **warm start 优先级**：AlignScore(≈355M) → MiniCheck-Flan-T5-Large(770M) → 自训 2B。前两者任务形态（前提文档 + 声明 → 支持/不支持）与"转译是否忠实覆盖要点"完全同构。**但两者许可证都必须先自行核实。**
2. **中文医疗判别的唯一近似对标是 MedFact 两篇，不是 MedHallu**（后者英文 PubMed 域）。其报告的 **over-criticism 现象应当在红线 verifier 里预先设计对照臂**——否则"宁可错杀"的判别器会在离线指标上好看、上线后拦截率爆炸。
3. **RewardBench 2 / RM-Bench 更适合当噪声底与判别力体检，而非训练目标**：前者的 Factuality 域、后者的"细微差异 vs 风格偏见"设计，正好可以验证小判别器是不是在**学文风而不是学事实**。

---

## 5. 许可证红线清单

### ❌ 禁用（明确禁止或限制商用）

| 资源 | 许可证 |
|---|---|
| cMedQA2 | GPL-3.0，README 明示 "non-commercial research only" |
| MedDG | GPL-3.0 |
| C-Eval | CC BY-NC-SA 4.0 |
| CMMLU | CC BY-NC-SA 4.0 |
| ChatMed_Consult_Dataset | CC BY-NC 4.0 |
| ASSET | CC BY-NC 4.0 |
| Newsela | 需签 NDA + 限制性协议，禁止公开分享 split |
| **Qwen2.5-3B** | **qwen-research**（非 Apache-2.0，限制商用）——✅ 已在 HF 卡逐个确认 |
| Bespoke-MiniCheck-7B | ✅ 需联系 company@bespokelabs.ai 取得商用授权 |

### ⚠️ 需上游穿透审查 / 许可不明

| 资源 | 情况 |
|---|---|
| **DISC-Med-SFT** | HF 卡标 Apache-2.0，**但衍生自 MedDialog / cMedQA 等上游，上游许可更严**。商用前必须穿透审查 |
| CBLUE / PromptCBLUE | 代码 Apache-2.0；**数据本体许可未核实**（天池托管需登录同意协议）。PromptCBLUE README 未声明 license |
| IMCS-21 | **README 未声明 license** |
| MedDialog-CN、Yidu-S4K、CNMLEQA、webMedQA | 许可证均未核实 |
| MultiMSD | 论文 CC BY 4.0，**数据本体 license 未给出，需联系作者** |
| zispace/hanzi-chars、zispace/hanzi-words、AlphaReadabilityChinese、python-readability-cn | **仓库均未声明明确 license**，商用前需联系作者 |
| AlignScore、RM-Bench、JudgeBench、HaluEval / HalluQA / UHGEval、MedHallu、MedFact | 许可证均未核实（第 4 节网络受限） |

### ✅ 可商用（许可证已确认）

| 资源 | 许可证 |
|---|---|
| Huatuo-26M（2600 万 QA） | Apache-2.0 |
| CMB（Exam 280,839 + Clin 74） | Apache-2.0 |
| CMExam（68,119） | Apache-2.0 |
| MLEC-QA（136,236） | MIT |
| CliMedBench（33,735 / 14 场景） | MIT |
| Toyhom 中文医疗对话（792,099） | MIT |
| CMtMedQA（68,023） | MIT |
| MedXpertQA、HealthBench(simple-evals) | MIT |
| PLABA（750 摘要 / 7,643 句对） | CC BY 4.0 |
| Cochrane PLS (GEM) | CC BY 4.0 |
| CARE-Bench | CC BY 4.0 |
| elkmovie/hsk30、drkameleon/complete-hsk-vocabulary | MIT |
| cntext | MIT |
| **Qwen3.5 全系（0.8B/2B/4B/9B + MoE）、Qwen3 全系、Qwen2.5-1.5B** | **Apache-2.0** |

---

## 6. 未验证清单（诚实披露）

以下条目**未取得一手来源确认**，引用前必须自行核实。

**完全未找到一手来源：**

- 中文检验报告 / 化验单解读数据集（两轮定向检索）
- 公开的中文医嘱 / 处方 / 出院小结原文数据集
- 中文医学文本通俗化 / lay summary 平行语料
- NMPA 西药说明书公开结构化数据集
- DrugBank 中文版
- MMCU（本次检索完全无返回，**存在性未验证**）
- CELLS 数据集（仅在综述性结果中被提及）

**关键事实未验证：**

- CBLUE 天池数据使用协议 / 商用条款
- CBLUE、PromptCBLUE、CMB 榜单 2026 年当前是否仍可提交（相关页面 WebFetch 均失败）
- CHIP 历年赛事页（2021–2025）是否仍可访问（`cips-chip.org.cn` 全部抓取失败；但搜索结果显示主站已更新到 "CHIP 2026"，域名本身存活）
- HealthBench 是否含中文对话（openai.com 返回 403）
- CMMLU 的具体医学科目名单（README 未列出）
- MedQA-MCMLE 的确切规模与许可证
- Yidu-S4K、webMedQA、MidMed、ReMeDi 的许可与获取地址
- MedS-Bench / MedS-Ins、TCM-Eval、MTCMB、Medmarks、MedES 仅搜索摘要
- 义务教育常用词表是否存在任何公开机读版本（多轮检索均无，属**负面证据非绝对结论**）
- CRIE 3.0 官网当前是否可下载
- 教育部《现代汉语常用词表（草案）》页面附件 PDF 是否仍可下载
- 「字表一 2500 字用于第三学段评价」的课标原文表述（转载页 403，仅据搜索摘要）
- Cochrane Library 20 语种 / 6.1 万条 PLS 翻译的数字（页面 403）
- CLEF SimpleText 的 Cochrane 语料是否含中文（2026 任务页未提中文，**存疑**）
- TurkCorpus 单独许可；WikiLarge / Wiki-auto / Simple Wikipedia 各仓库 license 文件
- FactCC / QAFactEval 一手来源（本次未检索）
- **第 4 节几乎全部条目**（网络策略拦截 huggingface / arxiv / github），详见 4.x 表格的核实列

---

## 7. 对本项目的结论

1. **评测集必须自建**，且优先级按公开资源匮乏程度排序：**检验报告解读 > 医嘱转译 > 用药说明 > 健康科普**。
2. **「医嘱转译」在任何公开来源与本地数据中都没有真实原文**。文档中不得默认其为真实医嘱，必要时表述为「治疗方案转译」。
3. **可商用的干净训练底座**：Toyhom（MIT，79 万）+ CMtMedQA（MIT，6.8 万多轮）+ Huatuo-26M / Huatuo26M-Lite（Apache-2.0）。
4. **rubric 设计可直接组合三家**：MedBench v4 的五分制带锚点 + 维度加权；LLMEval-Med 的 Core/Secondary requirements checklist；HealthBench 的每样本 11–12 条医生手写行为标准。人工评价维度用 PLABA 官方四维（simplicity / accuracy / completeness / brevity）。
5. **追问闭环的标注 schema 直接抄 IMCS-21**：DAC（意图）+ SLI（阳性/阴性/不确定）+ DDP（下一步策略）三层，中文里最成熟；再叠 CARE-Bench 的"逐轮当前动作"标签防止过早给结论。
6. **可读性指标不需要词频代理表**：字级用 2022 课标字表一 2500（官方指定为第三学段评价依据），词级用 GF 0025-2021（MIT 机读版）。金标准《义务教育常用词表（草案）》无公开机读版，列为长期获取项。
7. **verifier 的 warm start 候选是 AlignScore(355M) 与 MiniCheck-Flan-T5-Large(770M)**，任务形态同构；但许可证必须先自行核实。中文医疗判别的对标是 MedFact 两篇，其 **over-criticism** 现象需在设计阶段就安排对照臂。
