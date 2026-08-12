# LITERATURE — 动手前的文献核对台账（铁律 15）

> 契约：动到 训练目标 / 损失 / 数据构造策略 / 模型结构 / 解码策略 / 评测方法 ⇒ 先有一条 `L<n>`，才许改代码。
> 时间盒 30 分钟 / ≤5 次检索。**文件里没有的检索视为没做过**；「未检索到」是合格产出，空着不是。
> 检索**没有否决权**：找到 refute 的论文 ≠ 不做，而是必须写清「本项目差异」并变成判读规则的一条。

<!-- L1–L3 检索由三个并行 research agent 执行（2026-08-13），主线程对全部关键 arXiv 编号
     做过 WebFetch 抽查核实；正文数字标注了来源等级。数据背景：ticket 11 负结果
     （66 条红线正例训不出 ≤5% 漏报的 verifier，2B/4B 打平 → 非容量问题）。 -->

## L1 — 定向合成红线正例扩充训练集 ⟶ ticket 12 候选方向 A

- **学名**: **synthetic error injection**（verifier/判别器语境）；本项目设定最精确的谱系是
  **weakly-supervised factual consistency classification via rule-based synthetic transformations**
  （FactCC 系）；负例侧亦称 hard negative synthesis。
- **检索式**（5 次）:
  1. `synthetic error injection training faithfulness verifier classifier factual consistency synthetic negatives`（钉学名）
  2. `FactCC synthetic transformations factual consistency checking arXiv Kryscinski entity swap weakly-supervised`
  3. `synthetic data factual consistency classifier fails does not transfer real errors Goyal Durrett fine-grained factuality Falsesum` **【阴性专搜】**
  4. `"hard negatives" LLM-generated distributional artifacts hurt training arXiv "when hard negatives"` **【阴性专搜】**
  5. `synthetic data ratio real data mixing proportion training classifier imbalanced minority class LLM augmentation "ratio" negative result overfit artifacts`
- **支持**:
  - Kryscinski et al., EMNLP 2020, arXiv:1910.12840（FactCC，论文）——**完全用规则变换（实体/数字/代词替换、否定）合成正负例训练 BERT 判别器**，超过 NLI/fact-checking 强监督基线。与红线 0「类别→具体值」模板化违规同构，实体替换正是其最有效的变换类型。
  - Kwon, Yoon & Hwang, arXiv:2604.01993（SAFE，多跳推理 step-level verifier，论文，agent 全文核对）——教师按错误分类学合成 **7.6K 负例 + 3.9K 正例**，再挖 **848 条真实失败样本**精化，整体 +8.8pp。配比实况：**合成 ≈93%、真实锚定 ≈7%**。
  - Utama et al., NAACL 2022, arXiv:2205.06009（Falsesum，论文）——更自然、可控的合成不一致样本显著改善向真实错误的迁移。
- **反对 / 已知负结果**:
  - Goyal & Durrett, NAACL 2021, arXiv:2104.04302（论文）——简单合成错误**不覆盖**真实系统（XSum 类自由生成）犯的错；真实错误形态越自由，合成越失灵。
  - Zhang et al., KDD 2026, arXiv:2606.01304（When Hard Negatives Hurt，论文）——朴素混入 LLM 生成的 hard negatives **降低**性能；失败模式：**source-dependent shortcuts**（模型靠分布伪影按"来源"而非语义区分）。降幅数字待补（摘要未给）。
  - SAFE 自身消融（§3.3 + 表 12，agent 全文核对）——仅合成不崩盘（MuSiQue 71.7%），真实锚定的边际增益 **+2.4pp 平均 / MuSiQue +3.3pp / Llama 3.1 8B +6.3pp**——真实样本是锚定增益，不是生死线。
  - Wu et al., arXiv:2512.02389（论文；任务是 self-correction，外推打折）——合成→真实错误的分布移位"即使合成覆盖良好"也显著劣化。
- **文献预测**: 对**模板化、实体级**违规，纯合成训练判别器可行且是标准做法（FactCC 谱系）；正例绝对量是主要杠杆。失效条件：① 真实违规形态漂出模板；② 合成管线留下来源伪影（模型学"这句像机器改的"）。真实锚定的边际收益量级为个位数 pp。
- **本项目差异**: 中文/医疗/2B 生成式 JSON（文献多为英文 encoder 二分类）——方向可外推、数字不可照搬。最有利：红线 0 高度模板化，恰落在 FactCC 实体替换最有效的区间；盘内数据也证实（红线 0 占全部违规实例 82–85%，见 docs/RESEARCH_verifier_data_scarcity.md）。最不利：真实正例仅 66 条，比 SAFE 的 848 条锚定池小一个量级，锚定与验收要抢同一批数据。
- **决断**: **做，改设计为「最小编辑对 + 真实锚定 + 真实-only 验收」**——合成负例用规则最小编辑（同一条真实 answer 只改"类别→具体值"一处，正负成对、同管线产出），不让 LLM 自由重写（防来源伪影的核心手段）。理由：唯一实测瓶颈是正例信息量，而红线 0 的模板性正是合成法文献战绩最好的地形。
- **已写进判读规则**: ticket 12 验收表——**分池报告：真实违规留出集（训练不可见）漏报率单独算，≤5% 门槛只认真实池；合成池与真实池漏报率差 >10pp 判"学到伪影"，逐条归因后重造**；对照臂「自由改写版 vs 最小编辑版」验证伪影假设。

## L2 — 红线 0 重定义为细粒度 grounding 检测子任务 ⟶ ticket 12 候选方向 B

- **学名**: **context-grounded / span-level (token-level) hallucination detection**；单条判断为
  **claim verification against grounding documents**。红线 0 对应文献专名子类
  **factually correct but unsupported**——它有专名，不是自造问题。faithfulness（对原文）与
  factuality（对世界）是文献明确区分的两轴，红线 0 只考 faithfulness 轴。
- **检索式**（6 次，第 4 条阴性专搜）:
  1. `fine-grained hallucination detection span-level entity-level factual consistency grounding claim verification survey arXiv`
  2. `MiniCheck efficient fact-checking grounding LLM outputs arXiv 2404 model size synthetic training data GPT-4 accuracy`
  3. `SetFit few-shot classification sentence transformers sample efficiency arXiv 2209 encoder classifier vs generative LLM few-shot`
  4. `factual consistency NLI entailment models limitations fails unsupported but factually correct world knowledge leakage specificity hyponym hallucination detection` **【阴性专搜】**
  5. `multilingual span-level hallucination detection Chinese Mu-SHROOM SemEval 2025 cross-lingual performance degradation low-resource languages`
  6. `LettuceDetect ModernBERT token-level hallucination detection RAG RAGTruth arXiv encoder outperforms LLM prompt-based`
- **支持**:
  - Tang, Laban & Durrett, EMNLP 2024, arXiv:2404.10774（MiniCheck，论文，主线程核实）——770M Flan-T5 达 GPT-4 级 grounding 判别、成本低 ~400×；**训练数据全部由 GPT-4 结构化合成**（专门构造"看似合理但无依据"错误）。直接证明「窄 grounding 二分类小模型 + 合成数据」可行。
  - arXiv:2502.17125（LettuceDetect，论文，主线程核实）——ModernBERT **token 级分类头**（非生成式）在 RAGTruth 上 example-level F1 79.2%（超前代 encoder +14.8pp），体积小 ~30×。「encoder+分类头逐 token 判 grounding」路线有实证 SOTA。
  - Tunstall et al., arXiv:2209.11055（SetFit，论文）——sentence-transformer+分类头，8 条/类逼近全量微调；间接支持「窄二分类样本效率高一个量级」（但见 L3 对成对任务的限定）。
  - arXiv:2504.11975（Mu-SHROOM, SemEval-2025 Task 3，论文）——span 级幻觉检测 14 语种共享任务，**含中文**，有公开标注数据可借。另 PsiloQA arXiv:2510.04849（多语言 span 级自动标注管线）、arXiv:2509.22582（LLM 做 context-grounded 细粒度定位）。
- **反对 / 已知负结果**:
  - McCoy et al., ACL 2019, arXiv:1902.01007（HANS，论文）——NLI 模型依赖词汇重叠启发式，挑战集上接近 0 准确率。对应风险：**LLM 系判别器带参数化医学知识，倾向把"医学上真"的 claim 判 supported**（faithfulness/factuality 混淆）。
  - Laban et al., TACL 2022, arXiv:2111.09525（SummaC，论文，主线程核实编号）——NLI 用错粒度（整文档 vs 句子）大幅失效，粒度切分本身是主要变量。
  - 跨语言：LettuceDetect 系英文训练，土耳其语需整套重训（arXiv:2509.17671）；Mu-SHROOM 报告 **span 边界标注分歧高**——「span 边界」不如「名词命中/不命中」二分稳。
- **文献预测**: 逐名词/数值二分类（encoder+分类头）较联合生成式 JSON 样本效率高（SetFit 量级 5–10×，间接证据）且可达 prompt-based LLM 水平；合成负例可把正例扩到千级（MiniCheck 路线）。失效条件：① 类别词→下位词判断恰是启发式弱区；② 中文无现成模型，零样本迁移显著退化；③ 判别器世界知识越强越易放行"医学正确但越界"。
- **本项目差异**: ① 中文：只能借方法与合成配方，底座须中文（Mu-SHROOM 中文子集可当外部验证）；② 医疗术语商品名/通用名同义干扰字面匹配；③ 关键错位「蕴含≠有原文依据」：方向有利——general→specific 逻辑上本不蕴含，且 MiniCheck/RAGTruth 任务定义就是 supported-by-document 而非 factually-true，与红线 0 同构；风险在 LLM 判别器的医学知识会系统性放行此类样本。**未检索到「联合多任务判断 vs 拆分二分类」的直接对比实验**——该主张只有间接证据链，属在案无人区。
- **决断**: **做**——红线 0（含红线 2 的数字变体）拆出为「answer 中每个具体名词/数值 → source_text 是否有依据」的逐候选二分类；候选由规则抽取（盘内实测：词表+数字规则 case 级召回已达 92.6%，见研究笔记），判别头可用 encoder 或教师在线判。理由：66 条正例在联合生成式设定下已实测失败，窄二分类+合成扩充两条路线各有独立实证支持。
- **已写进判读规则**: ticket 12 验收表——必设「**医学正确但原文未给出**」对抗子集（全部为类别词→下位词/数值细化样本，≥50 条），该子集漏报率**单独报告**、门槛同总集（≤5%）；任何候选 checker（含现成模型）该子集不过线即判不可用，不看总分。

## L3 — 66 条正例的文献水位：继续训练 vs 改走不训练路线 ⟶ ticket 12 战略判断

- **学名**: 训练侧 **safety guardrail / content moderation classifier training**（Llama Guard 系）；
  小样本侧 **few-shot text classification**（SetFit/PET 系）；本任务的成对判断最接近
  **NLI / sentence-pair classification**；不训练路线 **LLM-as-a-judge**。
- **检索式**（5 次）:
  1. `Llama Guard ShieldGemma WildGuard training data size safety guardrail classifier arXiv`
  2. `SetFit few-shot text classification sentence transformers 8 examples per class arXiv Tunstall`
  3. `LLM-as-a-judge GPT-4 zero-shot versus fine-tuned small classifier faithfulness hallucination detection comparison arXiv`
  4. `fine-tuning LLM classifier fails insufficient positive examples rare class "few positive" recall negative result low-resource safety` **【阴性专搜】**
  5. `few-shot learning NLI sentence pair tasks poor performance SetFit limitations entailment "pair classification"` **【阴性专搜】**
- **支持**（guardrail 的训练数据量级基准）:
  - Inan et al., 2023, arXiv:2312.06674（Llama Guard，论文，agent 从 PDF 正文核数字）——共 **13,997** 条标注，≈10.5k 训练；**最小类别 166 条**也能用——但那是 7B 全参微调 + 全类别联合训练摊薄的结果。
  - Zeng et al., 2024, arXiv:2407.21772（ShieldGemma，论文）——最终训练集 **10,500 条**，**以合成数据为主**，每类正例数百到近千。
  - Han et al., NeurIPS 2024, arXiv:2406.18495（WildGuard，论文）——WildGuardMix **92k** 标注样本。
  - Tunstall et al., arXiv:2209.11055（SetFit）——8 条/类逼近全量微调，但成功案例全是**单文本语义分类**。
  - Wang et al., 2021, arXiv:2104.14690（Entailment as Few-Shot Learner，论文）——任务重写为 entailment 后 8 样本 +12%，但**依赖 MNLI 级成对预训练底座**（中文医疗域无现成可白嫖）。
  - arXiv:2608.00033（SIRIN，contextual hallucination 检测工具包，论文）——含微调 judge 与 zero-shot 的对比（agent 引其正文 AUROC 91.8–93.5 vs 82.0–88.8；**主线程抽查摘要未见该数字，正文数字待复核**），方向上佐证"容量足够、数据是瓶颈"。
- **反对 / 已知负结果**:
  - arXiv:2605.15680（医疗 triage 分类，论文）——安全敏感场景微调与 few-shot prompting 对比，few-shot LLM 宏 F1 略优、且都"不能自主部署"（agent 引正文"微调伤召回 ~10pp"，**主线程抽查摘要未见具体数字，待复核**）——极少正例时微调的失败模式正是漏报，与本项目实测 60.9% 漏报同向。
  - Brown et al., 2020, arXiv:2005.14165（GPT-3）——few-shot 的已知弱区正是 NLI/成对推理（ANLI 接近随机）；arXiv:2306.08058（软工域 sentence-pair few-shot 实证，论文）——PET 等需**数百条**标注才达全量性能。⇒ **SetFit「8 条/类」不覆盖 source-answer 对齐类任务**。
  - FaithBench, arXiv:2410.13210（论文，主线程核实）——最好的专训 faithfulness 检测器在困难样本上也仅 ~50% 准确率；zero-shot judge 在 AggreFact/RAGTruth <80%（Datadog 博客，来源等级：博客）⇒ 不训练路线也不是免费午餐。
  - TrustJudge, arXiv:2509.21117（论文）——LLM judge 打分系统性不一致，需固定 rubric + 一致性校验。
- **文献预测**: 能用的 guardrail 正例水位是 **10³ 量级**（每类数百条起，总集 10k–92k）；66 条比已知最小可行配置（166 条/类 + 14k 联合摊薄）低 3–15 倍，比典型配置低 1.5–2 个数量级。**未检索到任何已知微调方法在成对 grounding 任务上以 ~66 正例达到 ≤5% 漏报的先例**；few-shot 捷径恰在成对任务失效。反向条件：正例到 500–1500 条（合成为主亦可，ShieldGemma 先例）即回到文献可行区。
- **本项目差异**: ① 红线 0 是机械规则（类别→具体值越界），比开放危害分类更可程序化，合成正例成本极低且不易漂移；② 中文医疗域无 MNLI 级成对底座；③ 教师（Claude）本来在环，"退回教师在线判"零迁移成本。量级差距太大，结论可外推。
- **决断**: **双轨**——(a) 停止在 66 条正例上做任何再训练（非容量问题 + 差 1.5–2 个数量级，已判死）；(b) 若继续蒸馏，前置条件是先把正例扩到千级（合成为主，按 L1 设计）；(c) 短期可用性靠教师在线判候选（固定 rubric + 抽检一致性，防 TrustJudge 型偏差）。
- **已写进判读规则**: ticket 12 立项门槛——「**红线正例 <500 条时，任何微调轮（含换方法/换底座/过采样）不予立项**；恢复训练的前置条件是正例池 ≥1000 条且真实困难样本锚定占比达标。教师在线判上线必须带固定 rubric + ≥50 条人工抽检一致性 ≥95% 的验收线。」

## L4 — 合成扩量三条技术路径（单案例多示例注入 / 同案例多点合成 / 数字类最小编辑）⟶ ticket 14

- **学名**: 多示例/多扰动的部分沿用 L1 已定的 **synthetic error injection**（FactCC 系）本身的既有做法，没有独立学名；核心风险术语是 **source-dependent shortcuts / distributional artifacts in synthetic negatives**（同 L1 已引 arXiv:2606.01304）。数字类合成对应 **numerical hallucination in data-to-text / NLG faithfulness**（有独立学名，是幻觉文献里被专门讨论的子类）。
- **检索式**（6 次，第 3 条为阴性专搜）:
  1. `multiple synthetic negatives from same source document data augmentation correlated samples diminishing diversity`（钉"同文档多产"这件事在文献里怎么称呼）
  2. `numerical hallucination detection generation data-to-text arXiv survey`（钉数字类幻觉学名）
  3. `FactCC entity swap sentence perturbations per document how many negative examples one article` **【定量支持，同时也是对"同文档多产"是否已知问题的反向验证】**
  4. `number swap perturbation factual consistency classifier fails limitations arithmetic numeric errors harder than entity` **【阴性/失效条件专搜】**
  5. `numeric hallucination financial report generation arXiv number-aware fact checking benchmark`
  6. `RAGTruth span-level hallucination type breakdown numbers vs entities detection accuracy LettuceDetect MiniCheck`（查 L2 已引的 grounding checker 文献有没有数字 vs 实体的细分数据——未检索到，见下）
- **支持**:
  - FactCC 原论文（arXiv:1910.12840，摘要已核对："training data is generated by applying a series of rule-based transformations to the sentences of source documents"）+ 官方扩展仓库 `yuhui-zh15/FactCCX`（README 已核对）——确认有 **7 种独立变换类型**（backtranslation / pronoun_swap / date_swap / number_swap / entity_swap / negation / noise），对同一批源文档的句子反复应用。**"同一文档多产不同类型的扰动样本"是这个方法家族本来就有的标准做法**，不是本项目独创的风险动作。⚠️ 网络检索一度给出"单 claim 最多 8 条负例"的具体数字，本次核查未能从论文 PDF/摘要页直接验证该数字（工具限制，PDF 正文无法解析），**该具体数字按"编号待补/数字未核实"处理，不作为决断依据**——决断只依据"多类型变换同源应用是标准做法"这个已核实的定性事实。
  - Ji et al., "Survey of Hallucination in Natural Language Generation", ACM Comput. Surv. 2023, arXiv:2202.03629（论文，标题与覆盖范围——含 data-to-text——已核对；"数字幻觉是独立子类 + 现有指标不特别处理数字"这一具体论点来自检索摘要的二次转述，PDF 正文段落未能直接核对，**来源等级降级为二手**，不当作已验证的一手引用）。
- **反对 / 已知负结果**:
  - When Hard Negatives Hurt（arXiv:2606.01304，L1 已引，本条从"多示例注入"角度重新解读）——朴素合成负例会让模型学会**按来源身份而非语义**区分正负例（source-dependent shortcuts）。这对"同一文档多产"是直接相关的警示：如果正例池里**大比例正例都来自极少数几篇源文档**，"这篇文档被编辑过"本身就可能变成一个可学捷径，跟文档内容无关。FactCC 家族对同一文档反复应用变换是标准做法，但**具体上限没有查到可引用的数字**——这条"多产有没有明确上限"的问题按无人区处理，本项目自己定一个保守值（见下方决断），不是从文献里抄来的。
  - 关于数字扰动的多篇 arithmetic-reasoning 文献（Numeric-Remapping Attacks arXiv:2606.03606、Causal Consistency Regularization arXiv:2509.01544、Validation Gap arXiv:2502.11771）——**数字错误在多步推理链场景下比实体错误更难判别**（局部扰动不破坏周围推导的连贯性，模型倾向记字符串而非做算术）。这是真实的已知负结果，但场景是**多步推理链**（如数学解题步骤），跟本项目"单句里一个具体数字有没有原文依据"的场景不同——本项目更接近 FactCC 的 number swap（同句静态替换），不是链式推理扰动。
  - 检索式 6（span-level grounding checker 的数字 vs 实体细分表现）**未检索到相关工作**——LettuceDetect 等现有 span 级检测器公开材料只用二分类标注，没有按幻觉类型（数字/实体）拆分表现数字。这是无人区，不构成阻塞。
- **文献预测**: (a)(b) 单案例注入更多示例、同案例多点合成，方向上安全（FactCC 家族本来就对同一文档反复应用多种变换）；真正的风险不是"多产"本身，是**源文档集中度**——如果合成正例池的大比例都来自极少数源文档，"来源身份"可能变成捷径（When Hard Negatives Hurt 的核心机制）。**具体安全上限没有文献数字可引用**，效应量级未量化，只能定性防范、自定保守阈值。(c) 数字类最小编辑合成方向上安全，可以直接沿用 L1 已定的规则最小编辑方法论（FactCC 的 number swap 与 entity swap 是同一套机制的两种参数化），但**验收时必须把红线0（实体）和红线2（数字）的漏报率分开报**——数字幻觉在更难的多步推理场景下有已知的检测短板，即使本项目场景更简单，分开报告能防止实体类的强表现掩盖数字类的弱点，而不是假设两者表现一致。
- **本项目差异**: FactCC/FactCCX 是英文新闻摘要场景，本项目是中文医疗问答场景——扰动机制（实体替换/数字替换）可以直接外推，但"同一文档能安全承载多少条扰动"这个具体密度没有可核实的文献数字，本项目源文本更长（40-1200字），密度上限只能自己定，不是从文献外推的。数字幻觉的"多步推理更难"文献场景（数学解题/金融推理链）与本项目"单句静态数字替换"场景有实质差异，不能直接套用"更难"的结论。
- **决断**: **三条都做**，但 (b) 加一条新约束、(c) 加一条新验收拆分：
  - (a) 单案例注入 3-4 个类别成员（做——FactCC 家族本就对同一文档反复应用多种变换，方向安全；密度本身不是从文献抄的数字，是本项目自定的保守值）。
  - (b) 同案例多标记词各出一条合成样本（做），**新增源文档集中度上限**：任一 `source_case_id` 贡献的合成正例数不得超过合成正例总数的一个封顶比例（本项目定 **≤5%**——没有文献量级可查，保守取一个防止"单文档主导"的粗上限，训练前写死，不训完再调）。
  - (c) 扩展到红线2数字类最小编辑合成（做，沿用 L1 规则最小编辑方法论，不新增文献负担——number swap 与 entity swap 同源）。
- **已写进判读规则**: ticket 14 门槛表新增两条——① 「任一 `source_case_id` 贡献的合成正例数 ≤ 合成正例总数 5%，超过则视为源文档集中度过高，需要扩大源文档池而非从同一文档多产」；② 「验收表红线0（实体类越界）与红线2（数字类编造）漏报率必须分开报告，不得合并成一个数字」。
