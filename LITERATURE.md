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
