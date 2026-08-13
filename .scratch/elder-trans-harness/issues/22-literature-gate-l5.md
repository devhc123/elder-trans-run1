# 22 — 铁律15 文献门：L1 追加 + L5 新增

**What to build:** 第二批要动数据构造策略与评测方法，按铁律15 必须先有文献条目才许改代码。
这张票产出两样东西。

**一、L1 追加一行落地说明（不新开条目）**。判定：本次的 source-grounded 注入负例与等价形式
注入负例，**命中 acad-research 的跳过规则 #3**——L1 的决断原文已经写死"合成负例用规则最小编辑
（同一条真实 answer 只改'类别→具体值'一处，**正负成对、同管线产出**）"，而"用同一套模板、
同一个插入位置产出正负例，只让内容的 grounded 与否决定标签"正是这句话的落地；FactCC 的
backtranslation 分支本来就产 label=CORRECT 的样本，等价形式改写是它在中文数字上的最小对应。
L1 写于 2026-08-13，在 90 天射程内。所以只在 L1 的"已写进判读规则"追加一行，不新开条目。

**二、新开 `L5`（这个必须真查）**。把"用退化基线当作数据集的预注册门"过一遍铁律15——这是
**评测方法**改动，不在 L1 的射程里。这件事在文献里有正经学名，不是自造的：**annotation
artifacts / partial-input（hypothesis-only）baselines**。

检索方向（≥3 次，含一次阴性专搜）：
- NLI 标注伪影与 hypothesis-only baseline（Gururangan 2018、Poliak 2018 一系）
- **阴性专搜**：partial-input baselines 本身会误导（Feng, Wallace & Boyd-Graber 2019,
  "Misleading Failures of Partial-input Baselines"）——高 partial-input 分数不等于数据集坏了，
  这条直接决定我们该不该把探针分数当硬门槛
- shortcut learning / dataset ablation 作为数据集验收门的既有做法

产出条目要含"本项目差异"与一条能写进 ticket 的判读规则。**检索没有否决权**：找到 refute 的
论文不等于不做，而是要写清差异并变成判读规则的一条。"未检索到"是合格产出，空着不是。

**Blocked by:** 无 —— 可立即开始（但硬性 gate 住 24 和 26）

**Status:** ready-for-agent

- [ ] L1 追加落地说明，写明命中跳过规则 #3 的理由与射程判断
- [ ] L5 条目完整：学名、检索式（含阴性专搜）、支持、反对/已知负结果、文献预测、
      本项目差异、决断、已写进判读规则
- [ ] 引用的编号做过抽查核实；核实不了的显式降级标注（沿用 L4 对"FactCCX 单 claim 8 条负例"
      那次的处理方式）
- [ ] Feng et al. 那条阴性结果必须体现在门槛设计上，不能只是列在"反对"里凑数
