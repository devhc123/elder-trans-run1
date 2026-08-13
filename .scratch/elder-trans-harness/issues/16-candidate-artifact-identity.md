# 16 — 候选级产物的身份与落点（第一批 prefactor）

**What to build:** 候选级条目的 id 在全仓库只有**一处**定义，候选级训练集默认落在部署预检
实际会去读的那个位置。这两件事都是"改一处忘改另一处不会有任何报错信号"的坑，第一批后面
三张票（18/19/20）都踩在它们上面，所以先做。

具体两条：

1. **候选 id 单一定义**。`f"{case_id}::{candidate_text}"` 这个格式目前只存在于
   `train_lora.make_candidate_dataset` 内部。候选级推理（18）要按同一个格式建索引，验收
   脚本（19）要按同一个格式 join 预测与 gold——三处各写一份字面量，格式一漂移就静默对不上，
   表现是"验收集里所有条目都缺预测"而不是报错。抽成一个函数，三处都调它。同
   `TEMPLATE_PREFIXES` / `STRUCTURAL_NEGATIVE_SUFFIX` 那两次去重的纪律。
2. **`--candidate-data-out` 默认路径**。现在默认写当前工作目录的 `candidate_train.jsonl`，
   而部署预检（20）读的是 `verifier/work/candidate_train.jsonl`。两者不一致时会出现
   "重建过了，但预检读的是几天前的旧文件，而且全绿"——正是这一批要堵的那类坑。默认改成
   仓库绝对路径。

**`--data` 的"必须显式传"语义不动**——那是 ticket 14 里刻意保留的行为（默认值是 cwd 相对
路径，从仓库根跑不传会 FileNotFoundError），不要顺手一起改。

**Blocked by:** 无 —— 可立即开始

**Status:** done（2026-08-13）

- [x] 候选 id 只有一处定义，`make_candidate_dataset` 改为调用它
- [x] `--candidate-data-out` 默认为仓库绝对路径 `verifier/work/candidate_train.jsonl`
- [x] `--data` 的显式传参语义保持不变，有测试固化
- [x] 重跑 `--make-candidate-data`，产物与已提交版本**逐字节一致**（这次改动不该动任何数字）
- [x] 全量测试 + pyflakes 干净
