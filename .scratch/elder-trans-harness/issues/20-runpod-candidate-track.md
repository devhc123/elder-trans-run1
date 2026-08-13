# 20 — RunPod 部署脚本接上候选级 track

**What to build:** `deploy/runpod_pilot.sh` 目前训的是 `verifier/train.jsonl`（案例级 1,456 条，
ticket 09/11 那次实验），推的是案例级 holdout，scp 清单里根本没有候选级产物——而
**零成本预检阶段照样全绿**。现在跑 `--create` 会在毫无提示的情况下花 $3-5 训一个跟 ticket 14
完全无关的旧实验。这张票让脚本训对东西，并且让预检在训错东西时**报错而不是放行**。

**默认走 candidate track**（ticket 09/11 已经跑完，旧实验要显式 `--track case`）。但默认值
本身也是一种"悄悄替你选了实验"，所以两道显式提示：预检**第一行就打印当前 track 与它将要
训的文件名**，`--create` 在建机前对 track 再确认一次。

**预检要新增一条"文件与 track 匹配"的硬校验**：读候选级训练集第一行，断言输出 schema 里是
`violated` 而不是 `verdict`（case track 反过来）。这是"预检全绿但训的是另一个实验"这个坑的
直接堵法——光检查文件存在不够，文件存在且是错的那一类才是真正会烧钱的情形。同时打印正例
条数与占比，占比异常（<5%）直接判 bad。

**第一次只跑单点**：`--oversample-fail 1`、3 epoch 不动。按 ticket 09 实测（340 条 3 epoch =
129 步 / 8分25秒，约 3.1 s/step）推算，候选级 7,578 条约 2,842 步 ≈ 2.4 小时，加建机装依赖
约 20 分、加 2,165 次推理约 35 分 ≈ **3.3 小时 ≈ $1.3**——$3-5 只够 2-3 个网格点。这一趟要买的
信息是"有没有学到伪影"，伪影若成立，网格搜索的钱全是白花的；epoch 不动是为了与 ticket 09/11
可比。倍数网格等真实池数字出来再决定值不值得。

**顺带修一个建机后才会炸的风险**：`SFTConfig(max_seq_length=...)` 在较新版 TRL 里改名成了
`max_length`。`verifier/work/pilot_train.log` 里记了 Unsloth 2026.8.15 / Transformers 5.5.0 /
Torch 2.11.0+cu130 / Triton 3.6.0，**唯独没有 TRL 版本**——所以"锁定当年验证过的版本"对这个
具体风险无效。改成运行时探测参数名（本机可测，mock 掉 SFTConfig），已知的锁、不知道的探测。

**这张票不包含真的建机。** `--create` 需要用户单独明确确认，不在任何"我同意"的自动授权范围内。

**Blocked by:** 16（产物落点）、18（候选级推理）、19（候选级验收）

**Status:** done（2026-08-13）

- [x] `--track candidate|case`，默认 candidate
- [x] 预检第一行打印 track 与将要训练的文件名
- [x] 预检检查三份候选级产物齐全，缺文件时打印可直接粘贴的重建命令
- [x] **喂错 track 的文件时预检报错**（故意把候选级文件换成案例级，确认 fail 而非放行）
- [x] 预检打印正例条数与占比，<5% 判 bad
- [x] 本机测试清单加上候选池 / 对抗子集 / 验收脚本 / 探针四份测试
- [x] scp 清单按 track 分支，candidate 传训练集 + 两个池、不传 `to_label.json`
- [x] 训练调用指向候选级训练集，`--oversample-fail 1`、3 epoch
- [x] 推理跑两次（真实池 + 对抗子集），取回后跑候选级验收
- [x] `SFTConfig` 参数名运行时探测 + 本机回归测试
- [x] 装依赖锁定日志里有记录的三个版本号
- [x] `bash deploy/runpod_pilot.sh`（零成本段）在 candidate track 下全绿
- [x] **不建机**——`--create` 留给用户单独确认
