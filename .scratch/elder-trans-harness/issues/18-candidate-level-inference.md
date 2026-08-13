# 18 — predict_lora 的候选级推理模式

**What to build:** 喂一份候选池 JSON（可信池或对抗子集），在 RunPod 上跑出候选级预测
`{"case_id": "vt-xxxx::候选文本", "violated": bool}`。**目前这条路完全不存在**——
`predict_lora.py` 只 import 案例级 `SYSTEM`/`build_prompt`，`audit_prediction` 硬假设
`key_points`/`red_lines`/`verdict` 字段，`load_items` 按 `to_label.json` 的 case_id 找条目，
候选级 id 会直接在缺失检查那里失败。

推理 prompt 必须与训练时**字面一致**：走 `render_prompt(SYSTEM_CANDIDATE,
build_candidate_prompt(item))`，不要在这里重写模板（两份字面量不同步的偏差不报错，
只会让生成质量莫名下降）。`parse_prediction` 现成可用。

**两种模式共用同一个推理循环**——把循环抽成"接收 `(row_id, prompt)` 列表 + 一个 audit 回调"
的共享实现，checkpoint 落盘、`write_jsonl`、`generate` 都不复制第二份。案例级路径的行为
必须逐字节不变。

生成上限单独定：候选级目标就是 `{"violated": true}` 十几个 token，沿用案例级的 768 会让
2,100+ 次串行生成白白拖长机时（按 ticket 09 实测速度，这是真金白银的机时）。

预测行**不写 gold 标签**——验收脚本从 pool 文件取 gold，预测文件自带答案是给自己挖坑。

**Blocked by:** 16（候选 id 单一定义）

**Status:** done（2026-08-13）

- [x] `--candidates-from <pool.json>` 与 `--case-ids-from` 互斥，两条路径都能跑通
- [x] 候选级模式不读 `to_label.json`（该文件在候选 track 里根本不会传上机器）
- [x] 推理循环只有一份，案例级行为逐字节不变（现有测试全绿即证）
- [x] 候选级 schema 校验：`violated` 必须是布尔
- [x] 候选级生成上限单独设，不沿用案例级的 768
- [x] 预测行只含 id 与判定，不含 gold
- [x] 本机可测的部分（prompt 构造、id、解析、schema）都有测试；推理本身要 GPU，沿用
      `test_predict_lora.py` 既有纪律
