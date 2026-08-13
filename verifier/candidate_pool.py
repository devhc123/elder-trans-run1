#!/usr/bin/env python3
"""可信候选池构造器（ticket 14）。

**核心设计决定**：`verdict=pass` 案例是教师对整案例给出的"零违规"判断——
这个判断本身没有"只记一条 evidence"的结构性缺陷，段A在这类案例上抽出的
每个候选都是真负例，不需要任何反推。`verdict=fail` 案例只有 evidence 命中
的候选才是可信正例；**没命中的候选整条丢弃**，不当负例用——这是 ticket 12
P2 探针诊断出的方法论缺陷的直接修复：案例级标注只记一条 evidence，不穷举
同一红线下的所有违规词，把"没命中 evidence"当"没违规"会把系统性漏标
烤进训练数据。

train 和 holdout 用同一个构造函数——训练负例挖矿、held-out 候选级验收集
都从这里来，保证两处用的是同一套"可信"规则，不会各自漂移出一套。

用法：
    python3 verifier/candidate_pool.py --split train
    python3 verifier/candidate_pool.py --split holdout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.judge_candidates import derive_candidate_label  # noqa: E402
from verifier.redline_candidates import extract_candidates, load_jargon, parse_case_text  # noqa: E402

WORK = ROOT / "verifier" / "work"

# 注入负例的 case_id 后缀。**两类分开标**——它们的难度差一个量级
# （grounded 是字符串查表就能做对；等价形式要真读懂"值相同写法不同"），
# 混在一个误报率里看不出模型是靠哪一类过的关。`train_lora.py` 与
# `eval_candidate_verifier.py` 都从这里导入，不各写一份字面量。
STRUCTURAL_NEGATIVE_SUFFIX = "-structneg"    # grounded 实体注入
EQUIVALENT_FORM_SUFFIX = "-eqformneg"        # 等价形式数字注入


def build_trusted_candidate_pool(records: list[dict], jargon: set[str]) -> list[dict]:
    """对每条 (train 或 holdout) 记录抽候选，按案例 verdict 分流打可信标签。

    - `verdict=pass`：段A抽出的每个候选都标 `label=False`（可信负例）。
    - `verdict=fail`：只保留 `derive_candidate_label` 命中 evidence 的候选，
      标 `label=True`；未命中的候选**不进 pool**——真实标签不确定，不能
      当负例用，也不算正例。
    """
    pool: list[dict] = []
    for r in records:
        src, ans = parse_case_text(r["input"])
        gold = json.loads(r["output"])
        is_pass = gold["verdict"] == "pass"
        for c in extract_candidates(src, ans, jargon):
            if is_pass:
                label = False
            else:
                if not derive_candidate_label(c["text"], c["kind"], gold):
                    continue
                label = True
            pool.append({
                "case_id": r["case_id"],
                "source_text": src,
                "answer": ans,
                "candidate_text": c["text"],
                "kind": c["kind"],
                "red_line_guess": c["red_line_guess"],
                "label": label,
            })
    return pool


def build_uncertain_candidate_pool(records: list[dict], jargon: set[str]) -> list[dict]:
    """`build_trusted_candidate_pool` 从 `verdict=fail` 案例里整条丢弃的
    候选——没命中 evidence，真实标签不确定，不能当负例用。这里单独枚举
    出来，供教师直接候选级判定（不是从单条 evidence 反推，是真的问
    "这个具体候选有没有原文依据"）。

    第三轮独立审计（问题二）发现：只丢弃这些候选会让候选标签退化成
    "这个候选来自哪个答案"的代理——train 里 0/1,905 个、holdout 里
    0/696 个唯一真实答案同时含正例候选和负例候选。教师直接判定这批
    候选、把结果并回可信池，能在同一个 fail 答案里同时造出真正例和
    真负例，第一次真正逼模型做候选级区分而不是整答案分类。"""
    pool: list[dict] = []
    for r in records:
        src, ans = parse_case_text(r["input"])
        gold = json.loads(r["output"])
        if gold.get("verdict") != "fail":
            continue
        for c in extract_candidates(src, ans, jargon):
            if derive_candidate_label(c["text"], c["kind"], gold):
                continue  # 已经是可信正例，不属于"不确定"集合
            pool.append({
                "case_id": r["case_id"],
                "source_text": src,
                "answer": ans,
                "candidate_text": c["text"],
                "kind": c["kind"],
                "red_line_guess": c["red_line_guess"],
            })
    return pool


def build_injection_negatives(
    records: list[dict], jargon: set[str], *,
    max_entity: int | None = None, max_equivalent: int | None = None,
) -> list[dict]:
    """两类**注入负例**：与合成正例同模板、同插入位置、同措辞，只有内容不同。

    取代原来的"零候选安慰语"（ticket 26）。那一版只共享了括注前缀、内容刻意
    零候选，于是"括注在不在"失效了、"**候选**在不在括注里"照样把两类完全分开
    ——ticket 17 的探针实测对抗子集 J=1.000。现在括注里真的有候选：

    - **grounded 实体注入**（`STRUCTURAL_NEGATIVE_SUFFIX`）：注入 source 里确实
      有、answer 里没有的药名。杀"候选在括注内 ⇒ 违规"。
    - **等价形式注入**（`EQUIVALENT_FORM_SUFFIX`）：原文「3天」→ 注入「三天」，
      非 grounded 却有依据。杀"在括注内 ∧ 非grounded"这个合取——那是三格里
      唯一能用规则填上的一格，其余按红线定义不存在（见 `shortcut_probes`）。

    **两类分别设上限**：等价形式全是数字类、实体全是词表类，用一个总数上限截断
    会让 kind 分布随机倾斜（ticket 17 的探针⑦ 盯的就是这个）。

    只对 `verdict=pass` 的案例生效——fail 答案里本来就有真违规，再叠注入会让
    标签失去意义。按 case_id 排序遍历，确定性、不随机。"""
    from verifier.synth_minimal_edit import (
        build_equivalent_form_injection,
        build_grounded_injection,
    )

    entity: list[dict] = []
    equivalent: list[dict] = []
    for r in sorted(records, key=lambda r: r["case_id"]):
        if json.loads(r["output"]).get("verdict") != "pass":
            continue
        if max_entity is None or len(entity) < max_entity:
            for c in build_grounded_injection(r, jargon):
                entity.append({**c, "case_id": f"{r['case_id']}{STRUCTURAL_NEGATIVE_SUFFIX}"})
        if max_equivalent is None or len(equivalent) < max_equivalent:
            for c in build_equivalent_form_injection(r):
                equivalent.append({**c, "case_id": f"{r['case_id']}{EQUIVALENT_FORM_SUFFIX}"})
    # 硬截断：内层循环一次可能追加多个候选，卡在差 1 条时会整条超发
    # （`/code-review` 在上一版抓到过同款 bug，这里保留同样的防线）。
    if max_entity is not None:
        entity = entity[:max_entity]
    if max_equivalent is not None:
        equivalent = equivalent[:max_equivalent]
    return entity + equivalent


def sample_uncertain_candidates(
    pool: list[dict], n: int | None, prefix: str,
    exclude: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """从 `build_uncertain_candidate_pool` 的输出里确定性抽样 n 条并分配
    `id`（供派发教师标注、之后回填给 `apply_teacher_labels` 用）。

    `/code-review` 发现的可复现性缺口：250 条不确定候选（问题二修复用的
    那批）当初是靠一次性脚本抽样+分配 id，脚本本身没有提交进库——这里
    补一个有测试覆盖、可重跑的入口，不代表要复现历史上那批 250 条的
    精确 id（那批已经连同教师标注结果一起落盘在 `teacher_candidates_
    {train,holdout}.jsonl`，不需要也不应该重新生成）。

    按 `(case_id, candidate_text)` 的哈希排序取前 n 条——确定性、不随机，
    重跑同一个 pool 拿到同一批候选和同一批 id，超过 pool 大小就取全部。
    `n=None` 表示"剩下的全要"（ticket 23 的 train 侧是全量补标，不是抽样）。

    `exclude`：已经标注过的 `(case_id, candidate_text)` 集合。**必须按内容键排除，
    不能按名次排除**——历史上那 250 条是一次性脚本抽的、脚本没入库，它的名次顺序
    无从复现，"跳过前 100 名"会跳错人（既可能重复派发已标过的，也可能永远漏掉
    某些条目）。内容键是唯一稳的锚。

    `prefix` 换一批要换一个——`apply_teacher_labels` 的 judgments 是按 id 查的，
    两批 id 撞了会张冠李戴，且不会报错。"""
    exclude = exclude or set()
    ranked = sorted(
        (it for it in pool if (it["case_id"], it["candidate_text"]) not in exclude),
        key=lambda it: hashlib.sha256(f"sample:{it['case_id']}:{it['candidate_text']}".encode()).hexdigest(),
    )
    chosen = ranked if n is None else ranked[:n]
    return [{**item, "id": f"{prefix}-{i:03d}"} for i, item in enumerate(chosen)]


def apply_teacher_labels(sampled_uncertain: list[dict], judgments: dict[str, bool]) -> list[dict]:
    """把教师对"不确定候选"（`build_uncertain_candidate_pool` 的输出，
    经采样后每条带了 `id`）的直接判定结果并回可信候选池的形状。

    `judgments` 是候选 `id` -> `violated` 布尔的映射（教师产物）。**没有
    对应判定的候选直接跳过**，不强行猜一个标签——教师标注不齐全时，
    宁可少几条真标签的候选，也不能把"没判到"悄悄当成某个默认值。"""
    out: list[dict] = []
    for item in sampled_uncertain:
        cid = item.get("id")
        if cid is None or cid not in judgments:
            continue
        merged = {k: v for k, v in item.items() if k != "id"}
        merged["label"] = judgments[cid]
        out.append(merged)
    return out


def load_teacher_candidate_labels(path: Path) -> list[dict]:
    """读回 `apply_teacher_labels` 的落盘产物（问题二修复：250 条"不确定
    候选"的一次性教师标注结果，`verifier/teacher_candidates_{train,
    holdout}.jsonl`，跟 `verifier/labels/*.jsonl` 同等地位）。

    教师标注是派发 4 个 agent 跑出来的人工产物，不是可以随时重算的纯
    函数——这里只是读文件，不重新触发标注流程。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "holdout"], required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--sample-uncertain", type=int, default=None, metavar="N",
                     help="不构建可信池，改为从这个切分的「不确定候选」"
                          "（build_uncertain_candidate_pool，fail 案例里没"
                          "命中 evidence、整条丢弃的那批）里确定性抽样 N 条"
                          "并分配 id，写出供派发教师标注（问题二修复的"
                          "复现入口，不是默认模式）")
    ap.add_argument("--id-prefix", default=None,
                     help="--sample-uncertain 用的 id 前缀，默认按 split 推 "
                          "utrain/uholdout。补标新一批时**必须换一个**，"
                          "两批 id 撞了 apply_teacher_labels 会张冠李戴且不报错")
    ap.add_argument("--exclude-labelled", action="store_true",
                     help="排除 teacher_candidates_<split>.jsonl 里已经标注过的"
                          "（按 case_id+候选文本，不是按名次）。ticket 23 补标用")
    ap.add_argument("--all-remaining", action="store_true",
                     help="不抽样，把排除之后剩下的全要（train 侧补标是全量）")
    args = ap.parse_args()

    records = load_split(ROOT / "verifier" / f"{args.split}.jsonl")
    jargon = load_jargon()

    if args.sample_uncertain is not None or args.all_remaining:
        uncertain = build_uncertain_candidate_pool(records, jargon)
        prefix = args.id_prefix or ("utrain" if args.split == "train" else "uholdout")
        exclude: set[tuple[str, str]] = set()
        if args.exclude_labelled:
            teacher_path = ROOT / "verifier" / f"teacher_candidates_{args.split}.jsonl"
            if teacher_path.exists():
                exclude = {(d["case_id"], d["candidate_text"])
                           for d in load_teacher_candidate_labels(teacher_path)}
        n = None if args.all_remaining else args.sample_uncertain
        sampled = sample_uncertain_candidates(uncertain, n, prefix, exclude=exclude)
        print(f"{args.split} 不确定候选池 {len(uncertain)} 条"
              + (f"，排除已标注 {len(exclude)} 条" if exclude else "")
              + f" -> 取 {len(sampled)} 条（前缀 {prefix}）")
        out = args.out or (WORK / f"uncertain_sample_{args.split}.json")
        WORK.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {out}")
        return 0

    pool = build_trusted_candidate_pool(records, jargon)

    # 问题二修复：把教师直接判定过的"不确定候选"（原本整条丢弃的那批）
    # 并回可信池——这是同一答案内第一次同时出现真正例和真负例的来源，
    # 没有它，候选标签退化成"来自哪个答案"的代理（详见模块顶部注释）。
    teacher_path = ROOT / "verifier" / f"teacher_candidates_{args.split}.jsonl"
    if teacher_path.exists():
        pool += load_teacher_candidate_labels(teacher_path)

    n_pos = sum(1 for it in pool if it["label"] is True)
    n_neg = sum(1 for it in pool if it["label"] is False)
    n_rl0 = sum(1 for it in pool if it["label"] is True and it["red_line_guess"] == 0)
    n_rl2 = sum(1 for it in pool if it["label"] is True and it["red_line_guess"] == 2)
    print(f"{args.split} {len(records)} 条 -> 可信候选池 {len(pool)} 条"
          f"（正例 {n_pos}：红线0 {n_rl0} / 红线2 {n_rl2}；负例 {n_neg}）")

    # 退化分类器探针（ticket 17）。可信池里没有任何注入结构，这张表在这里
    # 基本恒为 0——正因如此它是个有用的对照：同一组探针在对抗子集/训练池上
    # 一旦不为 0，差异就全部来自合成注入，不是来自真实数据本身。
    # 惰性导入：`shortcut_probes` 会 import `synth_minimal_edit`，而本模块被
    # `adversarial_subset` 顶层导入，顶层互相 import 会成环。
    from verifier.shortcut_probes import report as report_shortcut_probes
    report_shortcut_probes(pool, f"可信候选池（{args.split}）")

    out = args.out or (WORK / f"trusted_candidate_pool_{args.split}.json")
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
