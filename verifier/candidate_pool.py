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
from verifier.synth_minimal_edit import TEMPLATE_PREFIXES  # noqa: E402

WORK = ROOT / "verifier" / "work"

# **结构性负例**（第三轮独立审计发现的问题一的修复）：只看答案有没有含
# 固定括注模板这一个特征，不读原文、不做任何语义判断，就能在对抗子集上
# 拿到 100% 召回 / 0% 误报——train/holdout 的合成正例统一用"答案末尾追加
# 括注"的结构，模型可能只学会认这个结构。这里造"括注存在但内容真实无害"
# 的负例：对 `verdict=pass` 的真实案例，答案后面追加一句**不含任何可
# 提取候选**的安慰语，让"有没有括注"这个特征在正负例里都出现。
#
# **必须复用跟合成正例相同的括注前缀**（"（补充一句：""（顺带说一句，"
# "（这类"），不能自己另起一套不重叠的措辞——否则退化分类器只需要认
# "这三个具体前缀"，换一套新前缀完全不影响它继续 100% 命中原来的正例，
# 等于没堵。填充内容本身必须零候选（已用测试核对：无数字/中文数词/
# 分数/词表实体），这样附着在同一答案上的原有可信候选标签不受影响，
# 只是答案多了一句结构上像"编造注入"、内容上完全无害的括注。
STRUCTURAL_NEGATIVE_FILLERS: dict[str, list[str]] = {
    "train": [
        "（补充一句：具体请以医嘱为准，这里说的都是一类情况。）",
        "（顺带说一句，用药安全最重要，有疑问随时问医生。）",
        "（这类情况的具体细节，建议咨询医生。）",
    ],
    "holdout": [
        "（补充一句：以上仅供参考，一切遵医嘱执行。）",
        "（顺带说一句，安全用药最重要，别自己乱做主。）",
        "（这类问题的具体答案，还是要听医生的。）",
    ],
}
assert all(
    f.startswith(TEMPLATE_PREFIXES) for fillers in STRUCTURAL_NEGATIVE_FILLERS.values() for f in fillers
), "安慰语前缀必须复用 synth_minimal_edit.TEMPLATE_PREFIXES，不能另起一套"

# 结构性负例的 case_id 后缀——`train_lora.py` 靠它从最终候选池里数出
# 结构性负例的条数。两处都从这里导入，不各写一份字面量（`/code-review`
# 发现原来是两处独立硬编码同一个魔法字符串，改一处忘改另一处会让统计
# 静默算错，不报错）。
STRUCTURAL_NEGATIVE_SUFFIX = "-structneg"


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


def build_structural_negatives(
    records: list[dict], jargon: set[str], *, holdout: bool = False, max_items: int | None = None
) -> list[dict]:
    """造"括注结构存在但内容真实无害"的负例（问题一的修复，见模块顶部
    `STRUCTURAL_NEGATIVE_FILLERS` 的注释）。

    只对 `verdict=pass` 的真实案例生效——answer 后面追加一句零候选的
    安慰语，答案里**原本就有**的可信候选（本来就该标 False）原样保留，
    只是现在这些候选所在的答案里也出现了"括注"这个结构特征，让这个
    特征在正负例里都出现。`holdout=True` 用另一组安慰语（同一套
    train/holdout 隔离纪律：验收用的结构不能是训练时见过的那几句原话，
    虽然这里堵的是"结构"不是"内容"，用不同句子仍然更干净）。

    `max_items`：可信池里 pass 案例候选本来就有 5000+ 条，全部装饰一遍
    会让候选级训练集的正例占比从 15.8% 再腰斩到 8.6%，重新逼近 ticket
    09/11 坍缩过的量级区间——目的只是让"有没有括注"这个特征不再完美
    区分正负例，不需要每条负例都装饰一遍，按 case_id 排序取前 N 条即可
    （确定性、不随机）。默认 None 表示不设上限（供只关心正确性的测试和
    调用方自己控制规模）。"""
    fillers = STRUCTURAL_NEGATIVE_FILLERS["holdout" if holdout else "train"]
    out: list[dict] = []
    for r in sorted(records, key=lambda r: r["case_id"]):
        if max_items is not None and len(out) >= max_items:
            break
        src, ans = parse_case_text(r["input"])
        gold = json.loads(r["output"])
        if gold.get("verdict") != "pass":
            continue
        candidates = extract_candidates(src, ans, jargon)
        if not candidates:
            continue
        digest = hashlib.sha256(f"structneg:{r['case_id']}".encode()).hexdigest()
        filler = fillers[int(digest, 16) % len(fillers)]
        decorated_answer = ans.rstrip() + "\n\n" + filler
        for c in candidates:
            out.append({
                "case_id": f"{r['case_id']}{STRUCTURAL_NEGATIVE_SUFFIX}",
                "source_text": src,
                "answer": decorated_answer,
                "candidate_text": c["text"],
                "kind": c["kind"],
                "red_line_guess": c["red_line_guess"],
                "label": False,
            })
    # **`/code-review` 发现的真 bug**：上面每条记录只检查一次
    # `len(out) >= max_items`，但一条记录的内层循环可能一次性追加多个
    # 候选——卡在刚好差 1 条时，下一条记录如果有 N 个候选会整条超发
    # N-1 条（实测：train 全量 + max_items=1015 时曾返回 1016 条）。
    # 这里做硬截断，保证返回值绝不超过 max_items；不影响确定性前缀性质
    # （records 已按 case_id 排序，截断前的 out 本身就是确定性顺序）。
    if max_items is not None:
        out = out[:max_items]
    return out


def sample_uncertain_candidates(pool: list[dict], n: int, prefix: str) -> list[dict]:
    """从 `build_uncertain_candidate_pool` 的输出里确定性抽样 n 条并分配
    `id`（供派发教师标注、之后回填给 `apply_teacher_labels` 用）。

    `/code-review` 发现的可复现性缺口：250 条不确定候选（问题二修复用的
    那批）当初是靠一次性脚本抽样+分配 id，脚本本身没有提交进库——这里
    补一个有测试覆盖、可重跑的入口，不代表要复现历史上那批 250 条的
    精确 id（那批已经连同教师标注结果一起落盘在 `teacher_candidates_
    {train,holdout}.jsonl`，不需要也不应该重新生成）。

    按 `(case_id, candidate_text)` 的哈希排序取前 n 条——确定性、不随机，
    重跑同一个 pool 拿到同一批候选和同一批 id，超过 pool 大小就取全部。"""
    ranked = sorted(
        pool,
        key=lambda it: hashlib.sha256(f"sample:{it['case_id']}:{it['candidate_text']}".encode()).hexdigest(),
    )
    return [{**item, "id": f"{prefix}-{i:03d}"} for i, item in enumerate(ranked[:n])]


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
                          "utrain/uholdout")
    args = ap.parse_args()

    records = load_split(ROOT / "verifier" / f"{args.split}.jsonl")
    jargon = load_jargon()

    if args.sample_uncertain is not None:
        uncertain = build_uncertain_candidate_pool(records, jargon)
        prefix = args.id_prefix or ("utrain" if args.split == "train" else "uholdout")
        sampled = sample_uncertain_candidates(uncertain, args.sample_uncertain, prefix)
        print(f"{args.split} 不确定候选池 {len(uncertain)} 条 -> 抽样 {len(sampled)} 条（前缀 {prefix}）")
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

    out = args.out or (WORK / f"trusted_candidate_pool_{args.split}.json")
    WORK.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
