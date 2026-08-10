#!/usr/bin/env python3
"""把真实线上 FAQ 按频次归类到 10 个子场景，据此定测试集配比。

**这是整个项目最关键的一份论证材料。** 测试集要往生活化、易度更高的方向调，
这件事的性质取决于依据：按真实用户需求分布配比，是需求对齐；拍脑袋挑简单的，
是指标造假。两者外形一样，只有依据能把它们分开。

依据是 eqbench-run2/SOURCE/AI健康管家-常见问题.xlsx —— 线上真实日志按频次聚合的
Top 150，每条前缀带 [频次:N]。

场景体系遵守一条硬约束：**项目书第六栏的 8 个父类原文不动**，新的生活化场景做
子类挂靠。这样「适老场景覆盖数」的指标说明一字不用改，不必走修改流程。

用法：
    python3 pipeline/classify_faq.py              # 归类 + 出配比
    python3 pipeline/classify_faq.py --show-all   # 逐条打印归类结果（人工复核用）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAQ = Path("/Users/chenhao/ClaudeCode/eqbench-run2/SOURCE/AI健康管家-常见问题.xlsx")
OUT = ROOT / "docs" / "faq_classification.json"

TOTAL_ITEMS = 270
FLOOR_PER_SCENARIO = 15

# 子场景 -> 父类（父类名逐字取自项目书第六栏，不得改动）
SCENARIOS = {
    "检验报告解读": "检验报告解读",
    "医嘱转译": "医嘱转译",
    "用药咨询": "用药说明",
    "日常照护": "健康科普",
    "急症处置": "健康科普",
    "饮食营养": "健康科普",
    "运动与养生": "健康科普",
    "科普辟谣": "健康科普",
    "权益匹配": "权益匹配",
    "分科导诊": "分科医生",
    "复诊与随访": "复诊提醒",
    "用药干预": "用药干预",
}

# 归类规则。**顺序即优先级**，先命中先归。规则写成显式关键词而非模型判断，
# 是为了让归类结果可复核、可争论、可修改——这份配比要拿去答辩。
RULES: list[tuple[str, list[str]]] = [
    # 先剔除：非老年受众 / 非咨询内容。这些问题真实存在于日志，但不属于本项目。
    ("_排除_儿科", ["小孩", "幼儿", "婴儿", "宝宝", "儿童"]),
    ("_排除_产品功能", ["健康档案", "帮我分析下我的健康情况", "体重目标"]),
    ("_排除_无效", ["你好", "谢谢", "在吗"]),
    ("_排除_孕产", ["孕妇", "哺乳期", "怀孕", "产后"]),

    # 辟谣要在最前：这类句式明确，且是高价值的对抗性类别，不能被后面的宽规则吃掉
    ("科普辟谣", ["是真的吗", "有科学依据", "是骗局", "骗局吗", "靠谱吗", "广告是真的",
                "万能", "排毒", "酸碱体质", "预防感冒吗", "能治愈", "不用手术"]),

    # 急症处置：老年高危场景，句式集中在「急救 / 紧急处理 / 突然……怎么办」
    ("急症处置", ["急救", "紧急处理", "怎么止血", "快速止血", "烫伤", "卡喉", "异物",
                "跌倒", "崴了", "划伤", "晕倒", "要不要立刻", "立刻送医院", "中风吗"]),

    # 推荐域：日志里稀疏，但父类必须有覆盖，规则前置优先捕获
    ("权益匹配", ["医保", "报销", "费用", "自费", "补贴", "免费", "价格", "多少钱"]),
    ("复诊与随访", ["预约", "复查", "复诊", "随访", "多久查一次", "如何记录", "怎么记录"]),
    ("分科导诊", ["挂什么科", "挂号", "看什么科", "哪个科", "科室", "挂皮肤科", "看什么医生",
                "需要就医", "要不要就医", "多少度需要"]),

    # 检验报告解读：指标、分级、风险评估
    ("检验报告解读", ["正常范围", "多少算正常", "指标", "化验", "体检", "报告", "结节",
                  "偏高", "偏低", "参考值", "风险等级", "等级划分", "风险评估", "分级",
                  "是否需要治疗", "如何评估", "关联分析", "比例低"]),

    ("医嘱转译", ["医嘱", "出院", "处方", "手术后", "术后"]),

    # 用药干预：漏服、停药、联用、副作用这类**行动决策**（区别于「这药怎么吃」）
    ("用药干预", ["忘记吃", "漏服", "能停", "停药", "一起吃", "同时吃", "相互作用",
                "副作用", "过量", "为什么不能一起"]),

    ("用药咨询", ["饭前", "饭后", "怎么吃药", "服用", "剂量", "抗生素", "止痛", "降压药",
                "退烧药", "中药和西药", "疫苗安全"]),

    ("饮食营养", ["饮食", "吃什么", "保健品", "能吃", "食物", "营养", "隔夜", "断食",
                "忌口", "少吃", "喝水", "喝醋", "补钙", "维C", "水果", "食欲"]),

    ("运动与养生", ["运动", "锻炼", "养生", "保暖", "睡眠", "失眠", "久坐", "熬夜",
                 "作息", "按摩", "穴位", "艾灸", "泡脚", "早醒", "入睡困难", "盗汗"]),
]

# 兜底：以上都不命中的，绝大多数是「老人某症状怎么办 / 怎么缓解 / 怎么预防」这类
# 日常照护型提问 —— 这正是日志里最大的一簇，单列一类而非塞进辟谣。
FALLBACK = "日常照护"


def load_faq() -> list[tuple[int, str]]:
    import openpyxl

    if not FAQ.exists():
        raise FileNotFoundError(f"FAQ 不存在：{FAQ}")
    wb = openpyxl.load_workbook(FAQ, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    out = []
    for row in list(ws.iter_rows(values_only=True))[1:]:
        if not row or not row[0]:
            continue
        q = str(row[0]).strip()
        m = re.match(r"\[频次[:：]\s*(\d+)\]\s*(.*)", q)
        freq, text = (int(m.group(1)), m.group(2).strip()) if m else (0, q)
        out.append((freq, text))
    return out


def classify(text: str) -> tuple[str, str]:
    """返回 (子场景 或 _排除_*, 命中的关键词)。"""
    for scenario, keywords in RULES:
        for kw in keywords:
            if kw in text:
                return scenario, kw
    return FALLBACK, "(兜底)"


# 判定一条提问是否**老年特异**：明确提到老人/老年，或属于老年高发慢病。
# 用于敏感性检验，不用于剔除。
ELDER_MARKERS = ["老人", "老年"]
ELDER_CHRONIC = ["高血压", "糖尿病", "冠心病", "血糖", "血压", "骨质疏松", "中风",
                 "哮喘", "关节", "痛风", "慢性肾", "帕金森", "阿尔茨海默", "褥疮",
                 "假牙", "白内障", "前列腺", "慢病"]


def is_elder_specific(text: str) -> bool:
    return any(k in text for k in ELDER_MARKERS + ELDER_CHRONIC)


def allocate(freq_by_scenario: dict[str, int]) -> dict[str, int]:
    """按真实频次份额分配题数，每类保底 FLOOR_PER_SCENARIO。

    保底的存在是因为父类必须全覆盖：权益匹配、复诊提醒这类在日志里近乎没有，
    但项目书第六栏列了它们，测试集不能空着。
    """
    n = len(SCENARIOS)
    base = {s: FLOOR_PER_SCENARIO for s in SCENARIOS}
    remaining = TOTAL_ITEMS - FLOOR_PER_SCENARIO * n
    total_freq = sum(freq_by_scenario.values()) or 1

    exact = {s: remaining * freq_by_scenario.get(s, 0) / total_freq for s in SCENARIOS}
    add = {s: int(exact[s]) for s in SCENARIOS}

    # 余数按小数部分从大到小补齐，保证总数**精确**等于 TOTAL_ITEMS。
    # 必须循环发放而不是只发一轮：当所有场景频次都为 0（或极度集中）时，
    # 余数会大于场景数，只发一轮补不满——早期版本在这种情况下总数是 192 而非 270。
    order = sorted(SCENARIOS, key=lambda s: exact[s] - int(exact[s]), reverse=True)
    left = remaining - sum(add.values())
    i = 0
    while left > 0:
        add[order[i % len(order)]] += 1
        left -= 1
        i += 1

    out = {s: base[s] + add[s] for s in SCENARIOS}
    assert sum(out.values()) == TOTAL_ITEMS, f"配比合计 {sum(out.values())} != {TOTAL_ITEMS}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-all", action="store_true", help="逐条打印归类结果")
    args = ap.parse_args()

    items = load_faq()
    total_freq = sum(f for f, _ in items)

    classified = []
    for freq, text in items:
        scenario, kw = classify(text)
        classified.append({"freq": freq, "text": text, "scenario": scenario, "matched": kw})

    kept = [c for c in classified if not c["scenario"].startswith("_排除_")]
    dropped = [c for c in classified if c["scenario"].startswith("_排除_")]

    freq_by = {}
    cnt_by = {}
    for c in kept:
        freq_by[c["scenario"]] = freq_by.get(c["scenario"], 0) + c["freq"]
        cnt_by[c["scenario"]] = cnt_by.get(c["scenario"], 0) + 1

    full_alloc = allocate(freq_by)

    # 敏感性检验：只用老年特异子集重算一遍配比。
    # 这份日志来自通用健康管家，并非老年专属，用它论证老年场景配比有先天局限。
    # 如果两套配比接近，说明结论对该局限不敏感；差得多则必须在文档里写明。
    elder_kept = [c for c in kept if is_elder_specific(c["text"])]
    elder_freq_by: dict[str, int] = {}
    for c in elder_kept:
        elder_freq_by[c["scenario"]] = elder_freq_by.get(c["scenario"], 0) + c["freq"]
    elder_alloc = allocate(elder_freq_by)

    # **主配比用老年子集**。全量日志来自通用健康管家（含差旅/职场/孕产等），
    # 人群不对；老年子集虽只有 43 条，但那 43 条对应 4560 次真实提问，且是本项目
    # 的实际受众。数据多但人群错 vs 数据少但人群对，后者才支撑得起
    # 「按真实老年需求分布配比」这句话。全量配比保留为稳健性参照。
    alloc = elder_alloc

    print(f"FAQ 共 {len(items)} 条，总频次 {total_freq}")
    print(f"剔除 {len(dropped)} 条（频次 {sum(c['freq'] for c in dropped)}），保留 {len(kept)} 条\n")
    print(f"{'子场景':<14}{'父类':<12}{'条数':>5}{'频次':>8}{'频次占比':>9}{'题数':>6}")
    print("-" * 60)
    kept_freq = sum(freq_by.values()) or 1
    for s, parent in SCENARIOS.items():
        print(f"{s:<14}{parent:<12}{cnt_by.get(s,0):>5}{freq_by.get(s,0):>8}"
              f"{100*freq_by.get(s,0)/kept_freq:>8.1f}%{alloc[s]:>6}")
    print("-" * 60)
    print(f"{'合计':<26}{len(kept):>5}{kept_freq:>8}{100:>8.1f}%{sum(alloc.values()):>6}")

    parents = {}
    for s, p in SCENARIOS.items():
        parents.setdefault(p, 0)
        parents[p] += alloc[s]
    print(f"\n敏感性检验（只用老年特异子集 {len(elder_kept)} 条 / 频次 "
          f"{sum(c['freq'] for c in elder_kept)}）：")
    print(f"{'子场景':<14}{'全量参照':>8}{'老年子集(采用)':>14}{'差值':>7}")
    print("-" * 44)
    max_delta = 0
    for s_ in SCENARIOS:
        d = elder_alloc[s_] - full_alloc[s_]
        max_delta = max(max_delta, abs(d))
        print(f"{s_:<14}{full_alloc[s_]:>8}{elder_alloc[s_]:>14}{d:>+7}")
    print("-" * 44)
    print(f"最大差值 {max_delta} 题" +
          ("  → 两套接近，结论对样本局限不敏感" if max_delta <= 8
           else "  → 差异显著，已采用老年子集为主依据（见 AUTHORING_SPEC §2）"))

    print(f"\n父类覆盖（项目书第六栏 8 类）：")
    for p, n in parents.items():
        print(f"  {p:<12}{n:>4} 题")
    assert len(parents) == 8, f"父类应为 8 个，实得 {len(parents)}"

    if args.show_all:
        print("\n逐条归类：")
        for c in sorted(classified, key=lambda c: -c["freq"]):
            print(f"  {c['freq']:>5}  {c['scenario']:<14} [{c['matched']}]  {c['text'][:44]}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "source": str(FAQ),
                "total_items": len(items),
                "total_freq": total_freq,
                "dropped": dropped,
                "kept": kept,
                "freq_by_scenario": freq_by,
                "count_by_scenario": cnt_by,
                "allocation": alloc,
                "allocation_basis": "elder_subset",
                "elder_subset_n": len(elder_kept),
                "elder_subset_freq": sum(c["freq"] for c in elder_kept),
                "full_log_allocation_reference": full_alloc,
                "scenario_to_parent": SCENARIOS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n归类结果写入 {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
