#!/usr/bin/env python3
"""从 test 池抽取候选源文本，按 04 的配比分配到 12 个子场景。

每个场景取哪几个库的哪几个字段，见 SCENARIO_SOURCES —— 全部对照索引里的**实际
列名**定，不是凭印象。

三条硬约束：
  1. 候选只能来自 test 池。与 verifier_train / reserve 的 record_id 零交集。
  2. 每条带完整 provenance（record_id + 库 + 逻辑表 + 字段名），可翻回原始记录。
  3. 筛选体现「面向银发群体」：按老年高发病、老年常用药、老年相关指标过滤，
     不是随机抽。

用法：
    python3 pipeline/extract_candidates.py            # 抽取 + 按配比抽样
    python3 pipeline/extract_candidates.py --stats    # 只看各场景候选池规模
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CORPUS = Path(
    "/Users/chenhao/DATA/03_国内医疗语料与数据源/2025国内医疗爬虫结果/2025国内医疗爬虫结果"
)
CORPUS = Path(os.environ.get("ELDER_CORPUS") or DEFAULT_CORPUS)
DB_PATH = ROOT / "index" / "corpus.sqlite"
CAND_DIR = ROOT / "candidates"
ALLOC_PATH = ROOT / "docs" / "faq_classification.json"

MIN_LEN, MAX_LEN = 40, 1200
SAMPLE_SEED = 20260810

# 场景 -> [(库, 逻辑表, [必需字段...])]。字段名逐一对照 index 的 files.columns。
SCENARIO_SOURCES: dict[str, list[tuple[str, str, list[str]]]] = {
    "检验报告解读": [
        ("ylys", "指标", ["指标含义", "指标正常值"]),
        ("ylys", "指标", ["指标偏高"]),
        ("ylys", "指标", ["指标偏低"]),
        ("bdyd", "检查", ["报告解读"]),
        ("bdyd", "操作", ["报告解读"]),
        ("dxys", "检查", ["结果解读"]),
        ("ylys", "检查", ["表示异常的结果"]),
        ("ylys", "检查", ["表示正常的结果"]),
    ],
    # ⚠️ 全盘无真实医嘱单/处方笺/出院小结，只能用疾病库的治疗与就医段模拟。
    # 产出一律带 simulated 标记，文档中表述为「治疗方案转译」。
    "医嘱转译": [
        ("bdyd", "疾病", ["治疗"]),
        ("ylys", "疾病", ["治疗"]),
        ("zsys", "disease_info", ["method"]),
        ("zsys", "disease_info", ["advice"]),
    ],
    "用药咨询": [
        ("others", "drug_info", ["dosage", "indication"]),
        ("ylys", "药品信息", ["用法用量", "适应症"]),
        ("bdyd", "药品", ["用法用量"]),
        ("bdyd", "药品", ["用前须知"]),
        ("mkss", "药品", ["drug_indications"]),
    ],
    "日常照护": [
        ("bdyd", "疾病", ["日常"]),
        ("bdyd", "症状", ["日常"]),
        ("ylys", "疾病", ["护理"]),
        ("zsys", "disease_info", ["self_care"]),
        ("dxys", "疾病", ["lifestyle"]),
    ],
    "急症处置": [
        ("bdyd", "急救", ["情况识别", "紧急处理"]),
        ("bdyd", "急救", ["紧急处理"]),
        ("ylys", "急救", ["症状识别", "救治方法"]),
        ("ylys", "急救", ["救治方法"]),
    ],
    "饮食营养": [
        ("ylys", "饮食指导", ["宜吃", "少吃"]),
        ("ylys", "饮食指导", ["慎吃"]),
        ("others", "diseaseKg", ["do_eat", "not_eat"]),
        ("ylys", "疾病", ["饮食"]),
        ("bdyd", "营养素激素", ["功能", "缺乏"]),
    ],
    "运动与养生": [
        ("bdyd", "穴位", ["功效作用", "生活保健"]),
        ("ylys", "穴位", ["穴位疗法"]),
        ("bdyd", "养生保健", ["实施方法", "注意事项"]),
        ("bdyd", "中医疾病", ["调理"]),
    ],
    # 辟谣类没有现成字段：取权威事实段作「正确答案」底座，谣言前提由撰写者
    # 在 ticket 06 依据指南构造。这是 12 类里源数据最间接的一类。
    "科普辟谣": [
        ("bdyd", "药品", ["用药贴士"]),
        ("bdyd", "疾病", ["预后"]),
        ("bdyd", "检查", ["检查须知"]),
    ],
    # ⚠️ 医保侧只有布尔位与粗估费用，无政策条款原文，是 12 类里源数据最薄的。
    # 单取 yibao_status+cost_money 只有 40 余字，作为待转译原文不成立，
    # 故并入治疗方式、疗程、常用药等字段，凑成一段有实质内容的「就医花费与
    # 保障」说明。这仍不是医保政策条款，thin_source 标记保留。
    "权益匹配": [
        ("others", "diseaseKg",
         ["yibao_status", "cost_money", "cure_way", "cure_lasttime", "check"]),
        ("xywy", "medical",
         ["medicalInsurance", "treatmentMethods", "treatmentCycle", "cureRate"]),
        ("ylys", "疾病", ["是否医保：", "治疗", "治疗周期："]),
    ],
    "分科导诊": [
        ("bdyd", "症状", ["就诊科室", "原因"]),
        ("bdyd", "疾病", ["就诊科室", "症状"]),
        ("xywy", "medical", ["clinicalDepartment", "symptoms"]),
        ("zsys", "disease_info", ["vis_dep", "cli_fea"]),
    ],
    "复诊与随访": [
        ("xywy", "medical", ["treatmentCycle", "treatmentMethods"]),
        ("others", "diseaseKg", ["cure_lasttime", "cure_way"]),
        ("zsys", "disease_info", ["best_time", "method"]),
    ],
    "用药干预": [
        ("others", "drug_info", ["taboo", "warning"]),
        ("others", "drug_info", ["interaction"]),
        ("others", "drug_info", ["side_effect"]),
        ("bdyd", "药品", ["禁忌"]),
        ("mkss", "药品", ["contraindication"]),
        ("ylys", "药品信息", ["不良反应"]),
        ("ylys", "药品信息", ["同类药对比"]),
    ],
}

SIMULATED_SCENARIOS = {"医嘱转译"}
THIN_SCENARIOS = {"权益匹配", "科普辟谣"}

# 急症处置用一套**专属**关键词，而不是整类豁免。
#
# 通用老年词表在急救表上几乎全不命中（「鼻出血急救」「异物卡喉急救」都不含
# 慢病名），但整类豁免又太宽——会把「断指急救」这种明显偏工伤的条目也抽进来。
# 折中：列出老年高发的急症类型。跌倒、噎食、体位性低血压晕厥、烫伤、低血糖
# 在老年人群的发生率与后果都显著高于一般人群，这份名单是有依据的，不是凑数。
EMERGENCY_ELDER_TERMS = [
    "跌倒", "摔", "骨折", "晕厥", "昏迷", "昏倒", "意识丧失", "神志",
    "中风", "脑卒中", "偏瘫", "口齿不清", "心梗", "心肌梗", "胸痛", "心绞痛",
    "心跳骤停", "心脏骤停", "呼吸困难", "呼吸骤停", "窒息", "噎", "异物", "误吸",
    "误服", "烫伤", "烧伤", "中暑", "低血糖", "高血糖", "昏迷", "出血", "咯血",
    "呕血", "便血", "鼻出血", "血压", "血糖", "药物中毒", "一氧化碳", "触电",
    "抽搐", "惊厥", "癫痫", "过敏", "休克", "脱水", "尿潴留", "便秘", "腹痛",
]
SCENARIO_TERMS = {"急症处置": EMERGENCY_ELDER_TERMS}

# 急症类型排除表。急救库里混着大量野外/工伤/职业暴露类急症，它们的**正文**会
# 提到「昏迷」「出血」「呼吸困难」而被老年词表捞进来，但类型本身与老年居家
# 场景无关（实测抽样里出现「断指急救」「海蛇咬伤急救法」「砷中毒急救法」
# 「高原反应处理方法」）。
#
# 该场景的候选池很薄（急救表全库 276 行，落进 test 池约 55 条），无法靠"只要
# 记录名命中"来筛，所以改为反向排除明显不相关的类型。这个池薄的事实已写进
# AUTHORING_SPEC 的诚实披露。
EXCLUDE_EMERGENCY = [
    "断指", "断肢", "离断",                      # 工伤
    # 「蜇/蛰」是异体字，记录里两种都用（"蜂蜇伤" vs "蝎子蛰伤"），必须都列
    "蛇咬", "蛇伤", "虫咬", "蜂蜇", "蜇伤", "蛰伤", "咬伤", "蝎", "蜈蚣", "蜘蛛",  # 野外
    "溺水", "淹溺", "高原", "潜水", "冻伤", "雪盲",
    "砷", "汞", "铅中毒", "氰化", "农药", "有机磷", "百草枯",  # 职业/自杀性中毒
    "触电", "雷击", "辐射", "核",
    "枪", "爆炸", "挤压", "贯穿", "刺入",
    "分娩", "产", "新生儿",
]


def is_excluded_emergency(name: str) -> bool:
    return any(k in name for k in EXCLUDE_EMERGENCY)

# 记录名天然不具判别性的场景：穴位叫「三阳络穴」「颊车穴」，名字里不可能出现
# 老年词，真正的信号在功效正文（该穴位主治失眠/高血压才是老年相关的证据）。
# 这类场景的抽样按正文命中取，不强求记录名命中。
NAME_MATCH_NOT_MEANINGFUL = {"运动与养生"}

# 老年导向筛选。命中记录名或正文任一即保留。
ELDER_DISEASES = [
    "高血压", "糖尿病", "冠心病", "心绞痛", "心肌梗", "心力衰竭", "心衰", "房颤",
    "心律失常", "脑梗", "脑卒中", "中风", "动脉硬化", "高脂血", "高血脂", "高尿酸",
    "痛风", "骨质疏松", "骨关节", "关节炎", "颈椎", "腰椎", "椎间盘", "白内障",
    "青光眼", "黄斑", "老年", "前列腺", "帕金森", "阿尔茨海默", "痴呆", "认知障碍",
    "慢阻肺", "肺气肿", "慢性支气管", "慢性肾", "肾功能", "便秘", "失眠", "耳鸣",
    "听力", "跌倒", "衰弱", "肌少", "褥疮", "压疮", "吞咽", "白内障", "疝",
    "甲状腺", "贫血", "骨折", "静脉曲张", "胃食管反流", "消化性溃疡", "脂肪肝",
]
ELDER_DRUGS = [
    "氨氯地平", "硝苯地平", "缬沙坦", "替米沙坦", "厄贝沙坦", "氯沙坦", "培哚普利",
    "美托洛尔", "比索洛尔", "呋塞米", "螺内酯", "氢氯噻嗪", "二甲双胍", "阿卡波糖",
    "格列", "西格列汀", "胰岛素", "阿司匹林", "氯吡格雷", "他汀", "阿托伐", "瑞舒伐",
    "华法林", "利伐沙班", "达比加群", "地高辛", "硝酸甘油", "单硝酸异山梨酯",
    "别嘌醇", "非布司他", "秋水仙碱", "碳酸钙", "骨化三醇", "阿仑膦酸",
    "多奈哌齐", "美金刚", "多巴丝肼", "左旋多巴", "坦索罗辛", "非那雄胺",
    "奥美拉唑", "泮托拉唑", "布地奈德", "沙丁胺醇", "噻托溴铵",
]
ELDER_INDICATORS = [
    "血压", "血糖", "糖化血红蛋白", "血脂", "胆固醇", "甘油三酯", "低密度脂蛋白",
    "高密度脂蛋白", "尿酸", "肌酐", "尿素氮", "肾小球滤过", "转氨酶", "胆红素",
    "骨密度", "血红蛋白", "血小板", "白细胞", "心电图", "心率", "尿蛋白",
    "前列腺特异性抗原", "甲状腺激素", "同型半胱氨酸",
    # 单字词会造成子串假命中（「钙」命中降钙素、「钠」命中头孢曲松钠），
    # 一律写成不会误配的完整形式。
    "血钾", "血钠", "血钙", "补钙", "低钾", "高钾", "低钠", "高钙",
]
ELDER_TERMS = ELDER_DISEASES + ELDER_DRUGS + ELDER_INDICATORS


# **人群排除**：老年词表匹配的是病名（贫血/糖尿病/低血糖），不区分人群，
# 于是「小儿α-地中海贫血」「早产儿贫血」「妊娠期糖尿病」这类记录也被抽了进来
# ——病名对得上，人群完全不对（实测 270 条抽样里污染 14 条 / 5.2%）。
# 凡记录名带明确的非老年人群标记，一律排除。
#
# 刻意**不排除**「先天性」「遗传性」：先天性甲减、遗传性血色病这类，八十岁的
# 老人一样带着，排除它们是另一种错误。
EXCLUDE_POPULATION = [
    "小儿", "婴儿", "新生儿", "新生", "早产", "儿童", "幼儿", "胎儿", "婴幼儿",
    "妊娠", "孕妇", "孕期", "怀孕", "哺乳", "产后", "产褥", "分娩", "围产",
    "青少年", "学龄", "青春期", "少儿",
    # 孕产相关但不含「孕/产妇」字样的漏网词（实测漏掉「胎盘早剥」「空腹血糖（产检）」）
    "胎盘", "子痫", "产检", "羊水", "宫缩", "娩", "催产", "保胎",
]


# 记录名不含人群标记、但正文通篇在讲儿科的情况。
# 实测漏网：「先天性甲状腺功能减退症」记录名**不该**排除——老人也带着这个病——
# 但它的正文是"宝宝喂药""对小婴儿…压碎后加奶服用"，纯儿科养育内容。
#
# 强弱标记分开，是因为一刀切会大量误伤：「重度贫血」「缺铁性贫血」的正文里都有
# 一句"关注婴幼儿、青少年、孕妇营养保健"，那是在列高危人群，整条记录仍是成人
# 适用的。实测一刀切 6 命中里 3 条是误伤。
#
# 判据（按实际正文归纳，非拍脑袋）：
#   - 出现育儿语（宝宝/小婴儿/家长/小朋友）→ 直接排除
#   - 「患儿」出现 ≥2 次 → 整段以儿科视角写成，排除
#   - 只在人群清单里提一句「婴幼儿」→ 保留
PEDIATRIC_STRONG = ["宝宝", "小婴儿", "小朋友", "家长"]
PEDIATRIC_BODY_HEAD = 160  # 只看开头，避免成人药品禁忌里提一句"儿童禁用"就被误杀


def is_pediatric_body(text: str) -> bool:
    head = text[:PEDIATRIC_BODY_HEAD]
    if any(k in head for k in PEDIATRIC_STRONG):
        return True
    return head.count("患儿") >= 2


def is_excluded_population(name: str, text: str = "") -> bool:
    if any(k in name for k in EXCLUDE_POPULATION):
        return True
    return bool(text) and is_pediatric_body(text)


def is_elder_relevant(
    name: str, text: str, terms: list[str] | None = None
) -> tuple[bool, str]:
    """记录名命中优先于正文命中——名字里就是老年常见项，质量明显高于正文偶然提及。"""
    terms = terms or ELDER_TERMS
    for kw in terms:
        if kw in name:
            return True, f"name:{kw}"
    for kw in terms:
        if kw in text:
            return True, f"text:{kw}"
    return False, ""


_LEX = None


def _source_readability(text: str) -> float | None:
    """原文的 strict 可读性。

    记进候选是为了让下游能报「净提升」：**31% 的原文本身已 ≥0.90**，这些题
    「照抄原文」即通过、对 KPI-1 零区分度。只报绝对达标率会高估转译能力，
    必须把原文基线一并留档（见 docs/AUTHORING_SPEC.md §5）。
    """
    global _LEX
    try:
        if _LEX is None:
            from metrics.readability import Lexicon

            _LEX = Lexicon.load()
        from metrics.readability import score

        return round(score(text, _LEX).rate_strict, 4)
    except Exception:
        return None


def clean(s: str) -> str:
    s = s or ""
    # 部分库把多值字段直接 str(list) 存了进来，原样出现在正文里：
    #   ['药物治疗', '支持性治疗']  ->  药物治疗、支持性治疗
    s = re.sub(
        r"\[\s*'([^']*)'(?:\s*,\s*'([^']*)')*\s*\]",
        lambda m: "、".join(re.findall(r"'([^']*)'", m.group(0))),
        s,
    )
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ \t　]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def load_table(db: sqlite3.Connection, lib: str, logical: str) -> list[dict]:
    row = db.execute(
        "SELECT source_file, encoding FROM files WHERE lib=? AND logical=? LIMIT 1",
        (lib, logical),
    ).fetchone()
    if not row:
        return []
    src, enc = row
    with open(CORPUS / src, encoding=enc, newline="") as f:
        return list(csv.DictReader(f))


def extract(db: sqlite3.Connection) -> dict[str, list[dict]]:
    test_pool = {
        rid for (rid,) in db.execute("SELECT record_id FROM splits WHERE pool='test'")
    }
    print(f"test 池 {len(test_pool)} 条 record_id\n")

    cache: dict[tuple[str, str], list[dict]] = {}
    out: dict[str, list[dict]] = {}

    for scenario, rules in SCENARIO_SOURCES.items():
        seen_rid: set[str] = set()
        cands: list[dict] = []
        for lib, logical, fields in rules:
            key = (lib, logical)
            if key not in cache:
                cache[key] = load_table(db, lib, logical)
            rows = cache[key]
            if not rows:
                print(f"  [warn] {lib}/{logical} 读不到", file=sys.stderr)
                continue
            name_col = next(
                (c for c in ("名称", "name", "title", "disease_name", "common_name")
                 if c in rows[0]), None
            )
            for i, row in enumerate(rows):
                rid = f"{lib}_{logical}#row{i}"
                if rid not in test_pool or rid in seen_rid:
                    continue
                parts = [clean(row.get(f, "")) for f in fields]
                if not all(parts):
                    continue
                text = "\n".join(parts)
                if not (MIN_LEN <= len(text) <= MAX_LEN):
                    continue
                name = clean(row.get(name_col, "")) if name_col else ""
                if is_excluded_population(name, text):
                    continue
                if scenario == "急症处置" and is_excluded_emergency(name):
                    continue
                ok, why = is_elder_relevant(name, text, SCENARIO_TERMS.get(scenario))
                if not ok:
                    continue
                seen_rid.add(rid)
                cands.append(
                    {
                        "scenario": scenario,
                        "record_id": rid,
                        "name": name,
                        "source_text": text,
                        "n_chars": len(text),
                        "elder_match": why,
                        "source_readability": _source_readability(text),
                        "provenance": {
                            "lib": lib,
                            "logical": logical,
                            "row_idx": i,
                            "fields": fields,
                            "record_id": rid,
                        },
                        **({"simulated": True,
                            "simulated_note": "无真实医嘱语料，源自疾病库治疗/就医段；"
                                              "对外表述为「治疗方案转译」"}
                           if scenario in SIMULATED_SCENARIOS else {}),
                        **({"thin_source": True} if scenario in THIN_SCENARIOS else {}),
                    }
                )
        out[scenario] = cands
    return out


def sample(cands: list[dict], n: int, scenario: str) -> list[dict]:
    """确定性抽样：按 record_id 哈希排序取前 n，同种子必得同结果。"""
    # 记录名命中的排在前面：「二甲双胍片」的用法用量，质量高于某个冷门药的
    # 说明里恰好提到了「糖尿病」。同一档内按哈希定序，保证确定性。
    ranked = sorted(
        cands,
        key=lambda c: (
            0 if c["elder_match"].startswith("name:") else 1,
            hashlib.sha256(
                f"{SAMPLE_SEED}:{scenario}:{c['record_id']}".encode()
            ).hexdigest(),
        ),
    )
    return ranked[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="只看候选池规模")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print("索引不存在，先跑 pipeline/source_index.py", file=sys.stderr)
        return 1
    if not ALLOC_PATH.exists():
        print("配比不存在，先跑 pipeline/classify_faq.py", file=sys.stderr)
        return 1

    alloc = json.loads(ALLOC_PATH.read_text(encoding="utf-8"))["allocation"]

    with sqlite3.connect(DB_PATH) as db:
        pools = extract(db)

    CAND_DIR.mkdir(exist_ok=True)

    # **跨场景全局去重。** 同一条源记录可能同时满足多个场景的字段要求
    # （bdyd/疾病 一行既有「治疗」也有「日常」也有「就诊科室」），若不去重，
    # 两道题会共享同一段原文，测试集的实际多样性低于表面题数。
    # 分配顺序按候选池从小到大——稀缺场景先占位，否则它们会被富裕场景抢空。
    taken: set[str] = set()
    picked_by: dict[str, list[dict]] = {}
    for scenario in sorted(SCENARIO_SOURCES, key=lambda s: len(pools[s])):
        avail = [c for c in pools[scenario] if c["record_id"] not in taken]
        chosen = sample(avail, alloc[scenario], scenario)
        taken.update(c["record_id"] for c in chosen)
        picked_by[scenario] = chosen

    print(f"{'子场景':<14}{'候选池':>7}{'需要':>6}{'余量':>7}  状态")
    print("-" * 52)
    short = []
    for scenario in SCENARIO_SOURCES:
        cands = pools[scenario]
        need = alloc[scenario]
        # 余量按**去重后实际抽到的**算，不是按候选池大小——两个场景抢同一批
        # 记录时，池子看着够、实际可能抽不满。
        got = len(picked_by[scenario])
        margin = len(cands) - need
        if got < need:
            short.append((scenario, got, need))
        status = "OK" if margin >= need else ("紧张" if margin >= 0 else "不足")
        print(f"{scenario:<14}{len(cands):>7}{need:>6}{margin:>7}  {status}")

        if not args.stats:
            with open(CAND_DIR / f"{scenario}.jsonl", "w", encoding="utf-8") as f:
                for c in cands:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            with open(CAND_DIR / f"sampled_{scenario}.jsonl", "w", encoding="utf-8") as f:
                for c in picked_by[scenario]:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print("-" * 52)
    total_c = sum(len(v) for v in pools.values())
    print(f"{'合计':<14}{total_c:>7}{sum(alloc.values()):>6}")

    if short:
        print("\n[候选不足] 以下场景抽不满，需放宽筛选或补数据源：", file=sys.stderr)
        for s, got, need in short:
            print(f"  {s}: 候选 {got}，需要 {need}", file=sys.stderr)
        return 1

    if not args.stats:
        print(f"\n候选与抽样结果写入 {CAND_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
