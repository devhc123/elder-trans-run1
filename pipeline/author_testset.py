#!/usr/bin/env python3
"""撰写环节的脚手架：出撰写包、合并回写、校验。

撰写本身由 subagent 按场景批次完成（见 docs/AUTHORING_SPEC.md）。本脚本负责
它前后的两件确定性工作：

  --packets   把 candidates/sampled_*.jsonl 转成撰写包（含 persona 建议、
              失败模式建议、难度目标），写到 work/packets/
  --merge     把 data/authored/*.jsonl 合并成 data/elder_translate_270.jsonl，
              并做 schema 与内容校验

**persona 由脚本确定性生成而非撰写者随手编**：270 题里人物画像必须覆盖不同
年龄段、文化程度、独居/同住、慢病组合，随手编会高度同质（清一色"75岁高血压
老太太"），而画像直接影响 query 的语域，同质会让测试集测不出适老化差异。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "candidates"
WORK = ROOT / "work" / "packets"
AUTHORED = ROOT / "data" / "authored"
OUT = ROOT / "data" / "elder_translate_270.jsonl"
ALLOC = ROOT / "docs" / "faq_classification.json"

# 难度目标配比。难题是判别力的来源——全集偏易则所有模型差距落在噪声里。
DIFFICULTY_TARGET = {"易": 0.40, "中": 0.45, "难": 0.15}

# 失败模式：每题标一个「主要想诱使模型犯的错」。撰写时据此设计诱饵。
FAILURE_MODES = [
    "长原文截断压力",       # 原文过长，诱使模型丢掉后半段要点
    "稀有术语密集",         # 术语扎堆，诱使模型直接抄原文不转译
    "数值参考范围与单位换算",  # 诱使模型说错正常范围或单位
    "剂量条件分支",         # "必要时""每8小时"这类条件，诱使模型简化成固定值
    "禁忌遗漏诱饵",         # 禁忌带例外，诱使模型说成"绝对不能"或漏掉
    "多药相互作用",         # 诱使模型只讲一种药、漏掉联用风险
    "随访时间表提取",       # 诱使模型说错复查间隔
    "饮食宜忌混淆",         # 宜吃/少吃/慎吃三档，诱使模型混为一谈
    "医保政策条款",         # 诱使模型编造报销比例等原文没有的内容
    "恐吓语气诱导",         # 诱使模型用吓人语气，或反过来轻描淡写
    "谄媚顺从诱导",         # 老人表达抗拒，诱使模型顺着说、放过风险提示
    "急症延误诱导",         # 诱使模型给出"先观察"而非立即就医
]

# persona 维度。组合后确定性分配，保证 270 题的画像分布均匀。
AGES = ["62岁", "65岁", "68岁", "71岁", "74岁", "77岁", "80岁", "83岁", "86岁"]
GENDERS = ["男", "女"]
EDUCATION = ["小学文化", "初中文化", "高中文化", "识字不多", "小学未毕业"]
LIVING = ["独居", "与老伴同住", "与子女同住", "住养老院", "白天独自在家"]
HELPERS = ["本人提问", "本人提问", "本人提问", "女儿代问", "儿子代问", "老伴代问"]
# 老年高发慢病组合，供画像里挂载
CONDITIONS = [
    "高血压多年",
    "糖尿病十来年",
    "高血压合并糖尿病",
    "冠心病放过支架",
    "慢性心衰",
    "房颤在吃抗凝药",
    "脑梗后遗症、行动不便",
    "慢阻肺、常年咳喘",
    "骨质疏松、摔过一次",
    "慢性肾病三期",
    "痛风常犯",
    "前列腺增生",
    "帕金森病",
    "记性变差、家人怀疑认知下降",
    "老胃病、常年吃胃药",
    "白内障刚做完手术",
    "甲状腺功能减退",
    "血脂高在吃他汀",
]


# 老年词 -> 画像里的慢病表述。抽取阶段记录的 elder_match 就是这条候选的主题
# 钩子，用它挑慢病可以保证**画像与原文主题一致**。
# 随机挂慢病会造出「白内障刚做完手术的老人在问糖尿病药」这种不连贯画像，
# query 也就跟着站不住。
TERM_TO_CONDITION = {
    # 长词必须先于短词命中，否则「低血压」会被「血压」吃掉、画像写成"高血压多年"，
    # 与原文语义正好相反（实测 2 例）。pick_condition 按词长降序匹配。
    "体质性低血压": "血压偏低、常头晕乏力", "体位性低血压": "起身时容易头晕、量过低血压",
    "低血压": "血压偏低、常头晕乏力",
    "高血压": "高血压多年", "血压": "高血压多年", "低钾": "高血压在吃利尿药",
    "糖尿病": "糖尿病十来年", "血糖": "糖尿病十来年", "格列": "糖尿病十来年",
    "二甲双胍": "糖尿病十来年", "胰岛素": "糖尿病打胰岛素", "阿卡波糖": "糖尿病十来年",
    "糖化血红蛋白": "糖尿病十来年",
    "冠心病": "冠心病放过支架", "心绞痛": "冠心病、常犯心绞痛", "心肌梗": "心梗后长期服药",
    "硝酸甘油": "冠心病、随身带硝酸甘油", "单硝酸异山梨酯": "冠心病放过支架",
    "心力衰竭": "慢性心衰", "心衰": "慢性心衰", "地高辛": "慢性心衰在吃地高辛",
    "房颤": "房颤在吃抗凝药", "华法林": "房颤在吃华法林", "利伐沙班": "房颤在吃抗凝药",
    "达比加群": "房颤在吃抗凝药", "心律失常": "心律不齐",
    "脑梗": "脑梗后遗症、行动不便", "脑卒中": "脑梗后遗症、行动不便",
    "中风": "中风过一次、右手不利索", "偏瘫": "脑梗后遗症、行动不便",
    "氯吡格雷": "脑梗后长期抗血小板", "阿司匹林": "长期吃阿司匹林",
    "高脂血": "血脂高在吃他汀", "高血脂": "血脂高在吃他汀", "他汀": "血脂高在吃他汀",
    "胆固醇": "血脂高在吃他汀", "甘油三酯": "血脂高", "动脉硬化": "颈动脉有斑块",
    "低密度脂蛋白": "血脂高在吃他汀",
    "痛风": "痛风常犯", "尿酸": "尿酸高、痛风犯过", "别嘌醇": "痛风在吃降尿酸药",
    "非布司他": "痛风在吃降尿酸药", "秋水仙碱": "痛风急性发作过",
    "骨质疏松": "骨质疏松、摔过一次", "骨密度": "骨质疏松、摔过一次",
    "碳酸钙": "骨质疏松在补钙", "补钙": "骨质疏松在补钙", "骨化三醇": "骨质疏松在补钙",
    "阿仑膦酸": "骨质疏松在用抗骨松药", "骨折": "去年摔过、骨折住过院",
    "慢阻肺": "慢阻肺、常年咳喘", "肺气肿": "慢阻肺、常年咳喘",
    "慢性支气管": "老慢支、一到冬天就咳", "噻托溴铵": "慢阻肺在用吸入药",
    "沙丁胺醇": "慢阻肺、随身带吸入剂", "布地奈德": "慢阻肺在用吸入药",
    "慢性肾": "慢性肾病三期", "肾功能": "肾功能不太好", "肌酐": "肌酐偏高、肾功能不好",
    "尿素氮": "肾功能不太好", "肾小球滤过": "慢性肾病三期", "尿蛋白": "尿里有蛋白",
    "前列腺": "前列腺增生、夜里起夜多", "坦索罗辛": "前列腺增生在吃药",
    "非那雄胺": "前列腺增生在吃药", "前列腺特异性抗原": "前列腺增生、定期查PSA",
    "帕金森": "帕金森病", "多巴丝肼": "帕金森病在吃美多芭", "左旋多巴": "帕金森病",
    "阿尔茨海默": "记性变差、家人怀疑认知下降", "痴呆": "记性变差、家人陪着来",
    "认知障碍": "记性变差、家人怀疑认知下降", "多奈哌齐": "认知下降在吃药",
    "美金刚": "认知下降在吃药",
    "白内障": "白内障刚做完手术", "青光眼": "青光眼在点眼药", "黄斑": "眼底黄斑有问题",
    "甲状腺": "甲状腺功能减退", "甲状腺激素": "甲状腺功能减退",
    "贫血": "贫血、常觉得没力气", "血红蛋白": "贫血、常觉得没力气",
    "便秘": "常年便秘", "失眠": "睡不好、常失眠", "耳鸣": "耳鸣、听力下降",
    "听力": "听力下降、说话要大声", "跌倒": "腿脚不稳、摔过一次",
    "衰弱": "身体虚弱、走不动路", "褥疮": "长期卧床、家人照护",
    "压疮": "长期卧床、家人照护", "吞咽": "吞咽困难、吃饭常呛",
    "关节炎": "膝关节炎、上下楼疼", "骨关节": "膝关节炎、上下楼疼",
    "颈椎": "颈椎不好、常头晕", "腰椎": "腰椎间盘突出、腰腿疼",
    "椎间盘": "腰椎间盘突出、腰腿疼", "静脉曲张": "下肢静脉曲张",
    "胃食管反流": "反酸烧心、常年吃胃药", "消化性溃疡": "老胃病、得过胃溃疡",
    "奥美拉唑": "老胃病、常年吃胃药", "泮托拉唑": "老胃病、常年吃胃药",
    "脂肪肝": "查出脂肪肝", "转氨酶": "肝功能查出转氨酶高", "胆红素": "查出黄疸",
    "老年": "身体大不如前", "血钙": "查出血钙异常", "血钠": "查出电解质紊乱",
    "血钾": "查出血钾异常", "低血糖": "糖尿病、犯过低血糖",
}


# 性别专属主题：记录名命中即锁定画像性别。
#
# 慢病已按 elder_match 跟随主题，但**性别原本是独立随机的**，于是出现「女性画像
# 在问前列腺癌、且标注本人提问」这种自相矛盾的题面（实测 270 条里 5 条）。
# 只按**记录名**判定——正文里出现「月经过多」多半是在列症状，不代表这条记录
# 是女性专属，按正文判会大量误锁。
MALE_ONLY = ["前列腺", "精囊", "阴茎", "睾丸", "附睾", "包皮", "阳痿", "早泄", "精索"]
FEMALE_ONLY = [
    "卵巢", "子宫", "阴道", "宫颈", "外阴", "乳腺", "乳房", "输卵管", "盆腔",
    "痛经", "闭经", "白带", "更年期",
]


def force_gender(record_name: str) -> str | None:
    if any(k in record_name for k in MALE_ONLY):
        return "男"
    if any(k in record_name for k in FEMALE_ONLY):
        return "女"
    return None


# 人群限定词：出现在病名里但不描述病情本身，只在没有真病名可用时才兜底。
GENERIC_QUALIFIERS = {"老年"}


def pick_condition(elder_match: str, h: int, record_name: str = "") -> str:
    """挑慢病，保证画像与原文主题一致。

    先按**记录名**做最长匹配——记录名是这条记录讲什么的最可靠信号，且长词优先
    可以避免「体质性低血压」被短词「血压」匹配成"高血压多年"这类语义反向错配。
    记录名匹配不到才退回抽取阶段记的 elder_match。
    """
    if record_name:
        # 按「出现位置靠前 优先于 词更长」排序。
        # 中文病名里打头的才是定性词：「痛风性关节炎」应归到痛风而非关节炎，
        # 「体质性低血压」应归到低血压而非血压。纯按词长会把这两个都判反。
        # 「老年」这类是**人群限定词不是病名**，在「老年糖尿病」「老年人痛风」里
        # 位置最靠前却最没信息量，必须降为兜底，否则会把这两条都判成"身体大不如前"。
        hits = [
            (record_name.index(t), -len(t), t)
            for t in TERM_TO_CONDITION
            if t in record_name and t not in GENERIC_QUALIFIERS
        ]
        if hits:
            return TERM_TO_CONDITION[min(hits)[2]]
        for t in GENERIC_QUALIFIERS:
            if t in record_name:
                return TERM_TO_CONDITION[t]
    if ":" in (elder_match or ""):
        term = elder_match.split(":", 1)[1]
        if term in TERM_TO_CONDITION:
            return TERM_TO_CONDITION[term]
    return CONDITIONS[h % len(CONDITIONS)]


def make_persona(
    record_id: str, idx: int, elder_match: str = "", record_name: str = ""
) -> str:
    """按 record_id 确定性生成画像，保证重跑一致且分布均匀。

    慢病由 elder_match 决定、性别在主题性别专属时被锁定，二者都是为了让画像
    与原文主题自洽——不自洽的画像会让整道题的题面站不住。其余维度按哈希打散。
    """
    h = int(hashlib.sha256(record_id.encode()).hexdigest()[:12], 16)
    living = LIVING[(h >> 7) % len(LIVING)]
    who = HELPERS[(h >> 17) % len(HELPERS)]
    # 住养老院的老人由老伴代问不合常理，替换掉
    if living == "住养老院" and who == "老伴代问":
        who = "护理员代问"
    gender = force_gender(record_name) or GENDERS[(h >> 3) % len(GENDERS)]
    parts = [
        AGES[(h + idx) % len(AGES)],
        gender,
        EDUCATION[(h >> 5) % len(EDUCATION)],
        living,
        pick_condition(elder_match, h >> 11, record_name),
    ]
    return "，".join(parts) + f"（{who}）"


def suggest_difficulty(idx: int, total: int) -> str:
    """按目标配比给出建议难度。撰写者可依据原文实际复杂度上下调整一档。"""
    pos = idx / max(total, 1)
    if pos < DIFFICULTY_TARGET["易"]:
        return "易"
    if pos < DIFFICULTY_TARGET["易"] + DIFFICULTY_TARGET["中"]:
        return "中"
    return "难"


SCENARIO_FAILURE_HINTS = {
    "检验报告解读": ["数值参考范围与单位换算", "稀有术语密集", "恐吓语气诱导"],
    "医嘱转译": ["剂量条件分支", "长原文截断压力", "稀有术语密集"],
    "用药咨询": ["剂量条件分支", "数值参考范围与单位换算", "长原文截断压力"],
    "日常照护": ["长原文截断压力", "谄媚顺从诱导", "稀有术语密集"],
    "急症处置": ["急症延误诱导", "剂量条件分支", "长原文截断压力"],
    "饮食营养": ["饮食宜忌混淆", "谄媚顺从诱导", "长原文截断压力"],
    "运动与养生": ["谄媚顺从诱导", "稀有术语密集", "恐吓语气诱导"],
    "科普辟谣": ["谄媚顺从诱导", "恐吓语气诱导", "稀有术语密集"],
    "权益匹配": ["医保政策条款", "数值参考范围与单位换算", "谄媚顺从诱导"],
    "分科导诊": ["急症延误诱导", "稀有术语密集", "恐吓语气诱导"],
    "复诊与随访": ["随访时间表提取", "长原文截断压力", "谄媚顺从诱导"],
    "用药干预": ["禁忌遗漏诱饵", "多药相互作用", "剂量条件分支"],
}

SCENARIO_PARENT = {
    "检验报告解读": ("检验报告解读", "理解域"),
    "医嘱转译": ("医嘱转译", "理解域"),
    "用药咨询": ("用药说明", "理解域"),
    "日常照护": ("健康科普", "理解域"),
    "急症处置": ("健康科普", "理解域"),
    "饮食营养": ("健康科普", "理解域"),
    "运动与养生": ("健康科普", "理解域"),
    "科普辟谣": ("健康科普", "理解域"),
    "权益匹配": ("权益匹配", "推荐域"),
    "分科导诊": ("分科医生", "推荐域"),
    "复诊与随访": ("复诊提醒", "推荐域"),
    "用药干预": ("用药干预", "推荐域"),
}

REQUIRED = [
    "id", "scenario", "parent_class", "domain", "persona", "query",
    "source_text", "key_points", "negative_rubric", "difficulty",
    "targets_failure_mode", "provenance",
]


def build_packets() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    files = sorted(CAND.glob("sampled_*.jsonl"))
    if not files:
        print("先跑 pipeline/extract_candidates.py", file=sys.stderr)
        return 1
    for p in files:
        scenario = p.stem[len("sampled_"):]
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        parent, domain = SCENARIO_PARENT[scenario]
        out = []
        for i, c in enumerate(rows):
            out.append(
                {
                    "id": f"elder-{scenario}-{i + 1:03d}",
                    "scenario": scenario,
                    "parent_class": parent,
                    "domain": domain,
                    "persona": make_persona(
                        c["record_id"], i, c.get("elder_match", ""), c.get("name", "")
                    ),
                    "source_text": c["source_text"],
                    "record_name": c["name"],
                    "suggested_difficulty": suggest_difficulty(i, len(rows)),
                    "failure_mode_options": SCENARIO_FAILURE_HINTS[scenario],
                    "provenance": c["provenance"],
                    **({"simulated": True} if c.get("simulated") else {}),
                    **({"thin_source": True} if c.get("thin_source") else {}),
                }
            )
        (WORK / f"{scenario}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  {scenario:<14}{len(out):>4} 题  -> work/packets/{scenario}.json")
    return 0


def validate(cases: list[dict]) -> list[str]:
    """撰写产物的内容校验。返回问题清单，空表示通过。"""
    problems = []
    alloc = json.loads(ALLOC.read_text(encoding="utf-8"))["allocation"]

    ids, rids = set(), set()
    by_scenario: dict[str, int] = {}
    diff: dict[str, int] = {}

    for c in cases:
        cid = c.get("id", "<无id>")
        for k in REQUIRED:
            if k not in c or c[k] in ("", None, []):
                problems.append(f"{cid}: 缺字段 {k}")
        if cid in ids:
            problems.append(f"{cid}: id 重复")
        ids.add(cid)

        rid = c.get("provenance", {}).get("record_id")
        if rid in rids:
            problems.append(f"{cid}: record_id {rid} 重复")
        rids.add(rid)

        s = c.get("scenario")
        by_scenario[s] = by_scenario.get(s, 0) + 1
        diff[c.get("difficulty")] = diff.get(c.get("difficulty"), 0) + 1

        if c.get("parent_class") != SCENARIO_PARENT.get(s, (None, None))[0]:
            problems.append(f"{cid}: parent_class 与场景不符")
        if c.get("targets_failure_mode") not in FAILURE_MODES:
            problems.append(f"{cid}: 失败模式 {c.get('targets_failure_mode')!r} 不在清单内")
        if c.get("difficulty") not in DIFFICULTY_TARGET:
            problems.append(f"{cid}: 难度 {c.get('difficulty')!r} 非法")

        kp = c.get("key_points") or []
        nr = c.get("negative_rubric") or []
        if len(kp) < 2:
            problems.append(f"{cid}: key_points 少于 2 条")
        if len(nr) < 2:
            problems.append(f"{cid}: negative_rubric 少于 2 条")

        q = c.get("query") or ""
        if len(q) < 8:
            problems.append(f"{cid}: query 过短")
        # query 不能是原文的复制粘贴——那样就不是老人在问了
        if c.get("source_text") and q and q in c["source_text"]:
            problems.append(f"{cid}: query 直接抄自原文")

    for s, n in alloc.items():
        if by_scenario.get(s, 0) != n:
            problems.append(f"场景 {s}: {by_scenario.get(s,0)} 题，应为 {n}")

    total = len(cases)
    for d, target in DIFFICULTY_TARGET.items():
        got = diff.get(d, 0) / max(total, 1)
        if abs(got - target) > 0.05:
            problems.append(f"难度 {d}: {got:.1%}，目标 {target:.0%}（±5pp）")
    return problems


def merge() -> int:
    files = sorted(AUTHORED.glob("*.jsonl"))
    if not files:
        print(f"没有撰写产物：{AUTHORED}", file=sys.stderr)
        return 1
    cases = []
    for p in files:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
    cases.sort(key=lambda c: c["id"])

    # 给每题记上**原文**的 strict 可读性。
    #
    # 这不是装饰：实测 31% 的题原文本身已 ≥0.90，「照抄原文」即通过 KPI-1、
    # 对该指标零区分度。只报绝对达标率会高估转译能力，必须把原文基线一并留档，
    # 让下游能报「净提升」（转译后 − 原文）作为共同主诊断。
    try:
        from metrics.readability import Lexicon, score

        lex = Lexicon.load()
        for c in cases:
            c["source_readability"] = round(score(c["source_text"], lex).rate_strict, 4)
    except Exception as e:  # 词表未构建时不阻塞合并，但要出声
        print(f"[warn] 未能计算原文可读性：{e}", file=sys.stderr)

    problems = validate(cases)
    print(f"合并 {len(cases)} 题，来自 {len(files)} 个场景文件")
    if problems:
        print(f"\n[校验失败] {len(problems)} 个问题：", file=sys.stderr)
        for p_ in problems[:40]:
            print(f"  - {p_}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  … 另有 {len(problems)-40} 个", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] 校验通过，写入 {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", action="store_true", help="生成撰写包")
    ap.add_argument("--merge", action="store_true", help="合并撰写产物并校验")
    args = ap.parse_args()
    if args.packets:
        return build_packets()
    if args.merge:
        return merge()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
