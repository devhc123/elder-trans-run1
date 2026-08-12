#!/usr/bin/env python3
"""P3：最小编辑对合成（ticket 12）。

**规则最小编辑，不许 LLM 自由重写**——这是 `LITERATURE.md` L1 的判读规则：
朴素混入 LLM 生成的 hard negatives 会带来源伪影（arXiv:2606.01304），模型
学会的是"这句话像机器改的"而不是"这句话没有原文依据"。规则最小编辑的做法：
只在源文本本来就出现"类别"标记词（如"XX类药物"）时才动手，往**真实答案**
后面追加一句"举例说是XX、YY"，不改写答案原有的任何一个字——这是 FactCC
系规则合成（arXiv:1910.12840）在本项目的对应实现，只是把"实体替换"换成了
"实体追加"，因为源头本来就没有具体实体可替换。

**只用 train 切分。** 合成产物不进 holdout，避免污染验收集。

用法：
    python3 verifier/synth_minimal_edit.py --build
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import CATEGORY_MARKERS as _ALL_CATEGORY_MARKERS  # noqa: E402
from verifier.redline_candidates import normalize, parse_case_text  # noqa: E402

WORK = ROOT / "verifier" / "work"

# **有意收窄的子集，不是要跟 build_teacher_labels.CATEGORY_MARKERS 完全一致。**
# 那份列表是给 bucket() 用的——只要全文任意位置出现任一标记词就归类，标记词
# 本身不需要局部语义成立。本模块不一样：要在标记词的局部上下文里**注入具体
# 药名**，所以标记词本身必须蕴含"这是一类药"，不能是"这是一类XX"的任意 XX。
# 实测过把全量 8 个标记词直接搬过来会怎样：命中数从 292 涨到 516，但新增的
# 220 余条里绝大多数（"等症状"单独就贡献 206 条）触发在症状/疾病/食物枚举上
# （如"男性化、性早熟等症状""高尔夫球、棒球之类"），往这些位置硬塞"这类药
# 常见的还有阿司匹林"读起来语义不通——这是比"合成产物质量差"更明确的信号，
# 不能要。所以只保留 4 个带"药"字、局部语义必然指向"一类药物"的标记词。
# `test_synth_minimal_edit.py` 里有一条测试断言这仍是 `_ALL_CATEGORY_MARKERS`
# 的子集——防止这份收窄的列表本身悄悄手抄跑偏（同一类 join_fields() 分叉的坑，
# 只是这次收窄是刻意的，子集断言确保"刻意"不会退化成"忘了同步"）。
CATEGORY_MARKERS = [m for m in _ALL_CATEGORY_MARKERS if "药" in m]

CONTEXT_WINDOW_CHARS = 15  # 标记词前取多少字当匹配类别关键词的上下文

# 类别关键词 -> 两个示例具体药名（不追求医学精确，只追求"具体、原文没有"）。
# 判分纪律本身不看医学对不对（TEACHER_TASK.md：医学正确但原文没说仍算触犯），
# 所以这份映射只需要"看起来是这个类别的具体成员"，不需要专业审校。
CATEGORY_EXAMPLES: dict[str, tuple[str, str]] = {
    "激素": ("泼尼松", "地塞米松"),
    "抗凝": ("华法林", "利伐沙班"),
    "降压": ("硝苯地平", "氨氯地平"),
    "降糖": ("二甲双胍", "阿卡波糖"),
    "抗血小板": ("阿司匹林", "氯吡格雷"),
    "抗生素": ("阿莫西林", "头孢氨苄"),
    "抗过敏": ("氯雷他定", "西替利嗪"),
    "镇痛": ("布洛芬", "对乙酰氨基酚"),
    "麻醉": ("吗啡", "哌替啶"),
    "磺胺": ("磺胺甲噁唑", "复方新诺明"),
    "阿片": ("吗啡", "可待因"),
    "精神": ("地西泮", "氯丙嗪"),
    "细胞毒": ("环磷酰胺", "甲氨蝶呤"),
    # 以下 18 条是 ticket 13 新增——来自 train 切分「兜底对占比」诊断
    # （ticket 12 P3：293 条合成里 76.5% 落在兜底对，扩类别关键词覆盖面前
    # 先统计了兜底案例的类别词根频率，只补高频根，不逐字照抄某条具体样本）。
    "黄嘌呤": ("氨茶碱", "茶碱"),
    "二氨基嘧啶": ("甲氧苄啶", "乙胺嘧啶"),
    "感冒": ("泰诺感冒片", "999感冒灵"),
    "大环内酯": ("红霉素", "阿奇霉素"),
    "磺脲": ("格列本脲", "格列齐特"),
    "磺酰脲": ("格列本脲", "格列齐特"),
    "四环素": ("多西环素", "米诺环素"),
    "喹诺酮": ("左氧氟沙星", "环丙沙星"),
    "他汀": ("阿托伐他汀", "瑞舒伐他汀"),
    "心肌重构": ("沙库巴曲缬沙坦", "螺内酯"),
    "硫酸镁": ("氧化镁", "碳酸镁"),
    "氨基糖苷": ("庆大霉素", "阿米卡星"),
    "头孢菌素": ("头孢氨苄", "头孢克肟"),
    "巴比妥": ("苯妥英钠", "扑米酮"),
    "双胍": ("苯乙双胍", "二甲双胍"),
    "硝酸酯": ("硝酸甘油", "单硝酸异山梨酯"),
    "洋地黄": ("地高辛", "西地兰"),
    "吩噻嗪": ("氯丙嗪", "奋乃静"),
}
DEFAULT_EXAMPLES = ("阿司匹林", "布洛芬")

TEMPLATES = [
    "（补充一句：像{a}、{b}这类药，也是您问的这一类里常见的。）",
    "（顺带说一句，{a}、{b}这些药也属于这一类，供您参考。）",
    "（这类药常见的还有{a}和{b}。）",
]


def find_category_trigger(source_text: str) -> str | None:
    """source_text 里第一处类别标记词的上下文窗口（标记词前 N 字 + 标记词本身）；
    没有类别标记就返回 None——没有触发信号就不合成，不是每条都能造。"""
    for marker in CATEGORY_MARKERS:
        idx = source_text.find(marker)
        if idx == -1:
            continue
        start = max(0, idx - CONTEXT_WINDOW_CHARS)
        return source_text[start: idx + len(marker)]
    return None


def pick_examples(context_window: str) -> tuple[str, str]:
    """按上下文窗口里出现的类别关键词挑示例药名，没匹配到就用兜底对。"""
    for kw, examples in CATEGORY_EXAMPLES.items():
        if kw in context_window:
            return examples
    return DEFAULT_EXAMPLES


def _template_for(case_id: str) -> str:
    """用无盐哈希选模板——必须跨进程确定性，内置 `hash()` 每次进程带随机盐
    做不到这一点。"""
    digest = hashlib.sha256(case_id.encode()).hexdigest()
    return TEMPLATES[int(digest, 16) % len(TEMPLATES)]


def synthesize_minimal_edit(record: dict) -> dict | None:
    """对一条 train 记录做最小编辑合成；没有类别触发信号，或挑出的示例
    恰好已经在原文出现（会变成"巧合为真"），返回 None。"""
    source_text, answer = parse_case_text(record["input"])
    context = find_category_trigger(source_text)
    if context is None:
        return None

    a, b = pick_examples(context)
    nsrc = normalize(source_text)
    if normalize(a) in nsrc or normalize(b) in nsrc:
        return None

    injected = _template_for(record["case_id"]).format(a=a, b=b)
    synthetic_answer = answer.rstrip() + "\n\n" + injected

    return {
        "case_id": f"{record['case_id']}-synth",
        "source_case_id": record["case_id"],
        "source_text": source_text,
        "original_answer": answer,
        "synthetic_answer": synthetic_answer,
        "injected_examples": [a, b],
        "category_context": context,
        "red_line_idx": 0,
        "verdict": "fail",
    }


def synthesize_all(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        synth = synthesize_minimal_edit(r)
        if synth is not None:
            out.append(synth)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--out", type=Path, default=WORK / "synth_minimal_edit.json")
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return 1

    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    synth = synthesize_all(records)
    print(f"train {len(records)} 条 -> 命中类别触发 {len(synth)} 条")

    # 按 case_id 排序保证输出确定性（synthesize_all 本身按输入顺序，这里
    # 再显式排一次防止未来改成并行/乱序遍历时悄悄破坏可复现性）。
    synth.sort(key=lambda s: s["case_id"])

    WORK.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(synth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
