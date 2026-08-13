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
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.build_teacher_labels import CATEGORY_MARKERS as _ALL_CATEGORY_MARKERS  # noqa: E402
from verifier.build_teacher_labels import CONDITIONAL_MARKERS as _ALL_CONDITIONAL_MARKERS  # noqa: E402
from verifier.redline_candidates import (  # noqa: E402
    CN_NUMERAL_RE,
    DIGIT_RE,
    normalize,
    parse_case_text,
)

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

# 类别关键词 -> 该类别的真实成员药名（不追求医学精确，只追求"具体、原文
# 没有"）。判分纪律本身不看医学对不对（TEACHER_TASK.md：医学正确但原文
# 没说仍算触犯），所以这份映射只需要"看起来是这个类别的具体成员"，不需要
# 专业审校。**每条 2-4 个**（L4 决断：单案例注入 3-4 个成员方向安全，
# 但列表本身不能超过 4——更长的枚举句读起来不自然，反而更像合成伪影）。
CATEGORY_EXAMPLES: dict[str, tuple[str, ...]] = {
    "激素": ("泼尼松", "地塞米松", "氢化可的松", "甲泼尼龙"),
    "抗凝": ("华法林", "利伐沙班", "达比加群酯"),
    "降压": ("硝苯地平", "氨氯地平", "缬沙坦", "厄贝沙坦"),
    "降糖": ("阿卡波糖", "格列美脲", "西格列汀"),
    "抗血小板": ("阿司匹林", "氯吡格雷", "替格瑞洛"),
    "抗生素": ("阿莫西林", "头孢氨苄", "阿奇霉素"),
    "抗过敏": ("氯雷他定", "西替利嗪", "氯苯那敏"),
    "镇痛": ("布洛芬", "对乙酰氨基酚", "双氯芬酸钠"),
    "麻醉": ("吗啡", "哌替啶", "芬太尼"),
    "磺胺": ("磺胺甲噁唑", "复方新诺明", "柳氮磺吡啶"),
    "阿片": ("吗啡", "可待因", "羟考酮"),
    "精神": ("地西泮", "氯丙嗪", "奥氮平"),
    "细胞毒": ("环磷酰胺", "甲氨蝶呤", "阿糖胞苷"),
    # 以下条目是 ticket 12/14 累计新增——来自 train 切分「兜底对占比」诊断
    # （ticket 12 P3：293 条合成里 76.5% 落在兜底对，扩类别关键词覆盖面前
    # 先统计了兜底案例的类别词根频率，只补高频根，不逐字照抄某条具体样本）。
    "黄嘌呤": ("氨茶碱", "茶碱", "多索茶碱"),
    "二氨基嘧啶": ("甲氧苄啶", "乙胺嘧啶"),
    "感冒": ("泰诺感冒片", "999感冒灵", "白加黑"),
    "大环内酯": ("红霉素", "阿奇霉素", "克拉霉素"),
    "磺脲": ("格列本脲", "格列齐特", "格列吡嗪"),
    "磺酰脲": ("格列本脲", "格列齐特", "格列吡嗪"),
    "四环素": ("多西环素", "米诺环素", "土霉素"),
    "喹诺酮": ("左氧氟沙星", "环丙沙星", "莫西沙星"),
    "他汀": ("阿托伐他汀", "瑞舒伐他汀", "辛伐他汀"),
    "心肌重构": ("沙库巴曲缬沙坦", "螺内酯"),
    "硫酸镁": ("氧化镁", "碳酸镁"),
    "氨基糖苷": ("庆大霉素", "阿米卡星", "妥布霉素"),
    "头孢菌素": ("头孢氨苄", "头孢克肟", "头孢曲松"),
    "巴比妥": ("苯妥英钠", "扑米酮"),
    "双胍": ("苯乙双胍", "二甲双胍"),
    "硝酸酯": ("硝酸甘油", "单硝酸异山梨酯"),
    "洋地黄": ("地高辛", "西地兰"),
    "吩噻嗪": ("氯丙嗪", "奋乃静", "三氟拉嗪"),
}
DEFAULT_EXAMPLES = ("阿司匹林", "布洛芬", "对乙酰氨基酚")

TEMPLATES = [
    "（补充一句：像{names}这类药，也是您问的这一类里常见的。）",
    "（顺带说一句，{names}这些药也属于这一类，供您参考。）",
    "（这类药常见的还有{names}。）",
]

# 每个模板的括注前缀——`candidate_pool.STRUCTURAL_NEGATIVE_FILLERS` 必须
# 复用这几个前缀（不能自己另起一套），否则打掉"括注即违规"捷径的负例
# 跟正例用的不是同一批模板标记，退化分类器只需要多认几个新前缀就绕过去。
# 这里是唯一定义处，`candidate_pool.py` 和它的测试都从这里导入，不再
# 各自手抄一份字面量元组（第三份独立副本曾经导致"改了 TEMPLATES 却忘了
# 同步测试里的硬编码副本"这类静默漂移，`/code-review` 发现）。
TEMPLATE_PREFIXES = ("（补充一句：", "（顺带说一句，", "（这类")
assert all(t.startswith(TEMPLATE_PREFIXES) for t in TEMPLATES), (
    "TEMPLATES 改了措辞但 TEMPLATE_PREFIXES 没同步——结构性负例的括注前缀会跟正例脱节"
)

# 数字/频次触发标记——**有意收窄的子集**，不是要跟 build_teacher_labels
# .CONDITIONAL_MARKERS 完全一致（同 CATEGORY_MARKERS 那次收窄，见上方注释
# 的教训）。只保留局部语义必然蕴含"这里没给具体数字"的四个词；排除
# "如果""若""根据""以上""以下"——这些是通用连接词，不蕴含"缺具体数字"，
# 往它们的局部上下文硬塞一个编造数字同样会读起来语义不通。
NUMERIC_TRIGGER_MARKERS = [
    m for m in _ALL_CONDITIONAL_MARKERS if m in {"必要时", "视情况", "酌情", "遵医嘱"}
]

# 确定性挑选一条不与原文数字冲突的编造数字/频次说明——每条只编一个具体值，
# 保持"最小编辑"的克制（不像实体注入可以合理地一次列 3-4 个同类成员，
# 编造的复查/服药频次通常只有一个，堆多个反而不像人话）。
NUMERIC_INJECTIONS = [
    "一般建议每3天复查一次",
    "通常一天吃2次，一次1片",
    "间隔至少2小时再服用",
    "建议每半年复查一次相关指标",
    "一般连续用药不超过7天",
    "两次用药中间要隔4小时以上",
    "建议连服5天后复诊",
    "一般每次用量不超过2片",
]


_MARKER_RE_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _marker_alternation(markers: list[str]) -> re.Pattern:
    """按长度降序编译标记词的正则交替——长标记词优先匹配，防止"类药"抢在
    "类药物"前面吃掉后两个字，导致上下文窗口的标记词本身被错误截短。"""
    key = tuple(markers)
    if key not in _MARKER_RE_CACHE:
        alt = "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True))
        _MARKER_RE_CACHE[key] = re.compile(alt)
    return _MARKER_RE_CACHE[key]


def _resolve_category_key(context_window: str) -> str | None:
    """context_window 命中了 CATEGORY_EXAMPLES 的哪个 key；没命中返回 None
    （调用方据此决定是否落到兜底对）。"""
    for kw in CATEGORY_EXAMPLES:
        if kw in context_window:
            return kw
    return None


def find_category_triggers(source_text: str) -> list[str]:
    """source_text 里**全部**类别标记词出现位置的上下文窗口，按解析出的
    类别关键词去重——同一类别提两次只出一条（避免合成出内容几乎相同的
    重复样本），不同类别各出一条（L4 决断 (b)：多标记点各自合成）。
    没有任何类别标记就返回空列表。"""
    marker_re = _marker_alternation(CATEGORY_MARKERS)
    seen_keys: set[str | None] = set()
    out: list[str] = []
    for m in marker_re.finditer(source_text):
        start = max(0, m.start() - CONTEXT_WINDOW_CHARS)
        context = source_text[start: m.end()]
        key = _resolve_category_key(context)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(context)
    return out


# 每个"词"（不是每个类别！）只属于 train 或 holdout 一侧——独立第二意见
# 代码审计发现：train 的合成正例和 holdout 的对抗子集共用同一份固定
# CATEGORY_EXAMPLES，实测注入词汇 100% 重叠。第一版修复按"每个类别自己
# 的最后一个成员留给 holdout"分，结果还是有 2 个词漏网（"对乙酰氨基酚"
# 在"镇痛"类别是 train 成员、在 DEFAULT_EXAMPLES 兜底组又是 holdout 成员；
# "阿奇霉素"同理跨"抗生素"/"大环内酯"两个类别）——同一个具体药名会在多个
# 类别的示例列表里重复出现，按类别内部拆分防不住跨类别撞车。
#
# 改成先给**每个全局唯一的词**定死归属，再按类别取子集：贪心处理每个
# 类别，从后往前找一个"还没被别的类别定成 train"的成员当这个类别的
# holdout 代表；找不到（极端情况——这个类别的所有成员都已被更早处理的
# 类别定成 train）就退化为用最后一个成员（可能产生残余冲突，用测试兜底，
# 实测全部 32 组类别+兜底组都能找到不冲突的分配，不需要这条退化路径）。
def _build_word_pools() -> dict[str, str]:
    assignment: dict[str, str] = {}
    groups = list(CATEGORY_EXAMPLES.items()) + [("__default__", DEFAULT_EXAMPLES)]
    for _, members in groups:
        holdout_word = next((w for w in reversed(members) if assignment.get(w) != "train"), None)
        if holdout_word is None:
            holdout_word = members[-1]
        assignment[holdout_word] = "holdout"
        for w in members:
            if w != holdout_word and w not in assignment:
                assignment[w] = "train"
    return assignment


_WORD_POOL_ASSIGNMENT = _build_word_pools()


def _train_pool(members: tuple[str, ...]) -> tuple[str, ...]:
    pool = tuple(w for w in members if _WORD_POOL_ASSIGNMENT.get(w) != "holdout")
    return pool or members[:1]


def _holdout_pool(members: tuple[str, ...]) -> tuple[str, ...]:
    pool = tuple(w for w in members if _WORD_POOL_ASSIGNMENT.get(w) == "holdout")
    return pool or members[-1:]


def pick_examples(context_window: str, *, holdout: bool = False) -> tuple[str, ...]:
    """按上下文窗口里出现的类别关键词挑示例药名，没匹配到就用兜底组。
    `holdout=True` 时只从每个类别留给验收用的那部分成员里选，与 train
    用的成员在**全局词表级别**互不重叠（见上方注释）。"""
    key = _resolve_category_key(context_window)
    members = CATEGORY_EXAMPLES[key] if key is not None else DEFAULT_EXAMPLES
    return _holdout_pool(members) if holdout else _train_pool(members)


def find_numeric_trigger(source_text: str) -> str | None:
    """source_text 里第一处"缺具体数字"标记词的上下文窗口；没有就返回
    None。红线2的触发只取第一处——数字类编造不像药物类别那样天然对应
    "这案例提到了几种不同类别"，同一案例内多次"必要时"通常指向同一件事，
    多出几条边际信息量很低，不值得为此再引入去重逻辑。"""
    for marker in NUMERIC_TRIGGER_MARKERS:
        idx = source_text.find(marker)
        if idx == -1:
            continue
        start = max(0, idx - CONTEXT_WINDOW_CHARS)
        return source_text[start: idx + len(marker)]
    return None


def _template_for(seed_key: str) -> str:
    """用无盐哈希选模板——必须跨进程确定性，内置 `hash()` 每次进程带随机盐
    做不到这一点。"""
    digest = hashlib.sha256(seed_key.encode()).hexdigest()
    return TEMPLATES[int(digest, 16) % len(TEMPLATES)]


# 同 CATEGORY_EXAMPLES 的 train/holdout 拆分理由：数字类注入也不能让
# train 和 holdout 对抗子集共用同一份固定短语。
_NUMERIC_HOLDOUT_RESERVED = 3  # 最后 3 条只给 holdout 用，其余给 train


def _pick_numeric_injection(seed_key: str, nsrc: str, *, holdout: bool = False) -> str | None:
    """确定性挑一条编造数字/频次说明，且其中的数字/数词不能恰好已经在
    原文出现（否则就是巧合为真，不是违规）。全部候选都冲突则返回 None。
    `holdout=True` 时只从 train 未用过的那几条里选。"""
    pool = (NUMERIC_INJECTIONS[-_NUMERIC_HOLDOUT_RESERVED:] if holdout
            else NUMERIC_INJECTIONS[:-_NUMERIC_HOLDOUT_RESERVED])
    digest = hashlib.sha256(seed_key.encode()).hexdigest()
    start_idx = int(digest, 16) % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start_idx + offset) % len(pool)]
        nums = DIGIT_RE.findall(candidate) + CN_NUMERAL_RE.findall(candidate)
        if any(normalize(n) in nsrc for n in nums):
            continue
        return candidate
    return None


def synthesize_entity_edits(record: dict, *, holdout: bool = False) -> list[dict]:
    """对一条记录做红线0（类别→具体值越界）最小编辑合成——**一案例可产出
    多条**：每个不同的类别标记点各出一条（L4 决断 (b)），挑出的示例若恰好
    已在原文出现（会变成"巧合为真"），单独跳过那一条而不影响其他。没有
    任何类别触发信号，返回空列表。

    `holdout=True` 时只从每个类别留给验收用的成员里选（`pick_examples`），
    与 train 用的成员互不重叠——防止 train 合成正例和 holdout 对抗子集
    共用同一份固定词表（独立第二意见代码审计发现：实测原本 100% 重叠）。"""
    source_text, answer = parse_case_text(record["input"])
    contexts = find_category_triggers(source_text)
    if not contexts:
        return []

    nsrc = normalize(source_text)
    nans_original = normalize(answer)
    out: list[dict] = []
    for i, context in enumerate(contexts):
        examples = pick_examples(context, holdout=holdout)
        # 也要跳过"已在原答案里出现"的示例，不只是"已在原文里出现"——
        # 第二轮独立审计（Fable 5）发现：如果这个词已经合法出现在真实答案
        # 里（比如就是这条病例本来问的那个药，触发了红线0的身份例外条款，
        # 案例本身判 pass），再把它当"编造注入"标成 True，会在训练集里
        # 造出同一个 (case, 词) 组合一边标 False 一边标 True 的直接矛盾
        # ——实测train切分里有11条这样的矛盾。
        if any(normalize(e) in nsrc or normalize(e) in nans_original for e in examples):
            continue
        seed_key = f"{record['case_id']}:{i}"
        injected = _template_for(seed_key).format(names="、".join(examples))
        synthetic_answer = answer.rstrip() + "\n\n" + injected
        suffix = "-synth" if i == 0 else f"-synth{i + 1}"
        out.append({
            "case_id": f"{record['case_id']}{suffix}",
            "source_case_id": record["case_id"],
            "source_text": source_text,
            "original_answer": answer,
            "synthetic_answer": synthetic_answer,
            "injected_examples": list(examples),
            "category_context": context,
            "red_line_idx": 0,
            "verdict": "fail",
        })
    return out


def synthesize_numeric_edit(record: dict, *, holdout: bool = False) -> dict | None:
    """对一条记录做红线2（编造数字/频次）最小编辑合成（L4 决断 (c)）——
    沿用 L1 的规则最小编辑方法论，number swap 与 entity swap 是 FactCC
    同一套机制的两种参数化。没有数字触发信号，或所有候选编造值都与原文
    数字冲突，返回 None。`holdout=True` 见 `synthesize_entity_edits` 同款
    train/holdout 词表隔离理由。"""
    source_text, answer = parse_case_text(record["input"])
    context = find_numeric_trigger(source_text)
    if context is None:
        return None

    nsrc = normalize(source_text)
    nans_original = normalize(answer)
    injection = _pick_numeric_injection(f"numeric:{record['case_id']}", nsrc, holdout=holdout)
    if injection is None:
        return None
    # 同 synthesize_entity_edits：也要防止编造短语里的数字恰好已经合法
    # 出现在原答案里，避免同一 (case, 数字) 组合一边标 False 一边标 True。
    inj_nums = list(dict.fromkeys(DIGIT_RE.findall(injection) + CN_NUMERAL_RE.findall(injection)))
    if not inj_nums or any(normalize(n) in nans_original for n in inj_nums):
        return None

    synthetic_answer = answer.rstrip() + f"\n\n（补充一句：{injection}。）"
    # **候选文本用抽出的数字/数词 span，不用整句编造短语**——第二轮独立
    # 审计（Fable 5）发现：整句当 candidate_text（9-13 字）与生产环境段A
    # 实际能抽出的候选形状（1-5 字的 digit/cn_numeral span）不匹配，训练
    # 正例的候选文本在生产环境里永远不会出现，红线2的验收数字测的是一个
    # 段A根本不会喂给它的分布。`inj_nums` 已经是用同一套正则从注入短语里
    # 提取的 span，形状与真实候选一致。
    return {
        "case_id": f"{record['case_id']}-synthnum",
        "source_case_id": record["case_id"],
        "source_text": source_text,
        "original_answer": answer,
        "synthetic_answer": synthetic_answer,
        "injected_examples": inj_nums,
        "category_context": context,
        "red_line_idx": 2,
        "verdict": "fail",
    }


# L4 判读规则：任一 source_case_id 贡献的合成正例数不得超过合成正例总数的
# 这个比例——防止"来源身份"变成可学捷径（arXiv:2606.01304）。训练前写死，
# 没有文献量级可查，取一个保守值。
MAX_SOURCE_SHARE = 0.05


def _cap_source_concentration(items: list[dict], max_share: float) -> list[dict]:
    """任一 source_case_id 的产出数迭代收敛到不超过 `max_share * 裁剪后总数`。

    **回归（code review 发现）：单次裁剪按裁剪前总数算 cap 不够**——裁剪会
    缩小总数，缩小后同一个 cap 对应的份额可能重新超线（实测：3 条+5×1条，
    单次裁剪按 8 算出 cap=1，裁完剩 6 条，主导来源份额变成 1/6=16.7%，
    远超 5% 门槛却没有任何信号）。改成迭代到不动点：每轮按当前总数重算
    cap 并裁剪，直到某一轮没有任何来源被进一步裁剪。

    **不再对 cap 设 `max(1, ...)` 的下限。** 小样本下"每个来源至少留 1 条"
    和"任何来源不超过 5%"这两条约束可能同时无法满足（来源数 <20 时，
    每个来源留 1 条就必然 ≥5%）——下限保证只会让第二条约束（份额上限，
    这是本函数真正要守的不变量）失效。宁可小样本下裁到 0（安全网在生产
    规模 1000+ 条、cap 通常 ≥50 的场景里几乎不生效，这条边界行为不影响
    实际产出，只影响测试用的极端小样本）。"""
    current = items
    while current:
        total = len(current)
        cap = int(total * max_share)
        by_source: dict[str, list[dict]] = defaultdict(list)
        for it in current:
            by_source[it["source_case_id"]].append(it)
        trimmed: list[dict] = []
        changed = False
        for src_id in sorted(by_source):
            group = sorted(by_source[src_id], key=lambda x: x["case_id"])
            kept = group[:cap]
            if len(kept) != len(group):
                changed = True
            trimmed += kept
        if not changed:
            return trimmed
        current = trimmed
    return current


def synthesize_all(records: list[dict], *, holdout: bool = False) -> list[dict]:
    """红线0（实体，可一案例多条）+ 红线2（数字，一案例至多一条）合成正例，
    并强制执行源文档集中度上限（`MAX_SOURCE_SHARE`，L4）。

    `holdout=True`（`adversarial_subset.py` 用）与默认的 `holdout=False`
    （`train_lora.py` 的训练正例用）从每个类别/数字候选池里选不重叠的
    成员——两处调用方必须用不同的值调用，否则对抗子集会和训练正例共用
    注入词表，验收分数不代表真实泛化（独立第二意见代码审计发现的问题，
    已修）。"""
    out: list[dict] = []
    for r in records:
        out += synthesize_entity_edits(r, holdout=holdout)
        numeric = synthesize_numeric_edit(r, holdout=holdout)
        if numeric is not None:
            out.append(numeric)
    return _cap_source_concentration(out, MAX_SOURCE_SHARE)


def synthetic_records_to_candidates(synth_records: list[dict]) -> list[dict]:
    """把 `synthesize_all()` 的记录级输出展开成候选级条目——一条实体记录
    可能注入 2-4 个词，各自展开成一个独立候选；一条数字记录固定展开成 1 个。
    `label` 恒为 True——这些是"医学正确但原文未给出"的天然违规样本
    （P3 教师抽检 50/50 已确认确属违规，见 ticket 12）。

    两处调用方共用这份转换：`adversarial_subset.py`（holdout 源文本，用于
    L2 对抗子集验收）和 `train_lora.py` 的候选级训练集组装（train 源文本，
    用于训练正例）——两处除了输入的 `records` 来自哪个切分，转换逻辑完全
    一致，不该各写一份。"""
    out: list[dict] = []
    for s in synth_records:
        for text in s["injected_examples"]:
            out.append({
                "case_id": s["case_id"],
                "source_case_id": s["source_case_id"],
                "source_text": s["source_text"],
                "answer": s["synthetic_answer"],
                "candidate_text": text,
                "red_line_guess": s["red_line_idx"],
                "label": True,
            })
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
    n_rl0_records = sum(1 for s in synth if s["red_line_idx"] == 0)
    n_rl2_records = sum(1 for s in synth if s["red_line_idx"] == 2)
    # **候选级正例 ≠ 合成记录数**——一条红线0记录可能注入 2-4 个词，一条
    # 红线2记录也可能展开成多个数字/数词 span（A3 修复后 candidate_text
    # 改用从编造短语里抽出的真实 span，不再是整句，一条短语常含 1-4 个
    # span——之前这里假设"红线2每条记录固定1个候选"，A3 改完就不成立了，
    # 第三轮独立审计的教训：改了数据形状要连报表口径一起改，不能留旧假设。
    n_rl0_candidates = sum(len(s["injected_examples"]) for s in synth if s["red_line_idx"] == 0)
    n_rl2_candidates = sum(len(s["injected_examples"]) for s in synth if s["red_line_idx"] == 2)
    print(f"train {len(records)} 条 -> 合成记录 {len(synth)} 条"
          f"（红线0/实体 {n_rl0_records} 条记录，红线2/数字 {n_rl2_records} 条记录）")
    print(f"候选级正例（段B训练用的粒度，一条记录可能展开成多个候选）：\n"
          f"  红线0 {n_rl0_candidates} + 红线2 {n_rl2_candidates} = "
          f"{n_rl0_candidates + n_rl2_candidates}（——L4 判读规则要求分开报告）")

    source_counts = Counter(s["source_case_id"] for s in synth)
    max_share = max(source_counts.values()) / len(synth) if synth else 0.0
    print(f"源文档集中度：最大单一来源占比 {max_share:.1%}"
          f"（门槛 ≤{MAX_SOURCE_SHARE:.0%}，{'通过' if max_share <= MAX_SOURCE_SHARE else '未通过——检查 _cap_source_concentration'}）")

    pair_counter = Counter(tuple(s["injected_examples"]) for s in synth if s["red_line_idx"] == 0)
    top_pair, top_n = (pair_counter.most_common(1) or [(None, 0)])[0]
    if top_pair is not None:
        print(f"实体注入多样性：{len(pair_counter)} 种不同组合，"
              f"最集中的一组 {top_pair} 占 {top_n / n_rl0_records:.1%}")

    # 按 case_id 排序保证输出确定性（synthesize_all 本身按输入顺序，这里
    # 再显式排一次防止未来改成并行/乱序遍历时悄悄破坏可复现性）。
    synth.sort(key=lambda s: s["case_id"])

    WORK.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(synth, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
