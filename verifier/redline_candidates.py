#!/usr/bin/env python3
"""两段式红线判别的段 A：规则候选抽取（ticket 12 P1）。

只做候选抽取，不做判定——判定精度留给段 B（教师在线判 / 未来 encoder 分类头，
见 `judge_candidates.py`）。段 A 的唯一职责是「别漏」：answer 里出现了
source 没有的具体名词/数字/中文数词，就是一个候选，宁可错杀（低精度）
不可漏杀（低召回）。

召回校准只在 train 切分（`verifier/train.jsonl`）上做，冻结后对 holdout
只评一次——见 `redline_candidates_eval.py` 与
`docs/RESEARCH_verifier_data_scarcity.md`。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JARGON_PATH = ROOT / "lexicon" / "jargon.txt"
FOOD_WORDLIST_PATH = ROOT / "verifier" / "redline_food_wordlist.txt"

MIN_LEXICON_LEN = 3  # 与 lexicon/jargon.txt 现有筛选口径一致

# 两段式管线覆盖的红线 idx。红线 1（说反）全池仅 2 例，两段式管线不覆盖，
# 教师全文判兜底——见 docs/RESEARCH_verifier_data_scarcity.md。
# `judge_candidates.py` 与 `redline_candidates_eval.py` 都要用同一份，
# 不许各自手写一遍 `idx not in (0, 2)`——那是 join_fields() 分叉过的同一类坑。
COVERED_RED_LINE_IDXS = (0, 2)

CN_NUM = "零一二三四五六七八九十百千万两半"
_UNIT_WORDS = [
    "星期", "礼拜", "小时", "钟头", "分钟", "毫升", "毫克", "公斤",
    "天", "年", "月", "周", "秒", "斤", "克", "升", "片", "粒", "颗",
    "倍", "次", "岁", "分",
]
_UNIT_ALT = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))
# `(?!之)`：防止「三千分之一」这类分数被本规则错切成「三千分」——分数由
# FRACTION_RE 单独处理。
CN_NUMERAL_RE = re.compile(rf"[{CN_NUM}]+(?:个)?(?:{_UNIT_ALT})(?!之)")
FRACTION_RE = re.compile(rf"[{CN_NUM}]+分之[{CN_NUM}]+")
DIGIT_RE = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """全角转半角 + 去空白，便于原样子串比对（source 是否已含该候选）。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def parse_case_text(input_text: str) -> tuple[str, str]:
    """从 verifier train/holdout.jsonl 的 `input` 字段切出 (source_text, answer)。"""
    src = input_text.split("【回答】")[0].replace("【原文】", "")
    ans = input_text.split("【回答】")[1].split("【要点】")[0]
    return src, ans


def load_wordlist(path: Path) -> set[str]:
    """加载词表，跳过注释行与空行；不做长度过滤（调用方决定）。"""
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_jargon(min_len: int = MIN_LEXICON_LEN) -> set[str]:
    """医学实体词表 ∪ 食物/日常名词补充词表。

    长度过滤只作用于 jargon.txt——它是 4.3 万条开放爬取词，短条目噪声大。
    食物词表是几十条人工精选闭表（多为「鸡蛋」「豆腐」这类 2 字常见词），
    噪声风险有限，不该被同一把尺子砍掉。
    """
    medical = {w for w in load_wordlist(JARGON_PATH) if len(w) >= min_len}
    food = load_wordlist(FOOD_WORDLIST_PATH)
    return medical | food


def extract_candidates(source_text: str, answer: str, jargon: set[str]) -> list[dict]:
    """段 A：抽取 answer 里「source 没有」的候选 span。

    每个候选 `{"text": str, "kind": "lexicon"|"digit"|"cn_numeral"|"fraction",
    "red_line_guess": 0 或 2}`。`kind`/`red_line_guess` 只是溯源标签，不是
    最终判定——段 B 才决定这个候选是不是真违规。

    数字/中文数词/分数的「source 是否已含」判定必须按**整段匹配**，不能用
    朴素子串——"12" 是 "112" 的子串，但 12 ≠ 112，若用子串判定会把一个真正
    编造的数字误判成"原文已有"而漏掉（source 说"不超过112"，answer 编成
    "别超过12"，两者语义完全不同）。词表词（lexicon）不受此限：中文复合词
    的子串关系通常仍是同一实体（"他汀"是"阿托伐他汀"的子串，指向同一类
    实体），沿用朴素子串判定是对的，只有数字/数词这类"子串就是另一个值"
    的场景才需要整段匹配。
    """
    nsrc = normalize(source_text)
    nans = normalize(answer)
    src_digits = set(DIGIT_RE.findall(nsrc))
    src_fractions = set(FRACTION_RE.findall(nsrc))
    src_cn_numerals = set(CN_NUMERAL_RE.findall(nsrc))

    seen: set[str] = set()
    out: list[dict] = []

    def add(text: str, kind: str, red_line_guess: int, already_grounded: bool) -> None:
        if already_grounded or text in seen:
            return
        seen.add(text)
        out.append({"text": text, "kind": kind, "red_line_guess": red_line_guess})

    for w in jargon:
        # 词表词也要 normalize 再比较——nsrc/nans 都做过 NFKC+去空白，
        # 词表词若原样保留全角字符/空格（如"维生素 C"）就匹配不上规范化后
        # 的原文/回答，导致已在原文的实体被误判成"新增候选"（独立第二意见
        # 代码审计发现，当前词表未实测命中，但属于潜在的静默误判）。
        nw = normalize(w)
        if nw in nans:
            add(nw, "lexicon", 0, already_grounded=nw in nsrc)
    for m in FRACTION_RE.finditer(nans):
        t = m.group()
        add(t, "fraction", 0, already_grounded=t in src_fractions)
    for m in CN_NUMERAL_RE.finditer(nans):
        t = m.group()
        add(t, "cn_numeral", 0, already_grounded=t in src_cn_numerals)
    for m in DIGIT_RE.finditer(nans):
        t = m.group()
        add(t, "digit", 2, already_grounded=t in src_digits)

    return out


def flag_case(source_text: str, answer: str, jargon: set[str]) -> bool:
    """case 级别：只要有一个候选，就打旗标（供 P1 召回评测用）。"""
    return bool(extract_candidates(source_text, answer, jargon))
