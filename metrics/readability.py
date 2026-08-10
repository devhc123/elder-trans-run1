#!/usr/bin/env python3
"""适老化转译可读性打分器 —— KPI-1 的裁决器。

确定性计算，不含任何 LLM 调用：同输入同环境重跑，结果字节一致。

指标定义见 docs/METRIC_SPEC.md。核心是：

    达标率 = 达标计分词数 / 计分词总数

一个计分词达标 iff 满足下列任一条：
    1. 在计量单位白名单
    2. 在常见健康词白名单
    3. 全部汉字 ∈ 六年级字表  且  该词不在 jargon 术语集

三个口径：
    strict   —— 字级基准用课标常用字表一(2500)。**主 KPI**，医学实体不豁免
    lenient  —— 字级基准放宽到表一+表二(3500，义务教育全程档)。宽松诊断
    glossed  —— 在 strict 基础上，术语后紧跟 ≤15 字白话括注即记达标

本指标相对 FKGL / LENS 的核心优势是**可逐词追溯**：每个不达标词都带失败原因
（jargon / char / latin），可直接列出来给人看。这是刻意的设计，不是附加功能。

用法：
    python3 metrics/readability.py --selftest         # 自检（无网络）
    python3 metrics/readability.py --text "……"        # 打一段文本
    echo "……" | python3 metrics/readability.py        # 从 stdin 读
    python3 metrics/readability.py --text "…" --show-failures
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import functools
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON_DIR = ROOT / "lexicon"

# 括注被认作"有效白话解释"的最大长度。超过这个长度的括注本身就是另一段专业
# 文本，不能算把术语讲明白了。
MAX_GLOSS_LEN = 15

HAN_CHAR = re.compile(r"[一-鿿]")
ALL_HAN = re.compile(r"^[一-鿿]+$")
LATIN_CHAR = re.compile(r"[A-Za-z]")
SENT_SPLIT = re.compile(r"[。！？；\n!?;]+")

# 完整日期与钟点时间：报告签发日期之类不属于「转译文本」的可读性，计分前掩掉。
# 刻意只匹配完整形态，避免误伤剂量表达——"3个月""1日3次""7天"都不会被吃掉。
# 只匹配「明确是日历日期」的形态：四位年份（19xx/20xx）、月+日、钟点。
# 刻意不匹配 \d{2,4}年 —— "高血压20年""病史15年" 是病程不是日期，吃掉它们会
# 悄悄改变任何提到多年病史的文本的分母。
DATETIME = re.compile(
    r"(?:19|20)\d{2}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*[日号])?"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*[日号]"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
)


@dataclass(frozen=True)
class Failure:
    word: str
    reason: str  # jargon | char | latin


@dataclass
class Result:
    n_scored: int
    n_pass_strict: int
    n_pass_lenient: int
    n_pass_glossed: int
    rate_strict: float
    rate_lenient: float
    rate_glossed: float
    scored_words: list[str]
    failures: list[Failure]
    jargon_hits: list[str]
    jargon_density: float          # 每 100 计分词的术语数
    n_chars: int
    n_sentences: int
    avg_sentence_len: float
    segmenter: str

    def as_dict(self) -> dict:
        return {
            "n_scored": self.n_scored,
            "rate_strict": round(self.rate_strict, 6),
            "rate_lenient": round(self.rate_lenient, 6),
            "rate_glossed": round(self.rate_glossed, 6),
            "n_pass_strict": self.n_pass_strict,
            "jargon_density": round(self.jargon_density, 4),
            "n_chars": self.n_chars,
            "n_sentences": self.n_sentences,
            "avg_sentence_len": round(self.avg_sentence_len, 3),
            "segmenter": self.segmenter,
            "failures": [{"word": f.word, "reason": f.reason} for f in self.failures],
        }


@dataclass
class Lexicon:
    grade6_chars: set[str]
    ext_chars: set[str]
    jargon: set[str]
    units: set[str]
    health_common: set[str]
    hsk_words: set[str] = field(default_factory=set)

    @functools.cached_property
    def lenient_chars(self) -> set[str]:
        return self.grade6_chars | self.ext_chars

    @functools.cached_property
    def jargon_lens(self) -> list[int]:
        """术语表中实际出现过的词长，降序。跨度扫描只探测这些长度。"""
        return sorted({len(t) for t in self.jargon if len(t) >= 2}, reverse=True)

    @classmethod
    def load(cls, d: Path | None = None) -> "Lexicon":
        d = d or LEXICON_DIR

        def read(name: str) -> set[str]:
            p = d / name
            if not p.exists():
                raise FileNotFoundError(p)
            out = set()
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    out.add(line)
            return out

        def read_chars(name: str) -> set[str]:
            return {c for tok in read(name) for c in tok if HAN_CHAR.match(c)}

        lex = cls(
            grade6_chars=read_chars("grade6_chars.txt"),
            ext_chars=read_chars("edu_ext_chars.txt"),
            jargon=read("jargon.txt"),
            units=read("units_whitelist.txt"),
            health_common=read("health_common_whitelist.txt"),
            hsk_words=read("hsk_words.txt") if (d / "hsk_words.txt").exists() else set(),
        )
        lex.check_integrity()
        return lex

    # 官方字表的法定规模。载入时强校验——一份缩水的字表会静默改变每一个 KPI
    # 数字，而下游没有任何地方会因此报错。
    EXPECTED_SIZES = {"grade6_chars": 2500, "ext_chars": 1000}

    def check_integrity(self) -> None:
        for attr, expect in self.EXPECTED_SIZES.items():
            got = len(getattr(self, attr))
            if got != expect:
                raise ValueError(
                    f"词表 {attr} 规模不符：期望 {expect}，实得 {got}。"
                    "重跑 pipeline/build_lexicon.py。"
                )
        if not self.jargon:
            raise ValueError(
                "jargon.txt 为空：术语判定会整体失效，未转译的术语密集原文会被"
                "打到 ~0.9 并报告为达标。重跑 pipeline/build_lexicon.py。"
            )


# 术语跨度扫描不再设固定上限。早期版本硬编码 12，导致表中 1074 条（2.3%）
# 超长条目（最长 135 字，如"重组牛碱性成纤维细胞生长因子眼用凝胶"）永远无法
# 命中——含这些药名的文本只按字面判定，可能直接通过 strict。
# 改为按表中**实际存在的长度**降序探测，既无漏检也不做无谓回退。


def _plain_segment(text: str) -> tuple[list[str], str]:
    try:
        import jieba

        return list(jieba.cut(text)), "jieba"
    except ImportError:
        return _fmm(text), "fmm"


def _segment(text: str, jargon: set[str], lens: list[int] | None = None) -> tuple[list[str], str]:
    """术语感知分词。

    先对**原文**做最长匹配的术语跨度扫描，把命中的术语整体切出来，剩余片段再交给
    jieba（或 FMM 兜底）。

    为什么不直接用分词结果去查 jargon：jargon 收的是完整实体名（"预激综合征"），
    而通用分词器会把它切成"预激"+"综合征"，两者都匹配不上，导致术语被系统性漏检。
    指标的判定结果不能取决于分词器恰好怎么切——那样指标不可信也不可复现。
    """
    if not text:
        return [], "none"
    if not jargon:
        return _plain_segment(text)

    n = len(text)
    spans = _jargon_spans(text, jargon, lens)

    # 通用分词只调一次。早期版本在每个术语处切断文本、对每个碎片各调一次 jieba，
    # 1000 字要 ~390ms；改为单次调用后按术语边界重切，快一个量级。
    toks, seg_name = _plain_segment(text)

    owner = [-1] * n
    for sid, (a, b) in enumerate(spans):
        for k in range(a, b):
            owner[k] = sid

    cuts = {0, n}
    p = 0
    for t in toks:
        p += len(t)
        cuts.add(p)
    for a, b in spans:
        cuts.add(a)
        cuts.add(b)
    pts = sorted(c for c in cuts if 0 <= c <= n)

    out: list[str] = []
    i = 0
    while i < len(pts) - 1:
        a = pts[i]
        sid = owner[a] if a < n else -1
        if sid != -1:
            sa, sb = spans[sid]
            out.append(text[sa:sb])
            while i < len(pts) - 1 and pts[i] < sb:
                i += 1
        else:
            out.append(text[a : pts[i + 1]])
            i += 1
    return out, seg_name


def _jargon_spans(
    text: str, jargon: set[str], lens: list[int] | None = None
) -> list[tuple[int, int]]:
    """在原文上做最长匹配的术语跨度扫描，返回互不重叠的 (起, 止)。

    最长匹配优先："心肌梗死" 整体命中，而不是先命中 "心肌"。
    只探测表中**实际存在**的词长（降序），因此既不会漏掉超长条目，也不做无谓回退。
    """
    if lens is None:
        lens = sorted({len(t) for t in jargon if len(t) >= 2}, reverse=True)
    spans: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        hit = 0
        for ln in lens:
            if ln > n - i:
                continue
            if text[i : i + ln] in jargon:
                hit = ln
                break
        if hit:
            spans.append((i, i + hit))
            i += hit
        else:
            i += 1
    return spans


def _fmm(text: str, max_len: int = 4) -> list[str]:
    """兜底分词：正向最大匹配，仅按字面切汉字块，其余按字符分。"""
    out, i, n = [], 0, len(text)
    while i < n:
        if HAN_CHAR.match(text[i]):
            j = min(i + max_len, n)
            while j > i + 1 and not ALL_HAN.match(text[i:j]):
                j -= 1
            out.append(text[i:j])
            i = j
        else:
            m = re.match(r"[A-Za-z0-9]+|\s+|.", text[i:])
            out.append(m.group(0))
            i += len(m.group(0))
    return out


def _is_scored(tok: str) -> bool:
    """计分词的界定。

    含汉字，或以拉丁字母开头（HbA1c / ALT / mg）—— 计分。
    其余一律不计入分母：纯标点、纯数字、日期时间、百分比、空白、纯符号。
    按字符构成判定而非枚举标点，避免漏掉未预料到的符号。
    """
    tok = tok.strip()
    if not tok:
        return False
    if HAN_CHAR.search(tok):
        return True
    return bool(re.match(r"^[A-Za-z]", tok))


def _gloss_anchors(text: str) -> set[int]:
    """返回「有效白话括注」所解释的那个词的**结束偏移**。

    按位置锚定，而不是按词面登记。早期版本把括注前整段连续汉字及其**每一个后缀**
    都登记为"已解释"——"他得了慢性病（长期的病）" 会登记 {慢性病, 性病, 病, …}，
    于是文档里**别处**一个毫无解释的"性病"也被算作达标。等于在文档任意位置加一个
    括注，就能给它从未解释过的术语放行。

    锚定到位置后，只有紧贴括注左侧的那个词会被救，且救谁与它因何失败无关
    （jargon 与 char 两类失败都救）——括注是否有效不该取决于 jargon.txt 恰好
    收没收这个词。
    """
    out: set[int] = set()
    for m in re.finditer(r"[（(]([^（）()]{1,%d})[)）]" % MAX_GLOSS_LEN, text):
        if not m.group(1).strip():
            continue
        j = m.start()
        while j > 0 and text[j - 1].isspace():
            j -= 1
        if j > 0:
            out.add(j)
    return out


def _judge(word: str, lex: Lexicon, chars: set[str]) -> Failure | None:
    if word in lex.units or word in lex.health_common:
        return None
    # 含**任何**拉丁字母即判 latin，而不只是整词皆拉丁。
    # 否则 "CT检查" 通过而 "CT" 失败——同一个缩写是否计入 KPI 取决于分词器
    # 恰好有没有把它和后一个词粘起来，正是 §4.1 声明要杜绝的分词器依赖。
    if LATIN_CHAR.search(word):
        return Failure(word, "latin")
    if word in lex.jargon:
        return Failure(word, "jargon")
    if all((not HAN_CHAR.match(c)) or c in chars for c in word):
        return None
    return Failure(word, "char")


def score(text: str, lex: Lexicon) -> Result:
    raw = text or ""
    masked = DATETIME.sub(" ", raw)
    tokens, segmenter = _segment(masked, lex.jargon, lex.jargon_lens)

    # 括注锚点是**偏移**而非词面，所以要带着偏移遍历 token。
    # 分词无损（见 tests/test_tokenizer_properties.py），偏移可由累加长度得出。
    anchors = _gloss_anchors(masked)

    scored: list[str] = []
    failures: list[Failure] = []
    n_pass_lenient = 0
    n_pass_glossed = 0
    jargon_hits: list[str] = []

    pos = 0
    for tok in tokens:
        start, end = pos, pos + len(tok)
        pos = end
        if not _is_scored(tok):
            continue
        w = tok.strip()
        scored.append(w)

        f_strict = _judge(w, lex, lex.grade6_chars)
        if f_strict is None:
            n_pass_glossed += 1
        elif end in anchors:
            # glossed 口径：这个词紧贴一段 ≤15 字白话括注，视为已被解释。
            # 不论它因 jargon 还是 char 失败都救——括注是否有效，不该取决于
            # jargon.txt 恰好收没收这个词（该表自述不完整，见 METRIC_SPEC §7）。
            n_pass_glossed += 1

        if f_strict is not None:
            failures.append(f_strict)
            if f_strict.reason == "jargon":
                jargon_hits.append(w)

        if _judge(w, lex, lex.lenient_chars) is None:
            n_pass_lenient += 1

    n = len(scored)
    n_pass_strict = n - len(failures)

    # 与打分同源：都基于 masked。早期版本用原文算字数句数，日期时间被计入长度
    # 却未计入分母，导致"达标率与长度并排看"的两半算的是两段不同文本。
    sents = [s for s in SENT_SPLIT.split(masked) if s.strip()]
    n_chars = len(re.sub(r"\s", "", masked))

    return Result(
        n_scored=n,
        n_pass_strict=n_pass_strict,
        n_pass_lenient=n_pass_lenient,
        n_pass_glossed=n_pass_glossed,
        rate_strict=(n_pass_strict / n) if n else 0.0,
        rate_lenient=(n_pass_lenient / n) if n else 0.0,
        rate_glossed=(n_pass_glossed / n) if n else 0.0,
        scored_words=scored,
        failures=failures,
        jargon_hits=jargon_hits,
        jargon_density=(100.0 * len(jargon_hits) / n) if n else 0.0,
        n_chars=n_chars,
        n_sentences=len(sents),
        avg_sentence_len=(n_chars / len(sents)) if sents else 0.0,
        segmenter=segmenter,
    )


# --------------------------------------------------------------------------
# 自检：证明指标有区分度。没有区分度的可读性指标是坏指标，必须先修再用。
# --------------------------------------------------------------------------

SELFTEST_PLAIN = [
    "这个药一天吃两次，早上一次晚上一次，饭后吃。要是忘了吃，想起来就补上。",
    "医生说您血压有点高，平时少吃点咸的，每天出门走一走。",
    "检查结果大体还好，有两项比正常范围高一点，不用太担心，过三个月再来看看。",
]

SELFTEST_JARGON = [
    "对任何强心苷制剂中毒者禁用。室性心动过速、心室颤动者禁用。",
    "梗阻性肥厚型心肌病患者慎用，合并预激综合征者禁用本品。",
    "血清肌酐、尿素氮及内生肌酐清除率提示肾小球滤过功能减退。",
]


def selftest() -> int:
    try:
        lex = Lexicon.load()
    except FileNotFoundError as e:
        print(f"[FAIL] 词表缺失：{e}\n先跑：python3 pipeline/build_lexicon.py", file=sys.stderr)
        return 1

    print(f"词表规模：六年级字表 {len(lex.grade6_chars)} / 扩展字表 {len(lex.ext_chars)} / "
          f"jargon {len(lex.jargon)} / 单位 {len(lex.units)} / 常见健康词 {len(lex.health_common)}")
    if len(lex.grade6_chars) != 2500:
        print(f"[FAIL] 六年级字表应为 2500 字（课标常用字表一），实得 {len(lex.grade6_chars)}", file=sys.stderr)
        return 1

    plain = [score(t, lex) for t in SELFTEST_PLAIN]
    jarg = [score(t, lex) for t in SELFTEST_JARGON]
    pm = sum(r.rate_strict for r in plain) / len(plain)
    jm = sum(r.rate_strict for r in jarg) / len(jarg)

    print("\n通俗文本：")
    for t, r in zip(SELFTEST_PLAIN, plain):
        print(f"  {r.rate_strict:.3f}  {t[:28]}…")
    print("术语密集原文：")
    for t, r in zip(SELFTEST_JARGON, jarg):
        print(f"  {r.rate_strict:.3f}  {t[:28]}…  不达标词={[f.word for f in r.failures][:6]}")

    gap = pm - jm
    print(f"\n通俗均值 {pm:.3f} / 术语均值 {jm:.3f} / 区分度 {gap:.3f}")

    ok = True
    if gap < 0.20:
        print(f"[FAIL] 区分度不足（要求 ≥0.20，实得 {gap:.3f}）——指标无效，先修词表", file=sys.stderr)
        ok = False

    # 确定性
    t = SELFTEST_JARGON[0]
    if score(t, lex).as_dict() != score(t, lex).as_dict():
        print("[FAIL] 同输入两次结果不一致", file=sys.stderr)
        ok = False

    print("[OK] 自检通过" if ok else "[FAIL] 自检未通过")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="适老化转译可读性打分器")
    ap.add_argument("--selftest", action="store_true", help="跑自检（含区分度与确定性）")
    ap.add_argument("--text", help="要打分的文本；缺省从 stdin 读")
    ap.add_argument("--show-failures", action="store_true", help="逐条列出不达标词与原因")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    text = args.text if args.text is not None else sys.stdin.read()
    lex = Lexicon.load()
    r = score(text, lex)

    if args.json:
        print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"计分词数        {r.n_scored}")
    print(f"达标率 strict   {r.rate_strict:.4f}   ← 主 KPI（课标常用字表一 2500，医学实体不豁免）")
    print(f"达标率 glossed  {r.rate_glossed:.4f}   （术语带 ≤{MAX_GLOSS_LEN} 字白话括注记达标）")
    print(f"达标率 lenient  {r.rate_lenient:.4f}   （字表一+二 3500，义务教育全程档）")
    print(f"jargon 密度     {r.jargon_density:.2f} / 100 词")
    print(f"字数 {r.n_chars}   句数 {r.n_sentences}   平均句长 {r.avg_sentence_len:.1f}   分词器 {r.segmenter}")
    if args.show_failures and r.failures:
        print("\n不达标词：")
        for f in r.failures:
            print(f"  {f.word}\t{f.reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
