#!/usr/bin/env python3
"""构建可读性判定所需的词表。

产出全部落在 lexicon/ 下，供 metrics/readability.py 使用：

  grade6_chars.txt      《义务教育语文课程》(2022年版) 常用字表一，2500 字
                        —— 课标指定为第三学段(5-6年级)识字写字能力评价依据，
                           本项目「小学六年级可读水平」的字级基准
  edu_ext_chars.txt     同上 常用字表二，1000 字（义务教育全程档，宽松诊断口径用）
  hsk_words.txt         GF 0025-2021《国际中文教育中文水平等级标准》一~四级词
  jargon.txt            七个爬虫库实体名字段收割的医学术语
  units_whitelist.txt   计量单位（手工维护）
  health_common_whitelist.txt  老人本就懂的高频健康词（手工维护）

字表来源是国家标准本身（国务院/教育部颁布），此处仅从公开转写仓库获取机读版；
派生出的字表为标准内容，不含转写仓库的额外创作。转写仓库出处见 docs/METRIC_SPEC.md。

用法：
    python3 pipeline/build_lexicon.py              # 全部重建
    python3 pipeline/build_lexicon.py --skip-jargon  # 只建字/词表（不读语料，快）
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path


class BuildError(RuntimeError):
    """构建失败。绝不允许降级为警告后继续写出残缺词表——那会静默抬高 KPI。"""


ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "lexicon"

# 官方字表的机读转写来源（raw 文件）
RAW = "https://raw.githubusercontent.com/zispace/hanzi-chars/main/data-charlist/"
CHAR_SOURCES = {
    "grade6_chars.txt": (
        "《义务教育语文课程》（2022年版）常用字表一.txt",
        2500,
        "义务教育语文课程标准(2022年版)附录5 常用字表一；课标指定为第三学段(5-6年级)评价依据",
    ),
    "edu_ext_chars.txt": (
        "《义务教育语文课程》（2022年版）常用字表二.txt",
        1000,
        "义务教育语文课程标准(2022年版)附录5 常用字表二；与表一合计3500，义务教育全程档",
    ),
    "general_standard_L1.txt": (
        "《通用规范汉字表》（2013年）一级字.txt",
        3500,
        "通用规范汉字表(国务院2013)一级字表，常用字集；用于交叉核对",
    ),
}

HSK_URL = "https://raw.githubusercontent.com/elkmovie/hsk30/main/wordlist.txt"
# HSK 3.0 三等九级。取一~四级作为「小学六年级词汇量级」的词级基准。
HSK_LEVELS_KEPT = {"一级", "二级", "三级", "四级"}

DEFAULT_CORPUS = Path(
    "/Users/chenhao/DATA/03_国内医疗语料与数据源/2025国内医疗爬虫结果/2025国内医疗爬虫结果"
)
CORPUS = Path(os.environ.get("ELDER_CORPUS") or DEFAULT_CORPUS)

# 实体**正式名**列：无条件进 jargon。
ENTITY_COLUMNS = {
    "bdyd": ["名称"],
    "dxys": ["name"],
    "mkss": ["title"],
    "others": ["name", "common_name", "trade_name", "main_ingredient"],
    "xywy": ["name"],
    "ylys": ["name", "缩写"],
    "zsys": ["disease_name"],
}

# **别名**列：这些数据库的别名列同时混着两类东西 —— 正式同义词（"核磁共振检查"）
# 与口语说法（"拍片子""出汗"）。两类都收会让指标反向激励：转译时加一句白话解释，
# 反而因为解释里用了口语词而掉分，指标就在惩罚它本该奖励的行为。两类都不收又会
# 丢掉大量真术语（实测 jargon 从 46885 掉到 37913）。
# 所以分开收，再按下面的口语判据剔除。
ALIAS_COLUMNS = {
    "bdyd": ["别名"],
    "mkss": ["drug_aliases"],
}

# 口语判据：**仅**作为别名出现（从未作为正式名）、长度 ≤4、且全部汉字都在六年级
# 字表内。三个条件同时满足才算口语说法。"拍片子""出汗" 命中；"核磁共振检查"
# 因长度 6 不命中，仍按术语处理。
COLLOQUIAL_MAX_LEN = 4

HAN = re.compile(r"^[一-鿿]+$")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "elder-trans-run1/lexicon-build"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def parse_charlist(text: str) -> list[str]:
    """字表文件：'#' 开头为注释/分节，其余每行一个字。"""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for ch in line:
            if HAN.match(ch):
                out.append(ch)
    # 去重保序
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def parse_hsk(text: str) -> list[str]:
    """HSK 3.0 词表：分节标题形如 '一级词汇表'，词条形如 '123 词语'。"""
    out, keep = [], False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.+?)词汇表$", line)
        if m:
            keep = m.group(1) in HSK_LEVELS_KEPT
            continue
        m = re.match(r"^\d+\s+(.+)$", line)
        if m and keep:
            for w in re.split(r"[|/、，,]", m.group(1)):
                w = w.strip()
                # 去掉词表里的括注与拼音标注残留
                w = re.sub(r"[（(].*?[)）]", "", w).strip()
                if w and HAN.match(w):
                    out.append(w)
    return sorted(set(out))


def sniff_open(path: str):
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            f = open(path, encoding=enc, newline="")
            head = f.read(4096)
            f.seek(0)
            if head:
                return f
            f.close()
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _clean_terms(raw: str) -> list[str]:
    """实体名常含书名号/括号/罗马数字等噪声，统一清洗后按分隔符拆开。"""
    out = []
    for piece in re.split(r"[、,，;；/|]", raw):
        piece = re.sub(r"[（(].*?[)）]", "", piece)
        piece = re.sub(r"[《》\[\]【】\s]", "", piece).strip()
        if len(piece) >= 2 and HAN.match(piece):
            out.append(piece)
    return out


def harvest_jargon(exclude: set[str], grade6: set[str]) -> list[str]:
    """从七库实体名字段收割医学术语。

    正式名列与别名列分开收；别名中判定为口语说法的一律不计入术语（见
    ALIAS_COLUMNS / COLLOQUIAL_MAX_LEN 处的说明）。
    """
    formal: set[str] = set()
    alias: set[str] = set()
    if not CORPUS.exists():
        raise BuildError(
            f"语料目录不存在：{CORPUS}\n"
            "不能跳过 jargon 收割后继续——空的 jargon.txt 会让术语判定整体失效，\n"
            "术语密集的未转译原文会被打到 ~0.9 并报告为达标（实测区分度 0.313→0.114）。\n"
            "用 ELDER_CORPUS=/path/to/corpus 指定语料，或显式 --skip-jargon 保留现有 jargon.txt。"
        )

    for path in sorted(glob.glob(str(CORPUS / "**" / "*.csv"), recursive=True)):
        rel = os.path.relpath(path, CORPUS)
        lib = rel.split(os.sep)[0]
        f_cols, a_cols = ENTITY_COLUMNS.get(lib, []), ALIAS_COLUMNS.get(lib, [])
        if not f_cols and not a_cols:
            continue
        f = sniff_open(path)
        if f is None:
            print(f"[warn] 编码识别失败，跳过：{rel}", file=sys.stderr)
            continue
        try:
            reader = csv.DictReader(f)
            names = reader.fieldnames or []
            for row in reader:
                for c in (c for c in f_cols if c in names):
                    formal.update(_clean_terms((row.get(c) or "").strip()))
                for c in (c for c in a_cols if c in names):
                    alias.update(_clean_terms((row.get(c) or "").strip()))
        finally:
            f.close()

    colloquial = {
        t
        for t in alias - formal
        if len(t) <= COLLOQUIAL_MAX_LEN and all(ch in grade6 for ch in t)
    }
    print(f"  正式名 {len(formal)} / 别名 {len(alias)} / 判为口语剔除 {len(colloquial)}")
    return sorted((formal | alias) - colloquial - exclude)


def write_list(name: str, items: list[str], header: list[str]) -> None:
    p = LEXICON / name
    with open(p, "w", encoding="utf-8") as f:
        for h in header:
            f.write(f"# {h}\n")
        f.write(f"# 条目数: {len(items)}\n")
        for it in items:
            f.write(it + "\n")
    print(f"  {name}: {len(items)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-jargon", action="store_true", help="不读语料，只建字/词表")
    args = ap.parse_args()

    LEXICON.mkdir(exist_ok=True)

    print("拉取官方字表…")
    tables = {}
    for out_name, (src, expect, desc) in CHAR_SOURCES.items():
        chars = parse_charlist(fetch(RAW + urllib.parse.quote(src)))
        # 硬失败而非警告：上游改了分节标题、多了一行、或 URL 开始返回 404 页面，
        # 都会让字表规模变化。写出一份缩水的字表会静默改变每一个 KPI 数字，
        # 而下游没有任何地方会因此报错。
        if len(chars) != expect:
            raise BuildError(
                f"{out_name} 规模不符：期望 {expect} 字，实得 {len(chars)}。\n"
                f"来源 {src} 可能已变更。已中止，未覆盖现有词表。"
            )
        tables[out_name] = chars
        write_list(out_name, chars, [desc, f"来源文件: {src}", f"规模: {expect}（构建时强校验）"])

    # 交叉核对：字表一+二 应与《通用规范汉字表》一级字表高度一致
    edu = set(tables["grade6_chars.txt"]) | set(tables["edu_ext_chars.txt"])
    gs = set(tables["general_standard_L1.txt"])
    if edu != gs:
        raise BuildError(
            f"交叉核对失败：课标3500 与《通用规范汉字表》一级字表应完全一致，"
            f"实得 交集 {len(edu & gs)}、课标独有 {len(edu - gs)}、通用规范独有 {len(gs - edu)}。"
            "两个国家标准本应互证，不一致说明至少一份转写有误，已中止。"
        )
    print(f"  交叉核对 课标3500 vs 通用规范一级3500: 完全一致（{len(edu)} 字）")

    print("拉取 HSK 3.0 (GF 0025-2021) 词表…")
    hsk = parse_hsk(fetch(HSK_URL))
    write_list(
        "hsk_words.txt",
        hsk,
        [
            "GF 0025-2021《国际中文教育中文水平等级标准》一~四级词",
            "机读来源: elkmovie/hsk30 (MIT)。注意该仓库自述为 OCR 提取、未充分校对",
        ],
    )

    if not args.skip_jargon:
        print("收割 jargon（七库实体名字段）…")
        hc_path = LEXICON / "health_common_whitelist.txt"
        health_common = set()
        if hc_path.exists():
            health_common = {
                l.strip()
                for l in hc_path.read_text(encoding="utf-8").splitlines()
                if l.strip() and not l.startswith("#")
            }
        jargon = harvest_jargon(health_common | set(hsk), set(tables["grade6_chars.txt"]))
        if not jargon:
            raise BuildError(
                f"jargon 收割结果为空（语料 {CORPUS}）。未覆盖现有 jargon.txt。"
            )
        write_list(
            "jargon.txt",
            jargon,
            [
                "七个爬虫库实体名字段收割的医学术语（药名/病名/检查名/指标名/穴位/手术等）",
                "仅取实体**正式名**列；刻意不收别名列（口语说法聚集地）",
                "已剔除 health_common_whitelist 与 HSK 一~四级词",
            ],
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        print(f"\n[BUILD FAILED] {e}", file=sys.stderr)
        sys.exit(1)
