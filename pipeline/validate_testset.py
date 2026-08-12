#!/usr/bin/env python3
"""测试集校验与冻结（ticket 07）。

把 270 题从「一堆文件」变成「一个可被引用的、有版本的评测集」。

校验通过后写入 kpi.yaml 的 frozen 区：内容 hash、题数、场景数、切分种子。
此后任何改动都会让 hash 变化、被检出——这是评测集可信度的基础，评审要能确认
你报的分数和你给的数据是同一份。

同时切出 **hard 子集**（难度=难的那 15%）作为固定的判别力观测口。此后所有跑数
必须同时报全集分和 hard 子集分：只报全集分的 harness，三个月后没人知道自己在
优化什么。

用法：
    python3 pipeline/validate_testset.py            # 只校验
    python3 pipeline/validate_testset.py --freeze   # 校验通过后冻结进 kpi.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.author_testset import (  # noqa: E402
    DIFFICULTY_TARGET,
    SCENARIO_PARENT,
    validate,
)

TESTSET = ROOT / "data" / "elder_translate_270.jsonl"
HARD = ROOT / "data" / "hard_subset.jsonl"
KPI = ROOT / "kpi.yaml"
DB = ROOT / "index" / "corpus.sqlite"

# 项目书第六栏逐字列举的父类。核销指标说明时按这份名单逐条对。
PROJECT_PARENTS = [
    "检验报告解读", "医嘱转译", "用药说明", "健康科普",
    "权益匹配", "分科医生", "复诊提醒", "用药干预",
]


def load() -> list[dict]:
    if not TESTSET.exists():
        raise FileNotFoundError(f"{TESTSET}\n先跑 pipeline/author_testset.py --merge")
    return [json.loads(l) for l in TESTSET.read_text(encoding="utf-8").splitlines() if l.strip()]


def dataset_hash(cases: list[dict]) -> str:
    """内容 hash。按 id 排序后逐题取规范化 JSON，与文件里的行序、空白无关。"""
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c["id"]):
        h.update(json.dumps(c, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def check_leakage(cases: list[dict]) -> list[str]:
    """所有 record_id 必须落在 test 池内。泄漏不会报错，只会让分数好看。"""
    if not DB.exists():
        return ["索引不存在，跳过泄漏校验（这是一次降级，别当作通过）"]
    con = sqlite3.connect(DB)
    if not con.execute("SELECT name FROM sqlite_master WHERE name='splits'").fetchone():
        con.close()
        return ["尚未切分，跳过泄漏校验（这是一次降级，别当作通过）"]
    pools: dict[str, set[str]] = {}
    for p in ("test", "verifier_train", "reserve"):
        pools[p] = {r for (r,) in con.execute("SELECT record_id FROM splits WHERE pool=?", (p,))}
    con.close()

    problems = []
    used = {c["provenance"]["record_id"] for c in cases}
    for other in ("verifier_train", "reserve"):
        n = len(used & pools[other])
        if n:
            problems.append(f"{n} 条 record_id 落在 {other} 池，属泄漏")
    outside = used - pools["test"]
    if outside:
        problems.append(f"{len(outside)} 条 record_id 不在任何 test 池内，例：{sorted(outside)[:3]}")
    return problems


def check_coverage(cases: list[dict]) -> list[str]:
    problems = []
    parents = {c["parent_class"] for c in cases}
    missing = set(PROJECT_PARENTS) - parents
    if missing:
        problems.append(f"项目书父类未覆盖：{sorted(missing)}（指标说明核销会缺项）")
    scenarios = {c["scenario"] for c in cases}
    if len(scenarios) < 10:
        problems.append(f"子场景仅 {len(scenarios)} 个，KPI-2 目标 10")
    return problems


def check_flags(cases: list[dict]) -> list[str]:
    """数据缺口标记必须完好——它们是对外披露的依据。"""
    problems = []
    for c in cases:
        if c["scenario"] == "医嘱转译":
            if not c.get("simulated"):
                problems.append(f"{c['id']}: 医嘱转译题缺 simulated 标记")
            for banned in ("医嘱单", "处方单", "出院小结"):
                if banned in c["query"]:
                    problems.append(f"{c['id']}: query 含「{banned}」，会让人误以为有真实医嘱语料")
        if c["scenario"] in ("权益匹配", "科普辟谣") and not c.get("thin_source"):
            problems.append(f"{c['id']}: 缺 thin_source 标记")
    return problems


def write_kpi2_actual(cases: list[dict], kpi_path: Path = KPI) -> None:
    """回填 kpi2_scenario_coverage.actual——它的 command 就是本脚本本身，
    不需要 --freeze 才算数，此前一直是 null，没有脚本真正写过它。

    只做**局部文本替换**，不做整份 YAML 反序列化再写回：这份文件的大量
    注释是评审依据的一部分（比如 kpi1 那段"绝对达标率不得单独引用"的
    披露），`yaml.safe_dump` 会把注释全部吃掉。kpi1 与 guard_faithfulness
    也各有一处 `actual: null`，朴素全局替换会把它们一起改掉——用
    kpi2 独有的 `command: python3 pipeline/validate_testset.py` 这一行
    把替换范围锁定到 kpi2 那一处。
    """
    n = len({c["scenario"] for c in cases})
    text = kpi_path.read_text(encoding="utf-8")
    old = "    actual: null\n    command: python3 pipeline/validate_testset.py"
    new = f"    actual: {n}\n    command: python3 pipeline/validate_testset.py"
    if old in text:
        text = text.replace(old, new)
        kpi_path.write_text(text, encoding="utf-8")


def freeze(cases: list[dict]) -> None:
    import yaml

    text = KPI.read_text(encoding="utf-8")
    marker = "\n# --- frozen by validate_testset.py --freeze ---\n"
    text = text.split(marker)[0].rstrip() + "\n"

    seed = (yaml.safe_load(KPI.read_text(encoding="utf-8")) or {}).get("split_seed")
    hard = [c for c in cases if c["difficulty"] == "难"]
    scenarios = sorted({c["scenario"] for c in cases})

    text += marker
    text += f"dataset_sha256: {dataset_hash(cases)}\n"
    text += f"dataset_n_cases: {len(cases)}\n"
    text += f"dataset_n_scenarios: {len(scenarios)}\n"
    text += f"dataset_n_hard: {len(hard)}\n"
    text += f"dataset_split_seed: {seed}\n"
    KPI.write_text(text, encoding="utf-8")

    with open(HARD, "w", encoding="utf-8") as f:
        for c in sorted(hard, key=lambda c: c["id"]):
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="校验通过后冻结进 kpi.yaml")
    args = ap.parse_args()

    cases = load()
    problems = validate(cases) + check_leakage(cases) + check_coverage(cases) + check_flags(cases)

    print(f"测试集 {len(cases)} 题")
    by_s: dict[str, int] = {}
    by_d: dict[str, int] = {}
    for c in cases:
        by_s[c["scenario"]] = by_s.get(c["scenario"], 0) + 1
        by_d[c["difficulty"]] = by_d.get(c["difficulty"], 0) + 1

    print(f"\n{'子场景':<14}{'父类':<12}{'题数':>5}")
    print("-" * 34)
    for s in sorted(by_s, key=lambda s: -by_s[s]):
        print(f"{s:<14}{SCENARIO_PARENT[s][0]:<12}{by_s[s]:>5}")
    print("-" * 34)
    print(f"{'合计':<26}{len(cases):>5}   子场景 {len(by_s)}")

    print("\n难度：" + "  ".join(
        f"{d} {by_d.get(d,0)} ({by_d.get(d,0)/len(cases):.0%}，目标 {t:.0%})"
        for d, t in DIFFICULTY_TARGET.items()
    ))
    print(f"hash：{dataset_hash(cases)[:16]}…")

    if problems:
        print(f"\n[校验失败] {len(problems)} 个问题：", file=sys.stderr)
        for p in problems[:30]:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print("\n[OK] 校验通过")
    if KPI.exists():
        write_kpi2_actual(cases)
    if args.freeze:
        freeze(cases)
        print(f"已冻结进 kpi.yaml；hard 子集 {sum(1 for c in cases if c['difficulty']=='难')} 题 -> {HARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
