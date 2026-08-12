#!/usr/bin/env python3
"""两段式红线判别的段 B：候选池派生 gold + 分层抽样（ticket 12 P2）。

派生 gold 不是新标注，是把已有 case 级教师标注**翻译**成候选级的近似真值：
一个候选（词/数）如果落在某条已判违规的红线（0/2）evidence 子串里，就是
派生正例；否则是派生负例。用它来评「教师逐候选判定」这个更窄 prompt 的
一致率——教师本来就标过 case 级，这里只是换一种更细的粒度问同一件事。

用法：
    python3 verifier/judge_candidates.py --build   # 生成待判定候选包
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.redline_candidates import (  # noqa: E402
    COVERED_RED_LINE_IDXS,
    extract_candidates,
    load_jargon,
    parse_case_text,
)

WORK = ROOT / "verifier" / "work"
SEED = 20260813


def derive_candidate_label(candidate_text: str, gold: dict) -> bool:
    """候选文本是否落在某条**已判违规**的红线 0/2 evidence 子串里。"""
    for rl in gold["red_lines"]:
        if rl["idx"] not in COVERED_RED_LINE_IDXS:
            continue
        if rl["violated"] and candidate_text in rl["evidence"]:
            return True
    return False


def build_candidate_pool(records: list[dict], jargon: set[str]) -> list[dict]:
    """对每条 case 抽候选，挂上 case 上下文与派生 label。"""
    pool: list[dict] = []
    for r in records:
        src, ans = parse_case_text(r["input"])
        gold = json.loads(r["output"])
        for c in extract_candidates(src, ans, jargon):
            pool.append({
                "case_id": r["case_id"],
                "source_text": src,
                "answer": ans,
                "candidate_text": c["text"],
                "kind": c["kind"],
                "red_line_guess": c["red_line_guess"],
                "derived_label": derive_candidate_label(c["text"], gold),
            })
    return pool


def stratified_sample(pool: list[dict], n_per_class: int, seed: int = SEED) -> list[dict]:
    """按 derived_label 分层抽样，每类最多 n_per_class 条，类内按内容哈希
    排序取前 n（与 `build_teacher_labels.sample()` 同一套确定性抽样手法：
    结果只取决于 seed，不取决于 pool 的原始顺序）。"""
    def key_material(it: dict) -> str:
        # 真实候选池用 case_id+candidate_text 唯一标识；测试 fixture 里直接
        # 带了 "id"——两种输入都要能排出确定性顺序。
        return it.get("id") or f"{it.get('case_id', '')}|{it.get('candidate_text', '')}"

    out: list[dict] = []
    for label in (True, False):
        cls = [item for item in pool if item["derived_label"] is label]
        cls.sort(key=lambda it: hashlib.sha256(f"{seed}:{label}:{key_material(it)}".encode()).hexdigest())
        out += cls[:n_per_class]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="生成待判定候选包")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--split", choices=["train", "holdout"], default="holdout")
    ap.add_argument("--out", type=Path, default=WORK / "candidate_judge_packet.json")
    args = ap.parse_args()
    if not args.build:
        ap.print_help()
        return 1

    records = [
        json.loads(line)
        for line in (ROOT / "verifier" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    jargon = load_jargon()
    pool = build_candidate_pool(records, jargon)
    sampled = stratified_sample(pool, n_per_class=args.n_per_class)
    for i, item in enumerate(sampled):
        item["id"] = f"cand-{i:04d}"

    n_pos = sum(1 for i in sampled if i["derived_label"] is True)
    n_neg = sum(1 for i in sampled if i["derived_label"] is False)
    print(f"候选池 {len(pool)} 条（{args.split}），抽样 {len(sampled)} 条"
          f"（正例 {n_pos} / 负例 {n_neg}）")

    WORK.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
