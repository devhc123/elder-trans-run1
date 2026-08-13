#!/usr/bin/env python3
"""把小模型与 deepseek 的逐条判定拼成级联，离线算端到端指标。

两轮跑的是**同一批 80 条**（抽样函数逐条一致），所以不用再调一次 deepseek。

级联语义（SPEC S3 的 (a) 形态）：
    小模型判「违规」-> 送 deepseek 复核，**以 deepseek 的判定为准**
    小模型判「没违规」-> 直接放行，**不再看**
因此：
    端到端漏报 = 小模型漏的 + 小模型抓到但 deepseek 放走的  ← 两段的漏报会叠加
    端到端误报 = 两段都判违规的负例                          ← 两段的误报会相乘（变少）
    送检率     = 小模型判违规的比例                          ← 决定成本与速度

**解析失败一律算「没判违规」**（与 eval_candidate_verifier.flagged 同口径）：
安全判别器不回答就是没拦住，不能算弃权。
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
DEEPSEEK_SEC = 3.29          # 实测中位单条耗时


def flag(row: dict) -> bool:
    return row.get("violated") is True


def main() -> int:
    ds = {r["id"]: r for r in json.loads((OUT / "poc_deepseek_result.json").read_text(encoding="utf-8"))}
    small = json.loads((OUT / "poc_ollama_result.json").read_text(encoding="utf-8"))

    # 先把 deepseek 单独的表现打出来当参照行
    d_pos = [r for r in ds.values() if r["label"]]
    d_neg = [r for r in ds.values() if not r["label"]]
    d_miss = sum(1 for r in d_pos if not flag(r)) / len(d_pos)
    d_fp = sum(1 for r in d_neg if flag(r)) / len(d_neg)

    print(f"{'配置':34s}{'漏报':>8}{'误报':>8}{'送检率':>8}{'端到端/条':>11}{'vs 全量ds':>10}")
    print("-" * 80)
    print(f"{'【全量 deepseek】(当前 harness 形态)':34s}{d_miss:>8.1%}{d_fp:>8.1%}"
          f"{'100%':>8}{DEEPSEEK_SEC:>10.2f}s{'1.0x':>10}")

    rows_out = []
    for model, rows in small.items():
        by_id = {r["id"]: r for r in rows}
        pos = [r for r in rows if r["label"]]
        neg = [r for r in rows if not r["label"]]
        s_miss = sum(1 for r in pos if not flag(r)) / len(pos)
        s_fp = sum(1 for r in neg if flag(r)) / len(neg)
        s_sec = sorted(r["seconds"] for r in rows)[len(rows) // 2]
        bad = sum(1 for r in rows if r["violated"] is None)

        # 级联
        sent = [r for r in rows if flag(r)]
        rate = len(sent) / len(rows)
        casc_tp = sum(1 for r in pos if flag(r) and flag(ds[r["id"]]))
        casc_fp = sum(1 for r in neg if flag(r) and flag(ds[r["id"]]))
        casc_miss = 1 - casc_tp / len(pos)
        casc_fpr = casc_fp / len(neg)
        casc_sec = s_sec + rate * DEEPSEEK_SEC

        print(f"{'  [单独] ' + model:34s}{s_miss:>8.1%}{s_fp:>8.1%}{'—':>8}"
              f"{s_sec:>10.2f}s{DEEPSEEK_SEC / s_sec:>9.1f}x"
              + (f"   解析失败 {bad}" if bad else ""))
        print(f"{'  [级联] ' + model + ' -> ds':34s}{casc_miss:>8.1%}{casc_fpr:>8.1%}"
              f"{rate:>8.0%}{casc_sec:>10.2f}s{DEEPSEEK_SEC / casc_sec:>9.1f}x")
        rows_out.append({"model": model, "solo_miss": s_miss, "solo_fp": s_fp,
                         "solo_sec": s_sec, "parse_fail": bad,
                         "cascade_miss": casc_miss, "cascade_fp": casc_fpr,
                         "send_rate": rate, "cascade_sec": casc_sec})

    print("\n门槛：漏报 ≤5%（硬门，不可恢复）/ 误报 ≤20%（成本旋钮）")
    print("**全部对 silver-ref 而言**——即「与本项目 rubric 一致」，不是「对」。")
    (OUT / "poc_cascade_summary.json").write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
