"""P2 一致率计分逻辑测试（ticket 12）。

按门槛要求"一致率按候选命中evidence/未命中两层分别报"——这里守的是分层
计分本身，不是教师判断准不准（那要等真实教师产物回来才知道）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifier.score_teacher_judgments import score  # noqa: E402


def test_agreement_split_by_derived_label():
    pool = [
        {"id": "a", "derived_label": True},
        {"id": "b", "derived_label": True},
        {"id": "c", "derived_label": False},
        {"id": "d", "derived_label": False},
    ]
    judged = [
        {"id": "a", "violated": True},   # 一致（正例）
        {"id": "b", "violated": False},  # 不一致（正例，教师漏判）
        {"id": "c", "violated": False},  # 一致（负例）
        {"id": "d", "violated": False},  # 一致（负例）
    ]
    m = score(pool, judged)
    assert m["overall_agreement"] == 0.75
    assert m["positive_agreement"] == 0.5
    assert m["negative_agreement"] == 1.0
    assert m["n_positive"] == 2 and m["n_negative"] == 2


def test_missing_judgment_counts_as_disagreement_not_silently_dropped():
    pool = [{"id": "a", "derived_label": True}, {"id": "b", "derived_label": False}]
    judged = [{"id": "a", "violated": True}]  # b 没judged
    m = score(pool, judged)
    assert m["n_missing"] == 1
    assert m["overall_agreement"] == 0.5
