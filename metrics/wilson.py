"""Wilson score 区间。

比正态近似稳：样本小或比例贴近 0/1 时，正态近似会给出越界或过窄的区间，
而本项目的场景子集只有 15–47 题，正落在那个区间。
"""
from __future__ import annotations

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))
