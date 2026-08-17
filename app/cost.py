"""deepseek 开销记账 + 预算门（finetune-gguf 北极星推论二：每 batch ≤ ¥5）。

单价默认按 deepseek-v4-flash **峰时**输出 $1.32/M ≈ ¥9.5/M（谷时一半），输入 ¥1.0/M（粗略）；
用环境变量 DEEPSEEK_OUT_CNY_PER_M / DEEPSEEK_IN_CNY_PER_M 覆盖。宁可高估，别低估。
"""
from __future__ import annotations

import os
import threading

PRICE_OUT = float(os.environ.get("DEEPSEEK_OUT_CNY_PER_M", "9.5"))
PRICE_IN = float(os.environ.get("DEEPSEEK_IN_CNY_PER_M", "1.0"))
DEFAULT_BUDGET_CNY = 5.0


def est_cny(usage: dict | None) -> float:
    u = usage or {}
    return (u.get("completion_tokens", 0) * PRICE_OUT + u.get("prompt_tokens", 0) * PRICE_IN) / 1e6


class Budget:
    """线程安全的累计器。add() 返回是否仍在预算内；exceeded 一旦为 True 不再翻回。"""

    def __init__(self, limit_cny: float = DEFAULT_BUDGET_CNY, label: str = "batch"):
        self.limit = limit_cny
        self.label = label
        self.spent = 0.0
        self.prompt = 0
        self.completion = 0
        self.n = 0
        self.exceeded = False
        self._lock = threading.Lock()

    def add(self, usage: dict | None) -> bool:
        with self._lock:
            u = usage or {}
            self.prompt += u.get("prompt_tokens", 0)
            self.completion += u.get("completion_tokens", 0)
            self.spent += est_cny(u)
            self.n += 1
            if self.spent > self.limit:
                self.exceeded = True
            return not self.exceeded

    def summary(self) -> str:
        return (f"{self.label}: {self.n} 次调用，prompt {self.prompt} / completion {self.completion} tok，"
                f"估算 ¥{self.spent:.2f} / 预算 ¥{self.limit:.2f}"
                f"（单价 out ¥{PRICE_OUT}/M in ¥{PRICE_IN}/M）"
                + ("  ⚠️ 已超预算，后续调用被截停" if self.exceeded else ""))
