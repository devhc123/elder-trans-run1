import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.cost import Budget, est_cny, PRICE_OUT, PRICE_IN  # noqa: E402


def test_est_cny_formula():
    u = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert abs(est_cny(u) - (PRICE_OUT + PRICE_IN)) < 1e-9
    assert est_cny(None) == 0.0 and est_cny({}) == 0.0


def test_budget_exceeds_and_sticks():
    b = Budget(limit_cny=0.01, label="t")
    assert b.add({"completion_tokens": 100}) is True          # 100 tok ≈ ¥0.00095
    assert b.add({"completion_tokens": 2000}) is False        # 累计 ≈ ¥0.02 > 0.01
    assert b.exceeded
    assert b.add({}) is False                                 # 不会翻回
    assert "已超预算" in b.summary() and b.n == 3
