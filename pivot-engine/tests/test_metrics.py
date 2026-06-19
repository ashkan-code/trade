"""Tests for Phase 4: metrics computation."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import config
from contracts import TradeResult
from backtest.metrics import compute, _wilson_ci


def _results(wins: int, losses: int, r_win: float = 2.0, r_loss: float = -1.0) -> list[TradeResult]:
    return (
        [TradeResult(won=True, r_multiple=r_win, bars_held=10)] * wins
        + [TradeResult(won=False, r_multiple=r_loss, bars_held=10)] * losses
    )


def test_empty_results():
    m = compute([])
    assert m.flag == "unreliable"
    assert m.trades == 0


def test_unreliable_below_min_trades():
    # MIN_TRADES=30; use 10 trades (clearly below threshold)
    m = compute(_results(5, 5))
    assert m.flag == "unreliable"
    assert m.trades == 10


def test_costs_reflected_in_r():
    """r_multiple must be net — already baked in by simulator; metrics just aggregate."""
    results = [TradeResult(won=True, r_multiple=1.94, bars_held=10)]  # gross 2.0, net 1.94 after costs
    m = compute(results)
    assert abs(m.expectancy_r - 1.94) < 1e-9


def test_wilson_ci_contains_wr():
    wins, n = 55, 100
    lo, hi = _wilson_ci(wins, n)
    wr = wins / n
    assert lo < wr < hi, f"Wilson CI [{lo:.3f}, {hi:.3f}] must contain WR {wr:.3f}"


def test_wilson_ci_bounds():
    lo, hi = _wilson_ci(0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    lo2, hi2 = _wilson_ci(10, 10)
    assert 0.0 <= lo2 <= hi2 <= 1.0


def test_valid_flag():
    """High enough trades with positive CI low → valid."""
    # 65% WR, 100 trades, Wilson lower CI > 0.5
    results = _results(65, 35, r_win=2.0, r_loss=-1.0)
    m = compute(results)
    lo, _ = _wilson_ci(65, 100)
    if lo >= 0.5:
        assert m.flag == "valid"


def test_expectancy_sign():
    winning = _results(70, 30, r_win=2.0, r_loss=-1.0)
    m = compute(winning)
    assert m.expectancy_r > 0

    losing = _results(20, 80, r_win=1.0, r_loss=-2.0)
    m2 = compute(losing)
    assert m2.expectancy_r < 0


def test_walk_forward_folds():
    results = _results(60, 40, r_win=2.0, r_loss=-1.0)
    m = compute(results)
    assert 0 <= m.folds_positive <= config.FOLDS
