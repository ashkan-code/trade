"""Tests for backtest/backtest.py."""
import pytest

from backtest.backtest import compute_metrics, TradeDict, BacktestResult


def _trade(pnl_r: float, direction: str = "bullish") -> TradeDict:
    return TradeDict(
        symbol="BTCUSDT", direction=direction,
        entry=100.0, stop=98.0, target=106.0, rr=3.0,
        entry_index=10, exit_index=15,
        exit_price=106.0 if pnl_r > 0 else 98.0,
        outcome="win" if pnl_r > 0 else "loss",
        pnl_r=pnl_r,
    )


def test_compute_metrics_empty():
    result = compute_metrics([])
    assert result["total_trades"] == 0
    assert result["win_rate"] == 0.0
    assert result["profit_factor"] == 0.0


def test_compute_metrics_all_wins():
    trades = [_trade(2.0), _trade(2.0), _trade(3.0)]
    result = compute_metrics(trades)
    assert result["total_trades"] == 3
    assert result["wins"] == 3
    assert result["losses"] == 0
    assert result["win_rate"] == pytest.approx(1.0)
    assert result["profit_factor"] == float("inf")


def test_compute_metrics_known_values():
    # 3 wins × +2R, 1 loss × -1R
    trades = [_trade(2.0), _trade(2.0), _trade(2.0), _trade(-1.0)]
    result = compute_metrics(trades)
    assert result["total_trades"] == 4
    assert result["wins"] == 3
    assert result["win_rate"] == pytest.approx(0.75)
    assert result["profit_factor"] == pytest.approx(6.0)


def test_compute_metrics_max_drawdown():
    # Sequential losses then recovery
    trades = [_trade(2.0), _trade(-1.0), _trade(-1.0), _trade(3.0)]
    result = compute_metrics(trades)
    assert result["max_drawdown_r"] == pytest.approx(2.0)


def test_compute_metrics_avg_rr():
    trades = [_trade(3.0), _trade(-1.0)]
    result = compute_metrics(trades)
    assert result["avg_rr_realized"] == pytest.approx(1.0)
