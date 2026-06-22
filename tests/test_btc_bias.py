"""Tests for btc_bias() — Gate 0 current-state BTC direction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.indicators import btc_bias


def _make_btc_df(n: int = 200, trend: str = "up", base: float = 60000.0) -> pd.DataFrame:
    """Synthetic 4H OHLCV with controllable trend direction."""
    rng = np.random.default_rng(42)
    if trend == "up":
        drift = 0.003
    elif trend == "down":
        drift = -0.003
    else:
        drift = 0.0

    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + drift + rng.normal(0, 0.002)))

    closes = np.array(closes)
    highs  = closes * (1 + abs(rng.normal(0, 0.003, n)))
    lows   = closes * (1 - abs(rng.normal(0, 0.003, n)))
    opens  = np.roll(closes, 1); opens[0] = closes[0]

    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": rng.uniform(100, 500, n)},
        index=idx,
    )


def test_btc_bias_bullish():
    """Strong uptrend → 'long'."""
    df = _make_btc_df(n=200, trend="up")
    direction, dbg = btc_bias(df)
    assert direction == "long", f"Expected 'long' for uptrend, got '{direction}'. debug={dbg}"
    assert dbg["score"] >= config.BTC_SCORE_LONG


def test_btc_bias_bearish():
    """Strong downtrend → 'short'."""
    df = _make_btc_df(n=200, trend="down")
    direction, dbg = btc_bias(df)
    assert direction == "short", f"Expected 'short' for downtrend, got '{direction}'. debug={dbg}"
    assert dbg["score"] <= config.BTC_SCORE_SHORT


def test_btc_bias_neutral_flat_price():
    """Completely flat price → EMA cross=-1, slope=0, MACD=0, RSI=NaN(→0) → score=-1 → 'neutral'."""
    n = 200
    closes = np.full(n, 60000.0)
    highs  = closes + 50.0
    lows   = closes - 50.0
    opens  = closes.copy()
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": np.ones(n) * 300},
        index=idx,
    )
    direction, dbg = btc_bias(df)
    assert direction == "neutral", f"Expected 'neutral' for flat price, got '{direction}'. debug={dbg}"
    assert -2 < dbg["score"] < 2, f"Score {dbg['score']} should be inconclusive"


def test_btc_bias_insufficient_data():
    """Fewer than 60 bars → 'neutral' with reason 'insufficient_data'."""
    df = _make_btc_df(n=30, trend="up")
    direction, dbg = btc_bias(df)
    assert direction == "neutral"
    assert dbg.get("reason") == "insufficient_data"


def test_btc_bias_debug_dict_keys():
    """Debug dict must contain all keys used by _print_btc_bias_debug."""
    df = _make_btc_df(n=200, trend="up")
    _, dbg = btc_bias(df)
    for key in ("ema20", "ema50", "ema_cross", "ema_slope",
                "macd_hist", "macd_sig", "rsi", "rsi_sig", "score",
                "pos_pct", "range", "reason"):
        assert key in dbg, f"Missing key '{key}' in debug dict"


def test_btc_bias_score_components():
    """Each signal component is in {-1, 0, 1} and score equals their sum."""
    df = _make_btc_df(n=200, trend="up")
    _, dbg = btc_bias(df)
    for key in ("ema_cross", "ema_slope", "macd_sig", "rsi_sig"):
        assert dbg[key] in (-1, 0, 1), f"Signal '{key}' = {dbg[key]} not in {{-1,0,1}}"
    expected = sum(dbg[k] for k in ("ema_cross", "ema_slope", "macd_sig", "rsi_sig"))
    assert dbg["score"] == expected


def test_check_btc_direction_maps_neutral_to_none():
    """check_btc_direction returns None when btc_bias returns 'neutral' (scanner aborts)."""
    from core.confluence import check_btc_direction
    n = 200
    closes = np.full(n, 60000.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": closes, "high": closes + 50, "low": closes - 50,
         "close": closes, "volume": np.ones(n) * 300},
        index=idx,
    )
    result = check_btc_direction(df)
    assert result is None, f"Expected None for neutral BTC, got '{result}'"


def test_check_btc_direction_maps_long_to_bullish():
    """check_btc_direction maps 'long' → 'bullish'."""
    from core.confluence import check_btc_direction
    df = _make_btc_df(n=200, trend="up")
    result = check_btc_direction(df)
    assert result == "bullish", f"Expected 'bullish', got '{result}'"


def test_check_btc_direction_maps_short_to_bearish():
    """check_btc_direction maps 'short' → 'bearish'."""
    from core.confluence import check_btc_direction
    df = _make_btc_df(n=200, trend="down")
    result = check_btc_direction(df)
    assert result == "bearish", f"Expected 'bearish', got '{result}'"
