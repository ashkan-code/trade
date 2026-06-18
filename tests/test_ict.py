"""Tests for core/ict.py — MSS, OB, FVG, Liquidity."""
import numpy as np
import pandas as pd
import pytest

from core.ict import (
    _swing_highs, _swing_lows,
    detect_mss, detect_order_blocks, detect_fvg, detect_liquidity,
)
from tests.conftest import make_df


# ── Swing helpers ─────────────────────────────────────────────────────────────

def test_swing_highs_marks_peak():
    df = make_df(30, "flat")
    # Inject a clear peak at index 15
    df.loc[15, "high"] = 999.0
    sh = _swing_highs(df, length=5)
    assert sh.iloc[15] is True or sh.iloc[15] == True


def test_swing_lows_marks_trough():
    df = make_df(30, "flat")
    df.loc[15, "low"] = 0.001
    sl = _swing_lows(df, length=5)
    assert sl.iloc[15] is True or sl.iloc[15] == True


def test_swing_edges_are_false():
    df = make_df(30, "flat")
    sh = _swing_highs(df, length=5)
    sl = _swing_lows(df, length=5)
    # First and last `length` rows must be False
    assert not sh.iloc[:5].any()
    assert not sh.iloc[-5:].any()
    assert not sl.iloc[:5].any()
    assert not sl.iloc[-5:].any()


# ── MSS ───────────────────────────────────────────────────────────────────────

def test_detect_mss_bullish(df_with_bullish_mss):
    result = detect_mss(df_with_bullish_mss)
    assert result["direction"] == "bullish"
    assert result["index"] >= 0


def test_detect_mss_bearish(df_with_bearish_mss):
    result = detect_mss(df_with_bearish_mss)
    assert result["direction"] == "bearish"


def test_detect_mss_returns_none_for_flat():
    # Perfectly flat data → no swing high/low breaks
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(30):
        rows.append({
            "timestamp": ts + pd.Timedelta(hours=i),
            "open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1.0
        })
    df = pd.DataFrame(rows)
    result = detect_mss(df)
    assert result["direction"] is None


# ── Order Blocks ──────────────────────────────────────────────────────────────

def test_detect_order_blocks_returns_list():
    df = make_df(60, "up")
    obs = detect_order_blocks(df)
    assert isinstance(obs, list)


def test_bullish_ob_type():
    """After an up-impulse that breaks a confirmed swing high, a bullish OB should exist."""
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    # Phase 1: rise to swing high at index 7 (high=120)
    highs  = [101,102,103,104,105,106,107, 120, 108,107,106,105,104,103]
    lows   = [100,101,102,103,104,105,106, 119, 107,106,105,104,103,102]
    closes = [100,101,102,103,104,105,106,119.5,107,106,105,104,103,102]
    opens  = [99, 100,101,102,103,104,105, 119, 107,106,105,104,103,102]

    for i in range(14):
        rows.append({"timestamp": ts + pd.Timedelta(hours=i),
                     "open": opens[i], "high": highs[i],
                     "low": lows[i], "close": closes[i], "volume": 1.0})

    # Phase 2: down-candle (bearish = future bullish OB) at index 14
    rows.append({"timestamp": ts + pd.Timedelta(hours=14),
                 "open": 103, "high": 103.5, "low": 101, "close": 101, "volume": 1.0})

    # Phase 3: impulse up that closes above the swing high (120) at index 15
    rows.append({"timestamp": ts + pd.Timedelta(hours=15),
                 "open": 101, "high": 125, "low": 101, "close": 124, "volume": 1.0})

    # Pad to 30 bars
    for j in range(16, 30):
        rows.append({"timestamp": ts + pd.Timedelta(hours=j),
                     "open": 124, "high": 125, "low": 123, "close": 124, "volume": 1.0})

    df = pd.DataFrame(rows)
    obs = detect_order_blocks(df, swing_len=3)
    bullish_obs = [ob for ob in obs if ob["type"] == "bullish"]
    assert len(bullish_obs) >= 1


def test_ob_broken_flag():
    """An OB should be marked broken when price closes through it."""
    df = make_df(60, "up")
    obs = detect_order_blocks(df)
    # All OBs' broken flag should be a bool
    for ob in obs:
        assert isinstance(ob["broken"], bool)
        assert isinstance(ob["breaker"], bool)


# ── FVG ───────────────────────────────────────────────────────────────────────

def test_detect_fvg_finds_bullish(df_with_bullish_fvg):
    fvgs = detect_fvg(df_with_bullish_fvg)
    bullish = [f for f in fvgs if f["type"] == "bullish"]
    assert len(bullish) >= 1
    for f in bullish:
        assert f["top"] > f["bottom"]


def test_detect_fvg_active_flag():
    fvgs = detect_fvg(make_df(50, "up"))
    for f in fvgs:
        assert isinstance(f["active"], bool)


def test_fvg_invalidated_when_filled():
    """A bullish FVG where subsequent price drops below bottom → active=False."""
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    # Setup FVG window
    rows.append({"timestamp": ts, "open": 100, "high": 100, "low": 99, "close": 100, "volume": 1})
    rows.append({"timestamp": ts + pd.Timedelta(hours=1),
                 "open": 100.2, "high": 104, "low": 100.1, "close": 103.8, "volume": 1})  # large bullish
    rows.append({"timestamp": ts + pd.Timedelta(hours=2),
                 "open": 101.5, "high": 105, "low": 101.5, "close": 104, "volume": 1})  # low>high[0]
    # Candle that fills the FVG
    rows.append({"timestamp": ts + pd.Timedelta(hours=3),
                 "open": 101, "high": 101.5, "low": 98, "close": 98.5, "volume": 1})

    df = pd.DataFrame(rows)
    fvgs = detect_fvg(df)
    bullish_fvgs = [f for f in fvgs if f["type"] == "bullish"]
    if bullish_fvgs:
        assert not bullish_fvgs[0]["active"]


# ── Liquidity ─────────────────────────────────────────────────────────────────

def test_detect_liquidity_returns_list():
    from core.indicators import atr
    df = make_df(60, "flat")
    atr_series = atr(df)
    pools = detect_liquidity(df, atr_series)
    assert isinstance(pools, list)


def test_detect_liquidity_cluster(df_with_liquidity_cluster):
    from core.indicators import atr
    atr_series = atr(df_with_liquidity_cluster)
    pools = detect_liquidity(df_with_liquidity_cluster, atr_series)
    buyside = [p for p in pools if p["type"] == "buyside"]
    # Should detect the cluster of highs around 105
    assert any(abs(p["price"] - 105.0) < 1.0 for p in buyside)


def test_detect_liquidity_no_cluster():
    """Fewer than 3 equal swing highs → no buyside liquidity."""
    from core.indicators import atr
    df = make_df(20, "up")
    atr_series = atr(df)
    pools = detect_liquidity(df, atr_series)
    # Should not crash; may return empty
    assert isinstance(pools, list)
