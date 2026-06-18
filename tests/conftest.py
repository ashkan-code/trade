"""Shared test fixtures for synthetic OHLCV DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_candle(ts, o, h, l, c, v=1000.0):
    return {"timestamp": ts, "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": float(v)}


def make_df(n: int = 50, trend: str = "up", base: float = 100.0) -> pd.DataFrame:
    """Synthetic closed-candle OHLCV DataFrame."""
    rows = []
    price = base
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    for i in range(n):
        if trend == "up":
            delta = np.random.uniform(0.1, 1.0)
        elif trend == "down":
            delta = np.random.uniform(-1.0, -0.1)
        else:
            delta = np.random.uniform(-0.5, 0.5)

        o = price
        c = price + delta
        h = max(o, c) + np.random.uniform(0.05, 0.3)
        l = min(o, c) - np.random.uniform(0.05, 0.3)
        rows.append(_make_candle(ts + pd.Timedelta(hours=i), o, h, l, c))
        price = c

    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def df_flat():
    return make_df(50, trend="flat", base=100.0)


@pytest.fixture
def df_up():
    return make_df(50, trend="up", base=100.0)


@pytest.fixture
def df_down():
    return make_df(50, trend="down", base=100.0)


@pytest.fixture
def df_with_bullish_mss():
    """
    Clear swing high at index 7 (high=120, all neighbours < 115).
    At index 22, close=121 > 120 → bullish MSS.
    Total: 30 bars.
    """
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    highs  = [101,102,103,104,105,106,107,  120,  108,107,106,105,104,103,
              104,105,106,107,108,109,110,111,  121,  122,123,124,125,126,127,128]
    lows   = [100,101,102,103,104,105,106,  119,  107,106,105,104,103,102,
              103,104,105,106,107,108,109,110,  120,  121,122,123,124,125,126,127]
    closes = [100,101,102,103,104,105,106,  119.5,107,106,105,104,103,102,
              103,104,105,106,107,108,109,110,  121,  122,123,124,125,126,127,128]
    opens  = [99, 100,101,102,103,104,105,  119,  107,106,105,104,103,102,
              103,104,105,106,107,108,109,110,  120,  121,122,123,124,125,126,127]

    for i in range(30):
        rows.append(_make_candle(
            ts + pd.Timedelta(hours=i),
            opens[i], highs[i], lows[i], closes[i],
        ))

    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def df_with_bearish_mss():
    """
    Clear swing low at index 7 (low=80, all neighbours > 85).
    At index 22, close=79 < 80 → bearish MSS.
    """
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    highs  = [119,118,117,116,115,114,113,  81,  112,113,114,115,116,117,
              116,115,114,113,112,111,110,109,  79.5, 78, 77, 76, 75, 74, 73, 72]
    lows   = [118,117,116,115,114,113,112,  80,  111,112,113,114,115,116,
              115,114,113,112,111,110,109,108,  78,   77, 76, 75, 74, 73, 72, 71]
    closes = [118.5,117.5,116.5,115.5,114.5,113.5,112.5, 80.5, 111.5,112.5,
              113.5,114.5,115.5,116.5,115.5,114.5,113.5,112.5,111.5,110.5,109.5,
              108.5, 79, 78, 77, 76, 75, 74, 73, 72]
    opens  = [119,118,117,116,115,114,113,  81,  112,113,114,115,116,117,
              116,115,114,113,112,111,110,109,  80,   79, 78, 77, 76, 75, 74, 73]

    for i in range(30):
        rows.append(_make_candle(
            ts + pd.Timedelta(hours=i),
            opens[i], highs[i], lows[i], closes[i],
        ))

    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def df_with_bullish_fvg():
    """
    3-candle window with a bullish FVG:
    Candle i:   high=100
    Candle i+1: large bullish body
    Candle i+2: low=101.5  → gap between 100 and 101.5
    """
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    # Warm-up candles
    for i in range(20):
        p = 98 + i * 0.05
        rows.append(_make_candle(ts + pd.Timedelta(hours=i), p, p+0.4, p-0.4, p+0.05))

    # FVG window
    rows.append(_make_candle(ts + pd.Timedelta(hours=20), 99.5, 100.0, 99.0, 99.8))    # candle i
    rows.append(_make_candle(ts + pd.Timedelta(hours=21), 100.2, 103.5, 100.1, 103.3)) # large bullish
    rows.append(_make_candle(ts + pd.Timedelta(hours=22), 101.5, 104.0, 101.5, 103.8)) # candle i+2, low>high[i]

    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def df_with_liquidity_cluster():
    """Multiple candles with highs clustered around 105."""
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")

    # Create swing highs at ~105 level
    for i in range(40):
        if i % 8 == 0:  # periodic highs
            h = 105.0 + np.random.uniform(-0.1, 0.1)
            l = 103.0
            o, c = 103.5, 103.8
        else:
            h = 103.5 + np.random.uniform(0, 0.5)
            l = 102.0 + np.random.uniform(0, 0.5)
            o, c = l + 0.2, l + 0.5
        rows.append(_make_candle(ts + pd.Timedelta(hours=i), o, h, l, c))

    return pd.DataFrame(rows).reset_index(drop=True)
