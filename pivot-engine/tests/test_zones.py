"""Tests for Gate 2: OB/FVG rejection detection."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from contracts import Zone
from engine.zones import validate_rejection, find_rejection, find_active_zones, _has_prior_touch


def _make_candle(o: float, h: float, l: float, c: float) -> pd.Series:
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "volume": 1.0})


def _make_zone(direction="long", zh=100.0, zl=95.0) -> Zone:
    return Zone(
        zone_type="ob", direction=direction,
        zone_high=zh, zone_low=zl,
        origin_index=0, timeframe="4h",
    )


# ── validate_rejection ────────────────────────────────────────────────────────

def test_long_rejection_A_plus():
    """Body entirely above zone, shadow dips into zone → A+."""
    zone = _make_zone("long", zh=100.0, zl=95.0)
    # open=102, close=103, low=97 (enters zone), high=104
    row = _make_candle(102, 104, 97, 103)
    passed, grade = validate_rejection(row, zone)
    assert passed
    assert grade == "A+"


def test_long_rejection_grade_B():
    """Shadow enters zone, close above zone_high, open inside zone → B."""
    zone = _make_zone("long", zh=100.0, zl=95.0)
    # open=99 (inside zone), close=101 (above zone_high), low=96 (enters zone)
    row = _make_candle(99, 102, 96, 101)
    passed, grade = validate_rejection(row, zone)
    assert passed
    assert grade == "B"


def test_long_no_touch():
    """Low does not enter zone → no rejection."""
    zone = _make_zone("long", zh=100.0, zl=95.0)
    row = _make_candle(102, 105, 101, 103)  # low=101 > zone_high=100
    passed, _ = validate_rejection(row, zone)
    assert not passed


def test_long_close_inside_zone():
    """Shadow enters zone but close stays inside → no rejection (breakout, not rejection)."""
    zone = _make_zone("long", zh=100.0, zl=95.0)
    row = _make_candle(102, 103, 97, 98)  # close=98 < zone_high=100
    passed, _ = validate_rejection(row, zone)
    assert not passed


def test_short_rejection_A_plus():
    """Body entirely below zone, shadow spikes into zone → A+."""
    zone = _make_zone("short", zh=105.0, zl=100.0)
    # open=98, close=97, high=103 (enters zone from below), low=96
    row = _make_candle(98, 103, 96, 97)
    passed, grade = validate_rejection(row, zone)
    assert passed
    assert grade == "A+"


def test_short_no_close_below():
    """High enters zone but close stays above zone_low → no rejection."""
    zone = _make_zone("short", zh=105.0, zl=100.0)
    row = _make_candle(98, 103, 97, 101)  # close=101 > zone_low=100
    passed, _ = validate_rejection(row, zone)
    assert not passed


# ── find_rejection ────────────────────────────────────────────────────────────

def _make_df_with_rejection(direction="long") -> tuple[pd.DataFrame, Zone]:
    """Create a small DF where the last candle is a valid rejection."""
    n = 15
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")

    if direction == "long":
        # Trending down then rejection at 100–95 zone
        closes = np.linspace(120, 102, n)
        opens = closes + 2
        highs = closes + 5
        lows = closes - 3
        # Last candle: rejection (shadow enters zone, close above zone_high)
        closes[-1] = 101.0
        opens[-1] = 102.0
        highs[-1] = 103.0
        lows[-1] = 97.0  # enters zone (zone_high=100)
        zone = Zone("ob", "long", zone_high=100.0, zone_low=95.0, origin_index=5, timeframe="4h")
    else:
        closes = np.linspace(80, 98, n)
        opens = closes - 2
        highs = closes + 3
        lows = closes - 5
        closes[-1] = 99.0
        opens[-1] = 98.0
        highs[-1] = 103.0  # enters zone (zone_low=100)
        lows[-1] = 97.0
        zone = Zone("ob", "short", zone_high=105.0, zone_low=100.0, origin_index=5, timeframe="4h")

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n)},
        index=idx,
    )
    return df, zone


def test_find_rejection_finds_valid():
    df, zone = _make_df_with_rejection("long")
    rejection = find_rejection(df, len(df) - 1, [zone])
    assert rejection is not None
    assert rejection.candle_index == len(df) - 1


def test_find_rejection_single_touch_rule():
    """If zone was already touched, reject it."""
    df, zone = _make_df_with_rejection("long")
    # Inject a prior touch at bar 10
    df_mod = df.copy()
    df_mod.iloc[10, df_mod.columns.get_loc("low")] = 99.0  # enters zone_high=100
    rejection = find_rejection(df_mod, len(df_mod) - 1, [zone])
    # Prior touch should block the rejection
    assert rejection is None


# ── find_active_zones ──────────────────────────────────────────────────────────

def test_find_active_zones_returns_list():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    closes = np.linspace(100, 120, n)
    opens = closes - 1
    highs = closes + 2
    lows = closes - 2
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": np.ones(n)},
        index=idx,
    )
    zones = find_active_zones(df, "long", "4h")
    assert isinstance(zones, list)
