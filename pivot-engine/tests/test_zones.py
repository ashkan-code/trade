"""Tests for Gate 2: OB/FVG rejection detection and sweep detection."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from contracts import Zone
from engine.zones import (
    validate_rejection, find_rejection, find_active_zones, _has_prior_touch,
    find_sweep_rejection, diagnose_sweep_rejection, _volume_sma,
)


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
    # shadow_extreme must be the candle low for long
    assert rejection.shadow_extreme == float(df.iloc[-1]["low"])


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


# ── find_sweep_rejection ──────────────────────────────────────────────────────

def _make_sweep_df(
    direction: str,
    n: int = 30,
    vol_spike_bar: int | None = None,
    vol_spike_bullish: bool = True,
    rejection_vol_ratio: float = 1.5,
) -> pd.DataFrame:
    """Build a synthetic DF where the last candle is a sweep+reclaim.

    direction='short': price rallies, last bar spikes above recent high then closes below it.
    direction='long':  price drops, last bar spikes below recent low then closes above it.
    vol_spike_bar: if set, that bar gets a 3× avg volume candle in `vol_spike_bullish` direction.
    """
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    base_vol = 1_000_000.0
    avg_price = 100.0

    opens  = np.full(n, avg_price)
    closes = np.full(n, avg_price)
    highs  = np.full(n, avg_price + 1.0)
    lows   = np.full(n, avg_price - 1.0)
    vols   = np.full(n, base_vol)

    if direction == "short":
        # Establish a recent swing high at ~105 in the middle of the window
        mid = n - 8
        highs[mid] = avg_price + 5.0  # swing high = 105
        closes[mid] = avg_price + 4.0
        opens[mid]  = avg_price + 3.0
        # Last candle: sweeps above 105 and closes back below
        opens[-1]  = avg_price + 4.5
        highs[-1]  = avg_price + 6.0   # spike above 105
        lows[-1]   = avg_price + 2.0
        closes[-1] = avg_price + 2.0   # close BELOW 105 (reclaim)
        vols[-1]   = base_vol * rejection_vol_ratio
    else:  # long
        mid = n - 8
        lows[mid]  = avg_price - 5.0   # swing low = 95
        closes[mid]= avg_price - 4.0
        opens[mid] = avg_price - 3.0
        # Last candle: sweeps below 95 and closes back above
        opens[-1]  = avg_price - 4.5
        lows[-1]   = avg_price - 6.0   # spike below 95
        highs[-1]  = avg_price - 2.0
        closes[-1] = avg_price - 2.0   # close ABOVE 95 (reclaim)
        vols[-1]   = base_vol * rejection_vol_ratio

    if vol_spike_bar is not None:
        spike_vol = base_vol * 3.0
        vols[vol_spike_bar] = spike_vol
        if vol_spike_bullish:
            opens[vol_spike_bar]  = avg_price - 0.5
            closes[vol_spike_bar] = avg_price + 1.5
        else:
            opens[vol_spike_bar]  = avg_price + 0.5
            closes[vol_spike_bar] = avg_price - 1.5

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def test_sweep_short_passes_clean():
    """Clean SHORT sweep: spike above swing high, close below it, normal volume, no counter spike."""
    df = _make_sweep_df("short")
    result = find_sweep_rejection(df, "short", "1h")
    assert result is not None, "Expected sweep rejection to pass"
    assert result.zone.direction == "short"


def test_sweep_long_passes_clean():
    """Clean LONG sweep: spike below swing low, close above it, normal volume, no counter spike."""
    df = _make_sweep_df("long")
    result = find_sweep_rejection(df, "long", "1h")
    assert result is not None, "Expected sweep rejection to pass"
    assert result.zone.direction == "long"


def test_sweep_blocked_counter_momentum_short():
    """SHORT sweep blocked by a large BULLISH volume spike (counter-institutional momentum).

    This simulates the TOSHI scenario: a large institutional bullish candle in the
    lookback window should block the SHORT sweep signal.
    """
    # Place a bullish spike 3 bars before the last candle (within LOOKBACK_BARS=5)
    df = _make_sweep_df("short", vol_spike_bar=-4, vol_spike_bullish=True)
    result = find_sweep_rejection(df, "short", "1h")
    assert result is None, "Expected SHORT sweep to be BLOCKED by counter-momentum (bullish spike)"

    reason = diagnose_sweep_rejection(df, "short")
    assert reason == "counter_momentum", f"Expected 'counter_momentum', got '{reason}'"


def test_sweep_blocked_counter_momentum_long():
    """LONG sweep blocked by a large BEARISH volume spike."""
    df = _make_sweep_df("long", vol_spike_bar=-4, vol_spike_bullish=False)
    result = find_sweep_rejection(df, "long", "1h")
    assert result is None, "Expected LONG sweep to be BLOCKED by counter-momentum (bearish spike)"

    reason = diagnose_sweep_rejection(df, "long")
    assert reason == "counter_momentum", f"Expected 'counter_momentum', got '{reason}'"


def test_sweep_blocked_outside_lookback_not_blocked():
    """Counter-momentum spike OUTSIDE LOOKBACK_BARS should NOT block the signal."""
    import config
    # Place spike far outside the lookback window
    df = _make_sweep_df("short", vol_spike_bar=5, vol_spike_bullish=True)
    result = find_sweep_rejection(df, "short", "1h")
    assert result is not None, (
        "Counter-momentum outside LOOKBACK_BARS window should not block signal"
    )


def test_sweep_no_reclaim_blocked():
    """Last candle sweeps above recent high but does NOT close below → blocked."""
    df = _make_sweep_df("short")
    df_mod = df.copy()
    # Make close stay above the swept level (no reclaim)
    df_mod.iloc[-1, df_mod.columns.get_loc("close")] = 106.5
    result = find_sweep_rejection(df_mod, "short", "1h")
    assert result is None

    reason = diagnose_sweep_rejection(df_mod, "short")
    assert reason == "sweep_not_found"


def test_sweep_low_volume_blocked():
    """Rejection candle volume below VOLUME_MIN_RATIO × SMA → blocked."""
    df = _make_sweep_df("short", rejection_vol_ratio=0.3)  # only 30% of avg vol
    result = find_sweep_rejection(df, "short", "1h")
    assert result is None

    reason = diagnose_sweep_rejection(df, "short")
    assert reason == "volume_min_ratio"


def test_volume_sma():
    """_volume_sma returns correct simple average."""
    n = 15
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    vols = np.ones(n) * 100.0
    vols[5] = 200.0  # bump one bar
    df = pd.DataFrame({"volume": vols, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}, index=idx)
    # SMA over last 9 bars ending at idx=14 (bars 6..14) = average of 9 bars
    sma = _volume_sma(df, 14, period=9)
    expected = float(np.mean(vols[6:15]))
    assert abs(sma - expected) < 1e-9
