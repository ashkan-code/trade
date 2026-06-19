"""ICT structure detection: MSS, Order Blocks, FVGs, ATR, Liquidity.

All functions accept a CLOSED-candle DataFrame sliced to as_of (df[:as_of+1]).
The caller is responsible for causal slicing — no lookahead protection here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Zone


# ── helpers ──────────────────────────────────────────────────────────────────

def _swing_high_indices(highs: np.ndarray, length: int) -> list[int]:
    """Indices of confirmed swing highs (centered window, edges excluded)."""
    n = len(highs)
    out = []
    for i in range(length, n - length):
        window = np.concatenate([highs[i - length : i], highs[i + 1 : i + length + 1]])
        if len(window) and highs[i] > window.max():
            out.append(i)
    return out


def _swing_low_indices(lows: np.ndarray, length: int) -> list[int]:
    """Indices of confirmed swing lows (centered window, edges excluded)."""
    n = len(lows)
    out = []
    for i in range(length, n - length):
        window = np.concatenate([lows[i - length : i], lows[i + 1 : i + length + 1]])
        if len(window) and lows[i] < window.min():
            out.append(i)
    return out


# ── ATR ──────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's smoothed ATR as a Series aligned to df's index."""
    h = df["high"]
    l = df["low"]
    c = df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def atr_scalar(df: pd.DataFrame, period: int) -> float:
    """ATR at the last bar of df (scalar)."""
    s = atr(df, period)
    return float(s.iloc[-1]) if len(s) else 0.0


# ── MSS ──────────────────────────────────────────────────────────────────────

def detect_mss(df: pd.DataFrame) -> Direction | None:
    """Detect the most recent Market Structure Shift.

    Scans oldest→newest. Tracks last confirmed swing high/low.
    Bullish MSS: close > last swing high (and state not already bullish).
    Bearish MSS: close < last swing low (and state not already bearish).
    Returns the LAST detected direction, or None if never detected.
    """
    if len(df) < config.SWING_LEN * 2 + 1:
        return None

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)

    sh_idx = _swing_high_indices(highs, config.SWING_LEN)
    sl_idx = _swing_low_indices(lows, config.SWING_LEN)

    last_sh_price: float | None = None
    last_sl_price: float | None = None
    state: Direction | None = None
    result: Direction | None = None

    # Build sorted event list: (index, type, price)
    events: list[tuple[int, str, float]] = []
    for idx in sh_idx:
        events.append((idx, "sh", float(highs[idx])))
    for idx in sl_idx:
        events.append((idx, "sl", float(lows[idx])))
    events.sort(key=lambda e: e[0])

    event_cursor = 0

    for i in range(n):
        # Update swing references as they confirm
        while event_cursor < len(events) and events[event_cursor][0] <= i:
            _, etype, eprice = events[event_cursor]
            if etype == "sh":
                last_sh_price = eprice
            else:
                last_sl_price = eprice
            event_cursor += 1

        close = closes[i]
        if last_sh_price is not None and close > last_sh_price and state != "long":
            state = "long"
            result = "long"
        elif last_sl_price is not None and close < last_sl_price and state != "short":
            state = "short"
            result = "short"

    return result


# ── Order Blocks ──────────────────────────────────────────────────────────────

def detect_order_blocks(df: pd.DataFrame, direction: Direction, timeframe: str = "") -> list[Zone]:
    """Find unbroken order blocks aligned with direction.

    Bullish OB (long): last bearish candle before an up-impulse that breaks a swing high.
    Bearish OB (short): last bullish candle before a down-impulse that breaks a swing low.
    Only returns OBs where price has not subsequently closed through the far edge.
    """
    n = len(df)
    if n < config.OB_SWING_LOOKBACK + config.SWING_LEN + 2:
        return []

    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    sh_idx = _swing_high_indices(highs, config.SWING_LEN)
    sl_idx = _swing_low_indices(lows, config.SWING_LEN)

    zones: list[Zone] = []
    seen: set[int] = set()

    if direction == "long":
        for i in range(config.SWING_LEN + 1, n):
            # Look for breakout above a recent swing high
            recent = [j for j in sh_idx if i - config.OB_SWING_LOOKBACK <= j < i]
            if not recent:
                continue
            ref_price = max(float(highs[j]) for j in recent)
            if closes[i] <= ref_price:
                continue
            # Impulse confirmed: find last bearish candle in lookback window
            ob_idx: int | None = None
            for j in range(i - 1, max(-1, i - config.OB_SWING_LOOKBACK - 1), -1):
                if closes[j] < opens[j]:
                    ob_idx = j
                    break
            if ob_idx is None or ob_idx in seen:
                continue
            ob_high = float(max(opens[ob_idx], closes[ob_idx]))
            ob_low = float(lows[ob_idx])
            if ob_high <= ob_low:
                continue
            # Check unbroken: no close below ob_low after ob_idx
            if any(closes[k] < ob_low for k in range(ob_idx + 1, n)):
                continue
            seen.add(ob_idx)
            zones.append(Zone(
                zone_type="ob", direction="long",
                zone_high=ob_high, zone_low=ob_low,
                origin_index=ob_idx, timeframe=timeframe,
            ))

    else:  # short
        for i in range(config.SWING_LEN + 1, n):
            recent = [j for j in sl_idx if i - config.OB_SWING_LOOKBACK <= j < i]
            if not recent:
                continue
            ref_price = min(float(lows[j]) for j in recent)
            if closes[i] >= ref_price:
                continue
            ob_idx = None
            for j in range(i - 1, max(-1, i - config.OB_SWING_LOOKBACK - 1), -1):
                if closes[j] > opens[j]:
                    ob_idx = j
                    break
            if ob_idx is None or ob_idx in seen:
                continue
            ob_high = float(highs[ob_idx])
            ob_low = float(min(opens[ob_idx], closes[ob_idx]))
            if ob_high <= ob_low:
                continue
            if any(closes[k] > ob_high for k in range(ob_idx + 1, n)):
                continue
            seen.add(ob_idx)
            zones.append(Zone(
                zone_type="ob", direction="short",
                zone_high=ob_high, zone_low=ob_low,
                origin_index=ob_idx, timeframe=timeframe,
            ))

    return zones


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame, direction: Direction, timeframe: str = "") -> list[Zone]:
    """Find active (unfilled) Fair Value Gaps aligned with direction.

    3-candle window [i, i+1, i+2]:
    Bullish FVG: low[i+2] > high[i]  (gap above candle i, below candle i+2).
    Bearish FVG: high[i+2] < low[i]  (gap below candle i, above candle i+2).
    Middle candle (i+1) must have body >= FVG_BODY_PERC * range.
    Active = price has not completely filled the gap.
    """
    n = len(df)
    if n < 3:
        return []

    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    zones: list[Zone] = []

    for i in range(n - 2):
        mid = i + 1
        end = i + 2

        # Middle candle body check
        mid_range = highs[mid] - lows[mid]
        mid_body = abs(closes[mid] - opens[mid])
        if mid_range <= 0 or mid_body / mid_range < config.FVG_BODY_PERC:
            continue

        if direction == "long":
            gap_low = float(highs[i])
            gap_high = float(lows[end])
            if gap_high <= gap_low:
                continue
            # Check still active (price hasn't closed below gap_low)
            filled = any(lows[k] <= gap_low for k in range(end, n))
            if filled:
                continue
            zones.append(Zone(
                zone_type="fvg", direction="long",
                zone_high=gap_high, zone_low=gap_low,
                origin_index=mid, timeframe=timeframe,
            ))

        else:  # short
            gap_high = float(lows[i])
            gap_low = float(highs[end])
            if gap_low >= gap_high:
                continue
            filled = any(highs[k] >= gap_high for k in range(end, n))
            if filled:
                continue
            zones.append(Zone(
                zone_type="fvg", direction="short",
                zone_high=gap_high, zone_low=gap_low,
                origin_index=mid, timeframe=timeframe,
            ))

    return zones


# ── Liquidity Pools ───────────────────────────────────────────────────────────

def find_liquidity(df: pd.DataFrame, direction: Direction, entry: float) -> list[float]:
    """Find opposing liquidity levels (swing highs above entry for long, swing lows below for short).

    Returns sorted list of prices (ascending for short targets, descending for long targets).
    """
    if len(df) < config.SWING_LEN * 2 + 1:
        return []

    highs = df["high"].values
    lows = df["low"].values

    if direction == "long":
        sh_idx = _swing_high_indices(highs, config.SWING_LEN)
        levels = sorted(
            {float(highs[j]) for j in sh_idx if float(highs[j]) > entry},
        )
        return levels  # ascending; first = nearest target

    else:
        sl_idx = _swing_low_indices(lows, config.SWING_LEN)
        levels = sorted(
            {float(lows[j]) for j in sl_idx if float(lows[j]) < entry},
            reverse=True,
        )
        return levels  # descending; first = nearest target
