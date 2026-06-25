"""Causal pivot detection. The as_of parameter is the contract: never touch df.iloc[i > as_of]."""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Pivot


def find_pivots(df: pd.DataFrame, as_of: int, lookback: int) -> list[Pivot]:
    """Return all confirmed swing highs and lows visible at as_of.

    A swing low at index i is confirmed when:
      - low[i] < low[j] for all j in [i-lookback, i+lookback], j != i
      - confirm_index = i + lookback <= as_of  (causal gate)

    A swing high mirrors the above using high values.
    """
    view = df.iloc[: as_of + 1]
    lows = view["low"].to_numpy()
    highs = view["high"].to_numpy()
    n = len(view)
    pivots: list[Pivot] = []

    for i in range(lookback, n - lookback):
        confirm_idx = i + lookback
        if confirm_idx > as_of:
            break  # not confirmed yet — stop early (array is ordered)

        window_l = np.concatenate([lows[i - lookback : i], lows[i + 1 : i + lookback + 1]])
        if lows[i] < window_l.min():
            pivots.append(
                Pivot(index=i, confirm_index=confirm_idx, price=float(lows[i]), kind="low")
            )

        window_h = np.concatenate([highs[i - lookback : i], highs[i + 1 : i + lookback + 1]])
        if highs[i] > window_h.max():
            pivots.append(
                Pivot(index=i, confirm_index=confirm_idx, price=float(highs[i]), kind="high")
            )

    pivots.sort(key=lambda p: p.index)
    return pivots


def adaptive_lookback(df: pd.DataFrame, as_of: int) -> int:
    """Compute ATR-based lookback. High volatility → smaller lookback; low → larger.

    ATR% range for crypto ≈ 0.3–1.5. We map this linearly to [MAX_LB, MIN_LB]
    (inverted: high vol → small lookback so pivots confirm faster).
    """
    start = max(0, as_of - config.ATR_PERIOD)
    view = df.iloc[start : as_of + 1]
    if len(view) < 2:
        return config.MAX_LB

    tr = pd.concat(
        [
            view["high"] - view["low"],
            (view["high"] - view["close"].shift(1)).abs(),
            (view["low"] - view["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.mean()
    mid_price = view["close"].iloc[-1]
    if mid_price <= 0:
        return config.MAX_LB

    atr_pct = (atr / mid_price) * 100.0

    # Linear map: atr_pct=0.3 → MAX_LB, atr_pct=1.5 → MIN_LB
    span_vol = 1.5 - 0.3
    span_lb = config.MAX_LB - config.MIN_LB
    lb = config.MAX_LB - ((atr_pct - 0.3) / span_vol) * span_lb
    return int(max(config.MIN_LB, min(config.MAX_LB, round(lb))))


def predict_101_102(
    pivots: list[Pivot],
    as_of: int,
    df: pd.DataFrame,
    max_distance_pct: float | None = None,
) -> dict | None:
    """Predict entry zone (101) and target (102) via historical pattern analysis.

    Direction inferred from last 2 swing highs/lows:
      HH + HL → LONG,  LH + LL → SHORT,  else → None.

    Entry zone: EMA-weighted average pullback % across all historical
      (swing_low, next_swing_high) pairs for LONG — and the mirror for SHORT.
      entry_price = close × (1 − avg_pullback_pct); zone = ±0.15 ATR around it.
    Target: entry_price × (1 + avg_move_pct).
    Stop: entry_low − SL_BUFFER × ATR.

    max_distance_pct: when set, only patterns whose anchor swing is within this
      % of the current close are used (LONG: swing_low >= close*(1-pct/100);
      SHORT: swing_high <= close*(1+pct/100)). If no patterns survive, returns None.

    Returns None if fewer than 5 pattern instances exist or R:R < MIN_RR.
    No Fibonacci — all statistics derived from real confirmed structural pivots.
    """
    lows  = [p for p in pivots if p.kind == "low"]
    highs = [p for p in pivots if p.kind == "high"]

    if len(lows) < 2 or len(highs) < 2:
        return None

    lows  = sorted(lows,  key=lambda p: p.index)
    highs = sorted(highs, key=lambda p: p.index)

    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price  > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price  < lows[-2].price

    if hh and hl:
        direction: Direction = "long"
    elif lh and ll:
        direction = "short"
    else:
        return None

    close = float(df.iloc[as_of]["close"])
    atr_val = _atr(df, as_of)
    if atr_val <= 0:
        return None

    patterns = _collect_patterns(pivots, direction, df, as_of,
                                  max_distance_pct=max_distance_pct,
                                  current_close=close)
    if len(patterns) < 2:
        return None

    sample_size  = len(patterns)
    avg_pullback = _ema_w([p["pullback_pct"] for p in patterns])
    avg_move     = _ema_w([p["move_pct"]     for p in patterns])

    probability_entry  = sum(1 for p in patterns if p["reached_entry"])  / sample_size
    probability_target = sum(1 for p in patterns if p["reached_target"]) / sample_size

    if direction == "long":
        entry_price = close * (1.0 - avg_pullback)
        entry_low   = entry_price - atr_val * 0.15
        entry_high  = entry_price + atr_val * 0.15
        target      = entry_price * (1.0 + avg_move)
        stop        = entry_low - atr_val * config.SL_BUFFER
        risk        = entry_high - stop
        if risk <= 0:
            return None
        rr = (target - entry_high) / risk
    else:
        entry_price = close * (1.0 + avg_pullback)
        entry_high  = entry_price + atr_val * 0.15
        entry_low   = entry_price - atr_val * 0.15
        target      = entry_price * (1.0 - avg_move)
        stop        = entry_high + atr_val * config.SL_BUFFER
        risk        = stop - entry_low
        if risk <= 0:
            return None
        rr = (entry_low - target) / risk

    if rr < config.MIN_RR:
        return None

    return {
        "direction":          direction,
        "entry_low":          round(entry_low,          6),
        "entry_high":         round(entry_high,         6),
        "target":             round(target,             6),
        "stop":               round(stop,               6),
        "rr":                 round(rr,                 4),
        "probability_entry":  round(probability_entry,  4),
        "probability_target": round(probability_target, 4),
        "sample_size":        sample_size,
    }


# ── private helpers ────────────────────────────────────────────────────────────

def _collect_patterns(
    pivots: list[Pivot],
    direction: Direction,
    df: pd.DataFrame,
    as_of: int,
    max_distance_pct: float | None = None,
    current_close: float = 0.0,
) -> list[dict]:
    """Collect historical pullback/move instances for the given direction.

    LONG: each (swing_low, nearest_next_swing_high) pair.
    SHORT: each (swing_high, nearest_next_swing_low) pair.
    Causal: only reads df.iloc[:as_of+1]; all pivots already confirmed ≤ as_of.

    max_distance_pct / current_close: when set, skips swings whose price is
      further than max_distance_pct% from current_close (LONG: too far below;
      SHORT: too far above). Allows callers to request only nearby-entry patterns.
    """
    patterns: list[dict] = []

    if direction == "long":
        lows_s  = sorted([p for p in pivots if p.kind == "low"],  key=lambda p: p.index)
        highs_s = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.index)

        for l_curr in lows_s:
            # Distance gate: swing low must be within max_distance_pct% below close
            if (max_distance_pct is not None and current_close > 0
                    and l_curr.price < current_close * (1.0 - max_distance_pct / 100.0)):
                continue

            h_nexts = [h for h in highs_s if h.index > l_curr.index]
            if not h_nexts:
                continue
            h_next = h_nexts[0]

            signal_bar   = l_curr.confirm_index
            signal_close = float(df.iloc[signal_bar]["close"])
            if signal_close <= 0 or l_curr.price <= 0:
                continue

            pullback_pct = (signal_close - l_curr.price) / signal_close
            if pullback_pct <= 0:
                continue

            move_pct = (h_next.price - l_curr.price) / l_curr.price
            if move_pct <= 0:
                continue

            end_bar  = min(h_next.confirm_index, as_of)
            segment  = df.iloc[signal_bar : end_bar + 1]
            patterns.append({
                "pullback_pct":   pullback_pct,
                "move_pct":       move_pct,
                "reached_entry":  bool((segment["low"]  <= l_curr.price).any()),
                "reached_target": bool((segment["high"] >= h_next.price).any()),
            })

    else:  # short
        highs_s = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.index)
        lows_s  = sorted([p for p in pivots if p.kind == "low"],  key=lambda p: p.index)

        for h_curr in highs_s:
            # Distance gate: swing high must be within max_distance_pct% above close
            if (max_distance_pct is not None and current_close > 0
                    and h_curr.price > current_close * (1.0 + max_distance_pct / 100.0)):
                continue

            l_nexts = [l for l in lows_s if l.index > h_curr.index]
            if not l_nexts:
                continue
            l_next = l_nexts[0]

            signal_bar   = h_curr.confirm_index
            signal_close = float(df.iloc[signal_bar]["close"])
            if signal_close <= 0 or h_curr.price <= 0:
                continue

            pullback_pct = (h_curr.price - signal_close) / signal_close
            if pullback_pct <= 0:
                continue

            move_pct = (h_curr.price - l_next.price) / h_curr.price
            if move_pct <= 0:
                continue

            end_bar  = min(l_next.confirm_index, as_of)
            segment  = df.iloc[signal_bar : end_bar + 1]
            patterns.append({
                "pullback_pct":   pullback_pct,
                "move_pct":       move_pct,
                "reached_entry":  bool((segment["high"] >= h_curr.price).any()),
                "reached_target": bool((segment["low"]  <= l_next.price).any()),
            })

    return patterns


def _ema_w(values: list[float]) -> float:
    """EMA-weighted average. Most recent element (end of list) has highest weight."""
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    alpha = 2.0 / (n + 1)
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def _atr(df: pd.DataFrame, as_of: int) -> float:
    """Average True Range over config.ATR_PERIOD candles ending at as_of (causal)."""
    start = max(0, as_of - config.ATR_PERIOD)
    view  = df.iloc[start : as_of + 1]
    if len(view) < 2:
        return float(view["high"].iloc[-1] - view["low"].iloc[-1]) if len(view) else 0.0
    tr = pd.concat(
        [
            view["high"] - view["low"],
            (view["high"] - view["close"].shift(1)).abs(),
            (view["low"]  - view["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.mean())
