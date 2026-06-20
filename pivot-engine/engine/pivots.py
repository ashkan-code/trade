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
) -> dict | None:
    """Predict 101 (entry zone) and 102 (target) from confirmed causal pivots.

    Direction is inferred from swing structure:
      HH + HL  (last 2 highs rising AND last 2 lows rising) → LONG
      LH + LL  (last 2 highs falling AND last 2 lows falling) → SHORT
      Ambiguous → None.

    101 (LONG) : nearest confirmed swing LOW below current close → entry zone.
    102 (LONG) : nearest confirmed swing HIGH above current close → target.
    101 (SHORT): nearest confirmed swing HIGH above current close → entry zone.
    102 (SHORT): nearest confirmed swing LOW below current close → target.

    Returns None if structure ambiguous, levels absent, or R:R < MIN_RR.
    No Fibonacci, no hardcoded targets — only real confirmed structural pivots.
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

    if direction == "long":
        cands_101 = [p for p in lows if p.price < close]
        if not cands_101:
            return None
        p101 = max(cands_101, key=lambda p: p.price)  # nearest low below close

        cands_102 = [p for p in highs if p.price > close]
        if not cands_102:
            return None
        p102 = min(cands_102, key=lambda p: p.price)  # nearest high above close

        entry_low  = p101.price
        entry_high = p101.price + atr_val * 0.3
        stop       = p101.price - atr_val * config.SL_BUFFER
        risk       = entry_high - stop
        if risk <= 0:
            return None
        rr = (p102.price - entry_high) / risk

    else:  # short
        cands_101 = [p for p in highs if p.price > close]
        if not cands_101:
            return None
        p101 = min(cands_101, key=lambda p: p.price)  # nearest high above close

        cands_102 = [p for p in lows if p.price < close]
        if not cands_102:
            return None
        p102 = max(cands_102, key=lambda p: p.price)  # nearest low below close

        entry_high = p101.price
        entry_low  = p101.price - atr_val * 0.3
        stop       = p101.price + atr_val * config.SL_BUFFER
        risk       = stop - entry_low
        if risk <= 0:
            return None
        rr = (entry_low - p102.price) / risk

    if rr < config.MIN_RR:
        return None

    return {
        "direction":     direction,
        "p101_low":      round(entry_low,  6),
        "p101_high":     round(entry_high, 6),
        "p102":          round(p102.price, 6),
        "stop":          round(stop,       6),
        "rr":            round(rr,         4),
        "pivot_101_idx": p101.index,
        "pivot_102_idx": p102.index,
    }


# ── private helpers ────────────────────────────────────────────────────────────

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
