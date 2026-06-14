"""Sweep+reclaim → 101 zone; structural swing opposite → 102 target; SL behind sweep."""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Pivot, Setup


def find_setup(
    df: pd.DataFrame,
    as_of: int,
    pivots: list[Pivot],
    direction: Direction,
) -> Setup | None:
    """Detect 101/102 setup at as_of. All inputs are already causal (as_of-bounded)."""
    view = df.iloc[: as_of + 1]
    atr = _atr(view)
    if atr is None:
        return None

    if direction == "long":
        return _long_setup(view, as_of, pivots, atr)
    return _short_setup(view, as_of, pivots, atr)


# ── long ──────────────────────────────────────────────────────────────────────

def _long_setup(
    view: pd.DataFrame, as_of: int, pivots: list[Pivot], atr: float
) -> Setup | None:
    swing_lows = [p for p in pivots if p.kind == "low"]
    if not swing_lows:
        return None

    # pivot_100: most recent confirmed swing low
    pivot_100 = swing_lows[-1]
    p100_price = pivot_100.price
    p100_idx = pivot_100.confirm_index  # search for sweep starting here

    # Detect sweep + reclaim: sweep can happen any time after confirmation;
    # reclaim must follow within SWEEP_MAX_CANDLES candles.
    # We use the MOST RECENT valid sweep+reclaim (most relevant for live signal).
    sweep_low: float | None = None
    for j in range(p100_idx, as_of + 1):
        row = view.iloc[j]
        if row["low"] < p100_price:
            candidate_sweep = float(row["low"])
            # reclaim: within SWEEP_MAX_CANDLES candles after sweep
            reclaimed = False
            for k in range(j, min(j + config.SWEEP_MAX_CANDLES + 1, as_of + 1)):
                if view.iloc[k]["close"] > p100_price:
                    reclaimed = True
                    break
            if reclaimed:
                sweep_low = candidate_sweep  # keep updating → most recent wins

    if sweep_low is None:
        return None

    entry_low = sweep_low
    entry_high = p100_price
    stop = sweep_low - atr * config.SL_BUFFER

    # pivot_102: nearest confirmed swing high above current close
    current_price = float(view.iloc[-1]["close"])
    swing_highs = [p for p in pivots if p.kind == "high" and p.price > current_price]
    if not swing_highs:
        return None
    target = min(swing_highs, key=lambda p: p.price).price  # nearest overhead resistance

    risk = entry_high - stop
    if risk <= 0:
        return None
    rr = (target - entry_high) / risk
    if rr < config.MIN_RR:
        return None

    return Setup(
        entry_low=entry_low,
        entry_high=entry_high,
        target=target,
        stop=stop,
        rr=rr,
        direction="long",
    )


# ── short ─────────────────────────────────────────────────────────────────────

def _short_setup(
    view: pd.DataFrame, as_of: int, pivots: list[Pivot], atr: float
) -> Setup | None:
    swing_highs = [p for p in pivots if p.kind == "high"]
    if not swing_highs:
        return None

    pivot_100 = swing_highs[-1]
    p100_price = pivot_100.price
    p100_idx = pivot_100.confirm_index

    sweep_high: float | None = None
    for j in range(p100_idx, as_of + 1):
        row = view.iloc[j]
        if row["high"] > p100_price:
            candidate_sweep = float(row["high"])
            reclaimed = False
            for k in range(j, min(j + config.SWEEP_MAX_CANDLES + 1, as_of + 1)):
                if view.iloc[k]["close"] < p100_price:
                    reclaimed = True
                    break
            if reclaimed:
                sweep_high = candidate_sweep  # keep updating → most recent wins

    if sweep_high is None:
        return None

    entry_high = sweep_high
    entry_low = p100_price
    stop = sweep_high + atr * config.SL_BUFFER

    current_price = float(view.iloc[-1]["close"])
    swing_lows = [p for p in pivots if p.kind == "low" and p.price < current_price]
    if not swing_lows:
        return None
    target = max(swing_lows, key=lambda p: p.price).price  # nearest support below

    risk = stop - entry_low
    if risk <= 0:
        return None
    rr = (entry_low - target) / risk
    if rr < config.MIN_RR:
        return None

    return Setup(
        entry_low=entry_low,
        entry_high=entry_high,
        target=target,
        stop=stop,
        rr=rr,
        direction="short",
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _atr(view: pd.DataFrame) -> float | None:
    if len(view) < 2:
        return None
    tr = pd.concat(
        [
            view["high"] - view["low"],
            (view["high"] - view["close"].shift(1)).abs(),
            (view["low"] - view["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    val = float(tr.tail(config.ATR_PERIOD).mean())
    return val if val > 0 else None
