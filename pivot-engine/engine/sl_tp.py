"""SL/TP computation for ICT rejection setups."""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Zone
from engine.ict import atr_scalar, find_liquidity


def compute_sl(direction: Direction, shadow_extreme: float, atr_val: float) -> float:
    """Stop loss: beyond the rejection candle's shadow extreme.

    shadow_extreme: candle low for long, candle high for short.
    Buffer = max(SL_BUFFER% of shadow price, 0.25 × ATR) — never zero.
    """
    buffer = max(shadow_extreme * (config.SL_BUFFER / 100), atr_val * 0.25)
    if direction == "long":
        return shadow_extreme - buffer
    else:
        return shadow_extreme + buffer


def find_tp(
    direction: Direction,
    entry: float,
    sl: float,
    df: pd.DataFrame,
) -> float | None:
    """Find nearest opposing liquidity that gives >= MIN_RR.

    Returns None if no valid target found.
    """
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    levels = find_liquidity(df, direction, entry)
    min_target_dist = risk * config.MIN_RR

    for lvl in levels:
        dist = abs(lvl - entry)
        if dist >= min_target_dist:
            return lvl

    return None


def compute_rr(entry: float, stop: float, target: float) -> float:
    """R:R ratio (always positive)."""
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(target - entry) / risk
