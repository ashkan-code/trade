"""SL/TP computation for ICT rejection setups."""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Zone
from engine.ict import atr_scalar, find_liquidity


def compute_sl(direction: Direction, zone: Zone, atr_val: float) -> float:
    """Stop loss: beyond zone edge + SL_BUFFER % of zone height."""
    zone_height = zone.zone_high - zone.zone_low
    buffer = max(zone_height * (config.SL_BUFFER / 100), atr_val * 0.1)
    if direction == "long":
        return zone.zone_low - buffer
    else:
        return zone.zone_high + buffer


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
