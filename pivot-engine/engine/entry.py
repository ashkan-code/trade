"""Gate 4: 5m entry optimizer.

Finds a rejection candle on 5m within the zone boundaries.
NEVER cancels a confirmed (Gates 0-3 passed) trade — falls back to zone edge.
"""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Zone
from engine.zones import validate_rejection


def optimize_entry(
    df_5m: pd.DataFrame,
    zone: Zone,
    direction: Direction,
    as_of_ts: pd.Timestamp,
    lookback_bars: int = 12,
) -> float:
    """Find best 5m rejection entry within zone, defaulting to zone edge.

    Looks backward `lookback_bars` from as_of_ts on the 5m chart.
    Returns entry price (zone_high for long, zone_low for short if no better found).
    """
    default_entry = zone.zone_high if direction == "long" else zone.zone_low

    # Slice 5m data up to as_of_ts
    view = df_5m[df_5m.index <= as_of_ts]
    if len(view) < 2:
        return default_entry

    # Search last `lookback_bars` bars
    search = view.iloc[-lookback_bars:]

    # Find the most recent valid rejection candle inside zone
    best_entry: float | None = None
    for i in range(len(search) - 1, -1, -1):
        row = search.iloc[i]
        passed, _ = validate_rejection(row, zone)
        if passed:
            if direction == "long":
                best_entry = float(row["close"])
            else:
                best_entry = float(row["close"])
            break

    return best_entry if best_entry is not None else default_entry
