"""Anti-repaint audit: proves find_pivots never uses future data.

Theorem: since confirm_index = i + lookback <= as_of always,
adding candles after as_of cannot change the set of confirmed pivots at as_of.
This function verifies that empirically over the full DataFrame.
"""

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from engine.pivots import find_pivots, adaptive_lookback
from contracts import Pivot


def audit_no_lookahead(df: pd.DataFrame) -> tuple[bool, str]:
    """For every as_of from WARMUP to end, assert that using full df vs truncated df
    produces identical pivot lists at that as_of. Returns (passed, detail)."""
    mismatches: list[str] = []

    for k in range(config.WARMUP, len(df)):
        lb = adaptive_lookback(df, k)

        # Call with full df (the engine must ignore future candles)
        pivots_full = find_pivots(df, as_of=k, lookback=lb)

        # Call with df truncated at k (simulates true no-lookahead)
        df_trunc = df.iloc[: k + 1]
        pivots_trunc = find_pivots(df_trunc, as_of=k, lookback=lb)

        if pivots_full != pivots_trunc:
            mismatches.append(
                f"as_of={k}: full={len(pivots_full)} pivots, trunc={len(pivots_trunc)} pivots"
            )
            if len(mismatches) >= 5:
                break  # report first few, not thousands

    if mismatches:
        return False, "REPAINT DETECTED at: " + "; ".join(mismatches)
    return True, f"PASS — {len(df) - config.WARMUP} as_of positions verified, no lookahead"
