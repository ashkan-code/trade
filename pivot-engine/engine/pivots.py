"""Causal pivot detection. The as_of parameter is the contract: never touch df.iloc[i > as_of]."""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Pivot


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
