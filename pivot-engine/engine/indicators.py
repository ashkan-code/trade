"""Gate 3: RSI + MACD confluence confirmation."""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI using EWM (alpha = 1/length)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_len: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Standard MACD. Returns (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi_confirms(series: pd.Series, direction: Direction) -> bool:
    """RSI confirmation in last RSI_WINDOW bars.

    Long: RSI dipped below OS level then crossed back above it.
    Short: RSI rose above OB level then crossed back below it.
    """
    window = config.RSI_WINDOW
    r = rsi(series, config.RSI_LEN)
    if len(r) < window + 1:
        return False
    r_window = r.iloc[-(window + 1):]

    if direction == "long":
        dipped = (r_window < config.RSI_OS).any()
        crossed_back = float(r_window.iloc[-1]) > config.RSI_OS
        return bool(dipped and crossed_back)
    else:
        surged = (r_window > config.RSI_OB).any()
        crossed_back = float(r_window.iloc[-1]) < config.RSI_OB
        return bool(surged and crossed_back)


def macd_confirms(histogram: pd.Series, direction: Direction) -> bool:
    """MACD histogram confirmation: shrinking toward zero in last RSI_WINDOW bars.

    Long: histogram was negative, bars are growing (becoming less negative).
    Short: histogram was positive, bars are shrinking (becoming less positive).
    """
    window = config.RSI_WINDOW
    if len(histogram) < window + 1:
        return False
    h = histogram.iloc[-(window):]

    diffs = h.diff().dropna()
    if direction == "long":
        return bool((diffs > 0).sum() >= len(diffs) // 2 + 1)
    else:
        return bool((diffs < 0).sum() >= len(diffs) // 2 + 1)


def gate3_passes(df: pd.DataFrame, direction: Direction) -> bool:
    """Both RSI and MACD must confirm. Gate 3 is advisory (does not block signal by default)."""
    if len(df) < config.MACD_SLOW + config.RSI_WINDOW + 5:
        return False
    close = df["close"]
    _, _, hist = macd(close, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL_LEN)
    rsi_ok = rsi_confirms(close, direction)
    macd_ok = macd_confirms(hist, direction)
    return rsi_ok and macd_ok
