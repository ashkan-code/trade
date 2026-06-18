from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ── RSI (Wilder's smoothed) ───────────────────────────────────────────────────

def rsi(series: pd.Series, length: int = config.RSI_LEN) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, ignore_na=True).mean()
    # Divide directly; when avg_loss == 0 and avg_gain > 0 → rs = inf → RSI = 100
    rs = avg_gain / avg_loss
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    # 0/0 case (flat, no movement): rs=NaN → RSI=NaN (acceptable)
    return rsi_vals


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(
    series: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal_len: int = config.MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── ATR (Wilder's) ────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, length: int = config.ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


# ── RSI confirm helpers ───────────────────────────────────────────────────────

def rsi_confirm_long(
    rsi_series: pd.Series,
    window: int = config.RSI_WINDOW,
    low: float = config.RSI_LOW,
) -> bool:
    """RSI dipped below `low` then crossed back above within the last `window` bars."""
    tail = rsi_series.dropna().iloc[-window:]
    if len(tail) < 2:
        return False
    went_below = (tail < low).any()
    last_above = tail.iloc[-1] > low
    if not (went_below and last_above):
        return False
    # Must have a cross: at least one transition from below→above
    vals = tail.values
    for k in range(1, len(vals)):
        if vals[k - 1] < low and vals[k] > low:
            return True
    return False


def rsi_confirm_short(
    rsi_series: pd.Series,
    window: int = config.RSI_WINDOW,
    high: float = config.RSI_HIGH,
) -> bool:
    """RSI climbed above `high` then crossed back below within the last `window` bars."""
    tail = rsi_series.dropna().iloc[-window:]
    if len(tail) < 2:
        return False
    went_above = (tail > high).any()
    last_below = tail.iloc[-1] < high
    if not (went_above and last_below):
        return False
    vals = tail.values
    for k in range(1, len(vals)):
        if vals[k - 1] > high and vals[k] < high:
            return True
    return False


# ── MACD histogram-shrink helpers ─────────────────────────────────────────────

def macd_confirm_long(
    histogram: pd.Series,
    window: int = config.RSI_WINDOW,
) -> bool:
    """Negative histogram bars shrinking toward zero (majority of diffs are positive)."""
    tail = histogram.dropna().iloc[-window:]
    if len(tail) < 2:
        return False
    if not (tail < 0).all():
        return False
    diffs = tail.diff().dropna()
    return (diffs > 0).sum() > len(diffs) / 2


def macd_confirm_short(
    histogram: pd.Series,
    window: int = config.RSI_WINDOW,
) -> bool:
    """Positive histogram bars shrinking toward zero (majority of diffs are negative)."""
    tail = histogram.dropna().iloc[-window:]
    if len(tail) < 2:
        return False
    if not (tail > 0).all():
        return False
    diffs = tail.diff().dropna()
    return (diffs < 0).sum() > len(diffs) / 2
