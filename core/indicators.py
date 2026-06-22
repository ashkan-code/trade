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


# ── Gate 0: BTC current-state bias ───────────────────────────────────────────

def btc_bias(df: pd.DataFrame) -> "tuple[str, dict]":
    """Determine the CURRENT BTC market bias from EMA + MACD + RSI.

    Unlike detect_mss (which returns the last historical structure shift and
    can be stale by weeks), btc_bias scores the *present* state of four
    momentum indicators and requires a majority agreement before committing
    to a direction.

    Signals and scoring (+1 = bullish, -1 = bearish, 0 = neutral):
        ema_cross : EMA20 > EMA50 → +1 ; < → -1
        ema_slope : EMA20[-1] > EMA20[-5] → +1 ; < → -1 ; flat → 0
        macd_hist : MACD histogram > 0 → +1 ; < 0 → -1
        rsi_level : RSI > BTC_RSI_BULL(55) → +1 ; < BTC_RSI_BEAR(45) → -1 ; else 0

    Direction (score in [-4, +4]):
        score ≥ BTC_SCORE_LONG  (+2) → 'long'
        score ≤ BTC_SCORE_SHORT (-2) → 'short'
        else                          → 'neutral'

    Neutral override: if RSI is in [BTC_RSI_BEAR, BTC_RSI_BULL] AND price
    sits in [BTC_POS_LO%, BTC_POS_HI%] of the 20-bar range → 'neutral'
    regardless of score. This catches choppy, range-bound sessions.

    Returns (direction_str, debug_dict). direction_str is one of
    'long' | 'short' | 'neutral'.
    """
    n = len(df)
    if n < 60:
        return "neutral", {"reason": "insufficient_data", "bars": n}

    closes = df["close"]

    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema20_now  = float(ema20.iloc[-1])
    ema50_now  = float(ema50.iloc[-1])
    ema20_5ago = float(ema20.iloc[-6]) if n >= 6 else ema20_now

    _, _, hist = macd(closes)
    macd_hist = float(hist.iloc[-1])

    rsi_s   = rsi(closes)
    rsi_val = float(rsi_s.iloc[-1])

    recent_high = float(df["high"].tail(20).max())
    recent_low  = float(df["low"].tail(20).min())
    span        = recent_high - recent_low
    pos_pct = (float(closes.iloc[-1]) - recent_low) / span * 100 if span > 0 else 50.0

    ema_cross_sig = 1 if ema20_now > ema50_now else -1
    ema_slope_sig = (
        1  if ema20_now > ema20_5ago * 1.0001 else
        -1 if ema20_now < ema20_5ago * 0.9999 else
        0
    )
    macd_sig = 1 if macd_hist > 0 else (-1 if macd_hist < 0 else 0)
    rsi_sig  = (
        1  if rsi_val > config.BTC_RSI_BULL else
        -1 if rsi_val < config.BTC_RSI_BEAR else
        0
    )

    score = ema_cross_sig + ema_slope_sig + macd_sig + rsi_sig

    debug: dict = {
        "ema20":      round(ema20_now, 2),
        "ema50":      round(ema50_now, 2),
        "ema_cross":  ema_cross_sig,
        "ema_slope":  ema_slope_sig,
        "macd_hist":  round(macd_hist, 6),
        "macd_sig":   macd_sig,
        "rsi":        round(rsi_val, 1),
        "rsi_sig":    rsi_sig,
        "score":      score,
        "pos_pct":    round(pos_pct, 1),
        "range":      f"{recent_low:.0f}–{recent_high:.0f}",
    }

    rsi_mid = config.BTC_RSI_BEAR <= rsi_val <= config.BTC_RSI_BULL
    pos_mid = config.BTC_POS_LO   <= pos_pct  <= config.BTC_POS_HI
    if rsi_mid and pos_mid:
        debug["reason"] = "range_neutral(RSI+pos)"
        return "neutral", debug

    if score >= config.BTC_SCORE_LONG:
        debug["reason"] = f"score={score}>=+{config.BTC_SCORE_LONG}"
        return "long", debug
    if score <= config.BTC_SCORE_SHORT:
        debug["reason"] = f"score={score}<={config.BTC_SCORE_SHORT}"
        return "short", debug

    debug["reason"] = f"score={score}(inconclusive)"
    return "neutral", debug
