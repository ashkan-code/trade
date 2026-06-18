"""Tests for core/indicators.py."""
import numpy as np
import pandas as pd
import pytest

from core.indicators import rsi, macd, atr, rsi_confirm_long, rsi_confirm_short, macd_confirm_long, macd_confirm_short
from tests.conftest import make_df


def _series(values):
    return pd.Series(values, dtype=float)


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_range():
    df = make_df(100, "up")
    r = rsi(df["close"])
    valid = r.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_overbought_in_strong_uptrend():
    """Strong uptrend should yield RSI > 60 eventually."""
    prices = [100 + i * 2 for i in range(100)]
    r = rsi(_series(prices))
    valid = r.dropna()
    assert len(valid) > 0
    assert valid.iloc[-1] > 60


def test_rsi_length_controls_warmup():
    prices = list(range(1, 40))
    r14 = rsi(_series(prices), length=14)
    assert r14.iloc[:14].isna().all() or r14.iloc[0] != r14.iloc[0]  # NaN check


# ── MACD ──────────────────────────────────────────────────────────────────────

def test_macd_returns_three_series():
    df = make_df(100, "up")
    m, s, h = macd(df["close"])
    assert len(m) == len(df)
    assert len(s) == len(df)
    assert len(h) == len(df)


def test_macd_histogram_equals_macd_minus_signal():
    df = make_df(60, "flat")
    m, s, h = macd(df["close"])
    diff = (m - s - h).dropna().abs()
    assert (diff < 1e-10).all()


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_positive():
    df = make_df(50, "flat")
    a = atr(df)
    assert (a.dropna() > 0).all()


def test_atr_larger_for_volatile():
    df_v = make_df(50, "flat")
    df_v["high"] = df_v["high"] * 2
    df_v["low"] = df_v["low"] / 2
    a_volatile = atr(df_v).dropna().mean()
    df_q = make_df(50, "flat")
    a_quiet = atr(df_q).dropna().mean()
    assert a_volatile > a_quiet


# ── RSI confirm helpers ───────────────────────────────────────────────────────

def test_rsi_confirm_long_true():
    # RSI dips below 30 at index -3, crosses back above 30 at index -1
    vals = [50, 45, 40, 28, 25, 32, 45]
    r = _series(vals)
    assert rsi_confirm_long(r, window=5, low=30) is True


def test_rsi_confirm_long_false_no_cross():
    vals = [50, 45, 35, 32, 31]
    r = _series(vals)
    assert rsi_confirm_long(r, window=5, low=30) is False


def test_rsi_confirm_long_false_still_below():
    vals = [50, 28, 25, 22, 20]
    r = _series(vals)
    assert rsi_confirm_long(r, window=5, low=30) is False


def test_rsi_confirm_short_true():
    vals = [50, 65, 75, 80, 65, 55]
    r = _series(vals)
    assert rsi_confirm_short(r, window=5, high=70) is True


def test_rsi_confirm_short_false():
    vals = [60, 65, 68, 72, 75]
    r = _series(vals)
    assert rsi_confirm_short(r, window=5, high=70) is False


# ── MACD histogram shrink ─────────────────────────────────────────────────────

def test_macd_confirm_long_true():
    hist = _series([-0.8, -0.6, -0.4, -0.2, -0.05])
    assert bool(macd_confirm_long(hist, window=5))


def test_macd_confirm_long_false_positive_bars():
    hist = _series([0.1, 0.2, 0.3, 0.4, 0.5])
    assert not bool(macd_confirm_long(hist, window=5))


def test_macd_confirm_short_true():
    hist = _series([0.8, 0.6, 0.4, 0.2, 0.05])
    assert bool(macd_confirm_short(hist, window=5))


def test_macd_confirm_short_false_negative_bars():
    hist = _series([-0.1, -0.2, -0.3])
    assert not bool(macd_confirm_short(hist, window=5))
