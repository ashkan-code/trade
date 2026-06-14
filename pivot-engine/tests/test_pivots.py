"""Tests for Phase 2: causal pivot engine."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
import config
from engine.pivots import adaptive_lookback, find_pivots
from backtest.repaint_audit import audit_no_lookahead


def _make_df(seed: int = 42, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    t = np.linspace(0, 10 * np.pi, n)
    close = 40000 + 2000 * np.sin(t) + np.cumsum(rng.normal(0, 20, n))
    high = close + rng.uniform(50, 300, n)
    low = close - rng.uniform(50, 300, n)
    return pd.DataFrame(
        {"open": close - 30, "high": high, "low": low, "close": close, "volume": np.ones(n) * 5},
        index=idx,
    )


def test_confirm_index_never_exceeds_as_of():
    df = _make_df()
    as_of = 200
    lb = adaptive_lookback(df, as_of)
    pivots = find_pivots(df, as_of=as_of, lookback=lb)
    for p in pivots:
        assert p.confirm_index <= as_of, f"confirm_index {p.confirm_index} > as_of {as_of}"


def test_pivots_found():
    df = _make_df()
    pivots = find_pivots(df, as_of=250, lookback=5)
    assert len(pivots) > 0, "Expected pivots on oscillating data"


def test_adaptive_lookback_varies():
    df = _make_df(n=300)
    lbs = {adaptive_lookback(df, i) for i in range(config.WARMUP, 300, 10)}
    assert len(lbs) > 1, f"Lookback never varied: {lbs}"


def test_adaptive_lookback_within_bounds():
    df = _make_df(n=300)
    for i in range(config.WARMUP, 300, 5):
        lb = adaptive_lookback(df, i)
        assert config.MIN_LB <= lb <= config.MAX_LB, f"lb={lb} out of bounds at i={i}"


def test_repaint_audit_passes():
    df = _make_df(seed=7, n=300)
    passed, detail = audit_no_lookahead(df)
    assert passed, f"Repaint audit FAILED: {detail}"


def test_future_candles_dont_change_past_pivots():
    """Core causal invariant: adding candles after as_of must not change pivot list at as_of."""
    df = _make_df(seed=99, n=200)
    as_of = 100
    lb = adaptive_lockback = adaptive_lookback(df, as_of)
    pivots_full = find_pivots(df, as_of=as_of, lookback=lb)
    pivots_trunc = find_pivots(df.iloc[: as_of + 1], as_of=as_of, lookback=lb)
    assert pivots_full == pivots_trunc, "Adding future candles changed past pivot list — REPAINT"
