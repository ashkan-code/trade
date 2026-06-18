"""Tests for core/confluence.py."""
import pandas as pd
import pytest

from core.ict import OBDict, FVGDict, MSSDict
from core.confluence import (
    check_btc_direction, is_aligned, find_mtf_ob_overlap,
    find_best_zone, zone_triggered, score_zone,
)
from tests.conftest import make_df


def _ob(type_, top, bottom, index=5, broken=False):
    return OBDict(type=type_, top=top, bottom=bottom, index=index,
                  broken=broken, breaker=False)


def _fvg(type_, top, bottom, index=10, active=True):
    return FVGDict(type=type_, top=top, bottom=bottom, index=index, active=active)


# ── BTC direction ─────────────────────────────────────────────────────────────

def test_check_btc_direction_returns_string_or_none(df_with_bullish_mss):
    result = check_btc_direction(df_with_bullish_mss)
    assert result in ("bullish", "bearish", None)


# ── Alignment ─────────────────────────────────────────────────────────────────

def test_is_aligned_true():
    mss = MSSDict(direction="bullish", price=100.0, index=5)
    assert is_aligned(mss, "bullish") is True


def test_is_aligned_false_mismatch():
    mss = MSSDict(direction="bearish", price=100.0, index=5)
    assert is_aligned(mss, "bullish") is False


def test_is_aligned_false_none_btc():
    mss = MSSDict(direction="bullish", price=100.0, index=5)
    assert is_aligned(mss, None) is False


# ── MTF OB overlap ────────────────────────────────────────────────────────────

def test_find_mtf_ob_overlap_found():
    obs_1h = [_ob("bullish", top=105, bottom=102)]
    obs_4h = [_ob("bullish", top=104, bottom=101)]
    zone = find_mtf_ob_overlap(obs_1h, obs_4h, "bullish", "TESTUSDT")
    assert zone is not None
    assert zone["zone_type"] == "ob_overlap"
    assert zone["zone_top"] == 104  # min(105, 104)
    assert zone["zone_bottom"] == 102  # max(102, 101)


def test_find_mtf_ob_overlap_no_overlap():
    obs_1h = [_ob("bullish", top=100, bottom=98)]
    obs_4h = [_ob("bullish", top=107, bottom=103)]
    zone = find_mtf_ob_overlap(obs_1h, obs_4h, "bullish")
    assert zone is None


def test_find_mtf_ob_overlap_broken_ignored():
    obs_1h = [_ob("bullish", top=105, bottom=102, broken=True)]  # broken
    obs_4h = [_ob("bullish", top=104, bottom=101)]
    zone = find_mtf_ob_overlap(obs_1h, obs_4h, "bullish")
    assert zone is None


# ── find_best_zone priority ───────────────────────────────────────────────────

def test_find_best_zone_returns_overlap_first():
    obs_1h = [_ob("bullish", top=105, bottom=102)]
    obs_4h = [_ob("bullish", top=104, bottom=101)]
    fvgs = [_fvg("bullish", top=106, bottom=103)]
    zone = find_best_zone(obs_1h, obs_4h, fvgs, "bullish")
    assert zone["zone_type"] == "ob_overlap"


def test_find_best_zone_falls_back_to_single_ob():
    obs_1h = [_ob("bullish", top=105, bottom=102)]
    obs_4h = [_ob("bullish", top=200, bottom=150)]  # no overlap
    fvgs = []
    zone = find_best_zone(obs_1h, obs_4h, fvgs, "bullish")
    assert zone is not None
    assert zone["zone_type"] == "ob_single"


def test_find_best_zone_falls_back_to_fvg():
    obs_1h = []
    obs_4h = []
    fvgs = [_fvg("bullish", top=106, bottom=103)]
    zone = find_best_zone(obs_1h, obs_4h, fvgs, "bullish")
    assert zone is not None
    assert zone["zone_type"] == "fvg"


def test_find_best_zone_returns_none_when_nothing():
    zone = find_best_zone([], [], [], "bullish")
    assert zone is None


# ── Zone trigger ──────────────────────────────────────────────────────────────

def test_zone_triggered_long():
    zone = {
        "direction": "bullish",
        "zone_top": 102.0,
        "zone_bottom": 100.0,
    }
    rows = []
    ts = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(10):
        low = 101.0 if i == 3 else 103.0  # entered zone at bar 3
        close = 103.5 if i >= 5 else 101.5  # closed above zone_top at bar 5+
        rows.append({"timestamp": ts + pd.Timedelta(minutes=5*i),
                     "open": 102, "high": 104, "low": low, "close": close, "volume": 1})
    df = pd.DataFrame(rows)
    assert zone_triggered(zone, df) is True


def test_zone_triggered_false():
    zone = {"direction": "bullish", "zone_top": 110.0, "zone_bottom": 108.0}
    df = make_df(15, "flat", base=100.0)
    assert zone_triggered(zone, df) is False


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_score_ob_overlap_full_confirms_high_rr():
    zone = {
        "zone_type": "ob_overlap",
        "rsi_confirm": True,
        "macd_confirm": True,
    }
    score = score_zone(zone, rr=3.0)
    assert score >= 0.75


def test_score_fvg_no_confirms_low_rr():
    zone = {
        "zone_type": "fvg",
        "rsi_confirm": False,
        "macd_confirm": False,
    }
    score = score_zone(zone, rr=1.0)
    assert score < 0.75


def test_score_capped_at_1():
    zone = {"zone_type": "ob_overlap", "rsi_confirm": True, "macd_confirm": True}
    assert score_zone(zone, rr=10.0) <= 1.0
