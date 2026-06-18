"""Tests for engine/risk.py."""
import pytest

from core.ict import OBDict, LiqDict
from engine.risk import compute_risk, should_trail, compute_trail_stop
import config


def _ob(type_, top, bottom):
    return OBDict(type=type_, top=top, bottom=bottom, index=5, broken=False, breaker=False)


def _liq(ltype, price):
    return LiqDict(type=ltype, price=price, touches=3)


# ── compute_risk ──────────────────────────────────────────────────────────────

def test_long_valid_risk():
    ob = _ob("bullish", top=102, bottom=98)
    # stop ≈ 97.9 (98 - 0.1% of 103 ≈ 0.103), risk ≈ 5.1
    # need reward >= 5.1 * 2 = 10.2 → target >= 113.2
    pools = [_liq("buyside", 115), _liq("sellside", 90)]
    risk = compute_risk("bullish", entry=103, ob_1h=ob, liquidity_pools=pools)

    assert risk["valid"] is True
    assert risk["stop"] < 98
    assert risk["target"] == 115
    assert risk["rr"] >= config.MIN_RR


def test_short_valid_risk():
    ob = _ob("bearish", top=102, bottom=98)
    # stop ≈ 102.1 (102 + 0.1%), risk ≈ 5.1
    # need reward >= 10.2 → target <= 86.8
    pools = [_liq("sellside", 85), _liq("buyside", 110)]
    risk = compute_risk("bearish", entry=97, ob_1h=ob, liquidity_pools=pools)

    assert risk["valid"] is True
    assert risk["stop"] > 102
    assert risk["target"] == 85
    assert risk["rr"] >= config.MIN_RR


def test_no_liquidity_invalid():
    ob = _ob("bullish", top=102, bottom=98)
    risk = compute_risk("bullish", entry=103, ob_1h=ob, liquidity_pools=[])
    assert risk["valid"] is False


def test_low_rr_invalid():
    ob = _ob("bullish", top=102, bottom=99)
    # Target only 1 point away, stop 3 points away → RR < 1
    pools = [_liq("buyside", 104)]
    risk = compute_risk("bullish", entry=103, ob_1h=ob, liquidity_pools=pools)
    # RR = (104-103)/(103-98.901) ≈ 0.24 < MIN_RR
    assert risk["valid"] is False


def test_risk_pct_positive():
    ob = _ob("bullish", top=102, bottom=98)
    pools = [_liq("buyside", 115)]
    risk = compute_risk("bullish", entry=103, ob_1h=ob, liquidity_pools=pools)
    assert risk["risk_pct"] > 0


# ── should_trail ──────────────────────────────────────────────────────────────

def test_should_trail_long_triggered():
    risk = {"trail_trigger": 105.0, "stop": 98.0, "entry": 103.0}
    assert should_trail(106.0, "bullish", risk) is True


def test_should_trail_long_not_triggered():
    risk = {"trail_trigger": 105.0, "stop": 98.0, "entry": 103.0}
    assert should_trail(104.0, "bullish", risk) is False


def test_should_trail_short_triggered():
    risk = {"trail_trigger": 95.0, "stop": 103.0, "entry": 97.0}
    assert should_trail(94.0, "bearish", risk) is True


# ── compute_trail_stop ────────────────────────────────────────────────────────

def test_trail_stop_long_moves_up():
    risk = {"stop": 98.0, "trail_trigger": 105.0, "entry": 103.0}
    new_stop = compute_trail_stop(current_price=108.0, direction="bullish",
                                  risk=risk, atr_value=1.0)
    assert new_stop > risk["stop"]
    assert new_stop == pytest.approx(108.0 - 2.0)


def test_trail_stop_never_against_trade_long():
    risk = {"stop": 99.0, "trail_trigger": 105.0, "entry": 103.0}
    # If new calculated stop would be below current stop → keep current
    new_stop = compute_trail_stop(current_price=100.0, direction="bullish",
                                  risk=risk, atr_value=5.0)
    assert new_stop >= risk["stop"]
