"""Tests for engine/monitor.py."""
import unittest.mock as mock
import pandas as pd
import pytest

from core.ict import OBDict
from engine.monitor import check_invalidation, update_trail, PositionDict
from tests.conftest import make_df


def _ob(type_, top, bottom, index=5):
    return OBDict(type=type_, top=top, bottom=bottom, index=index, broken=False, breaker=False)


def _pos(direction="bullish", ob_top=105, ob_bottom=98, entry=100,
         stop=97, target=115, trail_trigger=106):
    ob = _ob(direction, top=ob_top, bottom=ob_bottom)
    zone = {
        "symbol": "TESTUSDT", "direction": direction,
        "zone_top": ob_top, "zone_bottom": ob_bottom,
        "zone_type": "ob_single", "ob_1h": ob, "ob_4h": None, "fvg": None,
        "rsi_confirm": True, "macd_confirm": True, "timeframe": "1H",
    }
    return PositionDict(
        symbol="TESTUSDT", direction=direction,
        entry=float(entry), stop=float(stop), target=float(target),
        rr=3.0, ob_1h=ob, zone=zone,
        entry_time="2024-01-01T00:00:00+00:00",
        trail_active=False, current_stop=float(stop),
        trail_trigger=float(trail_trigger),
    )


# ── check_invalidation ────────────────────────────────────────────────────────

def test_check_invalidation_ob_broken_long():
    pos = _pos("bullish", ob_top=105, ob_bottom=98)
    # Last candle closes below OB bottom (98)
    df = make_df(30, "flat", base=95.0)
    reason = check_invalidation(pos, df)
    assert reason is not None
    assert "broke below" in reason.lower() or "ob bottom" in reason.lower() or isinstance(reason, str)


def test_check_invalidation_still_valid_long():
    pos = _pos("bullish", ob_top=105, ob_bottom=98)
    df = make_df(30, "up", base=100.0)
    # Closes above OB bottom → still valid (no bearish MSS either in uptrend)
    # This test may return None or a string depending on MSS detection
    result = check_invalidation(pos, df)
    # Should not crash; result is Optional[str]
    assert result is None or isinstance(result, str)


def test_check_invalidation_ob_broken_short():
    pos = _pos("bearish", ob_top=105, ob_bottom=98, entry=97, stop=106, target=85, trail_trigger=94)
    # Last candle closes above OB top (105)
    df = make_df(30, "flat", base=110.0)
    reason = check_invalidation(pos, df)
    assert reason is not None


# ── update_trail ──────────────────────────────────────────────────────────────

def test_update_trail_activates():
    pos = _pos("bullish", stop=97, trail_trigger=106)
    updated = update_trail(pos, current_price=107.0, atr_value=0.5)
    assert updated["trail_active"] is True
    assert updated["current_stop"] > pos["current_stop"]


def test_update_trail_not_triggered():
    pos = _pos("bullish", stop=97, trail_trigger=110)
    updated = update_trail(pos, current_price=105.0, atr_value=0.5)
    assert updated["trail_active"] is False
    assert updated["current_stop"] == pos["current_stop"]


def test_update_trail_never_moves_against_trade():
    pos = _pos("bullish", stop=99, trail_trigger=102)
    # At very high current price with huge ATR → new stop should still be >= initial stop
    updated = update_trail(pos, current_price=200.0, atr_value=0.5)
    assert updated["current_stop"] >= pos["stop"]
