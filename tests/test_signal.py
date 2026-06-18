"""Tests for engine/signal.py."""
import unittest.mock as mock
import pandas as pd
import pytest

from tests.conftest import make_df
from engine.signal import generate_signal, SignalDict


def _make_dfs(n=80):
    """Return three synthetic DataFrames (4H, 1H, 5m)."""
    return make_df(n, "up"), make_df(n, "up"), make_df(n * 12, "up")


# ── Full pipeline happy-path ──────────────────────────────────────────────────

def test_generate_signal_structure_when_valid():
    """
    Mock sub-components so that all filters pass → verify SignalDict keys.
    """
    df_4h, df_1h, df_5m = _make_dfs()

    from core.ict import MSSDict, OBDict, LiqDict
    mss_bull = MSSDict(direction="bullish", price=102.0, index=20)
    ob = OBDict(type="bullish", top=105, bottom=102, index=10, broken=False, breaker=False)
    liq = [{"type": "buyside", "price": 130.0, "touches": 3}]
    zone = {
        "symbol": "TESTUSDT", "direction": "bullish",
        "zone_top": 106.0, "zone_bottom": 103.0,
        "zone_type": "ob_overlap", "ob_1h": ob, "ob_4h": ob,
        "fvg": None, "rsi_confirm": True, "macd_confirm": True, "timeframe": "1H",
    }

    with mock.patch("engine.signal.detect_mss", return_value=mss_bull), \
         mock.patch("engine.signal.is_aligned", return_value=True), \
         mock.patch("engine.signal.detect_order_blocks", return_value=[ob]), \
         mock.patch("engine.signal.detect_fvg", return_value=[]), \
         mock.patch("engine.signal.find_best_zone", return_value=zone), \
         mock.patch("engine.signal.zone_triggered", return_value=True), \
         mock.patch("engine.signal.compute_htf_confirms", return_value={"rsi": True, "macd": True}), \
         mock.patch("engine.signal._find_5m_entry", return_value=107.0), \
         mock.patch("engine.signal.detect_liquidity", return_value=liq):

        sig = generate_signal("TESTUSDT", "bullish", df_4h, df_1h, df_5m)

    assert sig is not None
    required_keys = ["symbol", "direction", "entry", "stop", "target", "rr",
                     "score", "zone_type", "rsi_confirm", "macd_confirm",
                     "timestamp", "ltf_entry_index"]
    for k in required_keys:
        assert k in sig, f"missing key: {k}"

    assert sig["direction"] in ("long", "short")
    assert sig["rr"] >= 0


# ── Filter rejection tests ────────────────────────────────────────────────────

def test_returns_none_when_no_mss():
    df_4h, df_1h, df_5m = _make_dfs()
    from core.ict import MSSDict
    no_mss = MSSDict(direction=None, price=float("nan"), index=-1)

    with mock.patch("engine.signal.detect_mss", return_value=no_mss):
        sig = generate_signal("TESTUSDT", "bullish", df_4h, df_1h, df_5m)
    assert sig is None


def test_returns_none_when_not_aligned():
    df_4h, df_1h, df_5m = _make_dfs()
    from core.ict import MSSDict
    mss_bear = MSSDict(direction="bearish", price=100.0, index=10)

    with mock.patch("engine.signal.detect_mss", return_value=mss_bear), \
         mock.patch("engine.signal.is_aligned", return_value=False):
        sig = generate_signal("TESTUSDT", "bullish", df_4h, df_1h, df_5m)
    assert sig is None


def test_returns_none_when_zone_not_triggered():
    df_4h, df_1h, df_5m = _make_dfs()
    from core.ict import MSSDict, OBDict
    mss_bull = MSSDict(direction="bullish", price=100.0, index=10)
    zone = {"symbol": "X", "direction": "bullish", "zone_top": 106, "zone_bottom": 103,
            "zone_type": "ob_overlap", "ob_1h": None, "ob_4h": None, "fvg": None,
            "rsi_confirm": False, "macd_confirm": False, "timeframe": "1H"}

    with mock.patch("engine.signal.detect_mss", return_value=mss_bull), \
         mock.patch("engine.signal.is_aligned", return_value=True), \
         mock.patch("engine.signal.detect_order_blocks", return_value=[]), \
         mock.patch("engine.signal.detect_fvg", return_value=[]), \
         mock.patch("engine.signal.find_best_zone", return_value=zone), \
         mock.patch("engine.signal.zone_triggered", return_value=False):
        sig = generate_signal("TESTUSDT", "bullish", df_4h, df_1h, df_5m)
    assert sig is None


def test_returns_none_when_rsi_fails():
    df_4h, df_1h, df_5m = _make_dfs()
    from core.ict import MSSDict, OBDict
    mss_bull = MSSDict(direction="bullish", price=100.0, index=10)
    ob = OBDict(type="bullish", top=105, bottom=102, index=10, broken=False, breaker=False)
    zone = {"symbol": "X", "direction": "bullish", "zone_top": 106, "zone_bottom": 103,
            "zone_type": "ob_overlap", "ob_1h": ob, "ob_4h": ob, "fvg": None,
            "rsi_confirm": False, "macd_confirm": False, "timeframe": "1H"}

    with mock.patch("engine.signal.detect_mss", return_value=mss_bull), \
         mock.patch("engine.signal.is_aligned", return_value=True), \
         mock.patch("engine.signal.detect_order_blocks", return_value=[ob]), \
         mock.patch("engine.signal.detect_fvg", return_value=[]), \
         mock.patch("engine.signal.find_best_zone", return_value=zone), \
         mock.patch("engine.signal.zone_triggered", return_value=True), \
         mock.patch("engine.signal.compute_htf_confirms", return_value={"rsi": False, "macd": True}):
        sig = generate_signal("TESTUSDT", "bullish", df_4h, df_1h, df_5m)
    assert sig is None
