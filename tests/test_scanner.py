"""Tests for engine/scanner.py."""
import unittest.mock as mock
import pytest

import config


def _make_signal(symbol, score):
    return {
        "symbol": symbol, "direction": "long", "entry": 100.0,
        "stop": 98.0, "target": 106.0, "rr": 3.0, "score": score,
        "zone_type": "ob_overlap", "rsi_confirm": True, "macd_confirm": True,
        "timestamp": "2024-01-01T00:00:00+00:00", "ltf_entry_index": 10,
    }


def test_run_scan_returns_sorted_by_score():
    from engine.scanner import run_scan
    from tests.conftest import make_df

    signals = [
        _make_signal("AAVEUSDT", 0.80),
        _make_signal("ETHUSDT", 0.95),
        _make_signal("SOLUSDT", 0.85),
    ]

    with mock.patch("engine.scanner.get_candles", return_value=make_df(60, "up")), \
         mock.patch("engine.scanner.get_symbols", return_value=["ETHUSDT", "AAVEUSDT", "SOLUSDT"]), \
         mock.patch("engine.scanner.check_btc_direction", return_value="bullish"), \
         mock.patch("engine.scanner.generate_signal", side_effect=signals):
        result = run_scan()

    scores = [s["score"] for s in result]
    assert scores == sorted(scores, reverse=True)


def test_run_scan_filters_below_threshold():
    from engine.scanner import run_scan
    from tests.conftest import make_df

    # Score below SCORE_THRESHOLD should be filtered
    low_score = config.SCORE_THRESHOLD - 0.1
    signals = [_make_signal("ETHUSDT", low_score)]

    with mock.patch("engine.scanner.get_candles", return_value=make_df(60, "up")), \
         mock.patch("engine.scanner.get_symbols", return_value=["ETHUSDT"]), \
         mock.patch("engine.scanner.check_btc_direction", return_value="bullish"), \
         mock.patch("engine.scanner.generate_signal", side_effect=signals):
        result = run_scan()

    assert result == []


def test_run_scan_skips_btc_symbol():
    from engine.scanner import run_scan
    from tests.conftest import make_df

    with mock.patch("engine.scanner.get_candles", return_value=make_df(60, "up")), \
         mock.patch("engine.scanner.get_symbols", return_value=[config.BTC_SYMBOL, "ETHUSDT"]), \
         mock.patch("engine.scanner.check_btc_direction", return_value="bullish"), \
         mock.patch("engine.scanner.generate_signal", return_value=_make_signal("ETHUSDT", 0.9)) as mock_gen:
        run_scan()

    # generate_signal should only be called for ETHUSDT, not BTCUSDT
    for call in mock_gen.call_args_list:
        assert call.args[0] != config.BTC_SYMBOL


def test_run_scan_handles_symbol_fetch_error():
    from engine.scanner import run_scan
    from tests.conftest import make_df

    with mock.patch("engine.scanner.get_candles", return_value=make_df(60, "up")), \
         mock.patch("engine.scanner.get_symbols", side_effect=RuntimeError("network error")), \
         mock.patch("engine.scanner.check_btc_direction", return_value="bullish"):
        result = run_scan()

    assert result == []
