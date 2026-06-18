"""Tests for data/bitunix.py — all HTTP calls mocked."""
import time
import unittest.mock as mock

import pandas as pd
import pytest

import config


@pytest.fixture(autouse=True)
def clear_caches():
    """Reset module-level TTL caches before each test."""
    import data.bitunix as b
    b._ticker_cache.clear()
    b._candle_cache.clear()
    yield
    b._ticker_cache.clear()
    b._candle_cache.clear()


def _ticker_response():
    return {
        "code": 0,
        "data": [
            {"symbol": "BTCUSDT", "lastPrice": "67000"},
            {"symbol": "ETHUSDT", "lastPrice": "3500"},
            {"symbol": "BTCBTC", "lastPrice": "1"},  # should be filtered out
        ],
    }


def _kline_response(n=10):
    """n+1 rows (last will be dropped as open candle)."""
    rows = []
    base_ts = 1700000000000
    for i in range(n + 1):
        ts = base_ts + i * 3600000
        rows.append([ts, str(100 + i), str(101 + i), str(99 + i), str(100.5 + i), "500"])
    return {"code": 0, "data": rows}


def test_get_symbols_returns_usdt_only():
    from data.bitunix import get_symbols
    with mock.patch("data.bitunix._get", return_value=_ticker_response()):
        syms = get_symbols()
    assert "BTCUSDT" in syms
    assert "ETHUSDT" in syms
    assert "BTCBTC" not in syms
    assert all(s.endswith("USDT") for s in syms)


def test_get_symbols_cached():
    from data.bitunix import get_symbols
    with mock.patch("data.bitunix._get", return_value=_ticker_response()) as mock_get:
        get_symbols()
        get_symbols()
    assert mock_get.call_count == 1


def test_get_candles_shape_and_dtypes():
    from data.bitunix import get_candles
    resp = _kline_response(n=15)
    with mock.patch("data.bitunix._get", return_value=resp):
        df = get_candles("BTCUSDT", "1H", limit=16)

    # last row (open candle) should be dropped → 15 rows
    assert len(df) == 15
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert df["open"].dtype == float
    assert str(df["timestamp"].dtype).startswith("datetime64")
    assert df["timestamp"].dt.tz is not None  # UTC-aware


def test_get_candles_ascending_order():
    from data.bitunix import get_candles
    resp = _kline_response(n=20)
    # Reverse the rows to simulate descending API response
    resp["data"] = list(reversed(resp["data"]))
    with mock.patch("data.bitunix._get", return_value=resp):
        df = get_candles("ETHUSDT", "4H", limit=21)
    assert df["timestamp"].is_monotonic_increasing


def test_get_candles_cached():
    from data.bitunix import get_candles
    resp = _kline_response(n=15)
    with mock.patch("data.bitunix._get", return_value=resp) as mock_get:
        get_candles("BTCUSDT", "1H", limit=16)
        get_candles("BTCUSDT", "1H", limit=16)
    assert mock_get.call_count == 1


def test_get_candles_api_error_raises():
    from data.bitunix import get_candles
    err_resp = {"code": 10001, "msg": "rate limit"}

    with mock.patch("requests.get") as mock_req:
        mock_req.return_value.status_code = 200
        mock_req.return_value.raise_for_status = mock.Mock()
        mock_req.return_value.json.return_value = err_resp
        with pytest.raises(RuntimeError):
            get_candles("BTCUSDT", "4H")


def test_get_candles_invalid_interval():
    from data.bitunix import get_candles
    with pytest.raises(ValueError, match="Unknown interval"):
        get_candles("BTCUSDT", "99X")


def test_get_candles_too_few_rows():
    from data.bitunix import get_candles
    resp = _kline_response(n=3)  # only 3 rows → below SWING_LEN*2
    with mock.patch("data.bitunix._get", return_value=resp):
        with pytest.raises(RuntimeError, match="candles returned"):
            get_candles("BTCUSDT", "1H", limit=4)
