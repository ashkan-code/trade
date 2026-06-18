import time
import logging
from typing import Optional

import requests
import pandas as pd
from cachetools import TTLCache

import config

logger = logging.getLogger(__name__)

_ticker_cache: TTLCache = TTLCache(maxsize=1, ttl=config.CACHE_TTL_SECONDS)
_candle_cache: TTLCache = TTLCache(maxsize=512, ttl=config.CACHE_TTL_SECONDS)

# Bitunix interval strings that the API accepts
_INTERVAL_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h", "12H": "12h",
    "1D": "1d", "1W": "1w",
}


def _get(url: str, params: dict) -> dict:
    time.sleep(config.RATE_LIMIT_DELAY)
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Bitunix HTTP error: {exc}") from exc

    data = resp.json()
    code = data.get("code", -1)
    if code != 0:
        raise RuntimeError(f"Bitunix API error code={code}: {data.get('msg', '')}")
    return data


def get_symbols() -> list[str]:
    """Return all USDT-M perpetual futures symbols."""
    cache_key = "symbols"
    if cache_key in _ticker_cache:
        return _ticker_cache[cache_key]

    url = f"{config.BITUNIX_BASE_URL}/api/v1/futures/market/tickers"
    data = _get(url, {})

    raw = data.get("data") or []
    if isinstance(raw, dict):
        raw = raw.get("list") or raw.get("tickers") or list(raw.values())

    symbols = [
        item["symbol"]
        for item in raw
        if isinstance(item, dict) and str(item.get("symbol", "")).upper().endswith("USDT")
    ]
    if not symbols:
        raise RuntimeError("get_symbols: no USDT symbols found in response")

    _ticker_cache[cache_key] = symbols
    logger.info("Fetched %d USDT symbols from Bitunix", len(symbols))
    return symbols


def get_candles(
    symbol: str,
    interval: str,
    limit: int = config.CANDLE_LIMIT,
) -> pd.DataFrame:
    """Return a closed-candle OHLCV DataFrame (last open candle dropped)."""
    cache_key = (symbol, interval)
    if cache_key in _candle_cache:
        return _candle_cache[cache_key]

    api_interval = _INTERVAL_MAP.get(interval)
    if api_interval is None:
        raise ValueError(f"Unknown interval '{interval}'. Valid: {list(_INTERVAL_MAP)}")

    url = f"{config.BITUNIX_BASE_URL}/api/v1/futures/market/kline"
    data = _get(url, {"symbol": symbol, "interval": api_interval, "limit": limit})

    raw = data.get("data") or []
    if isinstance(raw, dict):
        raw = raw.get("list") or raw.get("klines") or []

    if len(raw) < config.SWING_LEN * 2:
        raise RuntimeError(
            f"get_candles({symbol},{interval}): only {len(raw)} candles returned "
            f"(need at least {config.SWING_LEN * 2})"
        )

    rows = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            ts_ms, o, h, l, c, v = item[0], item[1], item[2], item[3], item[4], item[5]
        elif isinstance(item, dict):
            ts_ms = item.get("t") or item.get("time") or item.get("openTime")
            o = item.get("o") or item.get("open")
            h = item.get("h") or item.get("high")
            l = item.get("l") or item.get("low")
            c = item.get("c") or item.get("close")
            v = item.get("v") or item.get("volume", 0)
        else:
            continue
        rows.append((int(ts_ms), float(o), float(h), float(l), float(c), float(v)))

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)

    # Drop the last row — it is the current (open, unclosed) candle
    df = df.iloc[:-1].reset_index(drop=True)

    _candle_cache[cache_key] = df
    return df
