"""Bitunix REST kline fetcher with retry logic."""

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# CONFIRM: verify exact endpoint and path from official Bitunix docs
KLINE_URL: str = "https://fapi.bitunix.com/api/v1/market/kline"

# CONFIRM: verify interval string mapping matches Bitunix API docs
_TF_MAP: dict[str, str] = {
    "1m": "1",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "4h": "240",
    "1d": "D",
}

_log = logging.getLogger(__name__)


def _setup_error_log() -> None:
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "errors.log")
    if not any(isinstance(h, logging.FileHandler) for h in _log.handlers):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _log.addHandler(handler)
        _log.setLevel(logging.ERROR)


def fetch_ohlcv(symbol: str, tf: str, limit: int) -> pd.DataFrame | None:
    """Fetch OHLCV klines from Bitunix. Returns DataFrame or None on failure."""
    _setup_error_log()

    interval = _TF_MAP.get(tf)
    if interval is None:
        _log.error("%s unsupported timeframe: %s", _ts(), tf)
        return None

    params = {
        "symbol": symbol,
        "period": interval,  # CONFIRM: param name may differ ("interval", "granularity", etc.)
        "limit": limit,
    }

    last_exc: Exception | None = None
    for attempt, delay in enumerate(
        [0] + config.RETRY_DELAYS[: config.RETRY_ATTEMPTS - 1], start=1
    ):
        if delay:
            time.sleep(delay)
        try:
            resp = requests.get(KLINE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return _parse_response(data, symbol)
        except Exception as exc:
            last_exc = exc
            _log.error("%s fetch attempt %d/%d failed for %s: %s", _ts(), attempt, config.RETRY_ATTEMPTS, symbol, exc)

    _log.error("%s all retries exhausted for %s", _ts(), symbol)
    return None


def _parse_response(data: dict | list, symbol: str) -> pd.DataFrame | None:
    """Parse raw API response into OHLCV DataFrame.

    CONFIRM: adapt column extraction to actual Bitunix response schema.
    Assumed structure: data["data"] is a list of lists or dicts with
    [timestamp, open, high, low, close, volume].
    """
    try:
        # CONFIRM: adjust key path and column order for actual Bitunix response
        rows = data.get("data", data) if isinstance(data, dict) else data
        df = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as exc:
        _log.error("%s parse error for %s: %s", _ts(), symbol, exc)
        return None


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
