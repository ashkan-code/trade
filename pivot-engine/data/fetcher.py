"""Bitunix REST kline fetcher with retry logic and pagination (max 200 candles/request)."""

import logging
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Confirmed from official docs search: /api/v1/futures/market/kline
KLINE_URL: str = "https://fapi.bitunix.com/api/v1/futures/market/kline"
TICKERS_URL: str = "https://fapi.bitunix.com/api/v1/futures/market/tickers"

# Interval strings from official Bitunix docs:
# 1m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d 3d 1w 1M
_TF_MAP: dict[str, str] = {
    "1m":  "1m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",
    "1h":  "1h",
    "2h":  "2h",
    "4h":  "4h",
    "6h":  "6h",
    "8h":  "8h",
    "12h": "12h",
    "1d":  "1d",
    "3d":  "3d",
    "1w":  "1w",
    "1M":  "1M",
}

# Bitunix caps at 200 candles per request — set by user
_BATCH_SIZE: int = 200

# Global HTTP concurrency cap — keeps total in-flight requests under Bitunix's 10 req/s limit
_HTTP_SEM = threading.Semaphore(8)

# Confirmed via live test: time field is 13-digit milliseconds (e.g. 1781492400000)
_TIME_UNIT: str = "ms"

_log = logging.getLogger(__name__)


def _setup_error_log() -> None:
    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "errors.log"
    )
    if not any(isinstance(h, logging.FileHandler) for h in _log.handlers):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _log.addHandler(handler)
        _log.setLevel(logging.ERROR)


def fetch_ohlcv(symbol: str, tf: str, limit: int) -> pd.DataFrame | None:
    """Fetch `limit` candles via pagination (200/batch). Returns DataFrame or None."""
    _setup_error_log()

    interval = _TF_MAP.get(tf)
    if interval is None:
        _log.error("%s unsupported timeframe: %s", _ts(), tf)
        return None

    tf_ms = _tf_to_ms(tf)
    if tf_ms is None:
        _log.error("%s cannot compute duration for tf=%s", _ts(), tf)
        return None

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    end_ms = now_ms
    batches: list[pd.DataFrame] = []
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, _BATCH_SIZE)
        start_ms = end_ms - batch_size * tf_ms

        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        }

        df_batch = _fetch_one(params, symbol)
        if df_batch is None or df_batch.empty:
            break

        batches.append(df_batch)
        remaining -= len(df_batch)
        end_ms = start_ms  # slide window back

        if remaining > 0:
            time.sleep(0.2)  # rate limit: 10 req/s per IP

    if not batches:
        _log.error("%s no data returned for %s", _ts(), symbol)
        return None

    df = pd.concat(batches).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.tail(limit)


def _fetch_one(params: dict, symbol: str) -> pd.DataFrame | None:
    """Single paginated request with retry."""
    last_exc: Exception | None = None
    for attempt, delay in enumerate(
        [0] + config.RETRY_DELAYS[: config.RETRY_ATTEMPTS - 1], start=1
    ):
        if delay:
            time.sleep(delay)
        try:
            with _HTTP_SEM:
                resp = requests.get(KLINE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code", -1) != 0:
                _log.error("%s API error for %s: %s", _ts(), symbol, data.get("msg"))
                return None
            return _parse_rows(data["data"], symbol)
        except Exception as exc:
            last_exc = exc
            _log.error(
                "%s fetch attempt %d/%d failed for %s: %s",
                _ts(), attempt, config.RETRY_ATTEMPTS, symbol, exc,
            )

    _log.error("%s all retries exhausted for %s", _ts(), symbol)
    return None


def _parse_rows(rows: list[dict], symbol: str) -> pd.DataFrame | None:
    """Parse Bitunix kline rows into OHLCV DataFrame.

    Confirmed response fields (live test, June 2026):
      time, open, high, close, low, baseVol, quoteVol

    WARNING — Bitunix field names are SWAPPED relative to standard convention:
      "baseVol"  = USDT value traded  (e.g. 78_057_172 for BTCUSDT)  ← do NOT use as volume
      "quoteVol" = coin amount traded  (e.g. 1_186.97 BTC)           ← this is the real volume
    Verified: quoteVol × close ≈ baseVol confirms the swap.
    """
    try:
        records = [
            {
                "timestamp": float(r["time"]),
                "open":      float(r["open"]),
                "high":      float(r["high"]),
                "low":       float(r["low"]),
                "close":     float(r["close"]),
                "volume":    float(r["quoteVol"]),  # coin volume — NOT baseVol (see docstring)
            }
            for r in rows
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=_TIME_UNIT, utc=True)
        return df.set_index("timestamp").sort_index()
    except Exception as exc:
        _log.error("%s parse error for %s: %s", _ts(), symbol, exc)
        return None


def _tf_to_ms(tf: str) -> int | None:
    """Return timeframe duration in milliseconds."""
    _map = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }
    return _map.get(tf)


def _fetch_tickers() -> list[dict]:
    """Fetch all USDT-M tickers from Bitunix. Returns [] on failure."""
    _setup_error_log()
    try:
        with _HTTP_SEM:
            resp = requests.get(TICKERS_URL, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code", -1) != 0:
            _log.error("%s tickers API error: %s", _ts(), data.get("msg"))
            return []
        tickers: list[dict] = data.get("data", [])
        return [t for t in tickers if str(t.get("symbol", "")).endswith("USDT")]
    except Exception as exc:
        _log.error("%s tickers fetch failed: %s", _ts(), exc)
        return []


def _usdt_volume(t: dict) -> float:
    """Extract 24h USDT turnover from a ticker dict (tries multiple field names)."""
    for field in ("amount24h", "quoteVol", "baseVol", "vol24h", "vol"):
        v = t.get(field)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


def _prepend_btc(symbols: list[str]) -> list[str]:
    """Ensure BTC is always first (direction filter)."""
    if config.BTC_SYMBOL in symbols:
        symbols.remove(config.BTC_SYMBOL)
    symbols.insert(0, config.BTC_SYMBOL)
    return symbols


def get_all_symbols(min_volume_usd: float | None = None) -> list[str]:
    """Return ALL USDT-M futures symbols with 24h USDT turnover >= min_volume_usd.

    Sorted descending by volume. BTCUSDT is always first.
    min_volume_usd defaults to config.MIN_VOLUME_USD.
    Returns [] on API failure.
    """
    if min_volume_usd is None:
        min_volume_usd = config.MIN_VOLUME_USD

    tickers = _fetch_tickers()
    if not tickers:
        return []

    filtered = [t for t in tickers if _usdt_volume(t) >= min_volume_usd]
    filtered.sort(key=_usdt_volume, reverse=True)
    symbols = [t["symbol"] for t in filtered if "symbol" in t]
    return _prepend_btc(symbols)


def get_top_symbols(n: int | None = None) -> list[str]:
    """Return top-n USDT-M futures symbols ranked by 24h USDT turnover.

    BTCUSDT is always first. n defaults to config.TOP_N.
    Returns [] on API failure.
    """
    if n is None:
        n = config.TOP_N

    tickers = _fetch_tickers()
    if not tickers:
        return []

    tickers.sort(key=_usdt_volume, reverse=True)
    symbols = [t["symbol"] for t in tickers[:n] if "symbol" in t]
    return _prepend_btc(symbols)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
