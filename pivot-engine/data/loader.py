"""
Offline data loader — reads pre-downloaded CSVs from data/historical/.

Use this instead of data/fetcher.py when running the backtest so no live
API calls are made. Falls back to fetcher if the CSV does not exist.

CSV format (written by scripts/download_history.py):
    timestamp,open,high,low,close,volume
    timestamp column: ISO8601 UTC string  e.g. 2025-12-20 00:00:00+00:00

Returned DataFrame:
    index  — DatetimeTZDtype UTC (same as fetcher.py)
    columns — open, high, low, close, volume  (float64)
    sorted ascending, last row dropped (open candle already excluded at download time)
"""

from __future__ import annotations

import logging
import os

import pandas as pd

_log = logging.getLogger(__name__)

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HIST_DIR = os.path.join(_BASE, "data", "historical")


def load_csv(symbol: str, tf: str) -> pd.DataFrame | None:
    """Load symbol+tf from CSV. Returns None if file missing or unreadable."""
    path = os.path.join(_HIST_DIR, f"{symbol}_{tf}.csv")
    if not os.path.exists(path):
        _log.error("CSV not found: %s — run scripts/download_history.py first", path)
        return None
    try:
        df = pd.read_csv(
            path,
            parse_dates=["timestamp"],
            dtype={
                "open": "float64",
                "high": "float64",
                "low":  "float64",
                "close": "float64",
                "volume": "float64",
            },
        )
        # Ensure UTC-aware index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        if df.empty:
            _log.error("CSV is empty: %s", path)
            return None
        return df
    except Exception as exc:
        _log.error("Failed to load %s: %s", path, exc)
        return None


def get_ohlcv(
    symbol: str,
    tf: str,
    limit: int | None = None,
    offline: bool = True,
) -> pd.DataFrame | None:
    """
    Main entry point for the backtest.

    offline=True (default): read from CSV, never hit the network.
    offline=False: fall back to live fetcher if CSV missing.

    `limit` trims to the last N rows after loading.
    """
    df = load_csv(symbol, tf)

    if df is None and not offline:
        # Lazy import so the live fetcher is not required when offline
        try:
            from data.fetcher import fetch_ohlcv
            _log.warning("CSV missing for %s %s — falling back to live fetch", symbol, tf)
            print(f"[WARN] CSV missing for {symbol} {tf} — falling back to live fetch")
            df = fetch_ohlcv(symbol, tf, limit=limit or 600)
        except Exception as exc:
            _log.error("Live fetch fallback failed for %s %s: %s", symbol, tf, exc)
            return None

    if df is None:
        return None

    if limit is not None:
        df = df.iloc[-limit:]

    return df
