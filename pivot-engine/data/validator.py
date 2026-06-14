"""CHECK_1: basic data integrity validation."""

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_OHLCV = ["open", "high", "low", "close", "volume"]
_MIN_LEN = config.WARMUP + config.MIN_TRADES


def check_1(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate DataFrame integrity. Returns (ok, reason)."""
    if df is None or df.empty:
        return False, "dataframe is None or empty"

    missing = [c for c in _OHLCV if c not in df.columns]
    if missing:
        return False, f"missing columns: {missing}"

    if df[_OHLCV].isnull().any().any():
        return False, "NaN values present in OHLCV"

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        return False, "zero or negative price values present"

    if (df["volume"] < 0).any():
        return False, "negative volume present"

    if not df.index.is_monotonic_increasing:
        return False, "timestamps are not strictly ascending"

    if df.index.duplicated().any():
        return False, "duplicate timestamps present"

    if len(df) < _MIN_LEN:
        return False, f"insufficient data: {len(df)} candles, need {_MIN_LEN}"

    return True, "ok"
