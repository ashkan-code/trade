from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

import config
from data.bitunix import get_symbols, get_candles
from core.confluence import check_btc_direction
from engine.signal import generate_signal, SignalDict

logger = logging.getLogger(__name__)


def _fetch_symbol_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch 4H, 1H, 5m candles for a symbol with rate-limit delay between calls."""
    df_4h = get_candles(symbol, "4H")
    time.sleep(config.RATE_LIMIT_DELAY)
    df_1h = get_candles(symbol, "1H")
    time.sleep(config.RATE_LIMIT_DELAY)
    df_5m = get_candles(symbol, "5m")
    return df_4h, df_1h, df_5m


def run_scan() -> list[SignalDict]:
    """
    Full market scan:
    1. BTC 4H direction.
    2. All Bitunix USDT-M futures.
    3. Generate signal for each aligned symbol.
    4. Filter by SCORE_THRESHOLD, sort by score desc.
    """
    logger.info("Starting market scan")

    # BTC direction (Gate 0) — btc_bias prints its own debug line
    try:
        df_btc_4h = get_candles(config.BTC_SYMBOL, "4H")
        btc_direction = check_btc_direction(df_btc_4h)
        if btc_direction is None:
            logger.info("BTC direction NEUTRAL — market ranging, scan aborted")
            return []
        logger.info("BTC 4H direction: %s", btc_direction)
    except Exception as exc:
        logger.error("Failed to get BTC direction: %s", exc)
        return []

    # Symbol list
    try:
        symbols = get_symbols()
    except Exception as exc:
        logger.error("Failed to get symbol list: %s", exc)
        return []

    signals: list[SignalDict] = []

    for symbol in symbols:
        if symbol == config.BTC_SYMBOL:
            continue

        try:
            df_4h, df_1h, df_5m = _fetch_symbol_data(symbol)
            signal = generate_signal(symbol, btc_direction, df_4h, df_1h, df_5m)
            if signal is not None:
                signals.append(signal)
                logger.info("Signal found: %s %s score=%.2f", symbol, signal["direction"], signal["score"])
        except Exception as exc:
            logger.debug("Skipping %s: %s", symbol, exc)
            continue

    # Filter and rank
    signals = [s for s in signals if s["score"] >= config.SCORE_THRESHOLD]
    signals.sort(key=lambda s: s["score"], reverse=True)

    logger.info("Scan complete: %d signals above threshold", len(signals))
    return signals
