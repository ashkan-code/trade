from __future__ import annotations

import logging
import time
from typing import Optional, TypedDict

import pandas as pd

import config
from core.ict import detect_mss, detect_order_blocks, OBDict
from core.indicators import atr
from core.confluence import ZoneDict
from engine.risk import RiskDict, should_trail, compute_trail_stop
from engine.telegram import send_alert

logger = logging.getLogger(__name__)


class PositionDict(TypedDict):
    symbol: str
    direction: str       # 'bullish' | 'bearish'
    entry: float
    stop: float
    target: float
    rr: float
    ob_1h: OBDict
    zone: ZoneDict
    entry_time: str      # ISO8601 UTC
    trail_active: bool
    current_stop: float
    trail_trigger: float


def check_invalidation(position: PositionDict, df_1h: pd.DataFrame) -> Optional[str]:
    """
    Returns a reason string if the trade structure is invalidated, else None.
    Conditions:
      LONG : 1H close below bullish OB bottom OR new bearish MSS
      SHORT: 1H close above bearish OB top  OR new bullish MSS
    """
    ob = position["ob_1h"]
    direction = position["direction"]
    last_close = float(df_1h["close"].iloc[-1])

    if direction == "bullish":
        if last_close < ob["bottom"]:
            return f"1H close {last_close:.4g} broke below OB bottom {ob['bottom']:.4g}"
        mss = detect_mss(df_1h)
        if mss["direction"] == "bearish" and mss["index"] > ob["index"]:
            return f"New bearish MSS at bar {mss['index']} invalidates long setup"

    else:  # bearish position
        if last_close > ob["top"]:
            return f"1H close {last_close:.4g} broke above OB top {ob['top']:.4g}"
        mss = detect_mss(df_1h)
        if mss["direction"] == "bullish" and mss["index"] > ob["index"]:
            return f"New bullish MSS at bar {mss['index']} invalidates short setup"

    return None


def update_trail(
    position: PositionDict,
    current_price: float,
    atr_value: float,
) -> PositionDict:
    """Activate or tighten the trailing stop. Never moves stop against the trade."""
    risk = RiskDict(
        entry=position["entry"],
        stop=position["current_stop"],
        target=position["target"],
        rr=position["rr"],
        risk_pct=0.0,
        trail_trigger=position["trail_trigger"],
        valid=True,
    )

    if should_trail(current_price, position["direction"], risk):
        new_stop = compute_trail_stop(current_price, position["direction"], risk, atr_value)
        position = dict(position)  # type: ignore[assignment]
        position["trail_active"] = True
        position["current_stop"] = new_stop

    return position  # type: ignore[return-value]


def monitor_loop(positions: list[PositionDict]) -> None:
    """
    Daemon loop — checks open positions every CACHE_TTL_SECONDS.
    Sends Telegram alert on invalidation or trail-stop update.
    """
    from data.bitunix import get_candles  # lazy import to avoid circular

    while True:
        time.sleep(config.CACHE_TTL_SECONDS)

        for i, pos in enumerate(list(positions)):
            symbol = pos["symbol"]
            try:
                df_1h = get_candles(symbol, "1H")
                atr_val = float(atr(df_1h).iloc[-1])
                current_price = float(df_1h["close"].iloc[-1])

                reason = check_invalidation(pos, df_1h)
                if reason:
                    msg = f"⚠️ POSITION INVALIDATED — {symbol} {pos['direction'].upper()}\n{reason}"
                    send_alert(msg)
                    logger.warning(msg)
                    positions.remove(pos)
                    continue

                updated = update_trail(pos, current_price, atr_val)
                if updated["trail_active"] and updated["current_stop"] != pos["current_stop"]:
                    msg = (
                        f"📌 TRAIL STOP UPDATE — {symbol}\n"
                        f"New stop: {updated['current_stop']:.6g}"
                    )
                    send_alert(msg)
                    positions[positions.index(pos)] = updated  # type: ignore[type-var]

            except Exception as exc:
                logger.error("Monitor error for %s: %s", symbol, exc)
