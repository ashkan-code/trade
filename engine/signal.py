from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, TypedDict

import pandas as pd

import config
from core.ict import (detect_mss, detect_order_blocks, detect_fvg,
                     detect_liquidity, _swing_highs, _swing_lows, LiqDict)
from core.indicators import atr
from core.confluence import (
    is_aligned, find_best_zone, zone_triggered, compute_htf_confirms,
    score_zone, ZoneDict,
)
from engine.risk import compute_risk

logger = logging.getLogger(__name__)


class SignalDict(TypedDict):
    symbol: str
    direction: str           # 'long' | 'short'
    entry: float
    stop: float
    target: float
    rr: float
    score: float
    zone_type: str
    rsi_confirm: bool
    macd_confirm: bool
    timestamp: str           # ISO8601 UTC
    ltf_entry_index: int


def generate_signal(
    symbol: str,
    btc_direction: Optional[str],
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
) -> Optional[SignalDict]:
    """Full 8-step ICT signal pipeline. Returns None at any failed filter."""

    # Step 1: Symbol-level MSS on 4H
    mss_4h = detect_mss(df_4h)
    if mss_4h["direction"] is None:
        logger.debug("%s: no MSS on 4H", symbol)
        return None

    # Step 2: Alignment with BTC
    if not is_aligned(mss_4h, btc_direction):
        logger.debug("%s: not aligned with BTC (%s vs %s)", symbol, mss_4h["direction"], btc_direction)
        return None

    direction = mss_4h["direction"]  # 'bullish' | 'bearish'

    # Step 3: Zone building
    obs_4h = detect_order_blocks(df_4h)
    obs_1h = detect_order_blocks(df_1h)
    fvgs_1h = detect_fvg(df_1h)

    zone = find_best_zone(obs_1h, obs_4h, fvgs_1h, direction, symbol)
    if zone is None:
        logger.debug("%s: no zone found", symbol)
        return None

    # Step 4: Zone trigger on closed 5m candle
    if not zone_triggered(zone, df_5m):
        logger.debug("%s: zone not triggered", symbol)
        return None

    # Step 5: HTF filter confirms
    confirms = compute_htf_confirms(df_1h, direction)
    zone["rsi_confirm"] = confirms["rsi"]
    zone["macd_confirm"] = confirms["macd"]

    if not confirms["rsi"]:
        logger.debug("%s: RSI confirm failed", symbol)
        return None
    if not confirms["macd"]:
        logger.debug("%s: MACD confirm failed", symbol)
        return None

    # Step 6: 5m sniper entry
    entry_price = _find_5m_entry(direction, df_5m, zone)
    if entry_price is None:
        logger.debug("%s: no 5m entry found", symbol)
        return None

    entry_idx = int(len(df_5m) - 1)

    # Step 7: Risk calculation
    atr_1h = atr(df_1h)
    liq_pools = detect_liquidity(df_1h, atr_1h)

    # Fallback: nearest swing extreme as target when no liquidity pool found
    if not liq_pools:
        sh = _swing_highs(df_4h, config.SWING_LEN)
        sl = _swing_lows(df_4h, config.SWING_LEN)
        if direction == "bullish":
            targets = [df_4h["high"].iloc[i] for i in range(len(df_4h))
                       if sh.iloc[i] and df_4h["high"].iloc[i] > entry_price]
            if targets:
                liq_pools = [LiqDict(type="buyside", price=min(targets), touches=1)]
        else:
            targets = [df_4h["low"].iloc[i] for i in range(len(df_4h))
                       if sl.iloc[i] and df_4h["low"].iloc[i] < entry_price]
            if targets:
                liq_pools = [LiqDict(type="sellside", price=max(targets), touches=1)]
        if liq_pools:
            logger.debug("%s: using swing-extreme fallback target %s", symbol, liq_pools[0]["price"])

    ob_1h = zone.get("ob_1h")
    if ob_1h is None:
        # Use a synthetic OB from zone bounds as fallback
        from core.ict import OBDict
        ob_1h = OBDict(
            type=direction,
            top=zone["zone_top"],
            bottom=zone["zone_bottom"],
            index=-1,
            broken=False,
            breaker=False,
        )

    risk = compute_risk(direction, entry_price, ob_1h, liq_pools)
    if not risk["valid"]:
        logger.debug("%s: R:R %.2f < MIN_RR %.2f", symbol, risk["rr"], config.MIN_RR)
        return None

    # Step 8: Score
    score = score_zone(zone, risk["rr"])

    human_dir = "long" if direction == "bullish" else "short"

    return SignalDict(
        symbol=symbol,
        direction=human_dir,
        entry=risk["entry"],
        stop=risk["stop"],
        target=risk["target"],
        rr=risk["rr"],
        score=round(score, 4),
        zone_type=zone["zone_type"],
        rsi_confirm=confirms["rsi"],
        macd_confirm=confirms["macd"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        ltf_entry_index=entry_idx,
    )


def _find_5m_entry(
    direction: str,
    df_5m: pd.DataFrame,
    zone: ZoneDict,
) -> Optional[float]:
    """
    Look for a 5m ICT setup (OB or FVG) inside the HTF zone.
    Falls back to zone midpoint if nothing found in the last 10 candles.
    """
    window_5m = df_5m.iloc[-10:]

    # Try 5m OB of same direction within zone
    obs_5m = detect_order_blocks(df_5m)
    active_5m = [
        ob for ob in obs_5m
        if ob["type"] == direction
        and not ob["broken"]
        and zone["zone_bottom"] <= ob["bottom"] <= zone["zone_top"]
        and ob["index"] >= len(df_5m) - 10
    ]
    if active_5m:
        best_ob = max(active_5m, key=lambda o: o["index"])
        return (best_ob["top"] + best_ob["bottom"]) / 2

    # Try 5m FVG within zone
    fvgs_5m = detect_fvg(df_5m)
    active_fvgs = [
        f for f in fvgs_5m
        if f["type"] == direction
        and f["active"]
        and zone["zone_bottom"] <= f["bottom"] <= zone["zone_top"]
        and f["index"] >= len(df_5m) - 10
    ]
    if active_fvgs:
        best_fvg = max(active_fvgs, key=lambda f: f["index"])
        return (best_fvg["top"] + best_fvg["bottom"]) / 2

    # Fallback: zone midpoint if current 5m price is inside the zone
    last_close = float(df_5m["close"].iloc[-1])
    if zone["zone_bottom"] <= last_close <= zone["zone_top"]:
        return last_close

    # Fallback: use last 5m close regardless (entry at market)
    return float(df_5m["close"].iloc[-1])
