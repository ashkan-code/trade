from __future__ import annotations

import logging
from typing import Optional, TypedDict

import pandas as pd

import config
from core.ict import (
    MSSDict, OBDict, FVGDict, LiqDict,
    detect_mss, detect_order_blocks, detect_fvg,
)
from core.indicators import rsi, macd, rsi_confirm_long, rsi_confirm_short, macd_confirm_long, macd_confirm_short, btc_bias

logger = logging.getLogger(__name__)


class ZoneDict(TypedDict):
    symbol: str
    direction: str          # 'bullish' | 'bearish'
    zone_top: float
    zone_bottom: float
    zone_type: str          # 'ob_overlap' | 'ob_single' | 'fvg'
    ob_1h: Optional[OBDict]
    ob_4h: Optional[OBDict]
    fvg: Optional[FVGDict]
    rsi_confirm: bool
    macd_confirm: bool
    timeframe: str


# ── BTC direction (Gate 0) ────────────────────────────────────────────────────

def check_btc_direction(df_btc_4h: pd.DataFrame) -> Optional[str]:
    """Return 'bullish'|'bearish'|None based on current-state bias, not stale MSS history.

    Uses btc_bias() which scores EMA cross, EMA slope, MACD histogram, and RSI.
    Returns None when neutral (caller should abort scan).
    Prints a one-line debug summary on every call.
    """
    direction, dbg = btc_bias(df_btc_4h)
    _print_btc_bias_debug(direction, dbg)

    if direction == "long":
        return "bullish"
    if direction == "short":
        return "bearish"
    return None  # neutral → scanner aborts


def _print_btc_bias_debug(direction: str, dbg: dict) -> None:
    cross_sym = "▲" if dbg.get("ema_cross", 0) > 0 else "▼"
    slope_sym = "↑" if dbg.get("ema_slope", 0) > 0 else ("↓" if dbg.get("ema_slope", 0) < 0 else "→")
    macd_sym  = "▲" if dbg.get("macd_sig",  0) > 0 else ("▼" if dbg.get("macd_sig",  0) < 0 else "−")
    rsi_sym   = "▲" if dbg.get("rsi_sig",   0) > 0 else ("▼" if dbg.get("rsi_sig",   0) < 0 else "−")
    print(
        f"[BTC BIAS] {direction.upper():<8}  score={dbg.get('score', '?')}  "
        f"EMAcross{cross_sym}({dbg.get('ema_cross',0):+d})  "
        f"EMAslope{slope_sym}({dbg.get('ema_slope',0):+d})  "
        f"MACD{macd_sym}({dbg.get('macd_sig',0):+d})  "
        f"RSI{rsi_sym}({dbg.get('rsi_sig',0):+d})={dbg.get('rsi','?')}  "
        f"pos={dbg.get('pos_pct','?')}%  "
        f"reason={dbg.get('reason','')}"
    )


# ── Alignment ─────────────────────────────────────────────────────────────────

def is_aligned(symbol_mss: MSSDict, btc_direction: Optional[str]) -> bool:
    return (
        btc_direction is not None
        and symbol_mss["direction"] is not None
        and symbol_mss["direction"] == btc_direction
    )


# ── Zone building ─────────────────────────────────────────────────────────────

def find_mtf_ob_overlap(
    obs_1h: list[OBDict],
    obs_4h: list[OBDict],
    direction: str,
    symbol: str = "",
) -> Optional[ZoneDict]:
    """Find the tightest price-range overlap between active 1H and 4H OBs."""
    active_1h = [ob for ob in obs_1h if ob["type"] == direction and not ob["broken"]]
    active_4h = [ob for ob in obs_4h if ob["type"] == direction and not ob["broken"]]

    best: Optional[ZoneDict] = None
    best_width = float("inf")

    for ob1 in active_1h:
        for ob4 in active_4h:
            overlap_top = min(ob1["top"], ob4["top"])
            overlap_bot = max(ob1["bottom"], ob4["bottom"])
            if overlap_top <= overlap_bot:
                continue  # no overlap

            width = overlap_top - overlap_bot
            if width < best_width:
                best_width = width
                best = ZoneDict(
                    symbol=symbol,
                    direction=direction,
                    zone_top=overlap_top,
                    zone_bottom=overlap_bot,
                    zone_type="ob_overlap",
                    ob_1h=ob1,
                    ob_4h=ob4,
                    fvg=None,
                    rsi_confirm=False,
                    macd_confirm=False,
                    timeframe="1H",
                )

    return best


def find_best_zone(
    obs_1h: list[OBDict],
    obs_4h: list[OBDict],
    fvgs: list[FVGDict],
    direction: str,
    symbol: str = "",
) -> Optional[ZoneDict]:
    """Priority: MTF OB overlap → best single active 1H OB → most recent active FVG."""

    # 1. MTF OB overlap
    zone = find_mtf_ob_overlap(obs_1h, obs_4h, direction, symbol)
    if zone:
        return zone

    # 2. Best single active 1H OB (most recent)
    active_1h = [ob for ob in obs_1h if ob["type"] == direction and not ob["broken"]]
    if active_1h:
        best_ob = max(active_1h, key=lambda o: o["index"])
        return ZoneDict(
            symbol=symbol,
            direction=direction,
            zone_top=best_ob["top"],
            zone_bottom=best_ob["bottom"],
            zone_type="ob_single",
            ob_1h=best_ob,
            ob_4h=None,
            fvg=None,
            rsi_confirm=False,
            macd_confirm=False,
            timeframe="1H",
        )

    # 3. Most recent active FVG of correct direction
    fvg_type = direction  # 'bullish' | 'bearish'
    active_fvgs = [f for f in fvgs if f["type"] == fvg_type and f["active"]]
    if active_fvgs:
        best_fvg = max(active_fvgs, key=lambda f: f["index"])
        return ZoneDict(
            symbol=symbol,
            direction=direction,
            zone_top=best_fvg["top"],
            zone_bottom=best_fvg["bottom"],
            zone_type="fvg",
            ob_1h=None,
            ob_4h=None,
            fvg=best_fvg,
            rsi_confirm=False,
            macd_confirm=False,
            timeframe="1H",
        )

    return None


# ── Zone trigger ──────────────────────────────────────────────────────────────

def zone_triggered(zone: ZoneDict, df_ltf: pd.DataFrame) -> bool:
    """
    Shadow-and-return on CLOSED 5m candles (last 10).
    LONG : price entered zone (low < zone_top) then a candle CLOSED ABOVE zone_top.
    SHORT: price entered zone (high > zone_bottom) then a candle CLOSED BELOW zone_bottom.
    """
    window = df_ltf.iloc[-10:]
    direction = zone["direction"]

    if direction == "bullish":
        entered = (window["low"] <= zone["zone_top"]).any()
        closed_above = (window["close"] > zone["zone_top"]).any()
        return bool(entered and closed_above)
    else:
        entered = (window["high"] >= zone["zone_bottom"]).any()
        closed_below = (window["close"] < zone["zone_bottom"]).any()
        return bool(entered and closed_below)


# ── HTF confirms ──────────────────────────────────────────────────────────────

def compute_htf_confirms(df_htf: pd.DataFrame, direction: str) -> dict:
    rsi_series = rsi(df_htf["close"])
    _, _, histogram = macd(df_htf["close"])

    if direction == "bullish":
        rsi_ok = rsi_confirm_long(rsi_series)
        macd_ok = macd_confirm_long(histogram)
    else:
        rsi_ok = rsi_confirm_short(rsi_series)
        macd_ok = macd_confirm_short(histogram)

    return {"rsi": rsi_ok, "macd": macd_ok}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_zone(zone: ZoneDict, rr: float) -> float:
    """
    Scoring formula (placeholder — tune via backtest):
      Zone type:     ob_overlap=0.40 | ob_single=0.25 | fvg=0.15
      RSI confirm:   +0.20
      MACD confirm:  +0.20
      R:R bonus:
        rr >= MIN_RR → +0.20 (safety dominates)
        rr <  MIN_RR → +min(rr/4, 0.20) (push for better R:R)
    """
    base = {"ob_overlap": 0.40, "ob_single": 0.25, "fvg": 0.15}.get(zone["zone_type"], 0.15)

    if zone["rsi_confirm"]:
        base += 0.20
    if zone["macd_confirm"]:
        base += 0.20

    if rr >= config.MIN_RR:
        base += 0.20
    else:
        base += min(rr / 4.0, 0.20)

    return min(base, 1.0)
