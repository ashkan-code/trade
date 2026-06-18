from __future__ import annotations

import logging
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ── Typed dicts ───────────────────────────────────────────────────────────────

class MSSDict(TypedDict):
    direction: Optional[str]   # 'bullish' | 'bearish' | None
    price: float
    index: int


class OBDict(TypedDict):
    type: str       # 'bullish' | 'bearish'
    top: float
    bottom: float
    index: int
    broken: bool
    breaker: bool


class FVGDict(TypedDict):
    type: str       # 'bullish' | 'bearish'
    top: float
    bottom: float
    index: int      # index of the large-body candle (middle candle)
    active: bool


class LiqDict(TypedDict):
    type: str       # 'buyside' | 'sellside'
    price: float
    touches: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _swing_highs(df: pd.DataFrame, length: int = config.SWING_LEN) -> pd.Series:
    """True at index i when high[i] is the rolling max over a centred window of 2*length+1."""
    highs = df["high"]
    rolling_max = highs.rolling(window=2 * length + 1, center=True, min_periods=2 * length + 1).max()
    is_swing = (highs == rolling_max)
    # Zero out edges where the window cannot be fully centred
    is_swing.iloc[:length] = False
    is_swing.iloc[-length:] = False
    return is_swing


def _swing_lows(df: pd.DataFrame, length: int = config.SWING_LEN) -> pd.Series:
    """True at index i when low[i] is the rolling min over a centred window of 2*length+1."""
    lows = df["low"]
    rolling_min = lows.rolling(window=2 * length + 1, center=True, min_periods=2 * length + 1).min()
    is_swing = (lows == rolling_min)
    is_swing.iloc[:length] = False
    is_swing.iloc[-length:] = False
    return is_swing


# ── MSS ───────────────────────────────────────────────────────────────────────

def detect_mss(df: pd.DataFrame, swing_len: int = config.SWING_LEN) -> MSSDict:
    """
    Scan oldest→newest closed candles tracking os state.
    Bullish MSS: close crosses above last swing high AND os != 0 (not already bullish).
    Bearish MSS: close crosses below last swing low  AND os != 1 (not already bearish).
    Returns the LAST detected MSS.
    """
    sh = _swing_highs(df, swing_len)
    sl = _swing_lows(df, swing_len)

    last_swing_high: Optional[float] = None
    last_swing_low: Optional[float] = None
    os: Optional[int] = None  # 0 = bullish structure, 1 = bearish structure

    result: MSSDict = {"direction": None, "price": float("nan"), "index": -1}

    for i in range(len(df)):
        # Update swing references first (confirmed swing = already printed length bars ago)
        if sh.iloc[i]:
            last_swing_high = df["high"].iloc[i]
        if sl.iloc[i]:
            last_swing_low = df["low"].iloc[i]

        close = df["close"].iloc[i]

        if last_swing_high is not None and close > last_swing_high and os != 0:
            os = 0
            result = {"direction": "bullish", "price": last_swing_high, "index": i}

        elif last_swing_low is not None and close < last_swing_low and os != 1:
            os = 1
            result = {"direction": "bearish", "price": last_swing_low, "index": i}

    return result


# ── Order Blocks ──────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    swing_len: int = config.OB_SWING_LOOKBACK,
) -> list[OBDict]:
    """
    For each impulse that breaks a swing, find the last opposing candle before
    the impulse — that is the Order Block.

    After identifying OBs, scan forward to:
    - Mark broken=True when price closes through the OB zone.
    - Mark breaker=True when price subsequently returns from the other side.
    """
    sh = _swing_highs(df, swing_len)
    sl = _swing_lows(df, swing_len)

    swing_high_prices: list[tuple[int, float]] = [
        (i, df["high"].iloc[i]) for i in range(len(df)) if sh.iloc[i]
    ]
    swing_low_prices: list[tuple[int, float]] = [
        (i, df["low"].iloc[i]) for i in range(len(df)) if sl.iloc[i]
    ]

    obs: list[OBDict] = []

    # ── Bullish OBs (impulse up breaks a swing high) ──────────────────────────
    for sh_idx, sh_price in swing_high_prices:
        # Find a close that breaks through this swing high (impulse candle index)
        break_idx: Optional[int] = None
        for j in range(sh_idx + 1, len(df)):
            if df["close"].iloc[j] > sh_price:
                break_idx = j
                break
        if break_idx is None:
            continue

        # Walk back from the candle before break_idx to find last bearish candle
        ob_idx: Optional[int] = None
        for k in range(break_idx - 1, sh_idx - 1, -1):
            if df["close"].iloc[k] < df["open"].iloc[k]:  # bearish candle
                ob_idx = k
                break

        if ob_idx is None:
            continue

        top = max(df["open"].iloc[ob_idx], df["close"].iloc[ob_idx])
        bottom = min(df["open"].iloc[ob_idx], df["close"].iloc[ob_idx])
        obs.append(OBDict(type="bullish", top=top, bottom=bottom, index=ob_idx,
                          broken=False, breaker=False))

    # ── Bearish OBs (impulse down breaks a swing low) ─────────────────────────
    for sl_idx, sl_price in swing_low_prices:
        break_idx = None
        for j in range(sl_idx + 1, len(df)):
            if df["close"].iloc[j] < sl_price:
                break_idx = j
                break
        if break_idx is None:
            continue

        ob_idx = None
        for k in range(break_idx - 1, sl_idx - 1, -1):
            if df["close"].iloc[k] > df["open"].iloc[k]:  # bullish candle
                ob_idx = k
                break

        if ob_idx is None:
            continue

        top = max(df["open"].iloc[ob_idx], df["close"].iloc[ob_idx])
        bottom = min(df["open"].iloc[ob_idx], df["close"].iloc[ob_idx])
        obs.append(OBDict(type="bearish", top=top, bottom=bottom, index=ob_idx,
                          broken=False, breaker=False))

    # ── Deduplicate by index ───────────────────────────────────────────────────
    seen: set[tuple[str, int]] = set()
    unique_obs: list[OBDict] = []
    for ob in sorted(obs, key=lambda o: o["index"]):
        key = (ob["type"], ob["index"])
        if key not in seen:
            seen.add(key)
            unique_obs.append(ob)
    obs = unique_obs

    # ── Mark broken / breaker ─────────────────────────────────────────────────
    for ob in obs:
        start = ob["index"] + 1
        broken_at: Optional[int] = None

        for j in range(start, len(df)):
            close_j = df["close"].iloc[j]
            if ob["type"] == "bullish" and close_j < ob["bottom"] and not ob["broken"]:
                ob["broken"] = True
                broken_at = j
            elif ob["type"] == "bearish" and close_j > ob["top"] and not ob["broken"]:
                ob["broken"] = True
                broken_at = j

            if ob["broken"] and broken_at is not None and j > broken_at:
                # Check if price returns from the other side (breaker flip)
                if ob["type"] == "bullish" and close_j > ob["top"]:
                    ob["breaker"] = True
                    break
                elif ob["type"] == "bearish" and close_j < ob["bottom"]:
                    ob["breaker"] = True
                    break

    return obs


# ── Fair Value Gaps ───────────────────────────────────────────────────────────

def detect_fvg(df: pd.DataFrame) -> list[FVGDict]:
    """
    Scan all 3-candle windows [i, i+1, i+2].

    Bullish FVG: candle i+1 is large bullish AND low[i+2] > high[i]
    Bearish FVG: candle i+1 is large bearish AND high[i+2] < low[i]

    Large-body: abs(body) > mean(abs(body)) * FVG_BODY_PERC
    """
    closes = df["close"].values
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values

    bodies = np.abs(closes - opens)
    body_mean = bodies.mean()
    threshold = body_mean * config.FVG_BODY_PERC

    fvgs: list[FVGDict] = []

    for i in range(len(df) - 2):
        mid = i + 1
        right = i + 2

        body_mid = bodies[mid]
        is_large = body_mid > threshold
        if not is_large:
            continue

        bullish_mid = closes[mid] > opens[mid]
        bearish_mid = closes[mid] < opens[mid]

        if bullish_mid and lows[right] > highs[i]:
            fvgs.append(FVGDict(
                type="bullish",
                top=lows[right],
                bottom=highs[i],
                index=mid,
                active=True,
            ))
        elif bearish_mid and highs[right] < lows[i]:
            fvgs.append(FVGDict(
                type="bearish",
                top=lows[i],
                bottom=highs[right],
                index=mid,
                active=True,
            ))

    # Mark FVGs inactive when price fully crosses through them
    for fvg in fvgs:
        start = fvg["index"] + 2
        for j in range(start, len(df)):
            c = closes[j]
            if fvg["type"] == "bullish" and c < fvg["bottom"]:
                fvg["active"] = False
                break
            elif fvg["type"] == "bearish" and c > fvg["top"]:
                fvg["active"] = False
                break

    return fvgs


# ── Liquidity ─────────────────────────────────────────────────────────────────

def detect_liquidity(df: pd.DataFrame, atr_series: pd.Series) -> list[LiqDict]:
    """
    Identify clusters of 3+ equal swing highs (buyside) or swing lows (sellside).
    Equal = price within ATR / LIQ_MARGIN_DIV of each other.
    """
    sh = _swing_highs(df, config.SWING_LEN)
    sl = _swing_lows(df, config.SWING_LEN)

    current_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
    if current_atr == 0.0 or np.isnan(current_atr):
        return []
    margin = current_atr / config.LIQ_MARGIN_DIV

    def cluster(prices: list[float], liq_type: str) -> list[LiqDict]:
        if not prices:
            return []
        sorted_prices = sorted(prices)
        result: list[LiqDict] = []
        group: list[float] = [sorted_prices[0]]
        for p in sorted_prices[1:]:
            if p - group[0] <= margin:
                group.append(p)
            else:
                if len(group) >= 3:
                    result.append(LiqDict(
                        type=liq_type,
                        price=float(np.mean(group)),
                        touches=len(group),
                    ))
                group = [p]
        if len(group) >= 3:
            result.append(LiqDict(
                type=liq_type,
                price=float(np.mean(group)),
                touches=len(group),
            ))
        return result

    sh_prices = [df["high"].iloc[i] for i in range(len(df)) if sh.iloc[i]]
    sl_prices = [df["low"].iloc[i] for i in range(len(df)) if sl.iloc[i]]

    pools = cluster(sh_prices, "buyside") + cluster(sl_prices, "sellside")
    pools.sort(key=lambda x: x["price"])
    return pools
