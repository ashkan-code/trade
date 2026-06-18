from __future__ import annotations

from typing import Optional, TypedDict

import config
from core.ict import OBDict, LiqDict


class RiskDict(TypedDict):
    entry: float
    stop: float
    target: float
    rr: float
    risk_pct: float
    trail_trigger: float
    valid: bool


def compute_risk(
    direction: str,
    entry: float,
    ob_1h: OBDict,
    liquidity_pools: list[LiqDict],
) -> RiskDict:
    """
    Stop: just beyond the 1H OB boundary (STOP_BUFFER_PCT buffer).
    Target: nearest opposing liquidity pool beyond entry.
    """
    buffer = entry * config.STOP_BUFFER_PCT

    if direction == "bullish":
        stop = ob_1h["bottom"] - buffer
        candidates = [p["price"] for p in liquidity_pools if p["type"] == "buyside" and p["price"] > entry]
        target = min(candidates) if candidates else None
    else:
        stop = ob_1h["top"] + buffer
        candidates = [p["price"] for p in liquidity_pools if p["type"] == "sellside" and p["price"] < entry]
        target = max(candidates) if candidates else None

    if target is None:
        return RiskDict(
            entry=entry, stop=stop, target=float("nan"),
            rr=0.0, risk_pct=0.0, trail_trigger=float("nan"), valid=False,
        )

    risk_pts = abs(entry - stop)
    reward_pts = abs(target - entry)
    rr = reward_pts / risk_pts if risk_pts > 0 else 0.0
    risk_pct = risk_pts / entry if entry > 0 else 0.0

    if direction == "bullish":
        trail_trigger = entry + config.TRAIL_START_R * risk_pts
    else:
        trail_trigger = entry - config.TRAIL_START_R * risk_pts

    return RiskDict(
        entry=entry,
        stop=stop,
        target=target,
        rr=round(rr, 4),
        risk_pct=round(risk_pct, 6),
        trail_trigger=trail_trigger,
        valid=rr >= config.MIN_RR,
    )


def should_trail(current_price: float, direction: str, risk: RiskDict) -> bool:
    if direction == "bullish":
        return current_price >= risk["trail_trigger"]
    return current_price <= risk["trail_trigger"]


def compute_trail_stop(
    current_price: float,
    direction: str,
    risk: RiskDict,
    atr_value: float,
) -> float:
    if direction == "bullish":
        new_stop = current_price - 2 * atr_value
        return max(new_stop, risk["stop"])  # never move stop against trade
    else:
        new_stop = current_price + 2 * atr_value
        return min(new_stop, risk["stop"])
