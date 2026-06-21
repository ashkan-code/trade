"""Gate 2: OB/FVG rejection detection.

Core rejection rule:
  LONG : low  <= zone_high  AND  close > zone_high   (shadow enters, close exits above)
  SHORT: high >= zone_low   AND  close < zone_low    (shadow enters, close exits below)

Grade A+: body entirely outside zone (open AND close both beyond zone edge).
Grade B : close beyond zone edge, body partially inside.
"""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, RejectionCandle, Zone
from engine.ict import detect_fvg, detect_order_blocks


def find_active_zones(
    df: pd.DataFrame,
    direction: Direction,
    timeframe: str = "",
) -> list[Zone]:
    """Return all active OBs + FVGs aligned with direction, sorted by recency."""
    obs = detect_order_blocks(df, direction, timeframe)
    fvgs = detect_fvg(df, direction, timeframe)
    all_zones = obs + fvgs
    all_zones.sort(key=lambda z: z.origin_index, reverse=True)
    return all_zones


def validate_rejection(
    row: pd.Series,
    zone: Zone,
) -> tuple[bool, str]:
    """Check if a single closed candle is a valid rejection of the zone.

    Returns (passed, grade_or_reason).
    """
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = abs(c - o)
    zone_h = zone.zone_high
    zone_l = zone.zone_low
    zone_height = zone_h - zone_l
    if zone_height <= 0:
        return False, "degenerate_zone"

    if zone.direction == "long":
        # Shadow enters zone from above: low touches or pierces zone_high
        if not (l <= zone_h):
            return False, "no_touch"
        # Close must exit zone upward
        if not (c > zone_h):
            return False, "close_inside_or_below"
        # Penetration depth check
        penetration = zone_h - l
        if penetration < config.PENETRATION_MIN * zone_height:
            return False, "penetration_too_shallow"
        # Wick ratio check: lower wick (shadow that entered zone) vs body
        lower_wick = min(o, c) - l
        if body > 0 and lower_wick < config.WICK_RATIO_MIN * body:
            return False, "wick_ratio_too_low"
        # Grade
        if o > zone_h and c > zone_h:
            return True, "A+"
        return True, "B"

    else:  # short
        if not (h >= zone_l):
            return False, "no_touch"
        if not (c < zone_l):
            return False, "close_inside_or_above"
        penetration = h - zone_l
        if penetration < config.PENETRATION_MIN * zone_height:
            return False, "penetration_too_shallow"
        upper_wick = h - max(o, c)
        if body > 0 and upper_wick < config.WICK_RATIO_MIN * body:
            return False, "wick_ratio_too_low"
        if o < zone_l and c < zone_l:
            return True, "A+"
        return True, "B"


def _has_prior_touch(
    df: pd.DataFrame,
    zone: Zone,
    candidate_idx: int,
) -> bool:
    """Return True if zone was already touched before candidate_idx."""
    start = max(zone.origin_index + 1, candidate_idx - config.OB_TOUCH_LOOKBACK)
    end = candidate_idx  # exclusive

    for i in range(start, end):
        row = df.iloc[i]
        if zone.direction == "long":
            if float(row["low"]) <= zone.zone_high:
                return True
        else:
            if float(row["high"]) >= zone.zone_low:
                return True
    return False


def diagnose_gate2_rejection(
    df: pd.DataFrame,
    candidate_idx: int,
    zones: list[Zone],
) -> str:
    """Return the primary reason find_rejection returned None for this df/zones.

    Priority: rejection_shape > prior_touch > grade_filter > zone_too_recent > no_active_zones
    """
    if not zones:
        return "no_active_zones"

    row = df.iloc[candidate_idx]
    any_shape   = False
    any_touched = False
    any_grade   = False
    any_future  = False

    for zone in zones:
        if zone.origin_index >= candidate_idx:
            any_future = True
            continue
        if _has_prior_touch(df, zone, candidate_idx):
            any_touched = True
            continue
        passed, grade_or_reason = validate_rejection(row, zone)
        if not passed:
            any_shape = True
            continue
        if config.MIN_SETUP_GRADE == "A+" and grade_or_reason != "A+":
            any_grade = True

    if any_shape:
        return "rejection_shape"
    if any_touched:
        return "prior_touch"
    if any_grade:
        return "grade_filter"
    if any_future:
        return "zone_too_recent"
    return "no_active_zones"


def find_rejection(
    df: pd.DataFrame,
    candidate_idx: int,
    zones: list[Zone],
) -> RejectionCandle | None:
    """Find the best rejection candle at candidate_idx across all zones.

    Prefers A+ grade over B. Returns None if no zone qualifies.
    """
    row = df.iloc[candidate_idx]
    best: RejectionCandle | None = None

    for zone in zones:
        # Zone must exist before candidate
        if zone.origin_index >= candidate_idx:
            continue
        # Single-touch rule: reject if zone was already touched
        if _has_prior_touch(df, zone, candidate_idx):
            continue

        passed, grade_or_reason = validate_rejection(row, zone)
        if not passed:
            continue

        grade = grade_or_reason  # "A+" or "B"

        # Enforce MIN_SETUP_GRADE: if config requires A+, skip Grade B
        if config.MIN_SETUP_GRADE == "A+" and grade != "A+":
            continue

        # Entry: zone edge; shadow_extreme: worst point of the rejection wick
        if zone.direction == "long":
            entry = zone.zone_high
            shadow_extreme = float(row["low"])
        else:
            entry = zone.zone_low
            shadow_extreme = float(row["high"])

        rejection = RejectionCandle(
            candle_index=candidate_idx,
            grade=grade,  # type: ignore[arg-type]
            zone=zone,
            entry=entry,
            shadow_extreme=shadow_extreme,
        )

        # Prefer A+ over B; within same grade, prefer most recent zone
        if best is None:
            best = rejection
        elif grade == "A+" and best.grade != "A+":
            best = rejection

    return best
