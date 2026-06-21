"""Gate 2: ICT/Wyckoff liquidity sweep + reclaim detection.

Sweep rule (all 4 must pass):
  1. Last candle swept a recent swing high (SHORT) or low (LONG) within SWEEP_LOOKBACK bars.
  2. Candle closed back below swept high (SHORT) / above swept low (LONG) — the reclaim.
  3. Rejection candle volume >= VOLUME_MIN_RATIO × SMA(volume, 9).
  4. No counter-institutional spike (>= VOLUME_SPIKE_MULTIPLIER × SMA9 in opposite direction)
     within LOOKBACK_BARS before the sweep.

Grade A+: body entirely outside swept level (open AND close both beyond).
Grade B : close reclaimed, body partially overlaps.

Legacy OB/FVG functions kept for backward compatibility.
"""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, RejectionCandle, Zone
from engine.ict import detect_fvg, detect_order_blocks


def _volume_sma(df: pd.DataFrame, idx: int, period: int = 9) -> float:
    """SMA of volume over up to `period` bars ending at idx (inclusive)."""
    if idx < 0:
        return 0.0
    start = max(0, idx - period + 1)
    vols = df["volume"].iloc[start : idx + 1]
    return float(vols.mean()) if len(vols) else 0.0


def find_sweep_rejection(
    df: pd.DataFrame,
    direction: Direction,
    timeframe: str = "",
) -> RejectionCandle | None:
    """Gate 2: liquidity sweep + reclaim + Wyckoff volume confirmation.

    Checks the LAST closed candle of df. Returns RejectionCandle on pass, None on failure.
    """
    n = len(df)
    if n < config.SWEEP_LOOKBACK + 10:
        return None

    candidate_idx = n - 1
    row = df.iloc[candidate_idx]
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    o = float(row["open"])
    vol = float(row["volume"])

    lb_start = max(0, candidate_idx - config.SWEEP_LOOKBACK)

    # 1 & 2: sweep + reclaim
    if direction == "short":
        swing_arr = df["high"].values[lb_start:candidate_idx]
        if len(swing_arr) == 0:
            return None
        swing_level = float(swing_arr.max())
        origin_int = lb_start + int(swing_arr.argmax())
        if not (h > swing_level and c < swing_level):
            return None
    else:
        swing_arr = df["low"].values[lb_start:candidate_idx]
        if len(swing_arr) == 0:
            return None
        swing_level = float(swing_arr.min())
        origin_int = lb_start + int(swing_arr.argmin())
        if not (l < swing_level and c > swing_level):
            return None

    # 3: rejection candle volume check
    vol_sma = _volume_sma(df, candidate_idx - 1)
    if vol_sma > 0 and vol < config.VOLUME_MIN_RATIO * vol_sma:
        return None

    # 4: no counter-institutional spike in LOOKBACK_BARS before the sweep
    check_start = max(0, candidate_idx - config.LOOKBACK_BARS)
    for i in range(check_start, candidate_idx):
        prev = df.iloc[i]
        prev_vol = float(prev["volume"])
        sma_i = _volume_sma(df, i - 1)
        if sma_i <= 0 or prev_vol < config.VOLUME_SPIKE_MULTIPLIER * sma_i:
            continue
        is_bullish = float(prev["close"]) > float(prev["open"])
        if direction == "short" and is_bullish:
            return None
        if direction == "long" and not is_bullish:
            return None

    # Build zone and grade
    if direction == "short":
        zone = Zone(
            zone_type="ob", direction="short",
            zone_high=h, zone_low=swing_level,
            origin_index=origin_int, timeframe=timeframe,
        )
        entry = swing_level
        shadow_extreme = h
    else:
        zone = Zone(
            zone_type="ob", direction="long",
            zone_high=swing_level, zone_low=l,
            origin_index=origin_int, timeframe=timeframe,
        )
        entry = swing_level
        shadow_extreme = l

    body = abs(c - o)
    wick = (h - max(o, c)) if direction == "short" else (min(o, c) - l)
    grade: str = "A+" if body > 0 and wick >= body else "B"

    return RejectionCandle(
        candle_index=candidate_idx,
        grade=grade,  # type: ignore[arg-type]
        zone=zone,
        entry=entry,
        shadow_extreme=shadow_extreme,
    )


def diagnose_sweep_rejection(df: pd.DataFrame, direction: Direction) -> str:
    """Return primary reason find_sweep_rejection returned None (for diagnostic logging)."""
    n = len(df)
    if n < config.SWEEP_LOOKBACK + 10:
        return "insufficient_data"

    candidate_idx = n - 1
    row = df.iloc[candidate_idx]
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    vol = float(row["volume"])

    lb_start = max(0, candidate_idx - config.SWEEP_LOOKBACK)

    if direction == "short":
        swing_arr = df["high"].values[lb_start:candidate_idx]
        swing_level = float(swing_arr.max()) if len(swing_arr) else 0.0
        swept_and_reclaimed = h > swing_level and c < swing_level
    else:
        swing_arr = df["low"].values[lb_start:candidate_idx]
        swing_level = float(swing_arr.min()) if len(swing_arr) else float("inf")
        swept_and_reclaimed = l < swing_level and c > swing_level

    if not swept_and_reclaimed:
        return "sweep_not_found"

    vol_sma = _volume_sma(df, candidate_idx - 1)
    if vol_sma > 0 and vol < config.VOLUME_MIN_RATIO * vol_sma:
        return "volume_min_ratio"

    check_start = max(0, candidate_idx - config.LOOKBACK_BARS)
    for i in range(check_start, candidate_idx):
        prev = df.iloc[i]
        prev_vol = float(prev["volume"])
        sma_i = _volume_sma(df, i - 1)
        if sma_i <= 0 or prev_vol < config.VOLUME_SPIKE_MULTIPLIER * sma_i:
            continue
        is_bullish = float(prev["close"]) > float(prev["open"])
        if direction == "short" and is_bullish:
            return "counter_momentum"
        if direction == "long" and not is_bullish:
            return "counter_momentum"

    return "passed"


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
