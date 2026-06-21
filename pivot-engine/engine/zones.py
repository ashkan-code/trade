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

    # 4: no counter-institutional spike in the full swing lookback window.
    # Deliberately uses lb_start (not candidate_idx - LOOKBACK_BARS) so the
    # entire period where the swing level was forming is covered. A 5-bar window
    # is too narrow: a 20-bar bullish rally elevates SMA9 so that individual
    # bars stop registering as spikes by the time they enter a short lookback.
    for i in range(lb_start, candidate_idx):
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

    check_start = lb_start  # full swing window, same logic as find_sweep_rejection
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


def _find_recent_sweep(
    df: pd.DataFrame,
    direction: Direction,
    lookback: int,
    timeframe: str,
) -> "tuple[int, float, Zone, str, float, float] | None":
    """Find the most recent valid sweep bar in the last `lookback` bars.

    Checks: sweep+reclaim, volume >= VOLUME_MIN_RATIO×SMA9,
    no counter-momentum spike in LOOKBACK_BARS before the sweep bar.
    Returns (bar_idx, swept_level, zone, grade, entry, shadow_extreme) or None.
    Works for any timeframe — pass SWEEP_LOOKBACK_4H or SWEEP_LOOKBACK_1H accordingly.
    """
    n = len(df)
    if n < lookback + config.LOOKBACK_BARS + 5:
        return None

    stop = max(n - 1 - lookback, lookback + config.LOOKBACK_BARS)
    for bar_idx in range(n - 1, stop - 1, -1):
        lb_start = max(0, bar_idx - lookback)
        row = df.iloc[bar_idx]
        h = float(row["high"]); l = float(row["low"])
        c = float(row["close"]); o = float(row["open"])
        vol = float(row["volume"])

        # 1 & 2: sweep + reclaim
        if direction == "short":
            swing_arr = df["high"].values[lb_start:bar_idx]
            if not len(swing_arr): continue
            swing_level = float(swing_arr.max())
            origin_int = lb_start + int(swing_arr.argmax())
            if not (h > swing_level and c < swing_level): continue
        else:
            swing_arr = df["low"].values[lb_start:bar_idx]
            if not len(swing_arr): continue
            swing_level = float(swing_arr.min())
            origin_int = lb_start + int(swing_arr.argmin())
            if not (l < swing_level and c > swing_level): continue

        # 3: sweep bar volume
        vol_sma = _volume_sma(df, bar_idx - 1)
        if vol_sma > 0 and vol < config.VOLUME_MIN_RATIO * vol_sma:
            continue

        # 4: counter-momentum in LOOKBACK_BARS before THIS bar
        cm_start = max(0, bar_idx - config.LOOKBACK_BARS)
        blocked = False
        for i in range(cm_start, bar_idx):
            prev = df.iloc[i]
            pv = float(prev["volume"])
            sma_i = _volume_sma(df, i - 1)
            if sma_i <= 0 or pv < config.VOLUME_SPIKE_MULTIPLIER * sma_i:
                continue
            is_bull = float(prev["close"]) > float(prev["open"])
            if direction == "short" and is_bull:
                blocked = True; break
            if direction == "long" and not is_bull:
                blocked = True; break
        if blocked:
            continue

        # Valid sweep — build zone and grade
        body = abs(c - o)
        if direction == "short":
            wick = h - max(o, c)
            zone = Zone("ob", "short", zone_high=h, zone_low=swing_level,
                        origin_index=origin_int, timeframe=timeframe)
            entry = swing_level; shadow_extreme = h
        else:
            wick = min(o, c) - l
            zone = Zone("ob", "long", zone_high=swing_level, zone_low=l,
                        origin_index=origin_int, timeframe=timeframe)
            entry = swing_level; shadow_extreme = l

        grade: str = "A+" if body > 0 and wick >= body else "B"
        return bar_idx, swing_level, zone, grade, entry, shadow_extreme

    return None


def _find_recent_4h_sweep(
    df_4h: pd.DataFrame,
    direction: Direction,
) -> "tuple[int, float, Zone, str, float, float] | None":
    """Backward-compat wrapper: find recent 4H sweep using SWEEP_LOOKBACK_4H."""
    return _find_recent_sweep(df_4h, direction, config.SWEEP_LOOKBACK_4H, "4h")


def _find_ltf_ob_after_sweep(
    df_ltf: pd.DataFrame,
    sweep_ts: "pd.Timestamp",
    direction: Direction,
    timeframe: str,
    sweep_tf_hours: int = 4,
) -> "tuple[Zone, str, float, float] | None":
    """Find a valid OB/FVG on LTF formed after sweep_ts.

    Time window: sweep_ts .. sweep_ts + SWEEP_TO_OB_MAX_BARS × sweep_tf_hours.
    For a 4H sweep pass sweep_tf_hours=4 (window=24h);
    for a 1H sweep pass sweep_tf_hours=1 (window=6h).
    Last bar of df_ltf must be rejecting from the found zone.
    Returns (zone, grade, entry, shadow_extreme) or None.
    """
    if df_ltf is None or df_ltf.empty:
        return None

    n = len(df_ltf)
    end_ts = sweep_ts + pd.Timedelta(hours=config.SWEEP_TO_OB_MAX_BARS * sweep_tf_hours)

    # Convert timestamps to integer indices
    sweep_pos = int(df_ltf.index.searchsorted(sweep_ts, side="left"))
    end_pos = int(min(df_ltf.index.searchsorted(end_ts, side="right"), n))

    if sweep_pos >= n:
        return None  # sweep is after all available LTF data

    valid_indices = set(range(sweep_pos, end_pos))
    if not valid_indices:
        return None

    # Detect on full history (need swing context)
    obs = detect_order_blocks(df_ltf, direction, timeframe)
    fvgs = detect_fvg(df_ltf, direction, timeframe)

    # Filter to zones formed inside the valid time window
    zones = [z for z in (obs + fvgs) if z.origin_index in valid_indices]
    if not zones:
        return None

    zones.sort(key=lambda z: z.origin_index, reverse=True)

    last_idx = n - 1
    last_row = df_ltf.iloc[last_idx]
    best_zone: Zone | None = None
    best_grade: str | None = None
    best_entry: float | None = None
    best_shadow: float | None = None

    for zone in zones:
        if _has_prior_touch(df_ltf, zone, last_idx):
            continue
        passed, grade_or_reason = validate_rejection(last_row, zone)
        if passed:
            grade = grade_or_reason
        else:
            # Relaxed fallback: accept if last bar's wick has entered the zone.
            # This covers FVGs and OBs where price is approaching/touching but
            # the close hasn't yet exited the zone edge (pending-entry signal).
            lh = float(last_row["high"])
            ll = float(last_row["low"])
            if direction == "short" and lh >= zone.zone_low:
                grade = "B"
            elif direction == "long" and ll <= zone.zone_high:
                grade = "B"
            else:
                continue
        if best_zone is None or (grade == "A+" and best_grade != "A+"):
            best_zone = zone
            best_grade = grade
            if direction == "long":
                best_entry = zone.zone_high
                best_shadow = float(last_row["low"])
            else:
                best_entry = zone.zone_low
                best_shadow = float(last_row["high"])

    if best_zone is None:
        return None

    return best_zone, best_grade, best_entry, best_shadow  # type: ignore[return-value]


def find_gate2_signal(
    df_4h: pd.DataFrame,
    df_1h: "pd.DataFrame | None",
    df_30m: "pd.DataFrame | None",
    df_15m: "pd.DataFrame | None",
    direction: Direction,
) -> "Gate2Result | None":
    """Hierarchical Gate 2: 4H sweep first, fallback to 1H sweep.

    Step 1: Look for 4H sweep (last SWEEP_LOOKBACK_4H bars).
      → If found: search 1H/30m/15m for OBs formed within SWEEP_TO_OB_MAX_BARS×4h.
        Stars: 4H base +1, +1 per LTF confirmed (min ★★, max ★★★).
    Step 2: If no 4H sweep: look for 1H sweep (last SWEEP_LOOKBACK_1H bars).
      → If found: search 30m/15m for OBs formed within SWEEP_TO_OB_MAX_BARS×1h.
        Stars: 1 per LTF confirmed (max ★★).
    Returns None if neither sweep found or no LTF OB confirms.
    """
    from contracts import Gate2Result

    # ── Step 1: 4H sweep ────────────────────────────────────────────────────────
    sweep4 = _find_recent_sweep(df_4h, direction, config.SWEEP_LOOKBACK_4H, "4h")
    if sweep4 is not None:
        bar_idx, swept_level, sweep_zone, _, _, _ = sweep4
        sweep_ts = df_4h.index[bar_idx]
        confirmed: dict[str, tuple] = {}
        for tf, df_ltf in [("1h", df_1h), ("30m", df_30m), ("15m", df_15m)]:
            if df_ltf is None or df_ltf.empty:
                continue
            res = _find_ltf_ob_after_sweep(df_ltf, sweep_ts, direction, tf, sweep_tf_hours=4)
            if res is not None:
                confirmed[tf] = res
        if confirmed:
            primary_tf = next(tf for tf in ["1h", "30m", "15m"] if tf in confirmed)
            p_zone, p_grade, p_entry, p_shadow = confirmed[primary_tf]
            # 4H sweep starts at 2★ for 1 LTF, caps at 3★
            stars = min(1 + len(confirmed), 3)
            return Gate2Result(
                sweep_bar_idx=bar_idx, swept_level=swept_level, sweep_zone=sweep_zone,
                primary_zone=p_zone, primary_tf=primary_tf,
                entry=p_entry, shadow_extreme=p_shadow, grade=p_grade,
                ltf_confirmed=list(confirmed.keys()), confluence_stars=stars,
                sweep_tf="4h",
            )

    # ── Step 2: 1H sweep fallback ────────────────────────────────────────────────
    if df_1h is None or df_1h.empty:
        return None

    sweep1 = _find_recent_sweep(df_1h, direction, config.SWEEP_LOOKBACK_1H, "1h")
    if sweep1 is None:
        return None

    bar_idx1, swept_level1, sweep_zone1, _, _, _ = sweep1
    sweep_ts1 = df_1h.index[bar_idx1]
    confirmed1: dict[str, tuple] = {}
    for tf, df_ltf in [("30m", df_30m), ("15m", df_15m)]:
        if df_ltf is None or df_ltf.empty:
            continue
        res = _find_ltf_ob_after_sweep(df_ltf, sweep_ts1, direction, tf, sweep_tf_hours=1)
        if res is not None:
            confirmed1[tf] = res

    if not confirmed1:
        return None

    primary_tf1 = next(tf for tf in ["30m", "15m"] if tf in confirmed1)
    p_zone1, p_grade1, p_entry1, p_shadow1 = confirmed1[primary_tf1]
    # 1H sweep: 1★ per LTF confirmed (max ★★ — always below 4H sweep signals)
    stars1 = min(len(confirmed1), 2)
    return Gate2Result(
        sweep_bar_idx=bar_idx1, swept_level=swept_level1, sweep_zone=sweep_zone1,
        primary_zone=p_zone1, primary_tf=primary_tf1,
        entry=p_entry1, shadow_extreme=p_shadow1, grade=p_grade1,
        ltf_confirmed=list(confirmed1.keys()), confluence_stars=stars1,
        sweep_tf="1h",
    )


def diagnose_gate2_signal(
    df_4h: pd.DataFrame,
    df_1h: "pd.DataFrame | None",
    df_30m: "pd.DataFrame | None",
    df_15m: "pd.DataFrame | None",
    direction: Direction,
) -> str:
    """Return primary reason find_gate2_signal returned None.

    Possible return values:
      "no_4h_sweep"               — 4H last bar has no sweep+reclaim
      "4h_sweep_low_volume"       — 4H sweep blocked by low volume
      "4h_sweep_counter_momentum" — 4H sweep blocked by counter-momentum
      "no_ob_after_4h_sweep"      — 4H sweep found but no LTF OB/FVG after it
      "no_ob_after_1h_sweep"      — 1H sweep found (4H failed) but no 30m/15m OB
      "insufficient_data"         — not enough 4H bars
    """
    # Check 4H sweep
    sweep4 = _find_recent_sweep(df_4h, direction, config.SWEEP_LOOKBACK_4H, "4h")
    if sweep4 is not None:
        sweep_ts = df_4h.index[sweep4[0]]
        for tf, df_ltf in [("1h", df_1h), ("30m", df_30m), ("15m", df_15m)]:
            if df_ltf is None or df_ltf.empty:
                continue
            if _find_ltf_ob_after_sweep(df_ltf, sweep_ts, direction, tf, sweep_tf_hours=4) is not None:
                return "passed"
        return "no_ob_after_4h_sweep"

    # Sub-diagnose why 4H sweep failed (check last bar only for a quick hint)
    n4 = len(df_4h)
    if n4 < config.SWEEP_LOOKBACK_4H + 10:
        return "insufficient_data"

    bar_idx = n4 - 1
    lb_start = max(0, bar_idx - config.SWEEP_LOOKBACK_4H)
    row4 = df_4h.iloc[bar_idx]
    h4 = float(row4["high"]); l4 = float(row4["low"])
    c4 = float(row4["close"]); vol4 = float(row4["volume"])
    if direction == "short":
        arr = df_4h["high"].values[lb_start:bar_idx]
        sl4 = float(arr.max()) if len(arr) else 0.0
        swept4 = h4 > sl4 and c4 < sl4
    else:
        arr = df_4h["low"].values[lb_start:bar_idx]
        sl4 = float(arr.min()) if len(arr) else float("inf")
        swept4 = l4 < sl4 and c4 > sl4

    if not swept4:
        four_h_reason = "no_4h_sweep"
    else:
        vsma = _volume_sma(df_4h, bar_idx - 1)
        four_h_reason = (
            "4h_sweep_low_volume"
            if vsma > 0 and vol4 < config.VOLUME_MIN_RATIO * vsma
            else "4h_sweep_counter_momentum"
        )

    # Try 1H sweep fallback — if it exists but no OB, report that
    if df_1h is not None and not df_1h.empty:
        sweep1 = _find_recent_sweep(df_1h, direction, config.SWEEP_LOOKBACK_1H, "1h")
        if sweep1 is not None:
            sweep_ts1 = df_1h.index[sweep1[0]]
            for tf, df_ltf in [("30m", df_30m), ("15m", df_15m)]:
                if df_ltf is None or df_ltf.empty:
                    continue
                if _find_ltf_ob_after_sweep(df_ltf, sweep_ts1, direction, tf, sweep_tf_hours=1) is not None:
                    return "passed"
            return "no_ob_after_1h_sweep"

    return four_h_reason


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
