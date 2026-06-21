"""Gate 4: micro entry optimizer — finds the best OB/FVG on LTF within an HTF zone.

Scoring formula (max 12), inspired by LuxAlgo ICT displacement logic:
  displacement_strength × 4   impulse after OB formation (ATR-relative, capped 0–1)
  rejection_quality     × 4   OB candle body/range ratio (capped 0–1)
  has_fvg_confluence    × 2   micro OB price range overlaps a LTF FVG (0 or 1)
  freshness             × 2   OB never touched since formation (0 or 1)

Only OBs where impulse >= config.DISPLACEMENT_ATR_MIN × ATR qualify.
Falls back to HTF zone edge if no qualifying micro OB found.
"""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, MicroEntry, Zone
from engine.ict import atr_scalar, detect_fvg, detect_order_blocks


def zones_overlap(z1: Zone, z2: Zone) -> bool:
    """True if the two zones share any price range."""
    return max(z1.zone_low, z2.zone_low) <= min(z1.zone_high, z2.zone_high)


def _score_micro_ob(
    df: pd.DataFrame,
    ob: Zone,
    fvg_zones: list[Zone],
    atr_val: float,
    direction: Direction,
) -> tuple[float, float]:
    """Score a micro OB. Returns (score 0–12, shadow_extreme for SL)."""
    idx = ob.origin_index
    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values

    # 1. Displacement: impulse in the 1-3 bars after OB formation
    end = min(idx + 4, n)
    if direction == "long":
        post = highs[idx + 1 : end]
        impulse = (float(post.max()) - ob.zone_high) if len(post) else 0.0
    else:
        post = lows[idx + 1 : end]
        impulse = (ob.zone_low - float(post.min())) if len(post) else 0.0
    displacement = min(max(impulse / atr_val, 0.0), 1.0) if atr_val > 0 else 0.0

    # 2. Rejection quality: body/range of the OB candle itself
    ob_range = float(highs[idx] - lows[idx])
    ob_body = abs(float(closes[idx]) - float(opens[idx]))
    rejection_quality = min(ob_body / ob_range, 1.0) if ob_range > 0 else 0.0

    # 3. FVG confluence: any LTF FVG overlaps this OB's price range
    has_fvg = int(any(zones_overlap(ob, fvg) for fvg in fvg_zones))

    # 4. Freshness: OB not touched (mitigated) since formation
    if direction == "long":
        touched = any(float(lows[k]) <= ob.zone_high for k in range(idx + 1, n))
    else:
        touched = any(float(highs[k]) >= ob.zone_low for k in range(idx + 1, n))
    freshness = 0 if touched else 1

    score = displacement * 4 + rejection_quality * 4 + has_fvg * 2 + freshness * 2

    shadow_extreme = float(lows[idx]) if direction == "long" else float(highs[idx])
    return score, shadow_extreme


def find_micro_entry(
    df_ltf: pd.DataFrame,
    htf_zone: Zone,
    direction: Direction,
    as_of_ts: pd.Timestamp,
    atr_ltf: float,
    ltf_name: str = "5m",
    lookback_bars: int = 48,
) -> MicroEntry | None:
    """Find the highest-scoring micro OB inside htf_zone on a lower timeframe.

    Only OBs with impulse >= DISPLACEMENT_ATR_MIN × ATR_LTF qualify.
    Returns None if nothing qualifies; callers fall back to HTF zone edge + HTF shadow.
    """
    view = df_ltf[df_ltf.index <= as_of_ts]
    min_bars = config.OB_SWING_LOOKBACK + config.SWING_LEN + 4
    if len(view) < min_bars:
        return None

    search = view.iloc[-lookback_bars:]
    if len(search) < min_bars:
        return None

    micro_obs = detect_order_blocks(search, direction, ltf_name)
    micro_fvgs = detect_fvg(search, direction, ltf_name)

    # Keep only micro OBs whose price range overlaps the HTF zone
    candidates = [ob for ob in micro_obs if zones_overlap(ob, htf_zone)]
    if not candidates:
        return None

    highs = search["high"].values
    lows = search["low"].values
    n = len(search)

    best: MicroEntry | None = None
    best_score = -1.0

    for ob in candidates:
        idx = ob.origin_index
        end = min(idx + 4, n)

        # Displacement threshold gate
        if direction == "long":
            post = highs[idx + 1 : end]
            impulse = (float(post.max()) - ob.zone_high) if len(post) else 0.0
        else:
            post = lows[idx + 1 : end]
            impulse = (ob.zone_low - float(post.min())) if len(post) else 0.0

        if atr_ltf > 0 and (impulse / atr_ltf) < config.DISPLACEMENT_ATR_MIN:
            continue

        score, shadow = _score_micro_ob(search, ob, micro_fvgs, atr_ltf, direction)
        if score > best_score:
            best_score = score
            entry = ob.zone_high if direction == "long" else ob.zone_low
            best = MicroEntry(
                entry=entry,
                shadow_extreme=shadow,
                score=round(score, 2),
                zone=ob,
            )

    return best


def optimize_entry(
    df_5m: pd.DataFrame,
    zone: Zone,
    direction: Direction,
    as_of_ts: pd.Timestamp,
    lookback_bars: int = 48,
) -> float:
    """Backward-compat wrapper: returns entry price only.

    Prefer find_micro_entry() directly when you also need the refined SL.
    """
    atr_val = atr_scalar(df_5m, config.ATR_5m_PERIOD)
    micro = find_micro_entry(df_5m, zone, direction, as_of_ts, atr_val, "5m", lookback_bars)
    return micro.entry if micro is not None else (
        zone.zone_high if direction == "long" else zone.zone_low
    )
