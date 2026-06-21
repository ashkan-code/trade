"""Trace why Gate 2 OB/FVG search returns None despite a valid sweep.

ROOT CAUSE: validate_rejection requires close to exit the zone edge
  (short: close < zone_low; long: close > zone_high).
  When price is approaching / inside the zone (typical entry scenario),
  the close is NOT yet beyond the edge → strict check rejects every zone.

FIX: If validate_rejection fails, fall back to wick-entered-zone (Grade B).
  This covers: FVG retrace, OB approach, pending-entry bars.
  Single-touch rule (_has_prior_touch) still applies.
  Grade A+ still requires full rejection.

Run with:  cd pivot-engine && python debug_no_ltf_ob.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import numpy as np
import config
from engine.zones import validate_rejection, _find_ltf_ob_after_sweep, _has_prior_touch
from engine.ict import detect_order_blocks, detect_fvg
from contracts import Zone


def _make_df(bars: list[tuple], freq_h: int = 1) -> pd.DataFrame:
    """Build UTC OHLCV DataFrame from (o,h,l,c,v) tuples at freq_h-hour spacing."""
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i, (o, h, l, c, v) in enumerate(bars):
        rows.append({"timestamp": base + pd.Timedelta(hours=i * freq_h),
                     "open": float(o), "high": float(h),
                     "low": float(l), "close": float(c), "volume": float(v)})
    return pd.DataFrame(rows).set_index("timestamp")


SECTION = "=" * 65

# ─── §1  WHY validate_rejection BLOCKS ALL SIGNALS ───────────────────────────
print(SECTION)
print("§1  ROOT CAUSE: validate_rejection too strict")
print(SECTION)

# SHORT zone: high=105, low=100  (price approaches from above)
zone_s = Zone(zone_type="ob", direction="short",
              zone_high=105.0, zone_low=100.0, origin_index=5, timeframe="1h")

# Case 1a: typical approach bar — wick touches zone edge, close still above
row_approach = pd.Series({"open": 102.0, "high": 101.0, "low": 100.5, "close": 100.8})
# h=101 >= zone_low=100 → wick in zone;  c=100.8 NOT < 100.0 → FAILS

# Wait — for short OB [high=105, low=100], price approaches from ABOVE.
# "short" means: price was above zone, enters zone.
# zone_high=105, zone_low=100.
# validate_rejection SHORT: h >= zone_l (100) AND c < zone_l (100)
# So wick entering zone means h >= 100, but close must be < 100 to confirm.

# Scenario: bar whose wick enters zone (h reaches 101, inside 100-105) but close=101 (above zone_low)
row_wick_only = pd.Series({"open": 103.0, "high": 103.5, "low": 100.2, "close": 101.5})
passed, reason = validate_rejection(row_wick_only, zone_s)
print(f"\n  SHORT zone [{zone_s.zone_low}–{zone_s.zone_high}]")
print(f"  Bar: o={row_wick_only['open']}, h={row_wick_only['high']}, "
      f"l={row_wick_only['low']}, c={row_wick_only['close']}")
print(f"  validate_rejection → passed={passed}, reason='{reason}'")
print(f"  WHY: strict check needs c < {zone_s.zone_low} (close must EXIT zone) "
      f"but c={row_wick_only['close']} is still inside")

# Relaxed logic (after fix):
lh = float(row_wick_only["high"])
wick_in = lh >= zone_s.zone_low
print(f"\n  RELAXED check: h({lh}) >= zone_low({zone_s.zone_low}) = {wick_in}")
print(f"  → Grade 'B' accepted (wick-entered zone; pending-entry signal)")

# Case 1b: FVG zone (long), price retracing into gap
print()
zone_fvg = Zone(zone_type="fvg", direction="long",
                zone_high=100.0, zone_low=95.0, origin_index=3, timeframe="30m")
row_retrace = pd.Series({"open": 101.5, "high": 102.0, "low": 99.0, "close": 100.3})
passed_l, reason_l = validate_rejection(row_retrace, zone_fvg)
print(f"  LONG FVG zone [{zone_fvg.zone_low}–{zone_fvg.zone_high}]")
print(f"  Bar: o={row_retrace['open']}, h={row_retrace['high']}, "
      f"l={row_retrace['low']}, c={row_retrace['close']}")
print(f"  validate_rejection → passed={passed_l}, reason='{reason_l}'")
ll = float(row_retrace["low"])
wick_in_l = ll <= zone_fvg.zone_high
print(f"  RELAXED check: l({ll}) <= zone_high({zone_fvg.zone_high}) = {wick_in_l}")
print(f"  → Grade 'B' accepted (wick-entered FVG; price retracing into gap)")


# ─── §2  FULL _find_ltf_ob_after_sweep TRACE ─────────────────────────────────
print(f"\n{SECTION}")
print("§2  Synthetic SHORT scenario: 4H sweep → LTF OB approach")
print(SECTION)

# Build 1H LTF data with proper ICT swing structure:
#   Bars  0-4 : descent (highs drop, lows > 97 to set up swing low)
#   Bar   5   : swing low l=97 (confirmed with 5 bars each side)
#   Bars  6-9 : ascent (lows > 97)
#   Bar  10   : OB candle (bullish, sweep_ts)  → becomes the SHORT OB
#   Bar  11   : bearish impulse, close=93 < swing_low(97) → OB confirmed
#   Bars 12-28: slow recovery (highs stay < zone_low=107 to avoid prior-touch)
#   Bar  29   : last bar, h=108 → wick enters OB zone [107,110]; close=107 (inside)
#               STRICT: c < 107? → 107 NOT < 107 → FAIL
#               RELAXED: h=108 >= 107 → PASS Grade B

V = 1000.0
bars = [
    # 0-4: descent, lows > 97
    (110, 113, 103, 109, V), (109, 112, 102, 108, V),
    (108, 111, 101, 107, V), (107, 110, 100, 106, V), (106, 109, 99, 105, V),
    # 5: swing low l=97
    (104, 108, 97, 99, V),
    # 6-9: ascent, lows > 97
    (99, 102, 98, 101, V), (101, 104, 99, 103, V),
    (103, 106, 100, 105, V), (105, 108, 101, 107, V),
    # 10: OB bullish candle (this is sweep_ts)
    (107, 110, 102, 109, V * 2),
    # 11: bearish impulse closes 93 < swing_low 97
    (109, 111, 92, 93, V * 3),
    # 12-28: slow recovery; keep highs < 107 to avoid prior-touch
    *[(93 + (i-12) * 0.75, 95 + (i-12) * 0.70,
       92 + (i-12) * 0.70, 94 + (i-12) * 0.72, V) for i in range(12, 29)],
    # 29: last bar — wick enters OB zone [107,110], close=107 still inside zone
    (106.0, 108.0, 105.5, 107.0, V * 1.2),
]
df_1h = _make_df(bars, freq_h=1)
sweep_ts = df_1h.index[10]

print(f"\n  Total 1H bars : {len(df_1h)}")
print(f"  Sweep bar ts  : {sweep_ts}  (bar 10)")
print(f"  SWEEP_TO_OB_MAX_BARS = {config.SWEEP_TO_OB_MAX_BARS}  (window = {config.SWEEP_TO_OB_MAX_BARS}h)")
end_ts = sweep_ts + pd.Timedelta(hours=config.SWEEP_TO_OB_MAX_BARS * 1)
print(f"  Search window : bar 10 → bar {min(10 + config.SWEEP_TO_OB_MAX_BARS, len(df_1h)-1)}")

# Inspect what OBs are detected
obs = detect_order_blocks(df_1h, "short", "1h")
fvgs = detect_fvg(df_1h, "short", "1h")
print(f"\n  detect_order_blocks found: {len(obs)} OBs")
for z in obs:
    in_win = sweep_ts <= df_1h.index[z.origin_index] <= end_ts
    print(f"    origin_idx={z.origin_index}, zone=[{z.zone_low:.1f},{z.zone_high:.1f}]  "
          f"in_window={in_win}")
print(f"  detect_fvg found: {len(fvgs)} FVGs")
for z in fvgs:
    print(f"    origin_idx={z.origin_index}, zone=[{z.zone_low:.1f},{z.zone_high:.1f}]")

# Inspect last bar
last_bar = df_1h.iloc[-1]
print(f"\n  Last bar (idx 29): o={last_bar['open']:.1f}, h={last_bar['high']:.1f}, "
      f"l={last_bar['low']:.1f}, c={last_bar['close']:.1f}")

# Run full function
result = _find_ltf_ob_after_sweep(df_1h, sweep_ts, "short", "1h", sweep_tf_hours=1)
if result is not None:
    z, g, entry, shadow = result
    print(f"\n  [RESULT] Zone accepted!")
    print(f"    type={z.zone_type}, zone=[{z.zone_low:.1f},{z.zone_high:.1f}]")
    print(f"    Grade={g}, Entry={entry:.2f}, SL anchor={shadow:.2f}")
else:
    print(f"\n  [RESULT] No zone found — check synthetic data structure")
    # Debug: why did each zone fail?
    n = len(df_1h)
    sweep_pos = int(df_1h.index.searchsorted(sweep_ts, side="left"))
    end_pos = int(min(df_1h.index.searchsorted(end_ts, side="right"), n))
    valid_indices = set(range(sweep_pos, end_pos))
    zones_in_window = [z for z in (obs + fvgs) if z.origin_index in valid_indices]
    print(f"  Zones in window (valid_indices={sorted(valid_indices)[:5]}..): {len(zones_in_window)}")
    last_idx = n - 1
    last_row = df_1h.iloc[last_idx]
    for z in zones_in_window:
        prior = _has_prior_touch(df_1h, z, last_idx)
        passed, reason = validate_rejection(last_row, z)
        lh2 = float(last_row["high"])
        relaxed = lh2 >= z.zone_low
        print(f"    zone [{z.zone_low:.1f},{z.zone_high:.1f}]: prior_touch={prior}, "
              f"strict=({passed},{reason}), relaxed_wick={relaxed}")


# ─── §3  CONFIG CHANGES ───────────────────────────────────────────────────────
print(f"\n{SECTION}")
print("§3  Config changes applied")
print(SECTION)
print(f"  SWEEP_TO_OB_MAX_BARS = {config.SWEEP_TO_OB_MAX_BARS}  "
      f"(increased from 6 → wider search window)")
print(f"  SWEEP_LOOKBACK_4H    = {config.SWEEP_LOOKBACK_4H}")
print(f"  SWEEP_LOOKBACK_1H    = {config.SWEEP_LOOKBACK_1H}")
print()
print("  engine/zones.py _find_ltf_ob_after_sweep:")
print("    BEFORE: strict validate_rejection required (close must exit zone) → 0 signals")
print("    AFTER:  relaxed fallback: wick-entered-zone = Grade B pending entry")
print("    Rule still enforced: single-touch (_has_prior_touch) + zone in time window")
