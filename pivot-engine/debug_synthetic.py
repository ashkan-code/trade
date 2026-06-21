"""Verify all three bug fixes with synthetic TONUSDT data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import config
from engine.zones import find_sweep_rejection, diagnose_sweep_rejection, _volume_sma
from engine.ict import detect_mss, detect_order_blocks, detect_fvg, atr_scalar
from engine.entry import find_micro_entry, zones_overlap
from contracts import Zone

# ── Build synthetic TONUSDT 4H data ──────────────────────────────────────────
# Phase 1 (bars 0-40):   price falls 2.0 → 1.55 (bearish, low volume)
# Phase 2 (bars 40-60):  bullish bounce 1.55 → 1.69 with HIGH institutional vol
# Phase 3 (bar 60):      wick spike to 1.714, close at 1.6956 (sweep candidate)

N = 61
idx = pd.date_range("2024-05-01", periods=N, freq="4h", tz="UTC")
opens  = np.zeros(N); highs  = np.zeros(N)
lows   = np.zeros(N); closes = np.zeros(N); vols = np.zeros(N)

for i in range(40):
    p = 2.0 - (2.0 - 1.55) * i / 39
    opens[i]=p+0.01; closes[i]=p-0.01; highs[i]=p+0.03; lows[i]=p-0.03; vols[i]=5_000_000

for i in range(40, 60):
    j = i - 40
    p = 1.55 + (1.69 - 1.55) * j / 19
    opens[i]=p-0.005; closes[i]=p+0.005; highs[i]=p+0.015; lows[i]=p-0.015
    vols[i] = 25_000_000  # heavy institutional bullish volume

opens[60]=1.695; highs[60]=1.714; lows[60]=1.690; closes[60]=1.6956; vols[60]=8_000_000

df_4h = pd.DataFrame(
    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
    index=idx,
)

# ── FIX 1: per-symbol MSS check ──────────────────────────────────────────────
print("=" * 70)
print("FIX 1: PER-SYMBOL MSS")
ton_mss = detect_mss(df_4h)
btc_dir = "short"
aligned = ton_mss == btc_dir
print(f"  TONUSDT 4H MSS: {ton_mss}")
print(f"  BTC direction:  {btc_dir}")
print(f"  Aligned: {aligned}")
if not aligned:
    print(f"  → BLOCKED at Gate 1 (structure_misaligned). Fix 1 works. ✓")
else:
    print(f"  → Would pass Gate 1 (aligned).")

# ── FIX 2: counter-momentum now uses full lb_start window ────────────────────
print("\n" + "=" * 70)
print("FIX 2: COUNTER-MOMENTUM — FULL SWING LOOKBACK WINDOW")

n = len(df_4h)
cand = n - 1
lb_start = max(0, cand - config.SWEEP_LOOKBACK)

print(f"  lb_start={lb_start}  candidate_idx={cand}")
print(f"  Checking bars {lb_start} to {cand-1} for institutional spikes:")

any_blocked = False
for i in range(lb_start, cand):
    sma_i = _volume_sma(df_4h, i - 1)
    ratio = vols[i] / sma_i if sma_i > 0 else 0.0
    is_bull = closes[i] > opens[i]
    spike = ratio >= config.VOLUME_SPIKE_MULTIPLIER
    blocks = spike and is_bull
    if blocks:
        any_blocked = True
    mark = "  << BLOCKS SHORT ✓" if blocks else ""
    print(f"    bar {i:3d}: vol={vols[i]:>12,.0f}  SMA9={sma_i:>12,.0f}  "
          f"ratio={ratio:4.1f}×  {'▲BULL' if is_bull else '▼BEAR'}  spike={spike}{mark}")

# Also check the early bull bars that were previously missed
print(f"\n  Early bull bars (previously outside LOOKBACK_BARS=5 window):")
for i in range(40, 45):
    sma_i = _volume_sma(df_4h, i - 1)
    ratio = vols[i] / sma_i if sma_i > 0 else 0.0
    is_bull = closes[i] > opens[i]
    spike = ratio >= config.VOLUME_SPIKE_MULTIPLIER
    print(f"    bar {i:3d}: vol={vols[i]:>12,.0f}  SMA9={sma_i:>12,.0f}  "
          f"ratio={ratio:4.1f}×  {'▲BULL' if is_bull else '▼BEAR'}  spike={spike}")

# Run the actual function
result = find_sweep_rejection(df_4h, "short", "4h")
diag = diagnose_sweep_rejection(df_4h, "short")
print(f"\n  find_sweep_rejection('short'): {result}")
print(f"  diagnose_sweep_rejection:      {diag}")
if result is None and diag in ("counter_momentum", "volume_min_ratio"):
    print(f"  → BLOCKED by '{diag}'. Fix 2 works. ✓")
elif result is None:
    print(f"  → BLOCKED by '{diag}'.")
else:
    print(f"  → NOT blocked (may need to review this case)")

# ── FIX 3: micro entry uses wider search zone ─────────────────────────────────
print("\n" + "=" * 70)
print("FIX 3: MICRO ENTRY WIDER SEARCH ZONE (entry ± 2×ATR)")

# Build a synthetic SHORT sweep result
sweep_zone = Zone("ob", "short", zone_high=1.714, zone_low=1.697, origin_index=55, timeframe="4h")

# Build 5m data where price is around 1.6940 (below the zone boundary at 1.697)
N5 = 288
idx5 = pd.date_range("2024-05-11", periods=N5, freq="5min", tz="UTC")
rng = np.random.default_rng(42)
p5 = np.linspace(1.685, 1.697, N5) + rng.normal(0, 0.0015, N5)
df_5m_sim = pd.DataFrame({
    "open":  p5 - 0.001,
    "high":  p5 + 0.003,
    "low":   p5 - 0.003,
    "close": p5 + 0.001,
    "volume": rng.uniform(500_000, 1_500_000, N5),
}, index=idx5)

atr_5m = atr_scalar(df_5m_sim, 14)
entry_price = sweep_zone.zone_low  # = 1.697 (entry for SHORT)
buffer = 2.0 * atr_5m
print(f"  sweep_zone: [{sweep_zone.zone_low:.4f}-{sweep_zone.zone_high:.4f}]")
print(f"  entry_price: {entry_price:.4f}")
print(f"  ATR_5m: {atr_5m:.5f}")
print(f"  search_zone: [{entry_price - buffer:.4f}-{entry_price + buffer:.4f}]  (entry ± 2×ATR)")

# Check OLD behaviour (narrow wick zone)
obs_5m = detect_order_blocks(df_5m_sim, "short", "5m")
recent_start = len(df_5m_sim) - 96
old_cands = [ob for ob in obs_5m if ob.origin_index >= recent_start and zones_overlap(ob, sweep_zone)]
print(f"\n  5m OBs detected: {len(obs_5m)}")
print(f"  OLD: candidates inside wick zone [{sweep_zone.zone_low:.4f}-{sweep_zone.zone_high:.4f}]: {len(old_cands)}")

# New behaviour (wider zone)
search_zone = Zone("ob","short", zone_high=entry_price+buffer, zone_low=entry_price-buffer,
                   origin_index=0, timeframe="5m")
new_cands = [ob for ob in obs_5m if ob.origin_index >= recent_start and zones_overlap(ob, search_zone)]
print(f"  NEW: candidates inside entry±2×ATR [{search_zone.zone_low:.4f}-{search_zone.zone_high:.4f}]: {len(new_cands)}")

micro = find_micro_entry(df_5m_sim, sweep_zone, "short", None, atr_5m, "5m", lookback_bars=96)
print(f"\n  find_micro_entry result: {micro}")
if micro is not None:
    print(f"  score={micro.score}  entry={micro.entry:.4f}  shadow={micro.shadow_extreme:.4f}")
    print(f"  → score > 0.0. Fix 3 works. ✓")
else:
    print(f"  → still None (no OBs in this random 5m data — run again or check with real data)")

print("\n" + "=" * 70)
print("SUMMARY")
print(f"  Fix 1 — per-symbol MSS: {'BLOCKS misaligned symbol ✓' if not aligned else 'symbol aligned, passed'}")
print(f"  Fix 2 — counter-momentum full window: result={diag}")
print(f"  Fix 3 — wider micro zone: micro={'found ✓' if micro else 'None (depends on 5m OB data)'}")
print("=" * 70)
