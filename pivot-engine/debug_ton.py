"""Debug script: investigate TONUSDT SHORT signal + SCORE=0 root causes."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config
from data.fetcher import fetch_ohlcv
from engine.ict import atr_scalar, detect_mss
from engine.zones import (
    find_sweep_rejection, diagnose_sweep_rejection, _volume_sma,
    find_active_zones, find_rejection,
)
from engine.entry import find_micro_entry, zones_overlap
from engine.ict import detect_order_blocks, detect_fvg

SYMBOL = "TONUSDT"
DIRECTION = "short"  # forced by BTC direction

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"FETCHING {SYMBOL} data …")
df_4h = fetch_ohlcv(SYMBOL, "4h", 300)
df_1h = fetch_ohlcv(SYMBOL, "1h", 500)
df_5m = fetch_ohlcv(SYMBOL, "5m", 288)

print(f"4H bars: {len(df_4h) if df_4h is not None else 'NONE'}")
print(f"1H bars: {len(df_1h) if df_1h is not None else 'NONE'}")
print(f"5m bars: {len(df_5m) if df_5m is not None else 'NONE'}")

# ── Bug 1: per-symbol MSS direction ──────────────────────────────────────────
print("\n" + "=" * 70)
print("BUG 1: PER-SYMBOL MSS DIRECTION")
ton_mss = detect_mss(df_4h)
print(f"  TONUSDT 4H MSS direction: {ton_mss}")
print(f"  BTC direction (forced):   {DIRECTION}")
print(f"  Alignment: {'YES' if ton_mss == DIRECTION else 'NO — this symbol should be FILTERED OUT'}")

# ── Bug 2: TONUSDT sweep detection + volume analysis ─────────────────────────
print("\n" + "=" * 70)
print("BUG 2: GATE 2 SWEEP DETECTION DETAILS")

for tf, df in [("4h", df_4h), ("1h", df_1h)]:
    if df is None or df.empty:
        print(f"  {tf}: NO DATA")
        continue

    n = len(df)
    candidate_idx = n - 1
    row = df.iloc[candidate_idx]
    lb_start = max(0, candidate_idx - config.SWEEP_LOOKBACK)

    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    o = float(row["open"])
    vol = float(row["volume"])

    # Find swing level
    swing_arr = df["high"].values[lb_start:candidate_idx]
    swing_level = float(swing_arr.max()) if len(swing_arr) else 0.0
    origin = lb_start + int(swing_arr.argmax())

    swept = h > swing_level
    reclaimed = c < swing_level

    vol_sma = _volume_sma(df, candidate_idx - 1)

    print(f"\n  [{tf}] Last candle @ {df.index[-1]}")
    print(f"    open={o:.4f}  high={h:.4f}  low={l:.4f}  close={c:.4f}")
    print(f"    SWEEP_LOOKBACK={config.SWEEP_LOOKBACK} bars → swing_high={swing_level:.4f} (bar {origin})")
    print(f"    Condition 1 (swept):    high {h:.4f} > {swing_level:.4f} → {swept}")
    print(f"    Condition 2 (reclaim):  close {c:.4f} < {swing_level:.4f} → {reclaimed}")
    print(f"    Condition 1+2 together: {swept and reclaimed}")
    print(f"    vol_candle={vol:,.0f}  SMA9={vol_sma:,.0f}  ratio={vol/vol_sma:.2f}x (need >={config.VOLUME_MIN_RATIO}x)")

    # Condition 4: counter-momentum check
    print(f"\n    COUNTER-MOMENTUM CHECK (last {config.LOOKBACK_BARS} bars):")
    check_start = max(0, candidate_idx - config.LOOKBACK_BARS)
    any_blocked = False
    for i in range(check_start, candidate_idx):
        prev = df.iloc[i]
        pv = float(prev["volume"])
        sma_i = _volume_sma(df, i - 1)
        ratio_i = pv / sma_i if sma_i > 0 else 0.0
        is_bull = float(prev["close"]) > float(prev["open"])
        spike = ratio_i >= config.VOLUME_SPIKE_MULTIPLIER
        blocks = spike and is_bull  # for SHORT: bullish spike blocks
        if blocks:
            any_blocked = True
        print(f"      bar {i}: vol={pv:,.0f} SMA9={sma_i:,.0f} ratio={ratio_i:.2f}x "
              f"{'BULL' if is_bull else 'BEAR'} spike={spike} blocks_short={blocks}")

    if not any_blocked:
        print(f"      → No counter-momentum spike found — condition 4 PASSES")

    # Volume of last 12 bars for context
    print(f"\n    VOLUME CONTEXT last 12 bars:")
    for i in range(max(0, n - 12), n):
        r = df.iloc[i]
        rv = float(r["volume"])
        sma_v = _volume_sma(df, i - 1)
        is_bull = float(r["close"]) > float(r["open"])
        print(f"      [{i:3d}] {df.index[i].strftime('%m-%d %H:%M')} "
              f"{'▲' if is_bull else '▼'} vol={rv:>12,.0f}  SMA9={sma_v:>12,.0f}  "
              f"ratio={rv/sma_v:.2f}x {'<< SPIKE' if rv >= config.VOLUME_SPIKE_MULTIPLIER * sma_v else ''}")

    result = find_sweep_rejection(df, DIRECTION, tf)
    diag = diagnose_sweep_rejection(df, DIRECTION)
    print(f"\n    find_sweep_rejection({tf}): {'PASS → ' + str(result.grade) if result else 'NONE'}")
    print(f"    diagnose_sweep_rejection({tf}): {diag}")

# ── Bug 3: SCORE = 0.0 ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BUG 3: MICRO SCORE = 0.0 INVESTIGATION")

# Get the primary zone from sweep (whichever TF passes)
sweep_4h = find_sweep_rejection(df_4h, DIRECTION, "4h")
sweep_1h = find_sweep_rejection(df_1h, DIRECTION, "1h") if df_1h is not None else None
primary = sweep_4h or sweep_1h

if primary is None:
    print("  Gate 2 returns None → micro never called → score=0 by definition")
    print("  (This is the expected path — score=0 is a symptom of gate2 state)")
else:
    print(f"  Primary zone: {primary.zone.zone_type} {primary.zone.timeframe} "
          f"[{primary.zone.zone_low:.4f} – {primary.zone.zone_high:.4f}]")

    if df_5m is not None and not df_5m.empty:
        atr_5m = atr_scalar(df_5m, config.ATR_5m_PERIOD)
        print(f"  ATR_5m = {atr_5m:.6f}")
        print(f"  DISPLACEMENT_ATR_MIN = {config.DISPLACEMENT_ATR_MIN} → threshold = {config.DISPLACEMENT_ATR_MIN * atr_5m:.6f}")

        # Run OB detection on 5m
        obs_5m = detect_order_blocks(df_5m, DIRECTION, "5m")
        fvgs_5m = detect_fvg(df_5m, DIRECTION, "5m")
        recent_start = len(df_5m) - 96
        print(f"\n  5m OBs detected: {len(obs_5m)} total")
        print(f"  5m FVGs detected: {len(fvgs_5m)} total")
        print(f"  lookback window: bars {recent_start} to {len(df_5m)-1}")

        candidates = [
            ob for ob in obs_5m
            if ob.origin_index >= recent_start and zones_overlap(ob, primary.zone)
        ]
        print(f"  Candidates in zone × recent: {len(candidates)}")
        if obs_5m:
            print(f"\n  All 5m OBs (checking zone overlap & recency):")
            for ob in obs_5m[-10:]:
                in_zone = zones_overlap(ob, primary.zone)
                recent = ob.origin_index >= recent_start
                n = len(df_5m)
                idx = ob.origin_index
                end = min(idx + 4, n)
                post = df_5m["low"].values[idx + 1: end]
                impulse = (ob.zone_low - float(post.min())) if len(post) else 0.0
                impulse_ratio = impulse / atr_5m if atr_5m > 0 else 0.0
                passes_disp = impulse_ratio >= config.DISPLACEMENT_ATR_MIN
                print(f"    bar {ob.origin_index}: zone [{ob.zone_low:.4f}-{ob.zone_high:.4f}] "
                      f"in_htf_zone={in_zone} recent={recent} "
                      f"impulse={impulse:.5f} ({impulse_ratio:.2f}x ATR) disp_pass={passes_disp}")

        micro = find_micro_entry(df_5m, primary.zone, DIRECTION, None, atr_5m, "5m", lookback_bars=96)
        print(f"\n  find_micro_entry result: {micro}")
        if micro:
            print(f"  score={micro.score}  entry={micro.entry:.4f}  shadow={micro.shadow_extreme:.4f}")
        else:
            print("  → micro=None → score=0.0 in live.py")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROOT CAUSE SUMMARY")
print(f"  Bug 1 — all SHORT: direction = btc_direction (line 185 live.py)")
print(f"          No per-symbol MSS check. TONUSDT MSS = {ton_mss} ≠ {DIRECTION}")
print(f"  Bug 2 — TONUSDT false signal: see diagnose above")
print(f"  Bug 3 — SCORE=0: micro=None because no 5m OB inside HTF sweep zone")
print("=" * 70)
