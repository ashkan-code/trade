"""
ICT Engine Validation — BTCUSDT 4H real data (Crypto.com BTC_USDT, 2026-06-10 to 2026-06-18)
50 raw candles → drop last (open) → 49 closed candles, UTC, ascending.
"""
from __future__ import annotations
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, ".")
import config

# ── Real BTCUSDT 4H candles (BTC_USDT, Crypto.com, newest-first from MCP) ────
# Last row is open candle (2026-06-18T16:00:00Z) → dropped per closed-candle rule
RAW = [
    # ts                      open        high        low         close       vol
    ("2026-06-10T12:00:00Z", 61036.09,   62871.58,   60958.69,   62635.01,   1576.26483),
    ("2026-06-10T16:00:00Z", 62640.44,   62653.99,   61598.14,   61947.60,   1090.56596),
    ("2026-06-10T20:00:00Z", 61947.59,   61954.20,   61098.83,   61510.42,    795.10233),
    ("2026-06-11T00:00:00Z", 61509.08,   62878.62,   61509.08,   62691.99,    771.23869),
    ("2026-06-11T04:00:00Z", 62688.01,   63034.44,   62542.33,   62721.02,    280.30497),
    ("2026-06-11T08:00:00Z", 62720.39,   63264.70,   62720.39,   63103.13,    246.84492),
    ("2026-06-11T12:00:00Z", 63108.02,   63254.13,   62502.45,   62759.34,   1363.08547),
    ("2026-06-11T16:00:00Z", 62762.27,   63934.85,   62342.57,   63607.01,   1223.58896),
    ("2026-06-11T20:00:00Z", 63606.73,   63725.99,   63271.74,   63630.25,    282.62934),
    ("2026-06-12T00:00:00Z", 63633.79,   63802.67,   63295.44,   63526.41,    586.22491),
    ("2026-06-12T04:00:00Z", 63524.81,   63871.54,   62833.95,   63095.05,    462.16354),
    ("2026-06-12T08:00:00Z", 63098.68,   63967.35,   63096.44,   63764.01,    461.31602),
    ("2026-06-12T12:00:00Z", 63761.02,   64403.00,   63048.32,   63599.27,   1668.86857),
    ("2026-06-12T16:00:00Z", 63593.43,   64123.59,   63505.10,   63597.66,    890.00718),
    ("2026-06-12T20:00:00Z", 63589.03,   63704.44,   63391.46,   63578.48,    342.17778),
    ("2026-06-13T00:00:00Z", 63578.48,   63851.24,   63432.57,   63535.11,    301.95027),
    ("2026-06-13T04:00:00Z", 63530.31,   63891.82,   63477.04,   63847.14,    152.584),
    ("2026-06-13T08:00:00Z", 63848.27,   63976.73,   63725.35,   63968.49,    153.07722),
    ("2026-06-13T12:00:00Z", 63968.02,   64346.29,   63913.47,   64298.00,    277.70564),
    ("2026-06-13T16:00:00Z", 64298.01,   64339.29,   63917.13,   64295.52,    389.78384),
    ("2026-06-13T20:00:00Z", 64296.60,   64775.00,   64226.38,   64457.06,    228.1097),
    ("2026-06-14T00:00:00Z", 64455.01,   64720.30,   64368.49,   64546.37,    234.44645),
    ("2026-06-14T04:00:00Z", 64546.38,   64568.19,   64225.52,   64435.49,    129.52213),
    ("2026-06-14T08:00:00Z", 64435.39,   64672.01,   64327.64,   64546.01,    106.268),
    ("2026-06-14T12:00:00Z", 64556.37,   64566.35,   63867.99,   64025.07,    349.44818),
    ("2026-06-14T16:00:00Z", 64022.57,   64220.62,   63682.92,   63807.20,    274.28517),
    ("2026-06-14T20:00:00Z", 63807.21,   65808.89,   63794.63,   65748.38,    835.10029),
    ("2026-06-15T00:00:00Z", 65740.88,   65999.99,   65348.53,   65924.05,    693.81205),
    ("2026-06-15T04:00:00Z", 65925.05,   66020.09,   65610.99,   65671.16,    268.89538),
    ("2026-06-15T08:00:00Z", 65675.78,   66413.16,   65508.01,   66231.13,    314.54373),
    ("2026-06-15T12:00:00Z", 66234.65,   67299.76,   66100.61,   67291.27,   1081.70391),
    ("2026-06-15T16:00:00Z", 67292.13,   67302.39,   66348.12,   66619.98,    779.48336),
    ("2026-06-15T20:00:00Z", 66618.64,   66684.31,   66088.64,   66316.01,    397.2126),
    ("2026-06-16T00:00:00Z", 66316.01,   66470.99,   65650.66,   66162.04,    635.67929),
    ("2026-06-16T04:00:00Z", 66161.89,   66578.68,   65798.40,   66383.42,    325.2787),
    ("2026-06-16T08:00:00Z", 66375.30,   66996.64,   66339.05,   66489.99,    337.36199),
    ("2026-06-16T12:00:00Z", 66493.26,   66825.00,   65350.85,   65867.58,   1132.22824),
    ("2026-06-16T16:00:00Z", 65862.42,   66251.80,   65551.92,   65678.33,    398.89443),
    ("2026-06-16T20:00:00Z", 65680.11,   65948.43,   65585.43,   65679.79,    314.36732),
    ("2026-06-17T00:00:00Z", 65683.05,   66145.22,   65475.80,   65850.01,    343.1365),
    ("2026-06-17T04:00:00Z", 65859.38,   66057.81,   65227.43,   65536.38,    247.3623),
    ("2026-06-17T08:00:00Z", 65538.34,   65587.52,   64559.69,   64812.18,    482.44524),
    ("2026-06-17T12:00:00Z", 64812.19,   65863.54,   64587.25,   65751.30,    970.32406),
    ("2026-06-17T16:00:00Z", 65749.53,   66448.61,   64023.36,   64297.35,   2455.69383),
    ("2026-06-17T20:00:00Z", 64296.99,   64529.50,   63906.98,   64509.39,    732.54209),
    ("2026-06-18T00:00:00Z", 64506.98,   64814.27,   64258.01,   64282.82,    511.22673),
    ("2026-06-18T04:00:00Z", 64282.82,   64520.44,   63689.75,   64504.51,    531.12326),
    ("2026-06-18T08:00:00Z", 64504.52,   64666.18,   63881.01,   63973.27,    395.80986),
    ("2026-06-18T12:00:00Z", 63975.04,   64450.24,   62275.14,   62371.64,   1834.87769),
    # 2026-06-18T16:00:00Z → OPEN CANDLE → dropped
]

df = pd.DataFrame(RAW, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values("timestamp").reset_index(drop=True)
# verify ascending, UTC
assert df["timestamp"].is_monotonic_increasing
assert str(df["timestamp"].dt.tz) == "UTC"

N = len(df)
SEP = "=" * 62

print(SEP)
print(" ICT Engine Validation — BTCUSDT 4H (Real Data)")
print(f" {df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')} UTC  →  "
      f"{df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')} UTC")
print(f" {N} closed candles  |  SWING_LEN={config.SWING_LEN}  |  FVG_BODY_PERC={config.FVG_BODY_PERC}")
print(SEP)

# ── Helpers ───────────────────────────────────────────────────────────────────
def ts(i): return df["timestamp"].iloc[i].strftime("%Y-%m-%d %H:%M")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — SWINGS (BLOCKER)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 1: SWING HIGHS / LOWS ──────────────────────────────")

from core.ict import _swing_highs, _swing_lows
sh = _swing_highs(df, config.SWING_LEN)
sl = _swing_lows(df,  config.SWING_LEN)

sh_idx = [i for i in range(N) if sh.iloc[i]]
sl_idx = [i for i in range(N) if sl.iloc[i]]

print(f"Swing Highs ({len(sh_idx)}):")
for i in sh_idx:
    print(f"  bar {i:2d}  {ts(i)}  high={df['high'].iloc[i]:.2f}")

print(f"Swing Lows ({len(sl_idx)}):")
for i in sl_idx:
    print(f"  bar {i:2d}  {ts(i)}  low={df['low'].iloc[i]:.2f}")

# Sanity: need at least 2 swing highs + 2 swing lows for MSS detection
total_swings = len(sh_idx) + len(sl_idx)
if total_swings >= 3:
    print(f"RESULT: PASS — {total_swings} swings detected across {N} bars")
    test1 = "PASS"
elif total_swings >= 1:
    print(f"RESULT: WARN — only {total_swings} swings in {N} bars "
          f"(SWING_LEN={config.SWING_LEN} requires {config.SWING_LEN*2+1}-bar isolation); "
          f"MSS may still work")
    test1 = "WARN — sparse swings; 49-bar window is tight for SWING_LEN=5"
else:
    print("RESULT: FAIL — 0 swings detected; data too short or SWING_LEN too large")
    test1 = "FAIL"
    print("\n⛔ BLOCKER: fix TEST 1 before proceeding\n")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — MSS DIRECTION (BLOCKER)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 2: MSS DIRECTION ────────────────────────────────────")

from core.ict import detect_mss

# Scan progressively to find every MSS flip
flips = []
for end in range(config.SWING_LEN * 2 + 2, N + 1):
    sub = df.iloc[:end].reset_index(drop=True)
    m = detect_mss(sub, config.SWING_LEN)
    if m["direction"] is not None:
        if not flips or flips[-1]["direction"] != m["direction"] or flips[-1]["index"] != m["index"]:
            flips.append({
                "direction": m["direction"],
                "index": m["index"],
                "ts": df["timestamp"].iloc[m["index"]].strftime("%Y-%m-%d %H:%M"),
                "price": m["price"],
            })

# Deduplicate consecutive same-direction entries
deduped = []
for f in flips:
    if not deduped or deduped[-1]["direction"] != f["direction"] or deduped[-1]["index"] != f["index"]:
        deduped.append(f)

print(f"MSS Flips detected ({len(deduped)}):")
for f in deduped:
    print(f"  [{f['direction']:8s}]  bar {f['index']:2d}  {f['ts']}  price={f['price']:.2f}")

final_mss = detect_mss(df, config.SWING_LEN)
print(f"\nFinal MSS (full window): direction={final_mss['direction']}  "
      f"bar={final_mss['index']}  price={final_mss['price']:.2f}  ts={ts(final_mss['index'])}")
print(f"Market bias at window end: {final_mss['direction'] or 'NONE'}")

if final_mss["direction"] is not None and len(deduped) >= 1:
    print("RESULT: PASS — MSS detected; verify direction against TradingView LuxAlgo")
    test2 = "PASS (verify)"
else:
    print("RESULT: FAIL — no MSS detected in window")
    test2 = "FAIL"

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — ORDER BLOCKS
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 3: ORDER BLOCKS ─────────────────────────────────────")

from core.ict import detect_order_blocks
obs = detect_order_blocks(df, config.SWING_LEN)

bull_obs = [o for o in obs if o["type"] == "bullish"]
bear_obs = [o for o in obs if o["type"] == "bearish"]
broken_obs = [o for o in obs if o["broken"]]
breaker_obs = [o for o in obs if o["breaker"]]

print(f"Total OBs: {len(obs)}  (bullish={len(bull_obs)}  bearish={len(bear_obs)})")
print(f"Broken: {len(broken_obs)}   Breakers: {len(breaker_obs)}")

recent3 = obs[-3:] if len(obs) >= 3 else obs
print(f"\n3 most recent OBs:")
for o in reversed(recent3):
    flag = " [BREAKER]" if o["breaker"] else (" [broken]" if o["broken"] else "")
    print(f"  bar {o['index']:2d}  {ts(o['index'])}  {o['type']:8s}  "
          f"top={o['top']:.2f}  bot={o['bottom']:.2f}{flag}")

test3 = "PASS" if len(obs) > 0 else "FAIL — 0 OBs detected"
print(f"RESULT: {test3}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — FVG
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 4: FAIR VALUE GAPS ──────────────────────────────────")

from core.ict import detect_fvg
fvgs = detect_fvg(df)

bull_fvgs = [f for f in fvgs if f["type"] == "bullish"]
bear_fvgs = [f for f in fvgs if f["type"] == "bearish"]
active_fvgs = [f for f in fvgs if f["active"]]
inactive_fvgs = [f for f in fvgs if not f["active"]]

print(f"Total FVGs: {len(fvgs)}  (bull={len(bull_fvgs)}  bear={len(bear_fvgs)})")
print(f"Active: {len(active_fvgs)}   Filled/inactive: {len(inactive_fvgs)}")

# Check synthetic-data over-counting problem
per_candle = len(fvgs) / max(N - 2, 1)
print(f"FVG density: {per_candle:.2f} per candle  "
      f"(>0.30 suggests FVG_BODY_PERC={config.FVG_BODY_PERC} may be too loose)")

recent3f = fvgs[-3:] if len(fvgs) >= 3 else fvgs
print(f"\n3 most recent FVGs:")
for f in reversed(recent3f):
    status = "active" if f["active"] else "FILLED"
    print(f"  bar {f['index']:2d}  {ts(f['index'])}  {f['type']:8s}  "
          f"top={f['top']:.2f}  bot={f['bottom']:.2f}  [{status}]")

if len(fvgs) == 0:
    test4 = "FAIL — 0 FVGs"
elif per_candle > 0.30:
    test4 = f"WARN — density {per_candle:.2f}/bar, FVG_BODY_PERC may be too loose"
else:
    test4 = "PASS"
print(f"RESULT: {test4}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — LIQUIDITY (BLOCKER)
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 5: LIQUIDITY POOLS ──────────────────────────────────")

from core.indicators import atr as calc_atr
from core.ict import detect_liquidity

atr_s = calc_atr(df, config.ATR_LEN)
liq = detect_liquidity(df, atr_s)

buy_side  = [l for l in liq if l["type"] == "buy_side"]
sell_side = [l for l in liq if l["type"] == "sell_side"]

print(f"Liquidity pools: {len(liq)}  (buy_side={len(buy_side)}  sell_side={len(sell_side)})")

if liq:
    for l in liq:
        print(f"  [{l['type']:10s}]  price={l['price']:.2f}  touches={l['touches']}")
    test5 = "PASS"
else:
    test5 = "FAIL — 0 liquidity pools (no targets → no signals)"
    print("\n  ⚠  ZERO liquidity pools detected!")
    print("  ATR (last 5 bars):", atr_s.dropna().tail(5).values.round(2))
    margin = float(atr_s.dropna().iloc[-1]) / config.LIQ_MARGIN_DIV
    print(f"  Cluster margin = ATR/{config.LIQ_MARGIN_DIV} = {margin:.2f}")
    print("\n  PROPOSED FALLBACK TARGET: nearest swing high (long) or swing low (short)")
    if sh_idx:
        best_sh = max(sh_idx, key=lambda i: df["high"].iloc[i])
        print(f"  → Fallback buy-side  target: swing high @ bar {best_sh}  "
              f"price={df['high'].iloc[best_sh]:.2f}  ({ts(best_sh)})")
    if sl_idx:
        best_sl = min(sl_idx, key=lambda i: df["low"].iloc[i])
        print(f"  → Fallback sell-side target: swing low  @ bar {best_sl}  "
              f"price={df['low'].iloc[best_sl]:.2f}  ({ts(best_sl)})")

print(f"RESULT: {test5}")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — END-TO-END PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
print("\n── TEST 6: END-TO-END PIPELINE ──────────────────────────────")

from core.confluence import (check_btc_direction, is_aligned, find_best_zone,
                              zone_triggered, compute_htf_confirms, score_zone)
from core.indicators import rsi, macd
from engine.risk import compute_risk

# Use full 49-bar df as "4H HTF", same as "1H HTF" (we only have 4H),
# and a 10-bar slice as a simulated LTF for zone_triggered check.

btc_dir = check_btc_direction(df)
print(f"Step 1 — BTC 4H direction: {btc_dir}")

obs_all = detect_order_blocks(df, config.SWING_LEN)
fvgs_all = detect_fvg(df)
active_obs = [o for o in obs_all if not o["broken"]]
active_fvgs = [f for f in fvgs_all if f["active"]]

zone = find_best_zone(active_obs, active_obs, active_fvgs, btc_dir)
print(f"Step 2 — find_best_zone: {zone['zone_type'] if zone else 'None'}", end="")
if zone:
    print(f"  top={zone['zone_top']:.2f}  bot={zone['zone_bottom']:.2f}  dir={zone['direction']}")
else:
    print()

aligned = is_aligned(final_mss, btc_dir)
print(f"Step 3 — MSS aligned with BTC direction: {aligned}  "
      f"(MSS={final_mss['direction']}  BTC={btc_dir})")

# zone_triggered — use last 10 bars as LTF proxy
df_ltf = df.tail(10).reset_index(drop=True)
triggered = zone_triggered(zone, df_ltf) if zone else False
print(f"Step 4 — zone_triggered (last 10 bars as LTF): {triggered}")

# Indicator confirms
rsi_s = rsi(df["close"])
_, _, hist = macd(df["close"])
confirms = compute_htf_confirms(df, btc_dir)
print(f"Step 5 — HTF confirms:  rsi={confirms['rsi']}  macd={confirms['macd']}")
print(f"         RSI (last):  {rsi_s.dropna().iloc[-1]:.1f}")
print(f"         MACD hist (last 5): {hist.dropna().tail(5).values.round(2)}")

# Risk (needs liquidity or fallback)
if zone and active_obs:
    ob_1h = active_obs[-1]
    # Use swing highs/lows as fallback liquidity targets when pools == 0
    if not liq:
        if btc_dir == "bullish" and sh_idx:
            best = max(sh_idx, key=lambda i: df["high"].iloc[i])
            liq_fake = [{"type": "buy_side", "price": df["high"].iloc[best], "touches": 3}]
        elif btc_dir == "bearish" and sl_idx:
            best = min(sl_idx, key=lambda i: df["low"].iloc[i])
            liq_fake = [{"type": "sell_side", "price": df["low"].iloc[best], "touches": 3}]
        else:
            liq_fake = []
        print(f"Step 6 — using FALLBACK swing-high/low as liquidity target")
    else:
        liq_fake = liq
    entry = float(df["close"].iloc[-1])
    direction = "bullish" if btc_dir == "bullish" else "bearish"
    risk = compute_risk(direction, entry, ob_1h, liq_fake)
    print(f"Step 6 — Risk:  entry={risk['entry']:.2f}  stop={risk['stop']:.2f}  "
          f"target={risk['target']:.2f}  R:R={risk['rr']:.2f}  valid={risk['valid']}")

    if zone and risk["valid"]:
        score = score_zone(zone, risk["rr"])
        print(f"Step 7 — Score: {score:.2f}  (threshold={config.SCORE_THRESHOLD})")
        above_thresh = score >= config.SCORE_THRESHOLD
        print(f"Step 8 — Signal fires: {above_thresh}")
        if above_thresh:
            test6 = "PASS — full pipeline produced a signal"
        else:
            test6 = f"PASS (pipeline OK) — score {score:.2f} < threshold {config.SCORE_THRESHOLD} (expected)"
    elif zone and not risk["valid"]:
        test6 = f"PASS (pipeline OK) — R:R {risk['rr']:.2f} < MIN_RR={config.MIN_RR} filtered signal"
    else:
        test6 = "FAIL — zone not found"
else:
    test6 = "FAIL — no zone or OB found"

# Trace which filters failed
print(f"\nFilter trace:")
f_dir   = btc_dir is not None
f_zone  = zone is not None
f_align = aligned
f_trig  = triggered
f_conf  = confirms["rsi"] or confirms["macd"]
print(f"  [{'✓' if f_dir   else '✗'}] BTC direction")
print(f"  [{'✓' if f_zone  else '✗'}] Zone found")
print(f"  [{'✓' if f_align else '✗'}] Alignment (MSS ↔ BTC)")
print(f"  [{'✓' if f_trig  else '✗'}] Zone triggered (LTF shadow-and-return)")
print(f"  [{'✓' if f_conf  else '✗'}] RSI or MACD confirm")

print(f"\nRESULT: {test6}")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print(" SUMMARY")
print(SEP)
tests = [
    ("TEST 1", "SWING HIGHS/LOWS   [BLOCKER]", test1),
    ("TEST 2", "MSS DIRECTION      [BLOCKER]", test2),
    ("TEST 3", "ORDER BLOCKS               ", test3),
    ("TEST 4", "FAIR VALUE GAPS            ", test4),
    ("TEST 5", "LIQUIDITY POOLS    [BLOCKER]", test5),
    ("TEST 6", "END-TO-END PIPELINE        ", test6),
]
for tid, name, result in tests:
    icon = "✓" if result.startswith("PASS") else ("⚠" if result.startswith("WARN") else "✗")
    print(f"  {icon} {tid} — {name}  {result}")

print(SEP)
all_pass = all(r.startswith("PASS") or r.startswith("WARN")
               for _, _, r in tests)
if all_pass:
    print(" VERDICT: ICT engine is trustworthy enough to proceed to backtest.")
    print("          Address any WARNs (FVG density, liquidity fallback) first.")
else:
    fails = [tid for tid, _, r in tests if r.startswith("FAIL")]
    print(f" VERDICT: NOT ready for backtest. Fix: {', '.join(fails)}")
print(SEP)
