"""Live scanner — scans USDT-M futures from Bitunix in real time.

Two-phase hierarchical scan (cuts API requests ~50%):
  Phase 1  (all symbols): fetch 4H+1H only → Gate 1 (MSS) + sweep check.
           Symbols WITHOUT a valid sweep are dropped here — no 30m/15m/5m fetch.
  Phase 2  (swept symbols only): fetch 30m+15m+5m → full Gates 2-4.

Example: 400 symbols × 5 TFs = 2000 fetch calls → now 400×2 + 40×3 = 920 calls.

Gates:
  Gate 0: BTC 4H MSS direction (fetched once, module-level cached 5 min)
  Gate 1: per-symbol 4H MSS must align with BTC direction (Phase 1)
  Gate 2: Hierarchical sweep — 4H sweep first, fallback to 1H sweep
          4H found → search 1H/30m/15m for OB/FVG  (max ★★★)
          1H found → search 30m/15m for OB/FVG      (max ★★)
  Gate 3: RSI + MACD advisory (does NOT block, only logged)
  Gate 4: 5m micro OB/FVG refinement — scored 0–12

ATR star scoring (Phase 2, NOT a gate — only re-ranks, never removes a signal):
  body > ATR_STRONG_MULT × ATR4H → confluence_stars +1 (strong momentum bar)
  body < ATR_WEAK_MULT   × ATR4H → confluence_stars −1 (weak / inside bar)

Usage:
    python live.py                # scan symbols >= MIN_VOLUME_USD (600k)
    python live.py --top 30       # top 30 by volume
    python live.py --min-vol 2m   # custom floor e.g. 2 million USD
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from contracts import Direction, MicroEntry
from data.fetcher import fetch_ohlcv, get_all_symbols, get_top_symbols
from engine.entry import find_micro_entry
from engine.ict import atr_scalar, detect_mss
from engine.indicators import gate3_passes, rsi as _rsi
from engine.sl_tp import compute_rr, compute_sl, find_tp
from engine.zones import (
    _find_recent_sweep,
    diagnose_gate2_signal,
    find_gate2_signal,
)

_BASE = os.path.dirname(os.path.abspath(__file__))
_SIGNALS_FILE = os.path.join(_BASE, "logs", "live_signals.json")
_ERROR_LOG = os.path.join(_BASE, "logs", "errors.log")

logging.basicConfig(
    filename=_ERROR_LOG,
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)

_CANDLES: dict[str, int] = {
    "4h":  300,   # ~50 days
    "1h":  500,   # ~21 days
    "30m": 240,   # ~5 days
    "15m": 192,   # ~2 days
    "5m":  288,   # ~1 day
}

_print_lock = threading.Lock()

# Gate 2 diagnostic counters (protected by _print_lock)
_gate2_stats: dict[str, int] = {}

# ── BTC module-level cache (item 3) ──────────────────────────────────────────
# Reused across repeated scan() calls within _BTC_CACHE_TTL seconds.
_btc_cache: dict = {}      # keys: "df" (DataFrame), "ts" (float epoch)
_BTC_CACHE_TTL: int = 300  # 5 minutes


def _fetch_btc_cached() -> "pd.DataFrame | None":
    """Return BTC 4H DataFrame from cache if < _BTC_CACHE_TTL s old, else fetch fresh."""
    now = time.time()
    if (
        _btc_cache.get("df") is not None
        and now - _btc_cache.get("ts", 0) < _BTC_CACHE_TTL
    ):
        age = int(now - _btc_cache["ts"])
        print(f"      BTC 4H data reused from cache (age {age}s)")
        return _btc_cache["df"]
    df = fetch_ohlcv(config.BTC_SYMBOL, config.HTF, limit=_CANDLES["4h"])
    if df is not None and not df.empty:
        _btc_cache["df"] = df
        _btc_cache["ts"] = now
    return df


# ── Gate 0: BTC current-state bias ───────────────────────────────────────────

def _btc_bias(df_btc: "pd.DataFrame") -> "tuple[str, dict]":
    """Score BTC *current* state from 4 momentum indicators.

    Returns (direction, debug_dict).
    direction is 'long' | 'short' | 'neutral'.

    Scoring (+1 bullish, -1 bearish, 0 neutral each):
        ema_cross : EMA20 > EMA50 → +1 ; else → -1
        ema_slope : EMA20[-1] > EMA20[-5] by 0.01% → +1 ; < → -1 ; flat → 0
        macd_hist : histogram last bar > 0 → +1 ; < 0 → -1 ; 0 → 0
        rsi_level : RSI > BTC_RSI_BULL(55) → +1 ; < BTC_RSI_BEAR(45) → -1 ; else → 0

    score ≥ +2 → 'long'
    score ≤ -2 → 'short'
    else       → 'neutral' (market inconclusive — scan aborts)

    Extra NEUTRAL override: if RSI in [45,55] AND pos in [35,65]% → 'neutral'
    (price mid-range, indicators flat — no edge).
    """
    import numpy as np
    from engine.indicators import macd as _macd

    n = len(df_btc)
    if n < 60:
        return "neutral", {"reason": "insufficient_data", "bars": n}

    closes = df_btc["close"]

    ema20 = closes.ewm(span=20, adjust=False).mean()
    ema50 = closes.ewm(span=50, adjust=False).mean()
    ema20_now  = float(ema20.iloc[-1])
    ema50_now  = float(ema50.iloc[-1])
    ema20_5ago = float(ema20.iloc[-6]) if n >= 6 else ema20_now

    _, _, hist = _macd(closes, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL_LEN)
    macd_hist_val = float(hist.iloc[-1])

    rsi_s   = _rsi(closes, config.RSI_LEN)
    rsi_val = float(rsi_s.iloc[-1])

    recent_high = float(df_btc["high"].tail(20).max())
    recent_low  = float(df_btc["low"].tail(20).min())
    span        = recent_high - recent_low
    pos_pct     = (float(closes.iloc[-1]) - recent_low) / span * 100 if span > 0 else 50.0

    ema_cross = 1 if ema20_now > ema50_now else -1
    ema_slope = (
        1  if ema20_now > ema20_5ago * 1.0001 else
        -1 if ema20_now < ema20_5ago * 0.9999 else
        0
    )
    macd_sig = 1 if macd_hist_val > 0 else (-1 if macd_hist_val < 0 else 0)
    rsi_sig  = (
        1  if rsi_val > config.BTC_RSI_BULL else
        -1 if rsi_val < config.BTC_RSI_BEAR else
        0
    )
    score = ema_cross + ema_slope + macd_sig + rsi_sig

    dbg = {
        "ema20":       round(ema20_now, 2),
        "ema50":       round(ema50_now, 2),
        "ema_cross":   ema_cross,
        "ema_slope":   ema_slope,
        "macd_hist":   round(macd_hist_val, 4),
        "macd_sig":    macd_sig,
        "rsi":         round(rsi_val, 1),
        "rsi_sig":     rsi_sig,
        "score":       score,
        "pos_pct":     round(pos_pct, 1),
        "close":       round(float(closes.iloc[-1]), 0),
    }

    # Range-neutral override
    if config.BTC_RSI_BEAR <= rsi_val <= config.BTC_RSI_BULL and config.BTC_POS_LO <= pos_pct <= config.BTC_POS_HI:
        dbg["reason"] = f"range_neutral(RSI={rsi_val:.1f} pos={pos_pct:.0f}%)"
        return "neutral", dbg

    if score >= config.BTC_SCORE_LONG:
        dbg["reason"] = f"score={score}>=+{config.BTC_SCORE_LONG}"
        return "long", dbg
    if score <= config.BTC_SCORE_SHORT:
        dbg["reason"] = f"score={score}<={config.BTC_SCORE_SHORT}"
        return "short", dbg

    dbg["reason"] = f"score={score}(inconclusive)"
    return "neutral", dbg


def _print_btc_debug(direction: str, dbg: dict) -> None:
    """Print Gate 0 bias line showing all 4 indicators + score."""
    c = {1: "▲", -1: "▼", 0: "→"}
    print(
        f"      [BTC BIAS] {direction.upper():<8}  score={dbg['score']}  "
        f"EMAcross{c[dbg['ema_cross']]}({dbg['ema_cross']:+d})  "
        f"EMAslope{c[dbg['ema_slope']]}({dbg['ema_slope']:+d})  "
        f"MACD{c[dbg['macd_sig']]}({dbg['macd_sig']:+d})={dbg['macd_hist']}  "
        f"RSI{c[dbg['rsi_sig']]}({dbg['rsi_sig']:+d})={dbg['rsi']}  "
        f"pos={dbg['pos_pct']}%  close={dbg['close']:.0f}  "
        f"reason={dbg['reason']}"
    )


# ── Main scan entry point ─────────────────────────────────────────────────────

def scan(symbols: list[str]) -> list[dict]:
    """Two-phase scan.

    Phase 1: ALL symbols → 4H+1H fetch → Gate 1 + sweep check (fast filter).
    Phase 2: ONLY swept symbols → 30m+15m+5m fetch → full Gates 2-4.
    """
    global _gate2_stats
    _gate2_stats = {}

    # ── item 5: blocklist filter (before any API call) ────────────────────────
    alts_raw = [s for s in symbols if s != config.BTC_SYMBOL]
    blocked_bl = [s for s in alts_raw if s in config.BLOCKLIST]
    alts = [s for s in alts_raw if s not in config.BLOCKLIST]

    print(f"\n{'='*65}")
    print(f"LIVE SCANNER  {_ts()}")
    print(
        f"Symbols: {len(alts)} alts  "
        f"(blocklist filtered: {len(blocked_bl)})  "
        f"MIN_VOL={config.MIN_VOLUME_USD:,.0f}  "
        f"workers={config.MAX_CONCURRENT}"
    )
    print(f"{'='*65}\n")

    # ── Gate 0: BTC direction (item 3: cached) ────────────────────────────────
    print(f"[0/3] BTC {config.HTF} direction …")
    df_btc = _fetch_btc_cached()
    if df_btc is None or df_btc.empty:
        print("      ERROR: BTC data unavailable — aborting scan.")
        return []

    btc_direction, btc_dbg = _btc_bias(df_btc)
    _print_btc_debug(btc_direction, btc_dbg)

    if btc_direction == "neutral":
        print("      BTC NEUTRAL — signals contradictory or market ranging. Aborting scan.")
        return []

    print(f"      BTC direction: {btc_direction.upper()}")

    # ── Phase 1: all symbols, 4H+1H only (item 1) ────────────────────────────
    t_scan_start = time.time()
    t1_start = time.time()
    print(f"\n[1/3] Phase 1 — {len(alts)} symbols, 4H+1H sweep check …")

    phase1_ok: dict[str, dict] = {}  # symbol → {"df_4h": df, "df_1h": df}
    p1_total = len(alts)
    p1_done = 0

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT) as pool:
        futs1 = {pool.submit(_scan_phase1, sym, btc_direction): sym for sym in alts}
        for fut in as_completed(futs1):
            sym = futs1[fut]
            p1_done += 1
            try:
                res = fut.result()
            except Exception as exc:
                _log.error("%s phase1 %s: %s", _ts(), sym, exc)
                res = None
            if res is not None:
                phase1_ok[sym] = res
            print(
                f"\r      [{p1_done:>4}/{p1_total}] checked  {len(phase1_ok)} passed sweep …",
                end="", flush=True,
            )

    elapsed1 = time.time() - t1_start
    print(
        f"\r      Phase 1: {p1_total} symbols → "
        f"{len(phase1_ok)} passed ({p1_total - len(phase1_ok)} dropped)  "
        f"[{elapsed1:.1f}s]"
    )

    if not phase1_ok:
        print("\n  No symbols passed Phase 1 sweep filter.")
        _print_summary([], p1_total, 0, btc_direction, p1_total, elapsed1, 0.0)
        return []

    # ── Phase 2: swept symbols only, full pipeline (item 1) ──────────────────
    t2_start = time.time()
    swept_syms = list(phase1_ok.keys())
    print(f"\n[2/3] Phase 2 — {len(swept_syms)} symbols, full Gates 2-4 …")

    active: list[dict] = []
    blocked_p2 = 0
    p2_done = 0

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT) as pool:
        futs2 = {
            pool.submit(
                _scan_phase2, sym, btc_direction,
                phase1_ok[sym]["df_4h"], phase1_ok[sym]["df_1h"],
            ): sym
            for sym in swept_syms
        }
        for fut in as_completed(futs2):
            sym = futs2[fut]
            p2_done += 1
            try:
                result = fut.result()
            except Exception as exc:
                _log.error("%s phase2 %s: %s", _ts(), sym, exc)
                result = {"symbol": sym, "signal": None, "reason": f"exception: {exc}"}

            if result.get("signal") == "active":
                active.append(result)
                conf = result.get("confluence", "★")
                dp = result.get("price_dp", _price_decimals(result["entry"]))
                with _print_lock:
                    print(
                        f"\n  {conf}  [{p2_done:>3}/{len(swept_syms)}]  {sym:<16} "
                        f"{btc_direction.upper():<5}  grade={result['grade']}  "
                        f"G3={'✓' if result.get('gate3') else '·'}  "
                        f"entry={result['entry']:.{dp}f}  "
                        f"R:R={result['rr']:.2f}  "
                        f"score={result.get('micro_score', 0):.1f}  "
                        f"atr={result.get('atr_adj', '~')}",
                        flush=True,
                    )
            else:
                blocked_p2 += 1

    elapsed2 = time.time() - t2_start
    total_elapsed = time.time() - t_scan_start

    print(
        f"\n      Phase 2: {len(swept_syms)} symbols → "
        f"{len(active)} signals ({blocked_p2} blocked)  [{elapsed2:.1f}s]"
    )
    print(f"      Total scan time: {total_elapsed:.1f}s")

    total_blocked = (p1_total - len(phase1_ok)) + blocked_p2
    _print_summary(active, total_blocked, blocked_p2, btc_direction, p1_total, elapsed1, elapsed2)
    _write_signals(active)
    return active


# ── Phase 1: fast sweep filter ────────────────────────────────────────────────

def _scan_phase1(symbol: str, btc_direction: Direction) -> "dict | None":
    """Phase 1: fetch 4H+1H in parallel → Gate 1 (MSS) + quick sweep check.

    Returns {"df_4h": df, "df_1h": df} if symbol has a valid sweep, else None.
    Does NOT search for OB/FVG — that is Phase 2 only.
    No 30m/15m/5m are fetched here.
    """
    # Fetch 4H and 1H in parallel (bounded by _HTTP_SEM in fetcher.py)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_4h = pool.submit(fetch_ohlcv, symbol, "4h", _CANDLES["4h"])
        fut_1h = pool.submit(fetch_ohlcv, symbol, "1h", _CANDLES["1h"])
        df_4h = fut_4h.result()
        df_1h = fut_1h.result()

    if df_4h is None or df_4h.empty:
        return None

    # Gate 1: symbol's own 4H MSS must align with BTC direction
    symbol_mss = detect_mss(df_4h)
    if symbol_mss is None or symbol_mss != btc_direction:
        return None

    # Quick sweep check — no OB/FVG search (Phase 2 handles that)
    sweep4 = _find_recent_sweep(df_4h, btc_direction, config.SWEEP_LOOKBACK_4H, "4h")
    if sweep4 is not None:
        return {"df_4h": df_4h, "df_1h": df_1h}

    if df_1h is not None and not df_1h.empty:
        sweep1 = _find_recent_sweep(df_1h, btc_direction, config.SWEEP_LOOKBACK_1H, "1h")
        if sweep1 is not None:
            return {"df_4h": df_4h, "df_1h": df_1h}

    return None  # no sweep → drop here, no further fetches


# ── Phase 2: full pipeline ────────────────────────────────────────────────────

def _scan_phase2(
    symbol: str,
    btc_direction: Direction,
    df_4h: "pd.DataFrame",
    df_1h: "pd.DataFrame | None",
) -> dict:
    """Phase 2: fetch 30m+15m+5m → Gates 2-4 + ATR star scoring.

    df_4h and df_1h already fetched in Phase 1 — no re-fetch.
    """
    def _block(reason: str) -> dict:
        return {"symbol": symbol, "signal": None, "reason": reason}

    direction: Direction = btc_direction

    # Fetch lower TFs in parallel (3 concurrent, bounded by _HTTP_SEM)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_30m = pool.submit(fetch_ohlcv, symbol, "30m", _CANDLES["30m"])
        fut_15m = pool.submit(fetch_ohlcv, symbol, "15m", _CANDLES["15m"])
        fut_5m  = pool.submit(fetch_ohlcv, symbol, "5m",  _CANDLES["5m"])
        df_30m = fut_30m.result()
        df_15m = fut_15m.result()
        df_5m  = fut_5m.result()

    # Gate 2: hierarchical sweep → LTF OB/FVG chain
    g2 = find_gate2_signal(df_4h, df_1h, df_30m, df_15m, direction)
    if g2 is None:
        reason = diagnose_gate2_signal(df_4h, df_1h, df_30m, df_15m, direction)
        with _print_lock:
            _gate2_stats[reason] = _gate2_stats.get(reason, 0) + 1
        return _block(f"gate2: {reason}")

    # Gate 3: RSI + MACD advisory (does NOT block)
    ref_df = df_1h if (df_1h is not None and not df_1h.empty) else df_4h
    gate3_ok = gate3_passes(ref_df, direction)

    # Base confluence from Gate 2
    confluence_stars = g2.confluence_stars

    # ── ATR star scoring (item 4): adjusts ranking, never blocks ─────────────
    atr_adj = "~"
    atr_4h = atr_scalar(df_4h, config.ATR_PERIOD)
    if atr_4h > 0:
        last = df_4h.iloc[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        if body > config.ATR_STRONG_MULT * atr_4h:
            confluence_stars = min(confluence_stars + 1, 3)
            atr_adj = "+1(strong)"
        elif body < config.ATR_WEAK_MULT * atr_4h:
            confluence_stars = max(confluence_stars - 1, 1)
            atr_adj = "-1(weak)"

    # Gate 4: 5m micro OB/FVG (scores entry quality 0-12)
    micro: MicroEntry | None = None
    if df_5m is not None and not df_5m.empty:
        atr_5m_val = atr_scalar(df_5m, config.ATR_5m_PERIOD)
        micro = find_micro_entry(
            df_5m, g2.primary_zone, direction, None, atr_5m_val, config.LTF,
            lookback_bars=96,
        )

    # Entry + SL
    if micro is not None:
        refined_entry = micro.entry
        sl_price = compute_sl(
            direction, micro.shadow_extreme,
            atr_scalar(df_5m, config.ATR_5m_PERIOD),   # type: ignore[arg-type]
        )
        micro_score = micro.score
    else:
        refined_entry = g2.entry
        sl_price = compute_sl(direction, g2.shadow_extreme, atr_scalar(df_4h, config.ATR_PERIOD))
        micro_score = 0.0

    # TP
    tp_ref = df_1h if (df_1h is not None and not df_1h.empty) else df_4h
    tp_price = find_tp(direction, refined_entry, sl_price, tp_ref)
    if tp_price is None:
        return _block(f"no TP at MIN_RR={config.MIN_RR}")

    rr = compute_rr(refined_entry, sl_price, tp_price)
    if rr < config.MIN_RR:
        return _block(f"R:R={rr:.2f} < {config.MIN_RR}")

    return {
        "symbol":           symbol,
        "direction":        direction,
        "signal":           "active",
        "grade":            g2.grade,
        "gate3":            gate3_ok,
        "confluence":       "★" * confluence_stars,
        "confluence_stars": confluence_stars,
        "atr_adj":          atr_adj,
        "micro_score":      round(micro_score, 2),
        "zone_type":        g2.primary_zone.zone_type,
        "zone_tf":          f"sweep{g2.sweep_tf}→{g2.primary_tf}",
        "zone_low":         g2.primary_zone.zone_low,
        "zone_high":        g2.primary_zone.zone_high,
        "entry":            refined_entry,
        "stop":             sl_price,
        "target":           tp_price,
        "rr":               round(rr, 2),
        "price_dp":         _price_decimals(refined_entry),
        "timestamp":        _ts(),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_summary(
    active: list[dict],
    total_blocked: int,
    blocked_p2: int,
    btc_direction: Direction,
    total: int,
    elapsed1: float,
    elapsed2: float,
) -> None:
    print(f"\n{'='*70}")
    print(f"SCAN COMPLETE  {_ts()}")
    print(
        f"Phase 1: {elapsed1:.1f}s   "
        f"Phase 2: {elapsed2:.1f}s   "
        f"Total: {elapsed1 + elapsed2:.1f}s"
    )
    print(f"Scanned: {total}  Active: {len(active)}  Blocked: {total_blocked}")
    print(f"BTC direction: {btc_direction.upper()}")

    if active:
        ranked = sorted(
            active,
            key=lambda s: (s["confluence_stars"], s["micro_score"]),
            reverse=True,
        )
        print(f"\n── ACTIVE SIGNALS (ranked by confluence + ATR scoring) ──────────────")
        hdr = (
            f"  {'CONF':<5} {'ATR':<12} {'SYMBOL':<16} {'DIR':<5} {'GRADE':<5} "
            f"{'G3':<3} {'SCORE':>5} {'ENTRY':>14} {'STOP':>14} {'TARGET':>14} {'R:R':>5}"
        )
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for s in ranked:
            g3 = "✓" if s.get("gate3") else "·"
            dp = s.get("price_dp", _price_decimals(s["entry"]))
            print(
                f"  {s['confluence']:<5} {s.get('atr_adj','~'):<12} {s['symbol']:<16} "
                f"{s['direction'].upper():<5} {s['grade']:<5} {g3:<3} "
                f"{s['micro_score']:>5.1f} "
                f"{s['entry']:>{14}.{dp}f} {s['stop']:>{14}.{dp}f} "
                f"{s['target']:>{14}.{dp}f} {s['rr']:>5.2f}"
            )
        print()
        print("  Zone detail:")
        for s in ranked:
            dp = s.get("price_dp", _price_decimals(s["zone_low"]))
            print(
                f"    {s['symbol']:<16} zone [{s['zone_type']} {s['zone_tf']}]  "
                f"{s['zone_low']:.{dp}f}–{s['zone_high']:.{dp}f}"
            )
    else:
        print("\n  No active signals.")

    # Gate 2 rejection breakdown (Phase 2 only)
    if _gate2_stats:
        print(f"\n── GATE 2 REJECTION BREAKDOWN (Phase 2 only) ────────────────────────")
        _reason_labels = {
            "no_4h_sweep":               "4H sweep absent (1H fallback also failed)",
            "4h_sweep_low_volume":        "4H sweep low volume (1H fallback also failed)",
            "4h_sweep_counter_momentum":  "Counter-momentum blocked 4H sweep",
            "no_ob_after_4h_sweep":       "4H sweep OK — no LTF OB/FVG in window",
            "no_ob_after_1h_sweep":       "1H sweep OK (4H failed) — no OB on 30m/15m",
            "no_ob_after_sweep":          "No LTF OB/FVG after sweep (legacy label)",
            "insufficient_data":          "Insufficient 4H data",
        }
        total_g2 = sum(_gate2_stats.values())
        for key in [
            "no_4h_sweep", "4h_sweep_low_volume", "4h_sweep_counter_momentum",
            "no_ob_after_4h_sweep", "no_ob_after_1h_sweep",
            "no_ob_after_sweep", "insufficient_data",
        ]:
            count = _gate2_stats.get(key, 0)
            if count:
                label = _reason_labels.get(key, key)
                print(f"  {label}: {count}  ({100 * count // total_g2}%)")
        print(f"  {'─'*50}")
        print(f"  Total gate2-blocked (Phase 2): {total_g2}")

    print(f"{'='*70}\n")


def _write_signals(active: list[dict]) -> None:
    import json
    os.makedirs(os.path.dirname(_SIGNALS_FILE), exist_ok=True)
    try:
        with open(_SIGNALS_FILE, "w") as f:
            json.dump(active, f, indent=2)
    except Exception as exc:
        _log.error("%s failed to write live_signals.json: %s", _ts(), exc)


def _price_decimals(price: float) -> int:
    """Decimal places to show ≥ 4 significant digits (min 2)."""
    if price <= 0:
        return 6
    mag = math.floor(math.log10(abs(price)))
    return max(4 - 1 - mag, 2)


def _ts() -> str:
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S Tehran")


def _parse_volume(s: str) -> float:
    """'5m' → 5_000_000, '500k' → 500_000, '1000000' → 1_000_000."""
    s = s.strip().lower()
    if s.endswith("m"):
        return float(s[:-1]) * 1_000_000
    if s.endswith("k"):
        return float(s[:-1]) * 1_000
    return float(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitunix ICT live scanner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--top", type=int, metavar="N",
        help="Scan only top N symbols by 24h volume",
    )
    group.add_argument(
        "--min-vol", type=str, metavar="USD", default=None,
        help=f"Min 24h USDT turnover e.g. 2m or 500k (default: {config.MIN_VOLUME_USD:,.0f})",
    )
    args = parser.parse_args()

    print("Fetching symbol list from Bitunix …")
    if args.top:
        symbols = get_top_symbols(args.top)
        print(f"Mode: top {args.top} symbols")
    else:
        min_vol = _parse_volume(args.min_vol) if args.min_vol else config.MIN_VOLUME_USD
        symbols = get_all_symbols(min_vol)
        print(f"Mode: ALL symbols >= ${min_vol:,.0f} 24h USDT volume")

    if not symbols:
        print("ERROR: could not fetch symbol list. Check network and retry.")
        sys.exit(1)

    print(f"Loaded {len(symbols) - 1} alts + BTC\n")
    scan(symbols)
