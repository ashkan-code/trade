"""Live scanner — scans USDT-M futures from Bitunix in real time.

Gates run on live API data (no CSVs):
  Gate 0: BTC 4H MSS direction
  Gate 2: ICT/Wyckoff sweep+reclaim on ANY of 4H / 1H / 30m / 15m
          Sweep conditions: liquidity sweep, reclaim close, volume ≥ SMA×ratio,
          no counter-institutional momentum spike in prior LOOKBACK_BARS
  Gate 3: RSI + MACD advisory (logged, does not block)
  Gate 4: micro OB/FVG on 5m within HTF sweep zone — scored and ranked

Multi-TF confluence levels (used for signal ranking):
  ★★★  4H + 1H sweep zones overlap + (30m or 15m also overlaps, or 5m micro found)
  ★★   4H + 1H sweep zones overlap in price
  ★    single-timeframe sweep

Usage:
    python live.py                # scan all symbols >= MIN_VOLUME_USD
    python live.py --top 30       # scan only top 30 by volume
    python live.py --min-vol 10m  # custom minimum volume (10 million USD)
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
from contracts import Direction, MicroEntry, RejectionCandle
from data.fetcher import fetch_ohlcv, get_all_symbols, get_top_symbols
from engine.entry import find_micro_entry, zones_overlap
from engine.ict import atr_scalar, detect_mss
from engine.indicators import gate3_passes
from engine.sl_tp import compute_rr, compute_sl, find_tp
from engine.zones import diagnose_sweep_rejection, find_sweep_rejection

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
    "4h": 300,   # ~50 days
    "1h": 500,   # ~21 days
    "30m": 240,  # ~5 days
    "15m": 192,  # ~2 days
    "5m": 288,   # ~1 day
}

_RATE_DELAY = 0.25  # seconds between sequential API requests per symbol (legacy; _fetch_all_tfs is parallel)

_print_lock = threading.Lock()

# Gate 2 diagnostic counters (protected by _print_lock)
_gate2_stats: dict[str, int] = {}


def scan(symbols: list[str]) -> list[dict]:
    """Scan a pre-built list of symbols. Returns list of active signal dicts."""
    global _gate2_stats
    _gate2_stats = {}
    alts = [s for s in symbols if s != config.BTC_SYMBOL]

    print(f"\n{'='*60}")
    print(f"LIVE SCANNER  {_ts()}")
    print(f"Symbols: {len(alts)} alts  MIN_RR={config.MIN_RR}  "
          f"HTF={config.HTF}  HTF_ALT={config.HTF_ALT}  "
          f"workers={config.MAX_CONCURRENT}")
    print(f"{'='*60}\n")

    # ── Step 1: BTC direction ──────────────────────────────────────────────────
    print(f"[1/3] Fetching BTC {config.HTF} …")
    df_btc = fetch_ohlcv(config.BTC_SYMBOL, config.HTF, limit=_CANDLES["4h"])
    if df_btc is None or df_btc.empty:
        print("      ERROR: BTC data unavailable — aborting scan.")
        return []

    btc_direction: Direction | None = detect_mss(df_btc)
    if btc_direction is None:
        print("      BTC MSS undetermined — no dominant structure. Aborting.")
        return []

    print(f"      BTC direction: {btc_direction.upper()}")

    # ── Step 2: symbol list already provided ──────────────────────────────────
    print(f"\n[2/3] {len(alts)} symbols ready\n")

    # ── Step 3: parallel scan ─────────────────────────────────────────────────
    print(f"[3/3] Scanning …  (. = no signal, ★ = signal)\n")
    active: list[dict] = []
    blocked_count = 0
    completed = 0
    total = len(alts)

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT) as pool:
        futures = {pool.submit(_scan_symbol, sym, btc_direction): sym for sym in alts}

        for fut in as_completed(futures):
            symbol = futures[fut]
            completed += 1
            try:
                result = fut.result()
            except Exception as exc:
                _log.error("%s unhandled error scanning %s: %s", _ts(), symbol, exc)
                result = {"symbol": symbol, "signal": None, "reason": f"exception: {exc}"}

            if result.get("signal") == "active":
                active.append(result)
                conf = result.get("confluence", "★")
                _dp = result.get("price_dp", _price_decimals(result["entry"]))
                with _print_lock:
                    print(
                        f"  {conf}  [{completed:>4}/{total}]  {symbol:<16} "
                        f"{btc_direction.upper():<5}  grade={result['grade']}  "
                        f"G3={'✓' if result.get('gate3') else '·'}  "
                        f"entry={result['entry']:.{_dp}f}  "
                        f"R:R={result['rr']:.2f}  "
                        f"score={result.get('micro_score', 0):.1f}"
                    )
            else:
                blocked_count += 1
                with _print_lock:
                    print(f"  .  [{completed:>4}/{total}]  {symbol}", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(active, blocked_count, btc_direction, total)
    _write_signals(active)
    return active


def _fetch_all_tfs(symbol: str) -> dict[str, "pd.DataFrame | None"]:
    """Fetch 4H, 1H, 30m, 15m, 5m concurrently (bounded by _HTTP_SEM in fetcher)."""
    tf_limits = [
        ("4h",  _CANDLES["4h"]),
        ("1h",  _CANDLES["1h"]),
        ("30m", _CANDLES["30m"]),
        ("15m", _CANDLES["15m"]),
        ("5m",  _CANDLES["5m"]),
    ]
    results: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fetch_ohlcv, symbol, tf, lim): tf for tf, lim in tf_limits}
        for fut in as_completed(futures):
            tf = futures[fut]
            try:
                results[tf] = fut.result()
            except Exception as exc:
                _log.error("%s fetch %s %s: %s", _ts(), symbol, tf, exc)
                results[tf] = None
    return results


def _scan_symbol(symbol: str, btc_direction: Direction) -> dict:
    """Fetch live data, run gates 2-4, compute multi-TF sweep confluence."""

    def _block(reason: str) -> dict:
        return {"symbol": symbol, "signal": None, "reason": reason}

    # ── Fetch all timeframes in parallel ────────────────────────────────────────
    dfs = _fetch_all_tfs(symbol)
    df_4h  = dfs.get("4h")
    df_1h  = dfs.get("1h")
    df_30m = dfs.get("30m")
    df_15m = dfs.get("15m")
    df_5m  = dfs.get("5m")

    if df_4h is None or df_4h.empty:
        return _block("4H data unavailable")

    direction: Direction = btc_direction

    # ── Gate 2: ICT sweep on each TF (passes if ANY TF passes) ─────────────────
    def _sw(df: "pd.DataFrame | None", tf: str) -> "RejectionCandle | None":
        if df is None or df.empty:
            return None
        return find_sweep_rejection(df, direction, tf)

    sweep_4h  = _sw(df_4h,  config.HTF)
    sweep_1h  = _sw(df_1h,  config.HTF_ALT)
    sweep_30m = _sw(df_30m, "30m")
    sweep_15m = _sw(df_15m, "15m")

    if not any([sweep_4h, sweep_1h, sweep_30m, sweep_15m]):
        # Diagnostic: best reason across TFs (counter_momentum > volume > sweep)
        best_reason = "sweep_not_found"
        for df_tf in [df_4h, df_1h, df_30m, df_15m]:
            if df_tf is None or df_tf.empty:
                continue
            r = diagnose_sweep_rejection(df_tf, direction)
            if r == "counter_momentum":
                best_reason = "counter_momentum"
                break
            if r == "volume_min_ratio" and best_reason == "sweep_not_found":
                best_reason = "volume_min_ratio"
        with _print_lock:
            _gate2_stats[best_reason] = _gate2_stats.get(best_reason, 0) + 1
        return _block(f"gate2: {best_reason}")

    # Primary sweep: prefer higher timeframe
    primary: RejectionCandle = (sweep_4h or sweep_1h or sweep_30m or sweep_15m)  # type: ignore[assignment]

    # ── Gate 3: RSI + MACD advisory ────────────────────────────────────────────
    ref_df = df_1h if (df_1h is not None and not df_1h.empty) else df_4h
    gate3_ok = gate3_passes(ref_df, direction)

    # ── Confluence level ────────────────────────────────────────────────────────
    # ★★:  4H + 1H sweep zones overlap in price
    # ★★★: 4H + 1H + (30m or 15m) zones overlap
    confluence_stars = 1
    if sweep_4h and sweep_1h and zones_overlap(sweep_4h.zone, sweep_1h.zone):
        confluence_stars = 2
        ltf_ok = (
            (sweep_30m and zones_overlap(sweep_4h.zone, sweep_30m.zone)) or
            (sweep_15m and zones_overlap(sweep_4h.zone, sweep_15m.zone))
        )
        if ltf_ok:
            confluence_stars = 3

    # ── Gate 4: micro OB/FVG on 5m within primary HTF zone ────────────────────
    micro: MicroEntry | None = None
    if df_5m is not None and not df_5m.empty:
        atr_5m_val = atr_scalar(df_5m, config.ATR_5m_PERIOD)
        micro = find_micro_entry(
            df_5m, primary.zone, direction, None, atr_5m_val, config.LTF,
            lookback_bars=96,
        )
        if micro is not None and confluence_stars == 2:
            confluence_stars = 3

    # ── Entry and SL ────────────────────────────────────────────────────────────
    if micro is not None:
        refined_entry = micro.entry
        sl_price = compute_sl(direction, micro.shadow_extreme, atr_scalar(df_5m, config.ATR_5m_PERIOD))  # type: ignore[arg-type]
        micro_score = micro.score
    else:
        refined_entry = primary.entry
        sl_price = compute_sl(direction, primary.shadow_extreme, atr_scalar(df_4h, config.ATR_PERIOD))
        micro_score = 0.0

    # ── TP ──────────────────────────────────────────────────────────────────────
    tp_ref = df_1h if (df_1h is not None and not df_1h.empty) else df_4h
    tp_price = find_tp(direction, refined_entry, sl_price, tp_ref)

    if tp_price is None:
        return _block(f"no TP at MIN_RR={config.MIN_RR}")

    rr = compute_rr(refined_entry, sl_price, tp_price)
    if rr < config.MIN_RR:
        return _block(f"R:R={rr:.2f} < {config.MIN_RR}")

    return {
        "symbol": symbol,
        "direction": direction,
        "signal": "active",
        "grade": primary.grade,
        "gate3": gate3_ok,
        "confluence": "★" * confluence_stars,
        "confluence_stars": confluence_stars,
        "micro_score": round(micro_score, 2),
        "zone_type": primary.zone.zone_type,
        "zone_tf": primary.zone.timeframe,
        "zone_low": primary.zone.zone_low,
        "zone_high": primary.zone.zone_high,
        "entry": refined_entry,
        "stop": sl_price,
        "target": tp_price,
        "rr": round(rr, 2),
        "price_dp": _price_decimals(refined_entry),
        "timestamp": _ts(),
    }


def _print_summary(
    active: list[dict],
    blocked_count: int,
    btc_direction: Direction,
    total: int,
) -> None:
    print(f"\n{'='*70}")
    print(f"SCAN COMPLETE  {_ts()}")
    print(f"Scanned: {total}  Active: {len(active)}  Blocked: {blocked_count}")
    print(f"BTC direction: {btc_direction.upper()}")

    if active:
        # Sort: confluence desc, then micro_score desc
        ranked = sorted(
            active,
            key=lambda s: (s["confluence_stars"], s["micro_score"]),
            reverse=True,
        )
        print(f"\n── ACTIVE SIGNALS (ranked by multi-TF confluence) ──────────────────")
        hdr = (
            f"  {'CONF':<5} {'SYMBOL':<16} {'DIR':<5} {'GRADE':<5} "
            f"{'G3':<3} {'SCORE':>5} {'ENTRY':>14} {'STOP':>14} {'TARGET':>14} {'R:R':>5}"
        )
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")
        for s in ranked:
            g3 = "✓" if s.get("gate3") else "·"
            dp = s.get("price_dp", _price_decimals(s["entry"]))
            print(
                f"  {s['confluence']:<5} {s['symbol']:<16} {s['direction'].upper():<5} "
                f"{s['grade']:<5} {g3:<3} {s['micro_score']:>5.1f} "
                f"{s['entry']:>{14}.{dp}f} {s['stop']:>{14}.{dp}f} {s['target']:>{14}.{dp}f} {s['rr']:>5.2f}"
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

    # ── Gate 2 sweep rejection breakdown ───────────────────────────────────────
    if _gate2_stats:
        print(f"\n── GATE 2 SWEEP REJECTION BREAKDOWN ───────────────────────────────")
        _reason_labels = {
            "sweep_not_found":   "No sweep+reclaim on any TF (4H/1H/30m/15m)",
            "volume_min_ratio":  "Rejection candle volume below SMA threshold",
            "counter_momentum":  "Counter-institutional momentum spike detected",
            "insufficient_data": "Insufficient data for sweep detection",
        }
        total_blocked = sum(_gate2_stats.values())
        for key in ["sweep_not_found", "volume_min_ratio", "counter_momentum", "insufficient_data"]:
            count = _gate2_stats.get(key, 0)
            if count:
                label = _reason_labels.get(key, key)
                print(f"  {label}: {count}  ({100*count//total_blocked}%)")
        print(f"  {'─'*50}")
        print(f"  Total gate2-blocked: {total_blocked}")

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
    """Decimal places needed to show at least 4 significant digits (minimum 2).

    Examples: 105000 → 2,  1.5 → 3,  0.0001234 → 7
    """
    if price <= 0:
        return 6
    mag = math.floor(math.log10(abs(price)))
    return max(4 - 1 - mag, 2)


def _ts() -> str:
    return datetime.now(ZoneInfo("Asia/Tehran")).strftime("%Y-%m-%d %H:%M:%S Tehran")


def _parse_volume(s: str) -> float:
    """Parse '5m' → 5_000_000, '500k' → 500_000, '1000000' → 1_000_000."""
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
        help=f"Min 24h USDT turnover e.g. 5m or 500k (default: {config.MIN_VOLUME_USD:,.0f})",
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
