"""Live scanner — scans USDT-M futures from Bitunix in real time.

Gates run on live API data (no CSVs):
  Gate 0: BTC 4H MSS direction
  Gate 2: OB/FVG rejection candle on 4H (fallback to 1H)
  Gate 3: RSI + MACD advisory (logged, does not block)
  Gate 4: 5m entry refinement

No backtest metrics gate — this is a live screener, not a backtest validator.

Usage:
    python live.py                # scan all symbols >= MIN_VOLUME_USD
    python live.py --top 30       # scan only top 30 by volume
    python live.py --min-vol 10m  # custom minimum volume (10 million USD)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetcher import fetch_ohlcv, get_all_symbols, get_top_symbols
from engine.ict import detect_mss, atr_scalar
from engine.zones import find_active_zones, find_rejection
from engine.indicators import gate3_passes
from engine.entry import optimize_entry
from engine.sl_tp import compute_sl, find_tp, compute_rr
from contracts import Direction

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
    "4h": 300,   # ~50 days — enough for OB/FVG detection + warmup
    "1h": 500,   # ~21 days
    "5m": 288,   # ~1 day — entry refinement only
}

_RATE_DELAY = 0.25  # seconds between API requests within one symbol's fetch sequence

_print_lock = threading.Lock()


def scan(symbols: list[str]) -> list[dict]:
    """Scan a pre-built list of symbols. Returns list of active signal dicts.

    `symbols` must include config.BTC_SYMBOL as the first entry.
    """
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
    print(f"[3/3] Scanning …  (. = no signal, S = signal)\n")
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
                with _print_lock:
                    g3 = "G3✓" if result.get("gate3") else "G3·"
                    print(
                        f"  SIGNAL  {symbol:<16} {btc_direction.upper():<5}  "
                        f"grade={result['grade']}  {g3}  "
                        f"entry={result['entry']:.4f}  "
                        f"stop={result['stop']:.4f}  "
                        f"target={result['target']:.4f}  "
                        f"R:R={result['rr']:.2f}  "
                        f"[{completed}/{total}]"
                    )
            else:
                blocked_count += 1
                with _print_lock:
                    print(f"  .  [{completed:>4}/{total}]  {symbol}", flush=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(active, blocked_count, btc_direction, total)
    _write_signals(active)
    return active


def _scan_symbol(symbol: str, btc_direction: Direction) -> dict:
    """Fetch live data and run gates 2-4 for one symbol."""

    def _block(reason: str) -> dict:
        return {"symbol": symbol, "signal": None, "reason": reason}

    df_4h = fetch_ohlcv(symbol, config.HTF, limit=_CANDLES["4h"])
    time.sleep(_RATE_DELAY)
    if df_4h is None or df_4h.empty:
        return _block("4H data unavailable")

    df_1h = fetch_ohlcv(symbol, config.HTF_ALT, limit=_CANDLES["1h"])
    time.sleep(_RATE_DELAY)

    df_5m = fetch_ohlcv(symbol, config.LTF, limit=_CANDLES["5m"])
    time.sleep(_RATE_DELAY)

    direction: Direction = btc_direction

    # Gate 2: rejection candle on 4H (fallback 1H)
    zones_4h = find_active_zones(df_4h, direction, config.HTF)
    rejection = find_rejection(df_4h, len(df_4h) - 1, zones_4h)

    if rejection is None and df_1h is not None and not df_1h.empty:
        zones_1h = find_active_zones(df_1h, direction, config.HTF_ALT)
        rejection = find_rejection(df_1h, len(df_1h) - 1, zones_1h)

    if rejection is None:
        return _block("gate2: no rejection candle")

    # Gate 3: RSI + MACD advisory
    ref_df = df_1h if (df_1h is not None and not df_1h.empty) else df_4h
    gate3_ok = gate3_passes(ref_df, direction)

    # Gate 4: 5m entry refinement
    zone = rejection.zone
    as_of_ts = df_4h.index[-1]
    if df_5m is not None and not df_5m.empty:
        refined_entry = optimize_entry(df_5m, zone, direction, as_of_ts)
    else:
        refined_entry = rejection.entry

    # SL / TP
    atr_val = atr_scalar(df_4h, config.ATR_PERIOD)
    sl_price = compute_sl(direction, rejection.shadow_extreme, atr_val)
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
        "grade": rejection.grade,
        "gate3": gate3_ok,
        "zone_type": zone.zone_type,
        "zone_tf": zone.timeframe,
        "zone_low": round(zone.zone_low, 6),
        "zone_high": round(zone.zone_high, 6),
        "entry": round(refined_entry, 6),
        "stop": round(sl_price, 6),
        "target": round(tp_price, 6),
        "rr": round(rr, 2),
        "timestamp": _ts(),
    }


def _print_summary(
    active: list[dict],
    blocked_count: int,
    btc_direction: Direction,
    total: int,
) -> None:
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE  {_ts()}")
    print(f"Scanned: {total}  Active: {len(active)}  Blocked: {blocked_count}")
    print(f"BTC direction: {btc_direction.upper()}")

    if active:
        print(f"\n── ACTIVE SIGNALS ──────────────────────────────────────")
        header = f"  {'SYMBOL':<16} {'DIR':<5} {'GRADE':<6} {'G3':<4} {'ZONE':<18} {'ENTRY':>10} {'STOP':>10} {'TARGET':>10} {'R:R':>5}"
        print(header)
        print(f"  {'-'*len(header.strip())}")
        for s in active:
            g3 = "✓" if s.get("gate3") else "·"
            zone_str = f"{s['zone_type']} {s['zone_tf']}"
            print(
                f"  {s['symbol']:<16} {s['direction'].upper():<5} "
                f"{s['grade']:<6} {g3:<4} {zone_str:<18} "
                f"{s['entry']:>10.4f} {s['stop']:>10.4f} {s['target']:>10.4f} {s['rr']:>5.2f}"
            )
    else:
        print("\n  No active signals.")
    print(f"{'='*60}\n")


def _write_signals(active: list[dict]) -> None:
    import json
    os.makedirs(os.path.dirname(_SIGNALS_FILE), exist_ok=True)
    try:
        with open(_SIGNALS_FILE, "w") as f:
            json.dump(active, f, indent=2)
    except Exception as exc:
        _log.error("%s failed to write live_signals.json: %s", _ts(), exc)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_volume(s: str) -> float:
    """Parse '5m' → 5_000_000, '1.5m' → 1_500_000, '500k' → 500_000, '1000000' → 1_000_000."""
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
        help=f"Scan only top N symbols by 24h volume (default: all >= MIN_VOLUME_USD)",
    )
    group.add_argument(
        "--min-vol", type=str, metavar="USD", default=None,
        help=f"Min 24h USDT turnover, e.g. 5m or 500k (default: {config.MIN_VOLUME_USD:,.0f})",
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
