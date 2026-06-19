"""Live scanner — scans top-N USDT-M futures from Bitunix in real time.

Gates run on live API data (no CSVs):
  Gate 0: BTC 4H MSS direction
  Gate 2: OB/FVG rejection candle on 4H (fallback to 1H)
  Gate 3: RSI + MACD advisory (logged, does not block)
  Gate 4: 5m entry refinement

No backtest metrics gate — this is a live screener, not a backtest validator.

Usage:
    python live.py
    python live.py --top 30      # override TOP_N for this run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.fetcher import fetch_ohlcv, get_top_symbols
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

# Candle counts for live fetch (enough for ICT detection + warmup)
_CANDLES: dict[str, int] = {
    "4h": 300,   # ~50 days
    "1h": 500,   # ~21 days
    "5m": 288,   # ~1 day (entry refinement only)
}

_RATE_DELAY = 0.25  # seconds between API requests


def scan(top_n: int) -> list[dict]:
    """Run full live scan. Returns list of active signal dicts."""

    print(f"\n{'='*60}")
    print(f"LIVE SCANNER  {_ts()}")
    print(f"TOP_N={top_n}  MIN_RR={config.MIN_RR}  "
          f"HTF={config.HTF}  HTF_ALT={config.HTF_ALT}")
    print(f"{'='*60}\n")

    # ── Step 1: BTC direction ──────────────────────────────────────────────────
    print(f"[1/3] Fetching BTC {config.HTF} …")
    df_btc = fetch_ohlcv(config.BTC_SYMBOL, config.HTF, limit=_CANDLES["4h"])
    if df_btc is None or df_btc.empty:
        print("      ERROR: BTC data unavailable — aborting scan.")
        return []

    btc_direction: Direction | None = detect_mss(df_btc)
    if btc_direction is None:
        print("      BTC MSS undetermined — no dominant structure found. Aborting.")
        return []

    print(f"      BTC direction: {btc_direction.upper()}")

    # ── Step 2: symbol list ────────────────────────────────────────────────────
    print(f"\n[2/3] Fetching top {top_n} symbols by 24h USDT turnover …")
    symbols = get_top_symbols(top_n)
    if not symbols:
        print("      ERROR: could not fetch symbol list from Bitunix API.")
        return []

    alts = [s for s in symbols if s != config.BTC_SYMBOL]
    print(f"      Got {len(alts)} alts (BTC excluded from signals)\n")

    # ── Step 3: scan each alt ──────────────────────────────────────────────────
    print(f"[3/3] Scanning {len(alts)} symbols …\n")
    active: list[dict] = []
    blocked: list[str] = []

    for i, symbol in enumerate(alts, 1):
        print(f"  [{i:2d}/{len(alts)}] {symbol:<15}", end="  ", flush=True)
        result = _scan_symbol(symbol, btc_direction, df_btc)

        if result["signal"] == "active":
            active.append(result)
            g3 = "✓G3" if result.get("gate3") else "·G3"
            print(
                f"SIGNAL  {btc_direction.upper():<5}  "
                f"grade={result['grade']}  {g3}  "
                f"entry={result['entry']:.4f}  "
                f"stop={result['stop']:.4f}  "
                f"target={result['target']:.4f}  "
                f"R:R={result['rr']:.2f}"
            )
        else:
            blocked.append(symbol)
            print(f"blocked  ({result.get('reason', '?')})")

        if i < len(alts):
            time.sleep(_RATE_DELAY)

    # ── Summary ────────────────────────────────────────────────────────────────
    _print_summary(active, blocked, top_n, btc_direction)
    _write_signals(active)
    return active


def _scan_symbol(
    symbol: str,
    btc_direction: Direction,
    df_btc: "pd.DataFrame",
) -> dict:
    """Run gates 0-4 on live data for one symbol. Returns signal dict."""
    import pandas as pd

    def _block(reason: str) -> dict:
        return {"symbol": symbol, "signal": None, "reason": reason}

    # Fetch live 4H
    df_4h = fetch_ohlcv(symbol, config.HTF, limit=_CANDLES["4h"])
    time.sleep(_RATE_DELAY)
    if df_4h is None or df_4h.empty:
        return _block("4H data unavailable")

    # Fetch live 1H
    df_1h = fetch_ohlcv(symbol, config.HTF_ALT, limit=_CANDLES["1h"])
    time.sleep(_RATE_DELAY)

    # Fetch live 5m
    df_5m = fetch_ohlcv(symbol, config.LTF, limit=_CANDLES["5m"])
    time.sleep(_RATE_DELAY)

    direction: Direction = btc_direction

    # Gate 2: rejection candle on 4H (fallback to 1H)
    zones_4h = find_active_zones(df_4h, direction, config.HTF)
    rejection = find_rejection(df_4h, len(df_4h) - 1, zones_4h)

    if rejection is None and df_1h is not None and not df_1h.empty:
        zones_1h = find_active_zones(df_1h, direction, config.HTF_ALT)
        rejection = find_rejection(df_1h, len(df_1h) - 1, zones_1h)

    if rejection is None:
        return _block("gate2: no rejection candle")

    # Gate 3: RSI + MACD (advisory — logged, does not block)
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
    blocked: list[str],
    top_n: int,
    btc_direction: Direction,
) -> None:
    total = len(active) + len(blocked)
    print(f"\n{'='*60}")
    print(f"SCAN COMPLETE  {_ts()}")
    print(f"Scanned: {total}  Active signals: {len(active)}  Blocked: {len(blocked)}")
    print(f"BTC direction: {btc_direction.upper()}")
    if active:
        print(f"\n── ACTIVE SIGNALS ──")
        for s in active:
            g3 = "G3✓" if s.get("gate3") else "G3·"
            print(
                f"  {s['symbol']:<15} {s['direction'].upper():<5} "
                f"grade={s['grade']} {g3}  "
                f"zone [{s['zone_type']} {s['zone_tf']}] "
                f"{s['zone_low']}–{s['zone_high']}"
            )
            print(
                f"  {'':15} entry={s['entry']}  stop={s['stop']}  "
                f"target={s['target']}  R:R={s['rr']}"
            )
    else:
        print("\n  No active signals — all symbols blocked at gate0-4.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bitunix ICT live scanner")
    parser.add_argument("--top", type=int, default=config.TOP_N,
                        help=f"Number of top symbols to scan (default: {config.TOP_N})")
    args = parser.parse_args()
    scan(top_n=args.top)
