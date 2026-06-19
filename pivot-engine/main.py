"""Pivot Engine entry point — multi-timeframe ICT rejection pipeline.

Two modes:
  OFFLINE=True  (default): load CSVs + full backtest + metrics validation.
  OFFLINE=False           : live scan — fetches symbols + OHLCV from Bitunix API,
                            runs gates 0-4, prints signals. Same as `python live.py`.

For each symbol:
  1. Load BTC 4H (direction filter — never traded).
  2. Detect BTC direction via MSS.
  3. Run 4-gate funnel: MSS → rejection candle → RSI/MACD → 5m entry.
  4. (Offline only) Backtest on 4H bars; compute honest metrics.
  5. Write signals to logs/signals.json or logs/live_signals.json.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backtest.metrics import compute
from backtest.repaint_audit import audit_no_lookahead
from backtest.simulator import replay
from contracts import Direction, Setup
from data.loader import get_ohlcv
from engine.ict import detect_mss, atr_scalar
from engine.zones import find_active_zones, find_rejection
from engine.indicators import gate3_passes
from engine.entry import optimize_entry
from engine.sl_tp import compute_sl, find_tp, compute_rr
from report.generator import generate
from sigbuild.builder import build

OFFLINE: bool = True   # False = live scan from Bitunix API (same as python live.py)

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCANNER_FILE = os.path.join(_BASE, "scanner_output.json")
_TOP_SYMBOLS_FILE = os.path.join(_BASE, "data", "top_symbols.json")
_SIGNALS_FILE = os.path.join(_BASE, "logs", "signals.json")
_ERROR_LOG = os.path.join(_BASE, "logs", "errors.log")

logging.basicConfig(
    filename=_ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)


def main() -> None:
    if not OFFLINE:
        # Live scan: fetch symbols from API, run gates on live data
        from live import scan
        from data.fetcher import get_all_symbols
        symbols = get_all_symbols()
        scan(symbols)
        return

    # Load BTC first — direction needed to build scanner items from top_symbols.json
    df_btc_4h = get_ohlcv(config.BTC_SYMBOL, config.HTF, offline=OFFLINE)
    if df_btc_4h is None:
        print("[WARN] BTC 4H data unavailable — BTC direction gate disabled")

    btc_direction = detect_mss(df_btc_4h) if df_btc_4h is not None else None
    if btc_direction:
        print(f"BTC direction: {btc_direction.upper()}")
    else:
        print("BTC direction: undetermined (no MSS detected)")

    scanner_items = _load_scanner(btc_direction)
    if not scanner_items:
        print(
            "No scanner items found.\n"
            "  Option A: create scanner_output.json  (manual symbol + direction list)\n"
            f"  Option B: run scripts/download_history.py to generate data/top_symbols.json"
        )
        return

    all_signals: list[dict] = []
    for item in scanner_items:
        symbol: str = item["symbol"]
        direction: Direction = item["direction"]
        print(f"\n{'─' * 60}")
        print(f"Processing {symbol} [{direction}]")
        result = _process(symbol, direction, df_btc_4h, btc_direction)
        all_signals.append(result)
        print(result["report"])

    _write_signals(all_signals)
    print(f"\nSignals written to {_SIGNALS_FILE}")


def _process(
    symbol: str,
    direction: Direction,
    df_btc_4h,
    btc_direction: Direction | None,
) -> dict:
    # 1. Load alt data
    df_4h = get_ohlcv(symbol, config.HTF, offline=OFFLINE)
    df_1h = get_ohlcv(symbol, config.HTF_ALT, offline=OFFLINE)
    df_5m = get_ohlcv(symbol, config.LTF, offline=OFFLINE)

    if df_4h is None:
        return _error_result(symbol, direction, "4H data load failed")
    if df_1h is None:
        print(f"  [WARN] 1H data missing for {symbol} — using 4H only")

    # 2. Gate 0: BTC direction
    if btc_direction is not None and btc_direction != direction:
        reason = f"gate0: BTC MSS={btc_direction}, scanner direction={direction}"
        return _error_result(symbol, direction, reason)

    # 3. Repaint audit (on 4H data)
    audit_passed, audit_detail = audit_no_lookahead(df_4h)

    # 4. Backtest
    results = replay(
        df_4h=df_4h,
        df_1h=df_1h,
        df_5m=df_5m,
        direction=direction,
        df_btc_4h=None,  # direction already fixed for this symbol
    )
    metrics = compute(results)

    # 5. Live setup at last bar
    as_of = len(df_4h) - 1
    zones_4h = find_active_zones(df_4h, direction, config.HTF)
    rejection = find_rejection(df_4h, as_of, zones_4h)

    if rejection is None and df_1h is not None:
        zones_1h = find_active_zones(df_1h, direction, config.HTF_ALT)
        last_1h_idx = len(df_1h) - 1
        rejection = find_rejection(df_1h, last_1h_idx, zones_1h)

    # Gate 3: RSI + MACD advisory (uses 1H if available, else 4H)
    ref_df = df_1h if df_1h is not None else df_4h
    gate3_ok = gate3_passes(ref_df, direction)
    if not gate3_ok:
        print(f"  [Gate3] RSI/MACD advisory: not confirmed for {symbol} [{direction}]")

    live_setup: Setup | None = None
    if rejection is not None:
        zone = rejection.zone
        atr_val = atr_scalar(df_4h, config.ATR_PERIOD)
        sl_price = compute_sl(direction, rejection.shadow_extreme, atr_val)

        # Gate 4: 5m entry refinement (never cancels a confirmed rejection)
        if df_5m is not None:
            refined_entry = optimize_entry(df_5m, zone, direction, df_4h.index[-1])
        else:
            refined_entry = rejection.entry

        ref_for_tp = df_1h if df_1h is not None else df_4h
        tp_price = find_tp(direction, refined_entry, sl_price, ref_for_tp)
        if tp_price is not None:
            rr = compute_rr(refined_entry, sl_price, tp_price)
            if rr >= config.MIN_RR:
                live_setup = Setup(
                    entry_low=zone.zone_low,
                    entry_high=zone.zone_high,
                    target=tp_price,
                    stop=sl_price,
                    rr=rr,
                    direction=direction,
                    zone=zone,
                    grade=rejection.grade,
                )

    # 6. Signal
    signal = build(symbol, direction, live_setup, metrics, audit_passed)

    # 7. Report
    report = generate(
        symbol, direction, live_setup, metrics,
        (audit_passed, audit_detail), signal,
    )

    return {
        "symbol": symbol,
        "direction": direction,
        "signal": signal,
        "report": report,
        "timestamp": _ts(),
    }


def _load_scanner(btc_direction: "Direction | None") -> list[dict]:
    """Load symbol+direction list.

    Priority:
      1. scanner_output.json  — manual list with explicit directions per symbol.
      2. data/top_symbols.json — generated by download_history.py; uses BTC MSS
         direction for all alts (symbols without a confirmed direction are skipped
         if btc_direction is None).
    """
    # 1. Manual override
    if os.path.exists(_SCANNER_FILE):
        try:
            with open(_SCANNER_FILE) as f:
                items = json.load(f)
            if items:
                print(f"[scanner] loaded {len(items)} items from scanner_output.json")
                return items
        except Exception as exc:
            _log.error("%s failed to load scanner_output.json: %s", _ts(), exc)

    # 2. Auto-scan from downloaded top-N list
    if not os.path.exists(_TOP_SYMBOLS_FILE):
        _log.error("%s neither scanner_output.json nor top_symbols.json found", _ts())
        return []

    if btc_direction is None:
        print("[WARN] BTC direction undetermined — cannot assign direction to top-symbols scan. Skipping.")
        return []

    try:
        with open(_TOP_SYMBOLS_FILE) as f:
            symbols: list[str] = json.load(f)
    except Exception as exc:
        _log.error("%s failed to load top_symbols.json: %s", _ts(), exc)
        return []

    # Exclude BTC itself — it is the direction filter, not an alt signal
    alts = [s for s in symbols if s != config.BTC_SYMBOL]
    items = [{"symbol": s, "direction": btc_direction} for s in alts]
    print(
        f"[scanner] loaded {len(items)} alts from top_symbols.json "
        f"(BTC direction = {btc_direction.upper()})"
    )
    return items


def _write_signals(all_signals: list[dict]) -> None:
    exportable = [
        {
            "symbol": s["symbol"],
            "direction": s["direction"],
            "signal": s["signal"],
            "timestamp": s["timestamp"],
        }
        for s in all_signals
    ]
    os.makedirs(os.path.dirname(_SIGNALS_FILE), exist_ok=True)
    try:
        with open(_SIGNALS_FILE, "w") as f:
            json.dump(exportable, f, indent=2)
    except Exception as exc:
        _log.error("%s failed to write signals.json: %s", _ts(), exc)


def _error_result(symbol: str, direction: Direction, reason: str) -> dict:
    _log.error("%s %s [%s]: %s", _ts(), symbol, direction, reason)
    from backtest.metrics import compute
    empty_metrics = compute([])
    return {
        "symbol": symbol,
        "direction": direction,
        "signal": {"symbol": symbol, "direction": direction, "signal": None, "blocked_reason": reason},
        "report": f"ERROR — {symbol}: {reason}",
        "timestamp": _ts(),
    }


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
