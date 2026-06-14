"""Pivot Engine entry point.

Reads scanner_output.json → per-symbol: fetch → validate → backtest → live setup →
gates → report. Writes signals to logs/signals.json.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Allow running as `python main.py` from pivot-engine/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from backtest.metrics import compute
from backtest.repaint_audit import audit_no_lookahead
from backtest.simulator import replay
from contracts import Direction
from data.fetcher import fetch_ohlcv
from data.validator import check_1
from engine.pivots import adaptive_lookback, find_pivots
from engine.structure import find_setup
from report.generator import generate
from signal.builder import build

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCANNER_FILE = os.path.join(_BASE, "scanner_output.json")
_SIGNALS_FILE = os.path.join(_BASE, "logs", "signals.json")
_ERROR_LOG = os.path.join(_BASE, "logs", "errors.log")

logging.basicConfig(
    filename=_ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)
_log = logging.getLogger(__name__)


def main() -> None:
    scanner_items = _load_scanner()
    if not scanner_items:
        print("No scanner items found — check scanner_output.json")
        return

    all_signals: list[dict] = []
    for item in scanner_items:
        symbol: str = item["symbol"]
        direction: Direction = item["direction"]
        print(f"\n{'─'*60}")
        print(f"Processing {symbol} [{direction}]")
        result = _process(symbol, direction)
        all_signals.append(result)
        print(result["report"])

    _write_signals(all_signals)
    print(f"\nSignals written to {_SIGNALS_FILE}")


def _process(symbol: str, direction: Direction) -> dict:
    # 1. Fetch
    df = fetch_ohlcv(symbol, config.EXECUTION_TF, limit=600)
    if df is None:
        return _error_result(symbol, direction, "fetch failed")

    # 2. Validate
    ok, reason = check_1(df)
    if not ok:
        return _error_result(symbol, direction, f"check_1 failed: {reason}")

    # 3. Repaint audit
    audit_passed, audit_detail = audit_no_lookahead(df)

    # 4. Backtest
    results = replay(df, direction)
    metrics = compute(results)

    # 5. Live setup (as_of = last candle)
    as_of = len(df) - 1
    lb = adaptive_lookback(df, as_of)
    pivots = find_pivots(df, as_of=as_of, lookback=lb)
    live_setup = find_setup(df, as_of=as_of, pivots=pivots, direction=direction)

    # 6. Gate check + signal
    signal = build(symbol, direction, live_setup, metrics, audit_passed)

    # 7. Report
    report = generate(symbol, direction, live_setup, metrics, (audit_passed, audit_detail), signal)

    return {
        "symbol": symbol,
        "direction": direction,
        "signal": signal,
        "report": report,
        "timestamp": _ts(),
    }


def _load_scanner() -> list[dict]:
    try:
        with open(_SCANNER_FILE) as f:
            return json.load(f)
    except Exception as exc:
        _log.error("%s failed to load scanner_output.json: %s", _ts(), exc)
        return []


def _write_signals(all_signals: list[dict]) -> None:
    exportable = [
        {"symbol": s["symbol"], "direction": s["direction"], "signal": s["signal"], "timestamp": s["timestamp"]}
        for s in all_signals
    ]
    try:
        with open(_SIGNALS_FILE, "w") as f:
            json.dump(exportable, f, indent=2)
    except Exception as exc:
        _log.error("%s failed to write signals.json: %s", _ts(), exc)


def _error_result(symbol: str, direction: str, reason: str) -> dict:
    _log.error("%s %s [%s]: %s", _ts(), symbol, direction, reason)
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
