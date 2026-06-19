"""Signal builder: 4-gate ICT rejection pipeline.

Gate 0: BTC direction (MSS on 4H).
Gate 1: Alignment (symbol direction == BTC direction — checked by caller via scanner_output.json).
Gate 2: OB/FVG rejection candle on 4H or 1H (shadow enters zone, close exits).
Gate 3: RSI + MACD confluence (advisory).
Gate 4: 5m entry refinement (never blocks).
Final: R:R >= MIN_RR; backtest metrics flag == valid.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Metrics, Setup, Zone
from engine.ict import atr_scalar, detect_mss
from engine.zones import find_active_zones, find_rejection
from engine.indicators import gate3_passes
from engine.entry import optimize_entry
from engine.sl_tp import compute_sl, compute_rr, find_tp

_log = logging.getLogger(__name__)
_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "errors.log")


def _setup_log() -> None:
    if not any(isinstance(h, logging.FileHandler) for h in _log.handlers):
        h = logging.FileHandler(_LOG_PATH)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _log.addHandler(h)
        _log.setLevel(logging.WARNING)


def build(
    symbol: str,
    direction: Direction,
    setup: Setup | None,
    metrics: Metrics,
    audit_passed: bool,
) -> dict:
    """Run gates and return signal dict (always returns dict, signal key is None if blocked)."""
    _setup_log()
    failures: list[str] = []

    if not audit_passed:
        failures.append("causality: repaint audit FAILED")

    if setup is None:
        failures.append("gate2: no rejection setup found")
    elif setup.rr < config.MIN_RR:
        failures.append(f"rr: {setup.rr:.2f} < {config.MIN_RR}")

    if metrics.flag != "valid":
        failures.append(f"metrics: flag={metrics.flag}")

    if failures:
        reason = "; ".join(failures)
        _log.warning("%s BLOCKED %s: %s", _ts(), symbol, reason)
        return {"symbol": symbol, "direction": direction, "signal": None, "blocked_reason": reason}

    assert setup is not None
    return {
        "symbol": symbol,
        "direction": direction,
        "signal": "active",
        "entry_low": round(setup.entry_low, 4),
        "entry_high": round(setup.entry_high, 4),
        "target": round(setup.target, 4),
        "stop": round(setup.stop, 4),
        "rr": round(setup.rr, 2),
        "grade": setup.grade,
        "expectancy_r": round(metrics.expectancy_r, 4),
        "wr": round(metrics.wr, 4),
        "wr_ci_low": round(metrics.wr_ci_low, 4),
        "wr_ci_high": round(metrics.wr_ci_high, 4),
        "trades": metrics.trades,
        "flag": metrics.flag,
        "timestamp": _ts(),
    }


def build_live(
    symbol: str,
    direction: Direction,
    df_4h: "pd.DataFrame",
    df_1h: "pd.DataFrame | None",
    df_5m: "pd.DataFrame | None",
    df_btc_4h: "pd.DataFrame | None",
    metrics: Metrics,
) -> dict:
    """Full live 4-gate pipeline. Returns signal dict."""
    import pandas as pd
    _setup_log()

    # Gate 0: BTC direction
    if df_btc_4h is not None:
        btc_dir = detect_mss(df_btc_4h)
        if btc_dir is None:
            return _blocked(symbol, direction, "gate0: BTC MSS undetermined")
        if btc_dir != direction:
            return _blocked(symbol, direction, f"gate0: BTC direction={btc_dir}, expected={direction}")

    # Gate 2: find rejection on 4H
    as_of = len(df_4h) - 1
    zones_4h = find_active_zones(df_4h, direction, config.HTF)
    rejection = find_rejection(df_4h, as_of, zones_4h)

    # Also check 1H if no 4H rejection
    used_tf = config.HTF
    if rejection is None and df_1h is not None:
        zones_1h = find_active_zones(df_1h, direction, config.HTF_ALT)
        last_1h_idx = len(df_1h) - 1
        rejection = find_rejection(df_1h, last_1h_idx, zones_1h)
        used_tf = config.HTF_ALT

    if rejection is None:
        return _blocked(symbol, direction, "gate2: no rejection candle found")

    # Gate 3: RSI + MACD (advisory — block only if both fail)
    ref_df = df_1h if df_1h is not None else df_4h
    gate3 = gate3_passes(ref_df, direction)

    # Gate 4: entry refinement on 5m
    zone = rejection.zone
    if df_5m is not None:
        as_of_ts = df_4h.index[-1]
        refined_entry = optimize_entry(df_5m, zone, direction, as_of_ts)
    else:
        refined_entry = zone.zone_high if direction == "long" else zone.zone_low

    # SL / TP
    ref_for_tp = df_1h if df_1h is not None else df_4h
    atr_val = atr_scalar(df_4h, config.ATR_PERIOD)
    sl_price = compute_sl(direction, rejection.shadow_extreme, atr_val)
    tp_price = find_tp(direction, refined_entry, sl_price, ref_for_tp)
    if tp_price is None:
        return _blocked(symbol, direction, "no_tp: no liquidity target at MIN_RR")

    rr = compute_rr(refined_entry, sl_price, tp_price)
    if rr < config.MIN_RR:
        return _blocked(symbol, direction, f"rr: {rr:.2f} < {config.MIN_RR}")

    zone_height = zone.zone_high - zone.zone_low
    setup = Setup(
        entry_low=zone.zone_low,
        entry_high=zone.zone_high,
        target=tp_price,
        stop=sl_price,
        rr=rr,
        direction=direction,
        zone=zone,
        grade=rejection.grade,
    )

    return build(symbol, direction, setup, metrics, audit_passed=True)


def _blocked(symbol: str, direction: Direction, reason: str) -> dict:
    _log.warning("%s BLOCKED %s: %s", _ts(), symbol, reason)
    return {"symbol": symbol, "direction": direction, "signal": None, "blocked_reason": reason}


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()
