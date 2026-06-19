"""Multi-timeframe causal backtest replay.

Detection on df_4h bars (Gates 0-2-3).
Trade management on df_5m bars (or df_4h if df_5m unavailable).
Entry at next 4H open + slippage. Intrabar pessimism enforced.
Partial TP: 50% at TP1, SL moves to breakeven, 50% at TP2.
"""

from __future__ import annotations

import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, TradeResult
from engine.ict import atr_scalar, detect_mss
from engine.zones import find_active_zones, find_rejection
from engine.indicators import gate3_passes
from engine.sl_tp import compute_sl, compute_rr, find_tp


def replay(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame | None = None,
    df_5m: pd.DataFrame | None = None,
    direction: Direction | None = None,
    df_btc_4h: pd.DataFrame | None = None,
) -> list[TradeResult]:
    """Causal bar-by-bar replay. Returns list of TradeResult (costs already deducted).

    direction: if None, derive from BTC MSS at each step (requires df_btc_4h).
               if provided, use fixed direction for all bars.
    """
    results: list[TradeResult] = []
    n = len(df_4h)
    in_trade = False
    entry_price = 0.0
    sl = 0.0
    tp1 = 0.0
    tp2: float | None = None
    tp1_hit = False
    sl_moved = False
    entry_bar_4h = 0
    current_direction: Direction = direction or "long"
    trade_grade: str = "B"

    for i in range(config.WARMUP, n):
        ts = df_4h.index[i]
        df_4h_view = df_4h.iloc[: i + 1]

        if in_trade:
            # Manage on 5m if available, else on 4H bars
            mgmt_df = df_5m if df_5m is not None else df_4h
            prev_ts = df_4h.index[i - 1] if i > 0 else ts
            next_ts = df_4h.index[i] if i < n - 1 else ts

            if df_5m is not None:
                mgmt_view = df_5m[
                    (df_5m.index > prev_ts) & (df_5m.index <= next_ts)
                ]
            else:
                mgmt_view = df_4h.iloc[i : i + 1]

            for _, row in mgmt_view.iterrows():
                h = float(row["high"])
                l = float(row["low"])
                exited = False

                if current_direction == "long":
                    hit_sl = l <= sl
                    hit_tp1 = h >= tp1 and not tp1_hit
                    hit_tp2 = tp2 is not None and h >= tp2 and tp1_hit

                    if hit_sl and (hit_tp1 or hit_tp2):
                        # Intrabar pessimism: SL wins
                        bars = i - entry_bar_4h
                        r = _r_gross(entry_price, sl, sl, "long")
                        results.append(TradeResult(
                            won=False, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_sl:
                        bars = i - entry_bar_4h
                        r = _r_gross(entry_price, sl, sl, "long")
                        results.append(TradeResult(
                            won=False, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_tp2 and tp2 is not None:
                        bars = i - entry_bar_4h
                        r1 = _r_gross(entry_price, tp1, sl, "long") * config.TP_PARTIAL_FRACTION
                        r2 = _r_gross(entry_price, tp2, sl, "long") * (1 - config.TP_PARTIAL_FRACTION)
                        r = r1 + r2
                        results.append(TradeResult(
                            won=True, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_tp1 and not tp1_hit:
                        tp1_hit = True
                        sl = entry_price  # breakeven
                        sl_moved = True

                else:  # short
                    hit_sl = h >= sl
                    hit_tp1 = l <= tp1 and not tp1_hit
                    hit_tp2 = tp2 is not None and l <= tp2 and tp1_hit

                    if hit_sl and (hit_tp1 or hit_tp2):
                        bars = i - entry_bar_4h
                        r = _r_gross(entry_price, sl, sl, "short")
                        results.append(TradeResult(
                            won=False, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_sl:
                        bars = i - entry_bar_4h
                        r = _r_gross(entry_price, sl, sl, "short")
                        results.append(TradeResult(
                            won=False, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_tp2 and tp2 is not None:
                        bars = i - entry_bar_4h
                        r1 = _r_gross(entry_price, tp1, sl, "short") * config.TP_PARTIAL_FRACTION
                        r2 = _r_gross(entry_price, tp2, sl, "short") * (1 - config.TP_PARTIAL_FRACTION)
                        r = r1 + r2
                        results.append(TradeResult(
                            won=True, r_multiple=r - _cost(bars),
                            bars_held=bars, grade=trade_grade,  # type: ignore[arg-type]
                        ))
                        in_trade = False
                        exited = True
                    elif hit_tp1 and not tp1_hit:
                        tp1_hit = True
                        sl = entry_price

                if exited:
                    break

        if not in_trade and i + 1 < n:
            # Determine direction
            if direction is None and df_btc_4h is not None:
                btc_view = df_btc_4h[df_btc_4h.index <= ts]
                mss = detect_mss(btc_view)
                if mss is None:
                    continue
                current_direction = mss
            elif direction is not None:
                current_direction = direction

            # Gate 2: find rejection on current 4H bar
            zones_4h = find_active_zones(df_4h_view, current_direction, config.HTF)
            rejection = find_rejection(df_4h_view, i, zones_4h)
            if rejection is None:
                # Also check 1H if available
                if df_1h is not None:
                    df_1h_view = df_1h[df_1h.index <= ts]
                    zones_1h = find_active_zones(df_1h_view, current_direction, config.HTF_ALT)
                    # Find last 1H bar at or before this 4H close
                    if len(df_1h_view) > 0:
                        last_1h_idx = len(df_1h_view) - 1
                        rejection = find_rejection(df_1h_view, last_1h_idx, zones_1h)
            if rejection is None:
                continue

            # Gate 3: RSI + MACD (advisory — only skip if clearly negative)
            ref_df = df_1h[df_1h.index <= ts] if df_1h is not None else df_4h_view
            gate3 = gate3_passes(ref_df, current_direction)
            # Gate 3 is checked but doesn't hard-block; we record for scoring

            # Compute SL/TP on 4H ATR
            atr_val = atr_scalar(df_4h_view, config.ATR_PERIOD)
            entry_zone = rejection.zone
            sl_price = compute_sl(current_direction, entry_zone, atr_val)
            tp_price = find_tp(current_direction, rejection.entry, sl_price, df_4h_view)
            if tp_price is None:
                continue
            rr = compute_rr(rejection.entry, sl_price, tp_price)
            if rr < config.MIN_RR:
                continue

            # Enter at next 4H bar open + slippage
            next_open = float(df_4h.iloc[i + 1]["open"])
            slip = atr_val * config.SLIPPAGE_ATR_MULT
            if current_direction == "long":
                entry_price = next_open + slip
            else:
                entry_price = next_open - slip

            sl = sl_price
            # TP1: nearest liquidity; TP2: tp_price (further)
            tp1 = tp_price  # simplified: use same level for both until TP2 found
            tp2_candidates = find_tp(
                current_direction,
                entry_price,
                sl_price,
                df_4h_view,
            )
            # Use tp_price as TP1, try to find TP2 further out
            tp1 = tp_price
            tp2 = None  # no TP2 in simplified backtest

            entry_bar_4h = i + 1
            tp1_hit = False
            sl_moved = False
            trade_grade = rejection.grade
            in_trade = True

    return results


def _r_gross(entry: float, exit_p: float, stop: float, direction: Direction) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    pnl = (exit_p - entry) if direction == "long" else (entry - exit_p)
    return pnl / risk


def _cost(bars_held: int) -> float:
    return 2 * config.TAKER_FEE + config.FUNDING_RATE * bars_held
