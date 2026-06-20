"""Multi-timeframe 101/102 pivot replay.

Entry logic (4h primary, 1h/1d confirmation):
  1. At each 4h bar i (as_of = i):
       a. Causal pivots on 4h  → find_setup → setup_4h  (REQUIRED)
       b. Causal pivots on 1h  → find_setup → setup_1h  (optional)
       c. Causal pivots on 1d  → find_setup → setup_1d  (optional)
  2. Confluence: setup_4h AND (setup_1d overlaps OR setup_1h overlaps).
       Overlap = entry zones share any price range.
  3. Target (102): longest-TF setup wins (1d > 4h).
  4. Entry: next 4h bar open ± slippage.
  5. Management: 1h bars (or 4h if unavailable), intrabar pessimism enforced.

Costs (deducted from every TradeResult.r_multiple):
  2 × TAKER_FEE  +  FUNDING_RATE × bars_held_in_4h_units
"""

from __future__ import annotations

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Setup, TradeResult
from engine.pivots import find_pivots, adaptive_lookback
from engine.structure import find_setup
from engine.ict import atr_scalar


def replay_mtf(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    direction: Direction,
) -> list[TradeResult]:
    """Bar-by-bar multi-TF replay. Returns list[TradeResult] with net R multiples."""
    results: list[TradeResult] = []
    n = len(df_4h)
    in_trade = False
    entry_price = sl = tp = 0.0
    entry_bar = 0

    for i in range(config.WARMUP, n):
        ts = df_4h.index[i]

        # ── trade management ──────────────────────────────────────────────────
        if in_trade:
            if df_1h is not None:
                prev_ts = df_4h.index[i - 1]
                mgmt = df_1h[(df_1h.index > prev_ts) & (df_1h.index <= ts)]
            else:
                mgmt = df_4h.iloc[i : i + 1]

            exited = False
            for _, row in mgmt.iterrows():
                h = float(row["high"])
                l = float(row["low"])

                if direction == "long":
                    hit_sl = l <= sl
                    hit_tp = h >= tp
                else:
                    hit_sl = h >= sl
                    hit_tp = l <= tp

                if hit_sl and hit_tp:
                    # Intrabar pessimism: SL always wins
                    exit_p = sl
                    won = False
                elif hit_sl:
                    exit_p = sl
                    won = False
                elif hit_tp:
                    exit_p = tp
                    won = True
                else:
                    continue

                bars = i - entry_bar
                r = _r_gross(entry_price, exit_p, sl, direction)
                results.append(TradeResult(
                    won=won, r_multiple=r - _cost(bars), bars_held=bars
                ))
                in_trade = False
                exited = True
                break

        # ── setup detection (only when flat) ─────────────────────────────────
        if not in_trade and i + 1 < n:
            df_4h_v = df_4h.iloc[: i + 1]

            # 4h: required
            lb4 = adaptive_lookback(df_4h_v, i)
            piv4 = find_pivots(df_4h_v, i, lb4)
            s4 = find_setup(df_4h_v, i, piv4, direction)
            if s4 is None:
                continue

            # 1d: optional confirmation
            s1d: Setup | None = None
            if df_1d is not None:
                df_1d_v = df_1d[df_1d.index <= ts]
                if len(df_1d_v) > config.WARMUP:
                    aoi1d = len(df_1d_v) - 1
                    lb1d = adaptive_lookback(df_1d_v, aoi1d)
                    piv1d = find_pivots(df_1d_v, aoi1d, lb1d)
                    s1d = find_setup(df_1d_v, aoi1d, piv1d, direction)

            # 1h: optional confirmation
            s1h: Setup | None = None
            if df_1h is not None:
                df_1h_v = df_1h[df_1h.index <= ts]
                if len(df_1h_v) > config.WARMUP:
                    aoi1h = len(df_1h_v) - 1
                    lb1h = adaptive_lookback(df_1h_v, aoi1h)
                    piv1h = find_pivots(df_1h_v, aoi1h, lb1h)
                    s1h = find_setup(df_1h_v, aoi1h, piv1h, direction)

            # Confluence: prefer 1d, fall back to 1h
            conf: Setup | None = None
            if s1d is not None and _zones_overlap(s4, s1d):
                conf = s1d
            elif s1h is not None and _zones_overlap(s4, s1h):
                conf = s1h

            if conf is None:
                continue

            # Target: longest TF (1d > 4h)
            if conf is s1d:
                final_target = conf.target
            else:
                # 1h confirms: take whichever target is farther (more structural)
                if direction == "long":
                    final_target = max(s4.target, conf.target)
                else:
                    final_target = min(s4.target, conf.target)

            final_stop = s4.stop

            # Re-check R:R with resolved target
            risk = abs(s4.entry_high - final_stop)
            if risk <= 0:
                continue
            if abs(final_target - s4.entry_high) / risk < config.MIN_RR:
                continue

            # Entry: next 4h bar open + slippage
            atr_val = atr_scalar(df_4h_v, config.ATR_PERIOD)
            slip = atr_val * config.SLIPPAGE_ATR_MULT
            next_o = float(df_4h.iloc[i + 1]["open"])
            if direction == "long":
                entry_price = next_o + slip
            else:
                entry_price = next_o - slip

            sl = final_stop
            tp = final_target
            entry_bar = i + 1
            in_trade = True

    return results


def replay_with_trace(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    direction: Direction,
    max_trace: int = 5,
) -> tuple[list[TradeResult], list[dict]]:
    """Same as replay_mtf but also returns trade traces (for debugging)."""
    results: list[TradeResult] = []
    traces: list[dict] = []
    n = len(df_4h)
    in_trade = False
    entry_price = sl = tp = 0.0
    entry_bar = 0
    pending_trace: dict = {}

    for i in range(config.WARMUP, n):
        ts = df_4h.index[i]

        if in_trade:
            if df_1h is not None:
                prev_ts = df_4h.index[i - 1]
                mgmt = df_1h[(df_1h.index > prev_ts) & (df_1h.index <= ts)]
            else:
                mgmt = df_4h.iloc[i : i + 1]

            for _, row in mgmt.iterrows():
                h = float(row["high"])
                l = float(row["low"])

                if direction == "long":
                    hit_sl = l <= sl
                    hit_tp = h >= tp
                else:
                    hit_sl = h >= sl
                    hit_tp = l <= tp

                if hit_sl and hit_tp:
                    exit_p, won, reason = sl, False, "intrabar_pessimism(SL)"
                elif hit_sl:
                    exit_p, won, reason = sl, False, "SL"
                elif hit_tp:
                    exit_p, won, reason = tp, True, "TP"
                else:
                    continue

                bars = i - entry_bar
                r_gross = _r_gross(entry_price, exit_p, sl, direction)
                cost = _cost(bars)
                r_net = r_gross - cost
                tr = TradeResult(won=won, r_multiple=r_net, bars_held=bars)
                results.append(tr)

                if len(traces) < max_trace:
                    pending_trace.update({
                        "exit_bar_4h": i,
                        "exit_price": exit_p,
                        "reason": reason,
                        "bars_held": bars,
                        "r_gross": round(r_gross, 4),
                        "cost": round(cost, 6),
                        "r_net": round(r_net, 4),
                        "won": won,
                    })
                    traces.append(pending_trace)

                in_trade = False
                break

        if not in_trade and i + 1 < n:
            df_4h_v = df_4h.iloc[: i + 1]
            lb4 = adaptive_lookback(df_4h_v, i)
            piv4 = find_pivots(df_4h_v, i, lb4)
            s4 = find_setup(df_4h_v, i, piv4, direction)
            if s4 is None:
                continue

            s1d = s1h = None
            if df_1d is not None:
                df_1d_v = df_1d[df_1d.index <= ts]
                if len(df_1d_v) > config.WARMUP:
                    aoi1d = len(df_1d_v) - 1
                    piv1d = find_pivots(df_1d_v, aoi1d, adaptive_lookback(df_1d_v, aoi1d))
                    s1d = find_setup(df_1d_v, aoi1d, piv1d, direction)

            if df_1h is not None:
                df_1h_v = df_1h[df_1h.index <= ts]
                if len(df_1h_v) > config.WARMUP:
                    aoi1h = len(df_1h_v) - 1
                    piv1h = find_pivots(df_1h_v, aoi1h, adaptive_lookback(df_1h_v, aoi1h))
                    s1h = find_setup(df_1h_v, aoi1h, piv1h, direction)

            conf = None
            conf_tf = ""
            if s1d is not None and _zones_overlap(s4, s1d):
                conf, conf_tf = s1d, "1d"
            elif s1h is not None and _zones_overlap(s4, s1h):
                conf, conf_tf = s1h, "1h"

            if conf is None:
                continue

            final_target = (conf.target if conf is s1d
                            else (max(s4.target, conf.target) if direction == "long"
                                  else min(s4.target, conf.target)))
            final_stop = s4.stop

            risk = abs(s4.entry_high - final_stop)
            if risk <= 0:
                continue
            rr = abs(final_target - s4.entry_high) / risk
            if rr < config.MIN_RR:
                continue

            atr_val = atr_scalar(df_4h_v, config.ATR_PERIOD)
            slip = atr_val * config.SLIPPAGE_ATR_MULT
            next_o = float(df_4h.iloc[i + 1]["open"])
            entry_price = next_o + slip if direction == "long" else next_o - slip

            sl = final_stop
            tp = final_target
            entry_bar = i + 1
            in_trade = True

            if len(traces) < max_trace:
                pending_trace = {
                    "signal_bar_4h": i,
                    "signal_ts": str(ts),
                    "entry_bar_4h": i + 1,
                    "entry_price": round(entry_price, 2),
                    "sl": round(sl, 2),
                    "tp": round(tp, 2),
                    "rr_planned": round(rr, 2),
                    "conf_tf": conf_tf,
                    "setup_4h_zone": (round(s4.entry_low, 2), round(s4.entry_high, 2)),
                    "lb_4h": lb4,
                    "pivot_count_4h": len(piv4),
                }

    return results, traces


# ── helpers ───────────────────────────────────────────────────────────────────

def _zones_overlap(s1: Setup, s2: Setup) -> bool:
    """Two entry zones overlap if they share any price range."""
    return max(s1.entry_low, s2.entry_low) < min(s1.entry_high, s2.entry_high)


def _r_gross(entry: float, exit_p: float, stop: float, direction: Direction) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    pnl = (exit_p - entry) if direction == "long" else (entry - exit_p)
    return pnl / risk


def _cost(bars_held: int) -> float:
    return 2 * config.TAKER_FEE + config.FUNDING_RATE * bars_held
