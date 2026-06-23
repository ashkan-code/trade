"""Multi-TF pivot 101/102 replay.

Confluence rule:
  At each 4h bar i, compute predict_101_102 on {4h, 1h, 1d} independently.
  Signal requires ≥ 2 of 3 TFs returning the SAME direction AND their 101
  zones being within 1 ATR of each other (overlapping or very near).

Entry trigger:
  The current 4h bar's wick enters the merged 101 zone from the correct side.
  (For LONG: bar low ≤ zone_high; for SHORT: bar high ≥ zone_low.)

Entry price: next 4h bar open ± slippage (ATR × SLIPPAGE_ATR_MULT).
Stop: 4h predict_101_102 stop.
Target: longest-TF p102 (1d wins over 4h/1h when 1d agrees).
Management: 1h bars with intrabar pessimism (SL always wins on tie).
Costs: 2 × TAKER_FEE + FUNDING_RATE × bars_held (same as replay_core).

No Fibonacci. No hardcoded WR. No blacklist.
"""

from __future__ import annotations

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from contracts import Direction, TradeResult
from engine.pivots import find_pivots, adaptive_lookback, predict_101_102
from engine.ict import atr_scalar


def replay_pivot_mtf(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    direction: Direction | None = None,
) -> list[TradeResult]:
    """Bar-by-bar multi-TF pivot 101/102 replay.

    direction=None → direction inferred from swing structure at each bar.
    direction='long'/'short' → only accept setups in that direction.
    Returns list[TradeResult] with net R multiples after all costs.
    """
    results: list[TradeResult] = []
    n = len(df_4h)
    in_trade = False
    entry_price = sl = tp = 0.0
    entry_bar = 0
    direction_active: Direction = direction or "long"  # overwritten on each entry

    for i in range(config.WARMUP, n):
        ts = df_4h.index[i]

        # ── trade management ──────────────────────────────────────────────────
        if in_trade:
            if df_1h is not None:
                prev_ts = df_4h.index[i - 1]
                mgmt = df_1h[(df_1h.index > prev_ts) & (df_1h.index <= ts)]
            else:
                mgmt = df_4h.iloc[i : i + 1]

            for _, row in mgmt.iterrows():
                h = float(row["high"])
                l = float(row["low"])

                if direction_active == "long":
                    hit_sl = l <= sl
                    hit_tp = h >= tp
                else:
                    hit_sl = h >= sl
                    hit_tp = l <= tp

                if hit_sl and hit_tp:
                    exit_p, won = sl, False       # intrabar pessimism: SL wins
                elif hit_sl:
                    exit_p, won = sl, False
                elif hit_tp:
                    exit_p, won = tp, True
                else:
                    continue

                bars = i - entry_bar
                r = _r_gross(entry_price, exit_p, sl, direction_active)
                results.append(
                    TradeResult(won=won, r_multiple=r - _cost(bars), bars_held=bars)
                )
                in_trade = False
                break

        # ── setup detection (only when flat) ─────────────────────────────────
        if in_trade or i + 1 >= n:
            continue

        df_4h_v = df_4h.iloc[: i + 1]

        # 4h — required
        lb4   = adaptive_lookback(df_4h_v, i)
        piv4  = find_pivots(df_4h_v, i, lb4)
        pred4 = predict_101_102(piv4, i, df_4h_v)
        if pred4 is None:
            continue

        target_dir: Direction = direction if direction is not None else pred4["direction"]
        if pred4["direction"] != target_dir:
            continue

        # 1d — optional confirmation
        pred1d: dict | None = None
        if df_1d is not None:
            df_1d_v = df_1d[df_1d.index <= ts]
            if len(df_1d_v) > config.WARMUP:
                aoi1d  = len(df_1d_v) - 1
                lb1d   = adaptive_lookback(df_1d_v, aoi1d)
                piv1d  = find_pivots(df_1d_v, aoi1d, lb1d)
                p1d    = predict_101_102(piv1d, aoi1d, df_1d_v)
                if p1d is not None and p1d["direction"] == target_dir:
                    pred1d = p1d

        # 1h — optional confirmation
        pred1h: dict | None = None
        if df_1h is not None:
            df_1h_v = df_1h[df_1h.index <= ts]
            if len(df_1h_v) > config.WARMUP:
                aoi1h  = len(df_1h_v) - 1
                lb1h   = adaptive_lookback(df_1h_v, aoi1h)
                piv1h  = find_pivots(df_1h_v, aoi1h, lb1h)
                p1h    = predict_101_102(piv1h, aoi1h, df_1h_v)
                if p1h is not None and p1h["direction"] == target_dir:
                    pred1h = p1h

        # Need ≥ 2 TFs agreeing (4h is always one; need at least 1 more)
        conf_preds = [pred4]
        if pred1d is not None:
            conf_preds.append(pred1d)
        if pred1h is not None:
            conf_preds.append(pred1h)

        if len(conf_preds) < 2:
            continue

        # 101 zone proximity: any pair within 1 ATR
        atr_val = atr_scalar(df_4h_v, config.ATR_PERIOD)
        if not _zones_near(conf_preds, atr_val):
            continue

        # Merged zone: take 4h zone (the confirmed trigger zone)
        zone_low  = pred4["entry_low"]
        zone_high = pred4["entry_high"]

        # Entry trigger: bar wick must touch the 101 zone
        bar = df_4h.iloc[i]
        if target_dir == "long":
            if float(bar["low"]) > zone_high:
                continue
        else:
            if float(bar["high"]) < zone_low:
                continue

        # Target: 1d wins (deepest structure), else 4h
        final_tp = pred1d["target"] if pred1d is not None else pred4["target"]
        final_sl = pred4["stop"]

        # Re-check R:R with resolved stop/target
        ref_entry = zone_high if target_dir == "long" else zone_low
        risk = abs(ref_entry - final_sl)
        if risk <= 0:
            continue
        rr = abs(final_tp - ref_entry) / risk
        if rr < config.MIN_RR:
            continue

        # Entry: next 4h bar open ± slippage
        slip    = atr_val * config.SLIPPAGE_ATR_MULT
        next_o  = float(df_4h.iloc[i + 1]["open"])
        entry_price    = next_o + slip if target_dir == "long" else next_o - slip
        sl             = final_sl
        tp             = final_tp
        entry_bar      = i + 1
        in_trade       = True
        direction_active = target_dir

    return results


def replay_pivot_with_trace(
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame | None,
    df_1d: pd.DataFrame | None,
    direction: Direction | None = None,
    max_trace: int = 5,
) -> tuple[list[TradeResult], list[dict]]:
    """Same as replay_pivot_mtf but also returns trade traces (for debugging)."""
    results: list[TradeResult] = []
    traces: list[dict] = []
    n = len(df_4h)
    in_trade = False
    entry_price = sl = tp = 0.0
    entry_bar = 0
    direction_active: Direction = direction or "long"
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

                if direction_active == "long":
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

                bars    = i - entry_bar
                r_gross = _r_gross(entry_price, exit_p, sl, direction_active)
                cost    = _cost(bars)
                r_net   = r_gross - cost
                results.append(TradeResult(won=won, r_multiple=r_net, bars_held=bars))

                if len(traces) < max_trace:
                    pending_trace.update({
                        "exit_bar_4h": i,
                        "exit_price":  round(exit_p, 2),
                        "reason":      reason,
                        "bars_held":   bars,
                        "r_gross":     round(r_gross, 4),
                        "cost":        round(cost, 6),
                        "r_net":       round(r_net, 4),
                        "won":         won,
                    })
                    traces.append(pending_trace)

                in_trade = False
                break

        if in_trade or i + 1 >= n:
            continue

        df_4h_v = df_4h.iloc[: i + 1]
        lb4   = adaptive_lookback(df_4h_v, i)
        piv4  = find_pivots(df_4h_v, i, lb4)
        pred4 = predict_101_102(piv4, i, df_4h_v)
        if pred4 is None:
            continue

        target_dir = direction if direction is not None else pred4["direction"]
        if pred4["direction"] != target_dir:
            continue

        pred1d = pred1h = None

        if df_1d is not None:
            df_1d_v = df_1d[df_1d.index <= ts]
            if len(df_1d_v) > config.WARMUP:
                aoi1d = len(df_1d_v) - 1
                p1d   = predict_101_102(
                    find_pivots(df_1d_v, aoi1d, adaptive_lookback(df_1d_v, aoi1d)),
                    aoi1d, df_1d_v,
                )
                if p1d is not None and p1d["direction"] == target_dir:
                    pred1d = p1d

        if df_1h is not None:
            df_1h_v = df_1h[df_1h.index <= ts]
            if len(df_1h_v) > config.WARMUP:
                aoi1h = len(df_1h_v) - 1
                p1h   = predict_101_102(
                    find_pivots(df_1h_v, aoi1h, adaptive_lookback(df_1h_v, aoi1h)),
                    aoi1h, df_1h_v,
                )
                if p1h is not None and p1h["direction"] == target_dir:
                    pred1h = p1h

        conf_preds = [pred4]
        if pred1d is not None:
            conf_preds.append(pred1d)
        if pred1h is not None:
            conf_preds.append(pred1h)

        if len(conf_preds) < 2:
            continue

        atr_val = atr_scalar(df_4h_v, config.ATR_PERIOD)
        if not _zones_near(conf_preds, atr_val):
            continue

        zone_low  = pred4["entry_low"]
        zone_high = pred4["entry_high"]

        bar = df_4h.iloc[i]
        if target_dir == "long":
            if float(bar["low"]) > zone_high:
                continue
        else:
            if float(bar["high"]) < zone_low:
                continue

        final_tp  = pred1d["target"] if pred1d is not None else pred4["target"]
        final_sl  = pred4["stop"]
        ref_entry = zone_high if target_dir == "long" else zone_low
        risk      = abs(ref_entry - final_sl)
        if risk <= 0:
            continue
        rr = abs(final_tp - ref_entry) / risk
        if rr < config.MIN_RR:
            continue

        slip        = atr_val * config.SLIPPAGE_ATR_MULT
        next_o      = float(df_4h.iloc[i + 1]["open"])
        entry_price = next_o + slip if target_dir == "long" else next_o - slip
        sl          = final_sl
        tp          = final_tp
        entry_bar   = i + 1
        in_trade    = True
        direction_active = target_dir

        if len(traces) < max_trace:
            conf_label = (
                "4h+1d" if pred1d is not None and pred1h is None else
                "4h+1h" if pred1h is not None and pred1d is None else
                "4h+1d+1h" if pred1d is not None and pred1h is not None else
                "4h-only"
            )
            pending_trace = {
                "signal_bar_4h": i,
                "signal_ts":     str(ts),
                "direction":     target_dir,
                "conf_tfs":      conf_label,
                "entry_zone":    (round(zone_low, 2), round(zone_high, 2)),
                "target":        round(final_tp, 2),
                "stop":          round(final_sl, 2),
                "rr_planned":    round(rr, 2),
                "entry_bar_4h":  i + 1,
                "entry_price":   round(entry_price, 2),
                "sl":            round(sl, 2),
                "tp":            round(tp, 2),
            }

    return results, traces


# ── helpers ───────────────────────────────────────────────────────────────────

def _zones_near(preds: list[dict], atr_val: float) -> bool:
    """True if any pair of entry zones overlap or are within 1 ATR of each other."""
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            p1, p2 = preds[i], preds[j]
            gap = max(p1["entry_low"], p2["entry_low"]) - min(p1["entry_high"], p2["entry_high"])
            if gap <= atr_val:
                return True
    return False


def _r_gross(entry: float, exit_p: float, stop: float, direction: Direction) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    pnl = (exit_p - entry) if direction == "long" else (entry - exit_p)
    return pnl / risk


def _cost(bars_held: int) -> float:
    return 2 * config.TAKER_FEE + config.FUNDING_RATE * bars_held
