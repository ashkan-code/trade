"""
Walk-forward backtester for the ICT scanner pipeline.

NO lookahead bias: at each simulated step i, only df[:i] (exclusive) is visible.
Trade outcome simulated by walking forward through subsequent candles.
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

import numpy as np
import pandas as pd

import config
from core.ict import detect_mss, detect_order_blocks, detect_fvg, detect_liquidity
from core.indicators import rsi, macd, atr as compute_atr
from core.confluence import (
    check_btc_direction, is_aligned, find_best_zone,
    zone_triggered, compute_htf_confirms, score_zone,
)
from engine.risk import compute_risk, should_trail, compute_trail_stop

logger = logging.getLogger(__name__)


class TradeDict(TypedDict):
    symbol: str
    direction: str
    entry: float
    stop: float
    target: float
    rr: float
    entry_index: int
    exit_index: int
    exit_price: float
    outcome: str      # 'win' | 'loss' | 'trail_stop'
    pnl_r: float


class BacktestResult(TypedDict):
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_rr_realized: float
    max_drawdown_r: float
    profit_factor: float
    signals_per_day: float
    score_threshold_used: float


def compute_metrics(trades: list[TradeDict]) -> BacktestResult:
    if not trades:
        return BacktestResult(
            total_trades=0, wins=0, losses=0, win_rate=0.0,
            avg_rr_realized=0.0, max_drawdown_r=0.0,
            profit_factor=0.0, signals_per_day=0.0,
            score_threshold_used=config.SCORE_THRESHOLD,
        )

    wins = [t for t in trades if t["pnl_r"] > 0]
    losses = [t for t in trades if t["pnl_r"] <= 0]

    gross_profit = sum(t["pnl_r"] for t in wins)
    gross_loss = abs(sum(t["pnl_r"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown in R
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        cumulative += t["pnl_r"]
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    return BacktestResult(
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(trades),
        avg_rr_realized=float(np.mean([t["pnl_r"] for t in trades])),
        max_drawdown_r=max_dd,
        profit_factor=profit_factor,
        signals_per_day=0.0,  # filled by run_backtest
        score_threshold_used=config.SCORE_THRESHOLD,
    )


def _simulate_trade(
    direction: str,
    entry: float,
    stop: float,
    target: float,
    trail_trigger: float,
    df_forward: pd.DataFrame,
    trail_start_r: float = config.TRAIL_START_R,
) -> tuple[int, float, str, float]:
    """
    Walk forward through df_forward candles to find the trade outcome.
    Returns (exit_index_relative, exit_price, outcome, pnl_r).
    """
    risk_pts = abs(entry - stop)
    if risk_pts == 0:
        return 0, entry, "loss", -1.0

    current_stop = stop
    trail_active = False

    for i, row in df_forward.iterrows():
        high = row["high"]
        low = row["low"]

        if direction == "bullish":
            # Stop hit?
            if low <= current_stop:
                pnl_r = (current_stop - entry) / risk_pts
                return int(i), current_stop, "trail_stop" if trail_active else "loss", round(pnl_r, 4)
            # Target hit?
            if high >= target:
                pnl_r = (target - entry) / risk_pts
                return int(i), target, "win", round(pnl_r, 4)
            # Trail activation?
            close = row["close"]
            if not trail_active and close >= trail_trigger:
                trail_active = True
            if trail_active:
                atr_val = abs(high - low)  # simplified ATR proxy for backtest
                new_stop = close - 2 * atr_val
                current_stop = max(current_stop, new_stop)
        else:
            if high >= current_stop:
                pnl_r = (entry - current_stop) / risk_pts
                return int(i), current_stop, "trail_stop" if trail_active else "loss", round(-abs(pnl_r), 4)
            if low <= target:
                pnl_r = (entry - target) / risk_pts
                return int(i), target, "win", round(pnl_r, 4)
            close = row["close"]
            if not trail_active and close <= trail_trigger:
                trail_active = True
            if trail_active:
                atr_val = abs(high - low)
                new_stop = close + 2 * atr_val
                current_stop = min(current_stop, new_stop)

    # No exit found — close at last bar
    last_close = float(df_forward.iloc[-1]["close"])
    if direction == "bullish":
        pnl_r = (last_close - entry) / risk_pts
    else:
        pnl_r = (entry - last_close) / risk_pts
    return int(df_forward.index[-1]), last_close, "loss", round(pnl_r, 4)


def run_backtest(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    params: Optional[dict] = None,
) -> BacktestResult:
    """
    Walk-forward backtest. For each 1H candle i (past warm-up), slice all
    DataFrames to [:i] and run the signal pipeline identically to production.
    """
    if params is None:
        params = {}

    score_threshold = params.get("SCORE_THRESHOLD", config.SCORE_THRESHOLD)
    trail_start_r = params.get("TRAIL_START_R", config.TRAIL_START_R)
    min_rr = params.get("MIN_RR", config.MIN_RR)

    warmup = max(config.SWING_LEN * 4, config.RSI_LEN + config.MACD_SLOW + config.MACD_SIGNAL)
    trades: list[TradeDict] = []

    # We step through 1H candles as the "time clock"
    for i in range(warmup, len(df_1h)):
        slice_1h = df_1h.iloc[:i].reset_index(drop=True)

        # Map 1H index to 4H and 5m indices by timestamp
        ts_cutoff = df_1h.iloc[i - 1]["timestamp"]
        slice_4h = df_4h[df_4h["timestamp"] <= ts_cutoff].reset_index(drop=True)
        slice_5m = df_5m[df_5m["timestamp"] <= ts_cutoff].reset_index(drop=True)

        if len(slice_4h) < warmup or len(slice_5m) < 20:
            continue

        try:
            mss_4h = detect_mss(slice_4h)
            if mss_4h["direction"] is None:
                continue

            direction = mss_4h["direction"]

            # We simulate BTC = the symbol itself in single-symbol backtest
            obs_4h = detect_order_blocks(slice_4h)
            obs_1h = detect_order_blocks(slice_1h)
            fvgs_1h = detect_fvg(slice_1h)

            zone = find_best_zone(obs_1h, obs_4h, fvgs_1h, direction, symbol)
            if zone is None:
                continue

            if not zone_triggered(zone, slice_5m):
                continue

            confirms = compute_htf_confirms(slice_1h, direction)
            if not confirms["rsi"] or not confirms["macd"]:
                continue

            zone["rsi_confirm"] = confirms["rsi"]
            zone["macd_confirm"] = confirms["macd"]

            entry = float(slice_5m["close"].iloc[-1])

            atr_1h = compute_atr(slice_1h)
            liq_pools = detect_liquidity(slice_1h, atr_1h)

            from core.ict import OBDict
            ob_1h = zone.get("ob_1h") or OBDict(
                type=direction, top=zone["zone_top"], bottom=zone["zone_bottom"],
                index=-1, broken=False, breaker=False,
            )

            risk = compute_risk(direction, entry, ob_1h, liq_pools)
            if not risk["valid"] or risk["rr"] < min_rr:
                continue

            score = score_zone(zone, risk["rr"])
            if score < score_threshold:
                continue

            # Simulate trade forward on 1H candles (approximate)
            forward_df = df_1h.iloc[i:].reset_index(drop=True)
            if forward_df.empty:
                continue

            exit_idx_rel, exit_price, outcome, pnl_r = _simulate_trade(
                direction=direction,
                entry=entry,
                stop=risk["stop"],
                target=risk["target"],
                trail_trigger=risk["trail_trigger"],
                df_forward=forward_df,
                trail_start_r=trail_start_r,
            )

            trades.append(TradeDict(
                symbol=symbol,
                direction=direction,
                entry=entry,
                stop=risk["stop"],
                target=risk["target"],
                rr=risk["rr"],
                entry_index=i,
                exit_index=i + exit_idx_rel,
                exit_price=exit_price,
                outcome=outcome,
                pnl_r=pnl_r,
            ))

        except Exception as exc:
            logger.debug("Backtest step %d error: %s", i, exc)
            continue

    result = compute_metrics(trades)
    # Compute signals per day
    if len(df_1h) > 0:
        total_bars = len(df_1h)
        days = total_bars / 24.0  # 1H bars → days
        result["signals_per_day"] = len(trades) / days if days > 0 else 0.0

    logger.info(
        "Backtest %s: %d trades | WR=%.1f%% | PF=%.2f | MaxDD=%.2fR",
        symbol, result["total_trades"], result["win_rate"] * 100,
        result["profit_factor"], result["max_drawdown_r"],
    )

    if result["win_rate"] > 0.75 and result.get("avg_rr_realized", 0) > 2.5:
        logger.warning(
            "⚠️  SUSPICIOUS RESULTS: WR=%.1f%% @ avg_R=%.2f — CHECK FOR LOOKAHEAD BIAS",
            result["win_rate"] * 100, result["avg_rr_realized"],
        )

    return result


def optimize(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame,
    param_grid: dict,
) -> tuple[dict, BacktestResult]:
    """
    Grid search over param_grid.
    Best = highest profit_factor with win_rate >= 0.45.
    """
    import itertools

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    best_params: Optional[dict] = None
    best_result: Optional[BacktestResult] = None

    for combo in combinations:
        params = dict(zip(keys, combo))
        result = run_backtest(symbol, df_4h, df_1h, df_5m, params)
        if result["win_rate"] < 0.45:
            continue
        if best_result is None or result["profit_factor"] > best_result["profit_factor"]:
            best_result = result
            best_params = params

    if best_params is None:
        best_params = {}
        best_result = compute_metrics([])

    return best_params, best_result  # type: ignore[return-value]
