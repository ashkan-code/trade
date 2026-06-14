"""Causal backtest replay. Cursor advances one candle at a time, as_of=i always."""

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Direction, Setup, TradeResult
from engine.pivots import adaptive_lookback, find_pivots
from engine.structure import find_setup


def replay(df: pd.DataFrame, direction: Direction) -> list[TradeResult]:
    """Replay full DataFrame causal bar-by-bar. Returns list of TradeResult (net costs)."""
    results: list[TradeResult] = []
    n = len(df)
    in_trade = False
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    entry_bar = 0
    setup_active: Setup | None = None

    for i in range(config.WARMUP, n):
        if in_trade:
            row = df.iloc[i]
            hit_sl = row["low"] <= sl
            hit_tp = row["high"] >= tp
            if direction == "short":
                hit_sl = row["high"] >= sl
                hit_tp = row["low"] <= tp

            if hit_sl or hit_tp:
                bars_held = i - entry_bar
                # Intrabar pessimism: if both triggered in same candle → SL first
                if hit_sl and hit_tp:
                    exit_price = sl
                    won = False
                elif hit_sl:
                    exit_price = sl
                    won = False
                else:
                    exit_price = tp
                    won = True

                r_gross = _r_gross(entry_price, exit_price, sl, direction)
                r_net = r_gross - _cost(bars_held)
                results.append(TradeResult(won=won, r_multiple=r_net, bars_held=bars_held))
                in_trade = False

        if not in_trade and i + 1 < n:
            lb = adaptive_lookback(df, i)
            pivots = find_pivots(df, as_of=i, lookback=lb)
            setup = find_setup(df, as_of=i, pivots=pivots, direction=direction)
            if setup is not None:
                atr = _atr_at(df, i)
                # Entry at next candle open + slippage
                next_open = float(df.iloc[i + 1]["open"])
                slip = atr * config.SLIPPAGE_ATR_MULT if atr else 0.0
                if direction == "long":
                    entry_price = next_open + slip
                else:
                    entry_price = next_open - slip
                sl = setup.stop
                tp = setup.target
                entry_bar = i + 1
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


def _atr_at(df: pd.DataFrame, as_of: int) -> float:
    start = max(0, as_of - config.ATR_PERIOD)
    view = df.iloc[start : as_of + 1]
    if len(view) < 2:
        return 0.0
    tr = pd.concat(
        [
            view["high"] - view["low"],
            (view["high"] - view["close"].shift(1)).abs(),
            (view["low"] - view["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.mean())
