"""
Local Bitunix run: multi-TF 101/102 core backtest on real BTC data.

Usage (in your terminal, NOT in cloud):
    cd /path/to/pivot-engine
    python scripts/run_core_btc.py

Downloads 1000 bars of 4h + 4000 bars of 1h + 365 bars of 1d for BTCUSDT.
Runs replay_core.replay_with_trace → prints raw net expectancy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import pandas as pd

import config
from data.fetcher import fetch_ohlcv
from data.validator import check_1
from backtest.replay_core import replay_with_trace
from backtest.metrics import compute

SYMBOL = "BTCUSDT"
DIRECTION = "long"   # change to "short" to test short side


def main() -> None:
    print("Fetching BTCUSDT data from Bitunix …")

    df_4h = fetch_ohlcv(SYMBOL, "4h", limit=1000)
    df_1h = fetch_ohlcv(SYMBOL, "1h", limit=4000)
    df_1d = fetch_ohlcv(SYMBOL, "1d", limit=365)

    for name, df in [("4h", df_4h), ("1h", df_1h), ("1d", df_1d)]:
        if df is None:
            print(f"FAIL: {name} data could not be fetched — check API connectivity")
            return
        ok, msg = check_1(df)
        print(f"  {name}: {len(df)} bars   validator: {msg}")
        if not ok:
            print(f"  FAIL: {name} data invalid — {msg}")
            return

    # Step 1: data quality
    print("\nOHLC sanity:")
    for name, df in [("4h", df_4h), ("1h", df_1h), ("1d", df_1d)]:
        mono = df.index.is_monotonic_increasing
        ohlc = ((df["high"] >= df["low"]) &
                (df["high"] >= df["close"]) &
                (df["low"]  <= df["close"])).all()
        last = df.index[-1]
        print(f"  {name}: monotonic={mono}  OHLC_ok={ohlc}  last_bar={last}")

    # Step 2: pivot counts at last 4h bar
    from engine.pivots import find_pivots, adaptive_lookback
    print("\nPivot extraction at last 4h bar:")
    for name, df in [("4h", df_4h), ("1h", df_1h), ("1d", df_1d)]:
        aof = len(df) - 1
        lb = adaptive_lookback(df, aof)
        pv = find_pivots(df, aof, lb)
        nl = sum(1 for p in pv if p.kind == "low")
        nh = sum(1 for p in pv if p.kind == "high")
        print(f"  {name}: lb={lb}  pivots={len(pv)} ({nl}L {nh}H)")

    # Step 3: live setup check (last bar)
    from engine.structure import find_setup
    from engine.pivots import find_pivots, adaptive_lookback

    ts_last = df_4h.index[-1]
    aof4 = len(df_4h) - 1
    lb4 = adaptive_lookback(df_4h, aof4)
    piv4 = find_pivots(df_4h, aof4, lb4)
    s4 = find_setup(df_4h, aof4, piv4, DIRECTION)

    df_1h_v = df_1h[df_1h.index <= ts_last]
    aof1h = len(df_1h_v) - 1
    lb1h = adaptive_lookback(df_1h_v, aof1h)
    piv1h = find_pivots(df_1h_v, aof1h, lb1h)
    s1h = find_setup(df_1h_v, aof1h, piv1h, DIRECTION)

    print(f"\nLive setup check at {ts_last}:")
    print(f"  4h setup  : {f'zone=[{s4.entry_low:.1f},{s4.entry_high:.1f}] SL={s4.stop:.1f} TP={s4.target:.1f}' if s4 else 'None'}")
    print(f"  1h setup  : {f'zone=[{s1h.entry_low:.1f},{s1h.entry_high:.1f}] SL={s1h.stop:.1f} TP={s1h.target:.1f}' if s1h else 'None'}")
    if s4 and s1h:
        overlap = max(s4.entry_low, s1h.entry_low) < min(s4.entry_high, s1h.entry_high)
        print(f"  Confluence: {overlap}")

    # Step 4: backtest
    print(f"\nRunning replay_mtf ({DIRECTION}) …")
    results, traces = replay_with_trace(df_4h, df_1h, df_1d, DIRECTION, max_trace=5)
    print(f"Total trades: {len(results)}")

    if not results:
        print("\nZero trades — check structure.py sweep detection:")
        print("  Possible reasons:")
        print("  1. pivot_100 never confirmed (need WARMUP bars)")
        print("  2. No sweep below pivot_100 (direction mismatch?)")
        print("  3. R:R < MIN_RR=2.0 on all candidate setups")
        print("  4. No confluence (4h setup exists but 1h/1d don't overlap)")
        return

    for j, t in enumerate(traces):
        print(f"\nTrade #{j + 1}:")
        print(f"  Signal     : 4h bar {t['signal_bar_4h']}  {t['signal_ts']}")
        print(f"  101 zone   : {t['setup_4h_zone']}")
        print(f"  Entry      : {t['entry_price']:.2f}  (next 4h open + slippage)")
        print(f"  SL / TP    : {t['sl']:.2f} / {t['tp']:.2f}")
        print(f"  R:R plan   : {t['rr_planned']:.2f}")
        print(f"  Conf TF    : {t['conf_tf']}")
        if "exit_bar_4h" in t:
            print(f"  Exit       : bar {t['exit_bar_4h']}  price {t['exit_price']:.2f}  [{t['reason']}]")
            print(f"  Bars held  : {t['bars_held']} × 4h")
            print(f"  R gross    : {t['r_gross']:+.4f}")
            print(f"  Cost       : -{t['cost']:.6f}  (taker + funding)")
            print(f"  R NET      : {t['r_net']:+.4f}  {'✓ WIN' if t['won'] else '✗ LOSS'}")

    m = compute(results)
    print("\n" + "=" * 50)
    print("BACKTEST RESULT (real data — net of all costs)")
    print("=" * 50)
    print(f"Trades     : {m.trades}  (need ≥{config.MIN_TRADES} for 'valid')")
    print(f"Win Rate   : {m.wr:.1%}  [Wilson 95%: {m.wr_ci_low:.1%} – {m.wr_ci_high:.1%}]")
    print(f"Expectancy : {m.expectancy_r:+.4f} R  ← primary metric")
    print(f"Walk-fwd   : {m.folds_positive}/{m.fold_count} folds positive")
    print(f"Flag       : {m.flag.upper()}")
    if m.trades < config.MIN_TRADES:
        print(f"\n⚠ Only {m.trades} trades on {len(df_4h)} 4h bars.")
        print("  Increase fetch limit or run on more symbols to reach MIN_TRADES.")


if __name__ == "__main__":
    main()
