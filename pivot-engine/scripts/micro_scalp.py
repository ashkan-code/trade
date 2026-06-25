"""
MICRO-SCALP-ENGINE: Live multi-symbol micro scalp scanner on 3m / 5m / 15m.

Layer 0 — BTC:  compute BTC bias from 15m swing structure.
Layer 1 — range filter: skip symbols where >=2 of 3 range checks fire on 3m.
Layer 2 — engine: run predict_101_102 on {3m, 5m, 15m}.
           >=2 TFs must agree on direction AND entry zones within 1 ATR.
           All 3 agreeing -> labelled STRONG.

Target : 15m predict_101_102 target; fallback chain 5m -> 3m.
Stop   : last confirmed swing on 15m (long -> swing low; short -> swing high).
Price  : Bitunix ticker lastPrice; fallback to last 3m close.
Time   : Asia/Tehran for all timestamps (format YYYY-MM-DD HH:MM Tehran).

NOT RUN in cloud — Bitunix API is blocked (403). Run locally:
    cd /path/to/pivot-engine && python scripts/micro_scalp.py
"""

import argparse
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _TEHRAN = ZoneInfo("Asia/Tehran")
except Exception:
    # Fallback: fixed UTC+3:30 offset (Iran standard time; no DST correction)
    from datetime import timezone, timedelta  # type: ignore[no-redef]
    _TEHRAN = timezone(timedelta(hours=3, minutes=30))  # type: ignore[assignment]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import requests

import config
from data.fetcher import fetch_ohlcv, get_all_symbols, get_top_symbols
from engine.pivots import find_pivots, predict_101_102
from engine.ict import atr_scalar
from contracts import Pivot

# Candle limits per symbol
_LIMIT_15M: int = 500
_LIMIT_5M:  int = 500
_LIMIT_3M:  int = 500

# Local MAX_LB override for micro TFs — wider window to filter noise.
# config.py MAX_LB (=10) is not touched.
_MICRO_MAX_LB: int = 15
_MICRO_MIN_LB: int = config.MIN_LB   # 3

# Bitunix single-ticker endpoint for live price
_TICKER_URL: str = "https://fapi.bitunix.com/api/v1/futures/tickers"

# TF display order for CONF label (shortest first)
_TF_ORDER: dict[str, int] = {"3m": 0, "5m": 1, "15m": 2}


# ── Tehran helpers ────────────────────────────────────────────────────────────

def _now_tehran() -> str:
    return datetime.now(_TEHRAN).strftime("%Y-%m-%d %H:%M Tehran")  # type: ignore[arg-type]


# ── Live price ────────────────────────────────────────────────────────────────

def _live_price(sym: str) -> float | None:
    """Fetch current lastPrice from Bitunix ticker API. Returns None on failure."""
    try:
        resp = requests.get(_TICKER_URL, params={"symbol": sym}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") == 0:
            items = data.get("data", [])
            if items:
                lp = items[0].get("lastPrice")
                if lp is not None:
                    return float(lp)
    except Exception:
        pass
    return None


# ── Adaptive lookback with local MAX_LB=40 ────────────────────────────────────

def _adaptive_lookback_micro(df: pd.DataFrame, as_of: int) -> int:
    """Same formula as engine.pivots.adaptive_lookback but uses _MICRO_MAX_LB=40.

    High vol -> smaller lookback (pivots confirm faster).
    Low  vol -> larger lookback (wider structural context).
    config.py MAX_LB (=10) is not touched.
    """
    start = max(0, as_of - config.ATR_PERIOD)
    view  = df.iloc[start : as_of + 1]
    if len(view) < 2:
        return _MICRO_MAX_LB

    tr = pd.concat([
        view["high"] - view["low"],
        (view["high"] - view["close"].shift(1)).abs(),
        (view["low"]  - view["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr       = tr.mean()
    mid_price = view["close"].iloc[-1]
    if mid_price <= 0:
        return _MICRO_MAX_LB

    atr_pct  = (atr / mid_price) * 100.0
    span_vol = 1.5 - 0.3
    span_lb  = _MICRO_MAX_LB - _MICRO_MIN_LB
    lb = _MICRO_MAX_LB - ((atr_pct - 0.3) / span_vol) * span_lb
    return int(max(_MICRO_MIN_LB, min(_MICRO_MAX_LB, round(lb))))


# ── BTC bias ──────────────────────────────────────────────────────────────────

def _btc_bias(pivots: list[Pivot]) -> str:
    """HH+HL -> 'long', LH+LL -> 'short', else -> 'neutral'."""
    lows  = sorted([p for p in pivots if p.kind == "low"],  key=lambda p: p.index)
    highs = sorted([p for p in pivots if p.kind == "high"], key=lambda p: p.index)
    if len(lows) < 2 or len(highs) < 2:
        return "neutral"
    hh = highs[-1].price > highs[-2].price
    hl = lows[-1].price  > lows[-2].price
    lh = highs[-1].price < highs[-2].price
    ll = lows[-1].price  < lows[-2].price
    if hh and hl:
        return "long"
    if lh and ll:
        return "short"
    return "neutral"


# ── Range filter ──────────────────────────────────────────────────────────────

def _is_ranging(df_3m: pd.DataFrame) -> bool:
    """Return True if >=2 of 3 range filters fire -> symbol is range-bound, skip it.

    Filter 1 — Body ratio: avg(|open-close|) / avg(high-low) on last 10 bars < 40%
    Filter 2 — HL range:   (max_high - min_low) / price on last 20 bars < 2%
    Filter 3 — EMA slope:  |EMA20[-1] - EMA20[-6]| / EMA20[-6] < 0.1%
    """
    n_range = 0

    # Filter 1
    last10     = df_3m.iloc[-10:]
    bodies     = (last10["close"] - last10["open"]).abs()
    bar_ranges = last10["high"] - last10["low"]
    avg_range  = bar_ranges.mean()
    if avg_range > 0 and (bodies.mean() / avg_range) < 0.40:
        n_range += 1

    # Filter 2
    last20   = df_3m.iloc[-20:]
    hl_range = last20["high"].max() - last20["low"].min()
    price    = float(df_3m.iloc[-1]["close"])
    if price > 0 and (hl_range / price) < 0.02:
        n_range += 1

    # Filter 3 — needs enough bars for EMA20 + 5-bar slope window
    if len(df_3m) >= config.WARMUP:
        ema20 = df_3m["close"].ewm(span=20, adjust=False).mean()
        ref   = float(ema20.iloc[-6])
        if ref > 0 and abs((float(ema20.iloc[-1]) - ref) / ref) < 0.001:
            n_range += 1

    return n_range >= 2


# ── CONF label ────────────────────────────────────────────────────────────────

def _conf_label(keys: list[str]) -> str:
    """Format agreeing-TF label in time order. All 3 -> STRONG(3m+5m+15m)."""
    ordered = sorted(keys, key=lambda t: _TF_ORDER.get(t, 99))
    joined  = "+".join(ordered)
    return f"STRONG({joined})" if len(ordered) == 3 else joined


# ── Micro confluence ──────────────────────────────────────────────────────────

def _micro_confluence(
    df_3m:  pd.DataFrame,
    df_5m:  pd.DataFrame | None,
    df_15m: pd.DataFrame | None,
    allowed_dir: str | None,
) -> dict | None:
    """Run predict_101_102 on 3m / 5m / 15m. >=2 must agree on direction.

    Entry zone : shortest agreeing TF (3m > 5m > 15m priority).
    Target     : longest agreeing TF (15m > 5m > 3m priority).
    Stop       : last confirmed swing on 15m (causal).
    R:R        : recomputed from resolved stop / target.
    Returns None if <2 agree, zones not proximate, or R:R < MIN_RR.
    """
    preds:      dict[str, dict] = {}
    pivots_15m: list[Pivot]     = []

    for label, df in [("3m", df_3m), ("5m", df_5m), ("15m", df_15m)]:
        if df is None or len(df) <= config.WARMUP:
            continue
        as_of = len(df) - 1
        lb    = _adaptive_lookback_micro(df, as_of)
        pivs  = find_pivots(df, as_of, lb)
        if label == "15m":
            pivots_15m = pivs
        pred  = predict_101_102(pivs, as_of, df, max_distance_pct=1.0)
        if pred is None:
            continue
        if allowed_dir is not None and pred["direction"] != allowed_dir:
            continue
        preds[label] = pred

    if len(preds) < 2:
        return None

    directions = {v["direction"] for v in preds.values()}
    if len(directions) > 1:
        return None
    direction = directions.pop()

    # Zone proximity: any pair within 1 ATR (use 15m ATR as reference; wider scale)
    atr_ref = df_15m if (df_15m is not None and len(df_15m) > config.ATR_PERIOD) else df_3m
    atr_val = atr_scalar(atr_ref, config.ATR_PERIOD)
    pred_list = list(preds.values())
    near = False
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            p1, p2 = pred_list[i], pred_list[j]
            gap = max(p1["entry_low"], p2["entry_low"]) - min(p1["entry_high"], p2["entry_high"])
            if gap <= atr_val:
                near = True
                break
        if near:
            break
    if not near:
        return None

    # Entry zone: shortest agreeing TF
    p_entry = preds.get("3m") or preds.get("5m") or preds["15m"]
    entry_low  = p_entry["entry_low"]
    entry_high = p_entry["entry_high"]

    # Target: longest agreeing TF (15m is longest here)
    p_target = preds.get("15m") or preds.get("5m") or preds["3m"]
    target = p_target["target"]

    # Stop: last confirmed swing on 15m (causal — pivots_15m already built above)
    p_ref = preds.get("15m") or preds.get("5m") or preds["3m"]
    if direction == "long":
        swing_lows = sorted([p for p in pivots_15m if p.kind == "low"], key=lambda p: p.index)
        stop = swing_lows[-1].price if swing_lows else p_ref["stop"]
    else:
        swing_highs = sorted([p for p in pivots_15m if p.kind == "high"], key=lambda p: p.index)
        stop = swing_highs[-1].price if swing_highs else p_ref["stop"]

    # Geometry sanity check
    if direction == "long":
        if stop >= entry_low or target <= entry_high:
            return None
        risk = entry_high - stop
        rr   = (target - entry_high) / risk
    else:
        if stop <= entry_high or target >= entry_low:
            return None
        risk = stop - entry_low
        rr   = (entry_low - target) / risk

    if risk <= 0 or rr < config.MIN_RR:
        return None

    return {
        "direction":          direction,
        "conf_tfs":           _conf_label(list(preds.keys())),
        "entry_low":          entry_low,
        "entry_high":         entry_high,
        "target":             target,
        "stop":               stop,
        "rr":                 round(rr, 4),
        "probability_entry":  p_ref["probability_entry"],
        "probability_target": p_ref["probability_target"],
        "sample_size":        p_ref["sample_size"],
    }


# ── Per-symbol worker ─────────────────────────────────────────────────────────

def _scan_symbol_micro(
    sym: str,
    scan_dir: str | None,
) -> tuple[bool, dict | None, str]:
    """Micro scalp scan for one symbol. Thread-safe: no shared mutable state."""
    try:
        df_3m = fetch_ohlcv(sym, "3m", limit=_LIMIT_3M)
        if df_3m is None or len(df_3m) < config.WARMUP + 10:
            return False, None, ""

        # Range filter checked first — avoids fetching 5m/15m for flat symbols
        if _is_ranging(df_3m):
            return True, None, f"  {sym:<12}  ranging — skip"

        df_5m  = fetch_ohlcv(sym, "5m",  limit=_LIMIT_5M)
        df_15m = fetch_ohlcv(sym, "15m", limit=_LIMIT_15M)

        setup = _micro_confluence(df_3m, df_5m, df_15m, scan_dir)

        if setup is not None:
            # ATR% and VOL_RATIO from 3m (smallest TF for micro sensitivity)
            _cl   = float(df_3m.iloc[-1]["close"])
            _atr  = atr_scalar(df_3m, config.ATR_PERIOD)
            _vols = df_3m["volume"].to_numpy()
            _avgv = float(np.mean(_vols[-20:])) if len(_vols) >= 20 else float(np.mean(_vols))
            setup["atr_pct"]   = round((_atr / _cl * 100.0) if _cl > 0 else 0.0, 4)
            setup["vol_ratio"] = round(float(_vols[-1]) / _avgv if _avgv > 0 else 0.0, 4)

            # Live price from ticker API; fallback to last close
            live_p = _live_price(sym)
            setup["current_price"] = live_p if live_p else _cl
            setup["price_source"]  = "ticker" if live_p else "close"
            setup["ts_tehran"]     = datetime.now(_TEHRAN).strftime("%Y-%m-%d %H:%M")  # type: ignore[arg-type]

        del df_3m, df_5m, df_15m

        if setup is None:
            return True, None, f"  {sym:<12}  no confluence"

        setup["symbol"] = sym
        log = (
            f"  {sym:<12}"
            f"  SIGNAL {setup['direction'].upper()}"
            f"  R:R={setup['rr']:.2f}"
            f"  [{setup['conf_tfs']}]"
        )
        return True, setup, log

    except Exception as exc:
        return False, None, f"  SKIP {sym}: {exc}"

    finally:
        time.sleep(0.1)  # rate-limit safety per worker


# ── Quality filter ────────────────────────────────────────────────────────────

def _classify_setups(setups: list[dict]) -> None:
    """Classify valid setups by ATR% (>0.5%) and volume ratio (>1.5×avg-20)."""
    confirmed: list[str] = []
    weak:      list[str] = []
    skip_:     list[str] = []

    for s in setups:
        atr_ok = s.get("atr_pct",   0.0) > 0.5
        vol_ok = s.get("vol_ratio", 0.0) > 1.5
        sym    = s["symbol"]
        if atr_ok and vol_ok:
            confirmed.append(sym)
        elif atr_ok or vol_ok:
            weak.append(sym)
        else:
            skip_.append(sym)

    print()
    print("=== BEST SETUPS ===")
    print(f"CONFIRMED: {confirmed}")
    print(f"WEAK:      {weak}")
    print(f"SKIP:      {skip_}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Micro scalp pivot scanner 3m/5m/15m")
    parser.add_argument("--top",   type=int, default=300,
                        help="Scan top-N symbols by 24h USDT volume (default: 300)")
    parser.add_argument("--long",  action="store_true", help="Long setups only")
    parser.add_argument("--short", action="store_true", help="Short setups only")
    args = parser.parse_args()

    forced_dir: str | None = None
    if args.long and not args.short:
        forced_dir = "long"
    elif args.short and not args.long:
        forced_dir = "short"

    print(f"=== MICRO SCALP SCANNER 3m/5m/15m  |  {_now_tehran()} ===")

    scan_dir = forced_dir  # None = both directions unless --long/--short given
    if scan_dir is not None:
        print(f"Direction forced: {scan_dir.upper()}")
    else:
        print("Scanning both directions")

    # ── Symbol list ───────────────────────────────────────────────────────────
    print("\n=== LAYER 1: Symbol list ===")
    if args.top is not None:
        symbols = get_top_symbols(args.top)
        print(f"Fetching top-{args.top} symbols by 24h USDT volume...")
    else:
        symbols = get_all_symbols()
        print("Fetching all USDT-M symbols...")

    alt_symbols = [s for s in symbols if s != "BTCUSDT"]
    print(f"Symbols to scan: {len(alt_symbols)} (BTC already done)")

    # ── Parallel scan ─────────────────────────────────────────────────────────
    n_scanned    = 0
    valid_setups: list[dict] = []
    _print_lock  = threading.Lock()

    workers = min(8, len(alt_symbols))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_symbol_micro, sym, scan_dir): sym
            for sym in alt_symbols
        }
        for fut in as_completed(futures):
            n_scanned += 1
            _, setup, log = fut.result()
            if setup is not None:
                valid_setups.append(setup)
            if log:
                with _print_lock:
                    print(log, flush=True)

    # ── Output ────────────────────────────────────────────────────────────────
    valid_setups.sort(key=lambda s: s["rr"], reverse=True)

    print()
    print("=" * 130)
    print(f"VALID MICRO SCALP SETUPS (sorted by R:R descending)  |  {_now_tehran()}")
    print("=" * 130)

    if valid_setups:
        hdr = (
            f"{'SYMBOL':<12} {'DIR':<6}  {'CONF':<22}  "
            f"{'ENTRY_LOW':>10} {'ENTRY_HIGH':>10}  {'TARGET':>10}  {'STOP':>10}  "
            f"{'R:R':>5}  {'P_ENTRY':>7}  {'P_TARGET':>8}  {'N':>4}  "
            f"{'PRICE':>12}  {'SRC':<6}  {'TS (Tehran)'}"
        )
        print(hdr)
        print("-" * 130)
        for s in valid_setups:
            print(
                f"{s['symbol']:<12} {s['direction'].upper():<6}  "
                f"{s['conf_tfs']:<22}  "
                f"{s['entry_low']:>10.4f} {s['entry_high']:>10.4f}  "
                f"{s['target']:>10.4f}  {s['stop']:>10.4f}  "
                f"{s['rr']:>5.2f}  {s['probability_entry']:>7.1%}  "
                f"{s['probability_target']:>8.1%}  {s['sample_size']:>4}  "
                f"{s['current_price']:>12.4f}  {s.get('price_source', '?'):<6}  "
                f"{s.get('ts_tehran', '')}"
            )
    else:
        print("No valid micro scalp setups found.")

    _classify_setups(valid_setups)

    print()
    print(
        f"Scanned {n_scanned} | Valid setups: {len(valid_setups)} | {_now_tehran()}"
    )


if __name__ == "__main__":
    main()
