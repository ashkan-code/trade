"""
PIVOT-ENGINE: Live multi-symbol scanner with 3-layer architecture.

  Layer 0 — BTC:    compute BTC bias (long/short/neutral) from swing structure.
                    compute BTC normalized pivot vector for similarity reference.
  Layer 1 — filter: skip any symbol whose 4h pivot vector Pearson-correlates < SIMILARITY_MIN
                    to BTC (different amplitude is fine; shape must match).
  Layer 2 — engine: run predict_101_102 on {4h, 1h, 1d} for each surviving symbol.
                    require >=2 TFs to agree on direction AND 101 zones within 1 ATR.
                    direction must align with BTC bias (both directions allowed if neutral).

Usage:
    python scripts/live_pivot_scan.py            # all symbols
    python scripts/live_pivot_scan.py --top 50   # top-50 by USDT volume
    python scripts/live_pivot_scan.py --long     # long setups only
    python scripts/live_pivot_scan.py --short    # short setups only

Output: English only. No Farsi.

NOT RUN in cloud — Bitunix API is blocked (403). Run locally:
    cd /path/to/pivot-engine && python scripts/live_pivot_scan.py
"""

import argparse
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from data.fetcher import fetch_ohlcv, get_all_symbols, get_top_symbols
from engine.pivots import find_pivots, adaptive_lookback, predict_101_102
from engine.ict import atr_scalar
from contracts import Pivot

# Candle limits per symbol (RAM-safe: process one at a time)
_LIMIT_4H: int = 1000
_LIMIT_1H: int = 1500
_LIMIT_1D: int = 365


# ── Layer 0 helpers ───────────────────────────────────────────────────────────

def _btc_bias(pivots: list[Pivot]) -> str:
    """Infer BTC directional bias from last 2 swing highs and lows.

    HH + HL -> 'long', LH + LL -> 'short', else -> 'neutral'.
    """
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


def _pivot_vector(pivots: list[Pivot], n_changes: int) -> list[float] | None:
    """Return the last n_changes relative % price changes between consecutive pivots.

    Normalizes out amplitude so that BTC (1% swings) and alts (10% swings) with
    the same rhythm produce the same vector shape.
    Returns None if too few pivots.
    """
    if len(pivots) < n_changes + 1:
        return None
    tail   = pivots[-(n_changes + 1):]
    prices = [p.price for p in tail]
    return [(prices[i + 1] - prices[i]) / prices[i] for i in range(len(prices) - 1)]


# ── Layer 1 helper ────────────────────────────────────────────────────────────

def _pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation between two lists (truncated to min length).

    Returns 0.0 on degenerate input (constant vector or n<2).
    """
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    va = np.array(a[:n], dtype=float)
    vb = np.array(b[:n], dtype=float)
    if va.std() == 0 or vb.std() == 0:
        return 0.0
    return float(np.corrcoef(va, vb)[0, 1])


# ── Layer 2 helper ────────────────────────────────────────────────────────────

def _confluence(
    df_4h, df_1h, df_1d,
    allowed_dir: str | None,
) -> dict | None:
    """Run predict_101_102 on each available TF.

    Returns a setup dict when:
      - >=2 TFs agree on direction (= allowed_dir, or any if allowed_dir is None)
      - At least one pair of 101 zones is within 1 ATR of each other
      - 4h is one of the agreeing TFs (required; it drives entry timing)

    as_of = last closed bar on each TF (len-1). The forming bar is excluded
    because fetch_ohlcv returns closed candles.
    """
    preds: dict[str, dict] = {}

    for label, df in [("4h", df_4h), ("1h", df_1h), ("1d", df_1d)]:
        if df is None or len(df) <= config.WARMUP:
            continue
        as_of = len(df) - 1
        lb    = adaptive_lookback(df, as_of)
        pivs  = find_pivots(df, as_of, lb)
        pred  = predict_101_102(pivs, as_of, df)
        if pred is None:
            continue
        if allowed_dir is not None and pred["direction"] != allowed_dir:
            continue
        preds[label] = pred

    if len(preds) < 2:
        return None
    if "4h" not in preds:
        return None  # 4h must be one of the agreeing TFs

    # All must share the same direction
    directions = {v["direction"] for v in preds.values()}
    if len(directions) > 1:
        return None
    direction = directions.pop()

    # Zone proximity: any pair of 101 zones within 1 ATR of the 4h ATR
    atr_val   = atr_scalar(df_4h, config.ATR_PERIOD)
    pred_list = list(preds.values())
    near = False
    for i in range(len(pred_list)):
        for j in range(i + 1, len(pred_list)):
            p1, p2 = pred_list[i], pred_list[j]
            gap = max(p1["p101_low"], p2["p101_low"]) - min(p1["p101_high"], p2["p101_high"])
            if gap <= atr_val:
                near = True
                break
        if near:
            break
    if not near:
        return None

    p4  = preds["4h"]
    p1d = preds.get("1d")
    return {
        "direction":     direction,
        "conf_tfs":      "+".join(sorted(preds.keys())),
        "p101_low":      p4["p101_low"],
        "p101_high":     p4["p101_high"],
        "p102":          p1d["p102"] if p1d is not None else p4["p102"],
        "stop":          p4["stop"],
        "rr":            p4["rr"],
        "current_price": float(df_4h.iloc[-1]["close"]),
    }


# ── per-symbol worker (runs inside ThreadPoolExecutor) ────────────────────────

def _scan_symbol(
    sym: str,
    btc_vec: list[float] | None,
    scan_dir: str | None,
) -> tuple[bool, dict | None, str]:
    """Layer 1 + 2 for a single symbol.

    Returns (passed_layer1, setup_or_None, log_line).
    Thread-safe: no shared mutable state written here.
    """
    try:
        df_4h = fetch_ohlcv(sym, "4h", limit=_LIMIT_4H)
        if df_4h is None or len(df_4h) < config.WARMUP + 10:
            return False, None, ""

        as_of   = len(df_4h) - 1
        lb      = adaptive_lookback(df_4h, as_of)
        pivs    = find_pivots(df_4h, as_of, lb)
        sym_vec = _pivot_vector(pivs, config.SIMILARITY_PIVOTS)

        # LAYER 1 — similarity gate
        if btc_vec is not None:
            if sym_vec is None:
                return False, None, ""
            sim = _pearson(btc_vec, sym_vec)
            if sim < config.SIMILARITY_MIN:
                return False, None, ""
        else:
            sim = float("nan")

        sim_str = f"{sim:.3f}" if sim == sim else "N/A"  # nan-safe

        # LAYER 2 — fetch remaining TFs and run confluence
        df_1h = fetch_ohlcv(sym, "1h", limit=_LIMIT_1H)
        df_1d = fetch_ohlcv(sym, "1d", limit=_LIMIT_1D)
        setup = _confluence(df_4h, df_1h, df_1d, scan_dir)
        del df_4h, df_1h, df_1d  # free RAM immediately (Termux-safe)

        if setup is None:
            log = f"  {sym:<12}  sim={sim_str}  pivots={len(pivs)}  no confluence"
            return True, None, log

        setup["symbol"]     = sym
        setup["similarity"] = round(sim, 3) if sim == sim else None
        log = (
            f"  {sym:<12}  sim={sim_str}  pivots={len(pivs)}"
            f"  SIGNAL {setup['direction'].upper()} R:R={setup['rr']:.2f}"
        )
        return True, setup, log

    except Exception as exc:
        return False, None, f"  SKIP {sym}: {exc}"

    finally:
        time.sleep(0.1)  # rate-limit safety per worker


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Live pivot 101/102 multi-symbol scanner")
    parser.add_argument("--top",   type=int, default=None,
                        help="Scan only top-N symbols by 24h USDT volume (default: all)")
    parser.add_argument("--long",  action="store_true", help="Long setups only")
    parser.add_argument("--short", action="store_true", help="Short setups only")
    args = parser.parse_args()

    forced_dir: str | None = None
    if args.long and not args.short:
        forced_dir = "long"
    elif args.short and not args.long:
        forced_dir = "short"

    # ── LAYER 0: BTC ─────────────────────────────────────────────────────────
    print("=== LAYER 0: BTC bias + reference pivot vector ===")
    df_btc_4h = fetch_ohlcv("BTCUSDT", "4h", limit=_LIMIT_4H)
    df_btc_1h = fetch_ohlcv("BTCUSDT", "1h", limit=_LIMIT_1H)
    df_btc_1d = fetch_ohlcv("BTCUSDT", "1d", limit=_LIMIT_1D)

    if df_btc_4h is None:
        print("FAIL: BTC 4h data unavailable — check network / API access. Aborting.")
        sys.exit(1)

    btc_aof4 = len(df_btc_4h) - 1
    btc_lb4  = adaptive_lookback(df_btc_4h, btc_aof4)
    btc_piv4 = find_pivots(df_btc_4h, btc_aof4, btc_lb4)

    n_piv_1h = n_piv_1d = 0
    if df_btc_1h is not None:
        aoi = len(df_btc_1h) - 1
        n_piv_1h = len(find_pivots(df_btc_1h, aoi, adaptive_lookback(df_btc_1h, aoi)))
    if df_btc_1d is not None:
        aoi = len(df_btc_1d) - 1
        n_piv_1d = len(find_pivots(df_btc_1d, aoi, adaptive_lookback(df_btc_1d, aoi)))

    btc_bias = _btc_bias(btc_piv4)
    btc_vec  = _pivot_vector(btc_piv4, config.SIMILARITY_PIVOTS)

    print(f"BTC bias: {btc_bias.upper()} | pivots: 4h={len(btc_piv4)} 1h={n_piv_1h} 1d={n_piv_1d}")
    if btc_vec is None:
        print(f"WARN: BTC has only {len(btc_piv4)} pivots — need >{config.SIMILARITY_PIVOTS}.")
        print("Similarity filter disabled; all symbols with enough pivots will proceed.")

    # Effective scan direction: CLI flag overrides BTC bias
    if forced_dir is not None:
        scan_dir = forced_dir
        print(f"Direction forced: {scan_dir.upper()}")
    elif btc_bias != "neutral":
        scan_dir = btc_bias
        print(f"Direction from BTC bias: {scan_dir.upper()}")
    else:
        scan_dir = None
        print("BTC bias: NEUTRAL — scanning both directions")

    # Free BTC DataFrames (RAM)
    del df_btc_1h, df_btc_1d

    # ── Symbol list ───────────────────────────────────────────────────────────
    print("\n=== LAYER 1: Symbol scan ===")
    if args.top is not None:
        symbols = get_top_symbols(args.top)
        print(f"Fetching top-{args.top} symbols by 24h USDT volume...")
    else:
        symbols = get_all_symbols()
        print("Fetching all USDT-M symbols...")

    alt_symbols = [s for s in symbols if s != "BTCUSDT"]
    print(f"Symbols to scan: {len(alt_symbols)} (BTC already done)")

    # ── Parallel per-symbol scan ──────────────────────────────────────────────
    n_scanned    = 0
    n_similar    = 0
    valid_setups: list[dict] = []
    _print_lock  = threading.Lock()

    workers = min(8, len(alt_symbols))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_symbol, sym, btc_vec, scan_dir): sym
            for sym in alt_symbols
        }
        for fut in as_completed(futures):
            n_scanned += 1
            similar, setup, log = fut.result()
            if similar:
                n_similar += 1
            if setup is not None:
                valid_setups.append(setup)
            if log:
                with _print_lock:
                    print(log, flush=True)

    # ── Output ────────────────────────────────────────────────────────────────
    valid_setups.sort(key=lambda s: s["rr"], reverse=True)

    print()
    print("=" * 110)
    print("VALID SETUPS (sorted by R:R descending)")
    print("=" * 110)

    if valid_setups:
        hdr = (
            f"{'SYMBOL':<12} {'DIR':<6} {'SIM':>6}  {'CONF':^9}  "
            f"{'101_LOW':>10} {'101_HIGH':>10}  {'102':>10}  {'STOP':>10}  {'R:R':>5}  {'PRICE':>12}"
        )
        print(hdr)
        print("-" * 110)
        for s in valid_setups:
            sim_d = f"{s['similarity']:.3f}" if s["similarity"] is not None else "  N/A"
            print(
                f"{s['symbol']:<12} {s['direction'].upper():<6} {sim_d:>6}  "
                f"{s['conf_tfs']:^9}  "
                f"{s['p101_low']:>10.2f} {s['p101_high']:>10.2f}  "
                f"{s['p102']:>10.2f}  {s['stop']:>10.2f}  "
                f"{s['rr']:>5.2f}  {s['current_price']:>12.2f}"
            )
    else:
        print("No valid setups found.")

    print()
    print(
        f"Scanned {n_scanned} | Similar to BTC: {n_similar} | Valid setups: {len(valid_setups)}"
    )


if __name__ == "__main__":
    main()
