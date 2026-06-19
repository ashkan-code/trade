"""
Download 6 months of OHLCV history from Bitunix and save to data/historical/.

Usage (from pivot-engine/ directory):
    python scripts/download_history.py

Output files:
    data/historical/BTCUSDT_4h.csv
    data/historical/ETHUSDT_4h.csv  ... and so on

Each CSV: timestamp,open,high,low,close,volume  (UTC, ascending)

Memory strategy: one (symbol, timeframe) at a time — each DataFrame is freed
before the next request. Safe on Termux with limited RAM.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

# Allow running as `python scripts/download_history.py` from pivot-engine/
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)

# ── Load .env before importing anything that reads config ──────────────────────
def _load_dotenv() -> None:
    env_path = os.path.join(_BASE, ".env")
    if not os.path.exists(env_path):
        print(f"[INFO] No .env found at {env_path} — using environment variables as-is.")
        print("       Market data is public; keys only needed for auth'd endpoints.")
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
    print(f"[INFO] Loaded .env from {env_path}")

_load_dotenv()

import requests
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("BITUNIX_BASE_URL", "https://fapi.bitunix.com").rstrip("/")
KLINE_URL = f"{BASE_URL}/api/v1/futures/market/kline"

# BTC is direction filter — still needs 4H and 1H for MSS/OB detection.
# 5m for BTC is skipped (scanner never uses BTC 5m).
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

# Timeframes to download per symbol.
# BTC gets 4h + 1h only (no 5m needed — it's a direction filter, not traded).
# Alts get all three.
TF_CONFIG: dict[str, list[str]] = {
    "BTCUSDT": ["4h", "1h"],
    "ETHUSDT": ["4h", "1h", "5m"],
    "SOLUSDT": ["4h", "1h", "5m"],
    "BNBUSDT": ["4h", "1h", "5m"],
    "XRPUSDT": ["4h", "1h", "5m"],
}

# 6 months ≈ 183 days
MONTHS = 6
DAYS = 183

# Candle counts for 6 months (used to compute expected rows for sanity check)
# 4h:  183 * 6  = 1098   bars
# 1h:  183 * 24 = 4392   bars
# 5m:  183 * 288= 52704  bars
EXPECTED: dict[str, int] = {
    "4h":  DAYS * 6,
    "1h":  DAYS * 24,
    "5m":  DAYS * 288,
}

BATCH_SIZE = 200       # Bitunix hard cap per request
RATE_DELAY = 0.25      # seconds between requests (4 req/s, well under 10 req/s limit)
RETRY_DELAYS = [5, 15, 30]

# Interval string map (Bitunix API values)
TF_API: dict[str, str] = {
    "5m":  "5m",
    "1h":  "1h",
    "4h":  "4h",
}

# Milliseconds per candle
TF_MS: dict[str, int] = {
    "5m":  5  * 60 * 1000,
    "1h":  60 * 60 * 1000,
    "4h":  4  * 60 * 60 * 1000,
}

OUT_DIR = os.path.join(_BASE, "data", "historical")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _fetch_batch(symbol: str, tf: str, start_ms: int, end_ms: int) -> list[dict] | None:
    """Single request. Returns raw row list or None on failure."""
    params = {
        "symbol": symbol,
        "interval": TF_API[tf],
        "startTime": start_ms,
        "endTime": end_ms,
    }
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0] + RETRY_DELAYS, start=1):
        if delay:
            print(f"        retry {attempt}/{len(RETRY_DELAYS)+1} in {delay}s …")
            time.sleep(delay)
        try:
            resp = requests.get(KLINE_URL, params=params, timeout=20)
            if resp.status_code != 200:
                print(f"  [ERROR] HTTP {resp.status_code} for {symbol}/{tf} "
                      f"start={start_ms} — body: {resp.text[:300]}")
                return None
            body = resp.json()
            code = body.get("code", -1)
            if code != 0:
                msg = body.get("msg", "(no msg)")
                print(f"  [ERROR] API code={code} for {symbol}/{tf}: {msg}")
                return None
            rows = body.get("data", [])
            return rows
        except requests.exceptions.Timeout:
            last_exc = Exception("timeout after 20s")
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
        except Exception as exc:
            last_exc = exc
        print(f"  [WARN]  attempt {attempt} failed for {symbol}/{tf}: {last_exc}")

    print(f"  [ERROR] all {len(RETRY_DELAYS)+1} attempts failed for {symbol}/{tf}: {last_exc}")
    return None


def _parse_rows(rows: list[dict]) -> list[dict]:
    """Parse Bitunix kline rows.

    Confirmed field names (live test June 2026):
      time, open, high, low, close, quoteVol (coin amount), baseVol (USDT value)
    quoteVol = coin volume (BTC qty, ETH qty, etc.) — this is what we store as 'volume'.
    """
    out = []
    for r in rows:
        try:
            out.append({
                "timestamp": int(float(r["time"])),
                "open":      float(r["open"]),
                "high":      float(r["high"]),
                "low":       float(r["low"]),
                "close":     float(r["close"]),
                "volume":    float(r["quoteVol"]),
            })
        except (KeyError, ValueError, TypeError) as exc:
            print(f"  [WARN]  skipped malformed row {r}: {exc}")
    return out


def download(symbol: str, tf: str) -> bool:
    """Download full 6-month history for one (symbol, tf) pair.

    Paginates backwards from now until DAYS * TF_MS[tf] covered.
    Frees all intermediate data after saving — one file at a time.
    Returns True on success.
    """
    tf_ms = TF_MS[tf]
    total_ms = DAYS * 24 * 60 * 60 * 1000
    end_ms = _now_ms()
    start_target = end_ms - total_ms

    all_rows: list[dict] = []
    batches_fetched = 0
    window_end = end_ms

    print(f"  Downloading {symbol} {tf} …")
    print(f"  Target: {DAYS} days = ~{EXPECTED[tf]} candles "
          f"({DAYS * 24 * 60 * 60 * 1000 // tf_ms} theoretical)")

    while window_end > start_target:
        window_start = max(window_end - BATCH_SIZE * tf_ms, start_target)

        rows = _fetch_batch(symbol, tf, window_start, window_end)
        if rows is None:
            return False  # error already printed
        if not rows:
            # No data in this window — could be exchange outage or data gap
            print(f"  [WARN]  empty batch at window {window_start}–{window_end}, skipping back")
            window_end = window_start
            time.sleep(RATE_DELAY)
            continue

        parsed = _parse_rows(rows)
        all_rows.extend(parsed)
        batches_fetched += 1

        oldest_ts = min(r["timestamp"] for r in parsed)
        pct = max(0, 100 * (end_ms - oldest_ts) / total_ms)
        print(f"  batch {batches_fetched:3d}: got {len(parsed):4d} rows, "
              f"oldest={_ms_to_utc(oldest_ts)}  [{pct:.0f}% covered]", end="\r")

        window_end = window_start
        time.sleep(RATE_DELAY)

    print()  # newline after \r progress

    if not all_rows:
        print(f"  [ERROR] zero rows collected for {symbol} {tf}")
        return False

    # Sort, deduplicate, drop open (last) candle
    all_rows.sort(key=lambda r: r["timestamp"])
    seen: set[int] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["timestamp"] not in seen:
            seen.add(r["timestamp"])
            deduped.append(r)
    # Drop last row (open/incomplete candle)
    if len(deduped) > 1:
        deduped = deduped[:-1]

    # Build DataFrame, convert timestamp to UTC string for CSV portability
    import pandas as pd
    df = pd.DataFrame(deduped)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Save
    fname = f"{symbol}_{tf}.csv"
    fpath = os.path.join(OUT_DIR, fname)
    df.to_csv(fpath, index=False)

    expected = EXPECTED[tf]
    pct_coverage = 100 * len(df) / expected if expected else 0
    status = "✓" if len(df) >= expected * 0.90 else "⚠ LOW"
    print(f"  {status}  {fname}: {len(df):,} rows  "
          f"(expected ~{expected:,}, {pct_coverage:.0f}% coverage)")
    print(f"     first: {df['timestamp'].iloc[0]}  last: {df['timestamp'].iloc[-1]}")

    # Free memory before next symbol
    del df, all_rows, deduped, seen
    return True


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nBitunix history downloader — target: {MONTHS} months ({DAYS} days)")
    print(f"Output dir: {OUT_DIR}\n")

    total = sum(len(tfs) for tfs in TF_CONFIG.values())
    done = 0
    failed: list[str] = []

    for symbol in SYMBOLS:
        tfs = TF_CONFIG[symbol]
        print(f"{'─'*60}")
        print(f"Symbol: {symbol}  timeframes: {', '.join(tfs)}")
        for tf in tfs:
            out_path = os.path.join(OUT_DIR, f"{symbol}_{tf}.csv")
            if os.path.exists(out_path):
                import pandas as pd
                existing = pd.read_csv(out_path, usecols=["timestamp"])
                row_count = len(existing)
                del existing
                print(f"  [SKIP] {symbol}_{tf}.csv already exists ({row_count:,} rows) "
                      f"— delete to re-download")
                done += 1
                continue

            ok = download(symbol, tf)
            done += 1
            if not ok:
                failed.append(f"{symbol}_{tf}")

            # Pause between timeframes for the same symbol
            if tf != tfs[-1]:
                time.sleep(1.0)

        # Longer pause between symbols
        if symbol != SYMBOLS[-1]:
            print(f"  [pause 2s before next symbol]")
            time.sleep(2.0)

    print(f"\n{'='*60}")
    print(f"Done: {done - len(failed)}/{total} succeeded")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("Re-run the script — it skips already-downloaded files.")
    else:
        print("All downloads complete. Run the backtest:")
        print("  python main.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
