"""
Download OHLCV history for the top-N USDT-M futures by 24h volume from Bitunix.

Usage (from pivot-engine/ directory):
    python scripts/download_history.py

Steps:
  1. Fetch live tickers → pick top TOP_N symbols by 24h USDT turnover.
  2. BTCUSDT always included first (direction filter, never traded).
  3. Download 6 months of 4h + 1h + 5m per alt; 4h + 1h for BTC.
  4. RESUMABLE: existing CSVs are skipped — interrupt and re-run freely.
  5. Saves data/top_symbols.json so main.py can auto-scan without scanner_output.json.

Output files:
    data/historical/<SYMBOL>_<tf>.csv   (timestamp,open,high,low,close,volume  UTC ascending)
    data/top_symbols.json               (ordered list of top-N symbols)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)


def _load_dotenv() -> None:
    env_path = os.path.join(_BASE, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


_load_dotenv()

import requests
import pandas as pd
import config
from data.fetcher import get_top_symbols

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_URL = os.environ.get("BITUNIX_BASE_URL", "https://fapi.bitunix.com").rstrip("/")
KLINE_URL = f"{BASE_URL}/api/v1/futures/market/kline"

MONTHS = 6
DAYS = 183

EXPECTED: dict[str, int] = {
    "4h": DAYS * 6,
    "1h": DAYS * 24,
    "5m": DAYS * 288,
}

BATCH_SIZE = 200
RATE_DELAY = 0.25
RETRY_DELAYS = [5, 15, 30]

TF_API: dict[str, str] = {"5m": "5m", "1h": "1h", "4h": "4h"}
TF_MS: dict[str, int] = {
    "5m":  5  * 60 * 1_000,
    "1h":  60 * 60 * 1_000,
    "4h":  4  * 60 * 60 * 1_000,
}

OUT_DIR = os.path.join(_BASE, "data", "historical")
TOP_SYMBOLS_JSON = os.path.join(_BASE, "data", "top_symbols.json")


# ── Symbol selection ───────────────────────────────────────────────────────────

def _build_tf_config(symbols: list[str]) -> dict[str, list[str]]:
    """BTC gets 4h+1h only (direction filter — 5m not needed).
    All alts get 4h+1h+5m.
    """
    cfg: dict[str, list[str]] = {}
    for sym in symbols:
        cfg[sym] = ["4h", "1h"] if sym == config.BTC_SYMBOL else ["4h", "1h", "5m"]
    return cfg


# ── Network helpers ────────────────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _ms_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fetch_batch(symbol: str, tf: str, start_ms: int, end_ms: int) -> list[dict] | None:
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
                print(f"  [ERROR] HTTP {resp.status_code} for {symbol}/{tf} — {resp.text[:200]}")
                return None
            body = resp.json()
            if body.get("code", -1) != 0:
                print(f"  [ERROR] API code={body.get('code')} for {symbol}/{tf}: {body.get('msg')}")
                return None
            return body.get("data", [])
        except requests.exceptions.Timeout:
            last_exc = Exception("timeout after 20s")
        except Exception as exc:
            last_exc = exc
        print(f"  [WARN]  attempt {attempt} failed for {symbol}/{tf}: {last_exc}")

    print(f"  [ERROR] all attempts exhausted for {symbol}/{tf}: {last_exc}")
    return None


def _parse_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            out.append({
                "timestamp": int(float(r["time"])),
                "open":   float(r["open"]),
                "high":   float(r["high"]),
                "low":    float(r["low"]),
                "close":  float(r["close"]),
                "volume": float(r["quoteVol"]),  # coin amount; baseVol = USDT value (naming is swapped)
            })
        except (KeyError, ValueError, TypeError) as exc:
            print(f"  [WARN]  skipped malformed row {r}: {exc}")
    return out


# ── Download one (symbol, tf) ──────────────────────────────────────────────────

def download(symbol: str, tf: str) -> bool:
    """Download full 6-month history for one (symbol, tf). Paginates backwards.

    Memory-safe: all rows held in a plain list; DataFrame built once at the end
    and freed after saving. Safe on Termux with limited RAM.
    Returns True on success.
    """
    tf_ms = TF_MS[tf]
    total_ms = DAYS * 24 * 60 * 60 * 1_000
    end_ms = _now_ms()
    start_target = end_ms - total_ms

    all_rows: list[dict] = []
    batches_fetched = 0
    window_end = end_ms

    print(f"  Downloading {symbol} {tf} … (target ~{EXPECTED[tf]:,} candles)")

    while window_end > start_target:
        window_start = max(window_end - BATCH_SIZE * tf_ms, start_target)

        rows = _fetch_batch(symbol, tf, window_start, window_end)
        if rows is None:
            return False
        if not rows:
            print(f"  [WARN]  empty batch at {_ms_to_utc(window_start)}–{_ms_to_utc(window_end)}, skipping")
            window_end = window_start
            time.sleep(RATE_DELAY)
            continue

        parsed = _parse_rows(rows)
        all_rows.extend(parsed)
        batches_fetched += 1

        oldest_ts = min(r["timestamp"] for r in parsed)
        pct = max(0, 100 * (end_ms - oldest_ts) / total_ms)
        print(
            f"  batch {batches_fetched:3d}: {len(parsed):4d} rows, "
            f"oldest={_ms_to_utc(oldest_ts)}  [{pct:.0f}%]",
            end="\r",
        )

        window_end = window_start
        time.sleep(RATE_DELAY)

    print()

    if not all_rows:
        print(f"  [ERROR] zero rows for {symbol} {tf}")
        return False

    all_rows.sort(key=lambda r: r["timestamp"])
    seen: set[int] = set()
    deduped: list[dict] = []
    for r in all_rows:
        if r["timestamp"] not in seen:
            seen.add(r["timestamp"])
            deduped.append(r)
    if len(deduped) > 1:
        deduped = deduped[:-1]  # drop last (open/incomplete) candle

    df = pd.DataFrame(deduped)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    fpath = os.path.join(OUT_DIR, f"{symbol}_{tf}.csv")
    df.to_csv(fpath, index=False)

    expected = EXPECTED[tf]
    pct_cov = 100 * len(df) / expected if expected else 0
    status = "✓" if len(df) >= expected * 0.90 else "⚠ LOW"
    print(
        f"  {status}  {symbol}_{tf}.csv: {len(df):,} rows  "
        f"(~{expected:,} expected, {pct_cov:.0f}% coverage)"
    )
    print(f"       first: {df['timestamp'].iloc[0]}  last: {df['timestamp'].iloc[-1]}")

    del df, all_rows, deduped, seen
    return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nBitunix history downloader — top {config.TOP_N} symbols by 24h USDT turnover")
    print(f"Period  : {MONTHS} months ({DAYS} days)")
    print(f"Output  : {OUT_DIR}\n")

    # ── Step 1: fetch ranked symbol list ──────────────────────────────────────
    print(f"Fetching top {config.TOP_N} symbols from Bitunix tickers …")
    symbols = get_top_symbols(config.TOP_N)
    if not symbols:
        print("[ERROR] Could not fetch symbol list from API. Check network and retry.")
        sys.exit(1)

    print(f"Got {len(symbols)} symbols. Top 10: {symbols[:10]}\n")

    # Save so main.py can auto-scan without scanner_output.json
    with open(TOP_SYMBOLS_JSON, "w") as f:
        json.dump(symbols, f, indent=2)
    print(f"Saved {TOP_SYMBOLS_JSON}\n")

    tf_config = _build_tf_config(symbols)
    total = sum(len(tfs) for tfs in tf_config.values())

    # ── Step 2: download (resumable) ──────────────────────────────────────────
    done = 0
    failed: list[str] = []

    for symbol in symbols:
        tfs = tf_config[symbol]
        print(f"{'─'*60}")
        print(f"Symbol: {symbol}  timeframes: {', '.join(tfs)}")

        for tf in tfs:
            out_path = os.path.join(OUT_DIR, f"{symbol}_{tf}.csv")
            if os.path.exists(out_path):
                try:
                    row_count = sum(1 for _ in open(out_path)) - 1  # fast line count
                except Exception:
                    row_count = -1
                print(f"  [SKIP] {symbol}_{tf}.csv already exists ({row_count:,} rows) — delete to re-download")
                done += 1
                continue

            ok = download(symbol, tf)
            done += 1
            if not ok:
                failed.append(f"{symbol}_{tf}")

            if tf != tfs[-1]:
                time.sleep(1.0)

        if symbol != symbols[-1]:
            time.sleep(2.0)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done: {done - len(failed)}/{total} files")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        print("Re-run the script — already-downloaded files are skipped automatically.")
    else:
        print("All downloads complete.")
    print(f"\nNext step:")
    print(f"  python main.py   (reads data/top_symbols.json if scanner_output.json absent)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
