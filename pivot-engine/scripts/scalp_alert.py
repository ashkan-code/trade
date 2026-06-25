"""
Scalp alert daemon: runs scalp.py every 3 minutes, diffs signals,
sends Telegram message for each new signal.

Usage (run locally — Bitunix API is blocked in cloud):
    cd /path/to/pivot-engine && python scripts/scalp_alert.py
    python scripts/scalp_alert.py --long
    python scripts/scalp_alert.py --short
    python scripts/scalp_alert.py --top 50
"""

import argparse
import subprocess
import sys
import os
import time
import json
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    _TEHRAN = ZoneInfo("Asia/Tehran")
except Exception:
    from datetime import timezone, timedelta  # type: ignore[no-redef]
    _TEHRAN = timezone(timedelta(hours=3, minutes=30))  # type: ignore[assignment]

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_INTERVAL: int = 180  # seconds between scans
_SCALP_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scalp.py")


# ── Telegram ──────────────────────────────────────────────────────────────────

def _send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"  [telegram] send failed: {exc}", flush=True)


# ── Signal key (dedup) ────────────────────────────────────────────────────────

def _sig_key(s: dict) -> str:
    """Unique key for a setup: symbol + direction + rounded entry_low."""
    return f"{s['symbol']}|{s['direction']}|{round(s['entry_low'], 2)}"


# ── Run scalp.py and parse its stdout ────────────────────────────────────────

def _run_scalp(extra_args: list[str]) -> list[dict]:
    """Execute scalp.py, parse table rows, return list of setup dicts."""
    cmd = [sys.executable, _SCALP_SCRIPT] + extra_args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("  [scan] timed out after 600s", flush=True)
        return []
    except Exception as exc:
        print(f"  [scan] error: {exc}", flush=True)
        return []

    setups: list[dict] = []
    in_table = False

    for line in result.stdout.splitlines():
        stripped = line.strip()

        # Table header detected by presence of "ENTRY_LOW"
        if "ENTRY_LOW" in stripped:
            in_table = True
            continue
        if in_table and stripped.startswith("-"):
            continue
        if not in_table:
            continue
        # Empty line or section header ends the table
        if not stripped or stripped.startswith("=") or stripped.startswith("Scanned"):
            in_table = False
            continue

        parts = stripped.split()
        # Expected columns: SYMBOL DIR CONF ENTRY_LOW ENTRY_HIGH TARGET STOP R:R P_ENTRY P_TARGET N PRICE SRC TS
        if len(parts) < 13:
            continue
        try:
            setups.append({
                "symbol":    parts[0],
                "direction": parts[1].lower(),
                "conf_tfs":  parts[2],
                "entry_low":  float(parts[3]),
                "entry_high": float(parts[4]),
                "target":     float(parts[5]),
                "stop":       float(parts[6]),
                "rr":         float(parts[7]),
            })
        except (ValueError, IndexError):
            continue

    return setups


# ── Format Telegram message ───────────────────────────────────────────────────

def _format_msg(s: dict) -> str:
    ts = datetime.now(_TEHRAN).strftime("%Y-%m-%d %H:%M Tehran")  # type: ignore[arg-type]
    return (
        f"SIGNAL: {s['symbol']} {s['direction'].upper()}\n"
        f"Entry: {s['entry_low']} - {s['entry_high']}\n"
        f"Target: {s['target']}\n"
        f"Stop: {s['stop']}\n"
        f"R:R: {s['rr']}\n"
        f"Conf: {s['conf_tfs']}\n"
        f"Time: {ts}"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Scalp alert daemon (runs scalp.py every 3 min)")
    parser.add_argument("--top",   type=int, default=None, help="Pass --top N to scalp.py")
    parser.add_argument("--long",  action="store_true", help="Long setups only")
    parser.add_argument("--short", action="store_true", help="Short setups only")
    args = parser.parse_args()

    extra: list[str] = []
    if args.top is not None:
        extra += ["--top", str(args.top)]
    if args.long:
        extra.append("--long")
    if args.short:
        extra.append("--short")

    print(f"Scalp alert daemon started. Scanning every {_INTERVAL}s.", flush=True)
    print(f"Token: ...{config.TELEGRAM_TOKEN[-6:]}  Chat: {config.TELEGRAM_CHAT_ID}", flush=True)

    seen: set[str] = set()

    while True:
        ts = datetime.now(_TEHRAN).strftime("%Y-%m-%d %H:%M")  # type: ignore[arg-type]
        print(f"\n[{ts} Tehran] Running scan...", flush=True)

        setups = _run_scalp(extra)
        print(f"  Found {len(setups)} setup(s).", flush=True)

        new_count = 0
        for s in setups:
            key = _sig_key(s)
            if key not in seen:
                seen.add(key)
                msg = _format_msg(s)
                print(f"  NEW -> {s['symbol']} {s['direction'].upper()}  R:R={s['rr']}", flush=True)
                _send_telegram(msg)
                new_count += 1

        print(f"  New alerts sent: {new_count}", flush=True)
        time.sleep(_INTERVAL)


if __name__ == "__main__":
    main()
