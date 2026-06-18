"""
ICT Multi-Timeframe Scanner — orchestrator.
Runs every 5 minutes, enforces ≤2 signals/day cap, monitors open positions.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timezone

import config
from engine.scanner import run_scan
from engine.telegram import send_signal, send_alert
from engine.monitor import monitor_loop, PositionDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("main")

SCAN_INTERVAL = 300  # 5 minutes (aligned with LTF=5m)


def main(dry_run: bool = False) -> None:
    """
    dry_run=True: print signals to stdout, do not send to Telegram.
    """
    daily_state = {
        "date": date.today(),
        "signals_sent": 0,
        "open_positions": [],
    }

    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(daily_state["open_positions"],),
        daemon=True,
        name="monitor",
    )
    monitor_thread.start()
    logger.info("Monitor thread started")

    while True:
        today = date.today()
        if today != daily_state["date"]:
            daily_state["date"] = today
            daily_state["signals_sent"] = 0
            logger.info("Daily state reset for %s", today)

        remaining = config.MAX_POSITIONS_PER_DAY - daily_state["signals_sent"]
        if remaining <= 0:
            logger.info("Daily cap reached (%d/%d). Next scan after sleep.",
                        daily_state["signals_sent"], config.MAX_POSITIONS_PER_DAY)
            time.sleep(SCAN_INTERVAL)
            continue

        try:
            signals = run_scan()
        except Exception as exc:
            logger.error("Scan error: %s", exc)
            time.sleep(SCAN_INTERVAL)
            continue

        top_signals = signals[:remaining]

        for sig in top_signals:
            if dry_run:
                print(f"\n{'='*50}")
                print(f"SIGNAL: {sig['symbol']} {sig['direction'].upper()}")
                print(f"  Entry:  {sig['entry']:.6g}")
                print(f"  Stop:   {sig['stop']:.6g}")
                print(f"  Target: {sig['target']:.6g}")
                print(f"  R:R:    {sig['rr']:.2f}")
                print(f"  Score:  {sig['score']*100:.0f}%")
                print(f"  Zone:   {sig['zone_type']}")
                daily_state["signals_sent"] += 1
            else:
                if send_signal(sig):
                    daily_state["signals_sent"] += 1
                    logger.info("Signal sent: %s %s score=%.2f",
                                sig["symbol"], sig["direction"], sig["score"])

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    if dry:
        logger.info("Running in DRY-RUN mode (no Telegram sends)")
    main(dry_run=dry)
