from __future__ import annotations

import logging

import requests

import config

logger = logging.getLogger(__name__)

_ESCAPE_CHARS = r"\_*[]()~`>#+-=|{}.!"


def _escape_md(text: str) -> str:
    for ch in _ESCAPE_CHARS:
        text = text.replace(ch, f"\\{ch}")
    return text


def format_signal(signal: dict) -> str:
    direction = signal["direction"].upper()
    emoji = "🟢" if signal["direction"] == "long" else "🔴"
    rsi_str = "YES ✅" if signal["rsi_confirm"] else "NO ❌"
    macd_str = "YES ✅" if signal["macd_confirm"] else "NO ❌"
    zone_label = signal["zone_type"].replace("_", " ").title()

    lines = [
        f"🚨 *ICT SIGNAL — {_escape_md(signal['symbol'])}*",
        f"Direction: {emoji} *{direction}*",
        f"Zone Type: {_escape_md(zone_label)}",
        "─────────────────────",
        f"Entry:  `{signal['entry']:.6g}`",
        f"Stop:   `{signal['stop']:.6g}`",
        f"Target: `{signal['target']:.6g}`",
        f"R:R:    `{signal['rr']:.2f}`",
        f"Score:  `{signal['score'] * 100:.0f}%`",
        "─────────────────────",
        f"RSI Confirm:  {rsi_str}",
        f"MACD Confirm: {macd_str}",
        "─────────────────────",
        f"⏰ `{_escape_md(signal['timestamp'])}` UTC",
    ]
    return "\n".join(lines)


def send_signal(signal: dict) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — signal not sent")
        return False

    text = format_signal(signal)
    return _post(text)


def send_alert(message: str) -> bool:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured — alert not sent")
        return False
    return _post(_escape_md(message))


def _post(text: str) -> bool:
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "MarkdownV2",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        logger.error("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
        return False
    except requests.RequestException as exc:
        logger.error("Telegram request error: %s", exc)
        return False
