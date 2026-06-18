"""Tests for engine/telegram.py."""
import unittest.mock as mock
import pytest


def _sample_signal():
    return {
        "symbol": "ETHUSDT", "direction": "long",
        "entry": 3500.12, "stop": 3420.0, "target": 3700.0,
        "rr": 2.5, "score": 0.82,
        "zone_type": "ob_overlap", "rsi_confirm": True, "macd_confirm": False,
        "timestamp": "2024-01-01T12:00:00+00:00", "ltf_entry_index": 5,
    }


def test_format_signal_contains_key_fields():
    from engine.telegram import format_signal
    text = format_signal(_sample_signal())
    assert "ETHUSDT" in text
    assert "LONG" in text
    assert "3500" in text   # entry price
    assert "R:R" in text
    assert "Score" in text
    assert "RSI" in text
    assert "MACD" in text


def test_format_signal_no_unescaped_special_chars():
    from engine.telegram import format_signal
    text = format_signal(_sample_signal())
    # In MarkdownV2 these must be escaped — raw unescaped periods/dashes are problematic
    # Basic check: text should not contain '**' (wrong MD syntax for v2)
    assert "**" not in text


def test_send_signal_no_credentials_returns_false():
    import config
    original_token = config.TELEGRAM_BOT_TOKEN
    original_chat = config.TELEGRAM_CHAT_ID
    try:
        config.TELEGRAM_BOT_TOKEN = ""
        config.TELEGRAM_CHAT_ID = ""
        from engine.telegram import send_signal
        result = send_signal(_sample_signal())
        assert result is False
    finally:
        config.TELEGRAM_BOT_TOKEN = original_token
        config.TELEGRAM_CHAT_ID = original_chat


def test_send_signal_posts_to_telegram():
    import config
    original_token = config.TELEGRAM_BOT_TOKEN
    original_chat = config.TELEGRAM_CHAT_ID
    try:
        config.TELEGRAM_BOT_TOKEN = "fake_token"
        config.TELEGRAM_CHAT_ID = "fake_chat"

        mock_resp = mock.Mock()
        mock_resp.status_code = 200

        with mock.patch("engine.telegram.requests.post", return_value=mock_resp) as mock_post:
            from engine.telegram import send_signal
            result = send_signal(_sample_signal())

        assert result is True
        assert mock_post.called
        # payload is passed as json= keyword arg
        call_kwargs = mock_post.call_args.kwargs
        payload = call_kwargs.get("json", {})
        assert payload.get("parse_mode") == "MarkdownV2"
        assert payload.get("chat_id") == "fake_chat"
    finally:
        config.TELEGRAM_BOT_TOKEN = original_token
        config.TELEGRAM_CHAT_ID = original_chat


def test_send_signal_returns_false_on_http_error():
    import config
    original_token = config.TELEGRAM_BOT_TOKEN
    original_chat = config.TELEGRAM_CHAT_ID
    try:
        config.TELEGRAM_BOT_TOKEN = "fake_token"
        config.TELEGRAM_CHAT_ID = "fake_chat"

        mock_resp = mock.Mock()
        mock_resp.status_code = 400
        mock_resp.text = "Bad Request"

        with mock.patch("engine.telegram.requests.post", return_value=mock_resp):
            from engine.telegram import send_signal
            result = send_signal(_sample_signal())

        assert result is False
    finally:
        config.TELEGRAM_BOT_TOKEN = original_token
        config.TELEGRAM_CHAT_ID = original_chat
