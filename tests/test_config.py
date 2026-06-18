import config


def test_required_attributes_exist():
    required = [
        "HTF", "MACRO_TF", "LTF",
        "SWING_LEN", "OB_SWING_LOOKBACK", "FVG_BODY_PERC",
        "LIQ_MARGIN_DIV", "ATR_LEN",
        "RSI_LEN", "RSI_WINDOW", "RSI_LOW", "RSI_HIGH",
        "MACD_FAST", "MACD_SLOW", "MACD_SIGNAL",
        "MIN_RR", "TRAIL_START_R", "STOP_BUFFER_PCT",
        "SCORE_THRESHOLD", "MAX_POSITIONS_PER_DAY",
        "CACHE_TTL_SECONDS", "RATE_LIMIT_DELAY", "CANDLE_LIMIT",
        "BITUNIX_BASE_URL", "BTC_SYMBOL",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ]
    for attr in required:
        assert hasattr(config, attr), f"config missing: {attr}"


def test_types():
    assert isinstance(config.HTF, list)
    assert isinstance(config.SWING_LEN, int)
    assert isinstance(config.MIN_RR, float)
    assert isinstance(config.SCORE_THRESHOLD, float)
    assert isinstance(config.BITUNIX_BASE_URL, str)


def test_sensible_values():
    assert config.SWING_LEN >= 2
    assert 0 < config.FVG_BODY_PERC < 1
    assert config.MIN_RR >= 1.0
    assert 0 < config.SCORE_THRESHOLD <= 1.0
    assert config.MAX_POSITIONS_PER_DAY >= 1
    assert config.RATE_LIMIT_DELAY >= 0
