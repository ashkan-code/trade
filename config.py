import os

# ── Timeframes ────────────────────────────────────────────────────────────────
HTF = ["4H", "1H"]
MACRO_TF = "1D"
LTF = "5m"

# ── ICT parameters (PLACEHOLDERS — tune via backtest) ─────────────────────────
SWING_LEN = 5            # MSS swing lookback
OB_SWING_LOOKBACK = 10   # OB detection swing lookback
FVG_BODY_PERC = 0.36     # minimum body fraction for large-body FVG candle
LIQ_MARGIN_DIV = 2.5     # liquidity cluster margin = ATR / LIQ_MARGIN_DIV
ATR_LEN = 10

# ── Indicators (PLACEHOLDERS) ─────────────────────────────────────────────────
RSI_LEN = 14
RSI_WINDOW = 5           # look back N candles for RSI confirm
RSI_LOW = 30
RSI_HIGH = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ── Risk (PLACEHOLDERS) ───────────────────────────────────────────────────────
MIN_RR = 2.0
TRAIL_START_R = 1.5      # trail activates after TRAIL_START_R × initial risk
STOP_BUFFER_PCT = 0.001  # 0.1% buffer beyond OB for stop placement

# ── Scoring & limits (PLACEHOLDERS) ──────────────────────────────────────────
SCORE_THRESHOLD = 0.75
MAX_POSITIONS_PER_DAY = 2

# ── Infrastructure ────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 60
RATE_LIMIT_DELAY = 0.2   # seconds between API requests
CANDLE_LIMIT = 200

# ── Bitunix API ───────────────────────────────────────────────────────────────
BITUNIX_BASE_URL = "https://fapi.bitunix.com"
BTC_SYMBOL = "BTCUSDT"

# ── Telegram (from environment — never hardcode) ──────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
