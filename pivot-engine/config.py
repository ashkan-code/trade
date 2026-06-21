"""All configuration constants. No magic numbers anywhere else in the codebase."""

# Timeframes
HTF: str = "4h"
HTF_ALT: str = "1h"
LTF: str = "5m"
EXECUTION_TF: str = "1h"       # kept for backward compat
BTC_SYMBOL: str = "BTCUSDT"

# Structure detection
SWING_LEN: int = 5
OB_SWING_LOOKBACK: int = 10
FVG_BODY_PERC: float = 0.36    # midcandle body / range must be >= this to form FVG
ATR_PERIOD: int = 14
ATR_5m_PERIOD: int = 14
MIN_LB: int = 3
MAX_LB: int = 20
SWEEP_MAX_CANDLES: int = 2      # kept for backward compat

# Gate 2 — sweep detection (ICT/Wyckoff)
SWEEP_LOOKBACK_4H: int = 10           # 4H bars to look back for a swept swing level
SWEEP_LOOKBACK_1H: int = 20           # 1H bars to look back (smaller candles → wider window)
SWEEP_LOOKBACK: int = SWEEP_LOOKBACK_4H   # backward compat (used by find_sweep_rejection)
SWEEP_TO_OB_MAX_BARS: int = 6      # max TF-bar-equivalents after sweep to find LTF OB
VOLUME_MIN_RATIO: float = 1.0         # rejection candle volume >= this × SMA(volume,9)
VOLUME_SPIKE_MULTIPLIER: float = 2.0  # counter-momentum spike threshold (opposite-direction candle)
LOOKBACK_BARS: int = 5                # bars before sweep to check for counter-institutional momentum

# Gate 2 — shape scoring (no longer hard filters; used for grade A+/B)
PENETRATION_MIN: float = 0.20   # shadow must enter >= 20% of zone height
WICK_RATIO_MIN: float = 1.0     # rejection wick must be >= 1× body
MIN_SETUP_GRADE: str = "B"      # "A+" or "B"; A+ = body entirely outside zone
OB_TOUCH_LOOKBACK: int = 10     # bars back to check for prior zone touch (single-touch rule)
CONFIRM_NEXT_BAR: bool = False   # require next bar to confirm (stricter; off by default)

# Gate 3 — RSI + MACD confluence
RSI_LEN: int = 14
RSI_WINDOW: int = 5
RSI_OB: float = 70.0
RSI_OS: float = 30.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL_LEN: int = 9

# Gate 4 — micro entry (LTF OB/FVG)
DISPLACEMENT_ATR_MIN: float = 0.25   # impulse after OB must be >= this × ATR_5m to qualify

# Risk management
SL_BUFFER: float = 0.5          # % (of zone height) buffer behind zone edge for SL
MIN_RR: float = 2.0
PARTIAL_TP: bool = True
TP_PARTIAL_FRACTION: float = 0.5  # close half at TP1, keep half for TP2

# Costs (mandatory in backtest)
TAKER_FEE: float = 0.0006       # per side
SLIPPAGE_ATR_MULT: float = 0.05
FUNDING_RATE: float = 0.0001

# Statistical validity
MIN_TRADES: int = 30             # lowered from 100 (multi-TF signals are rarer)
FOLDS: int = 5
WARMUP: int = 50

# Scanner
TOP_N: int = 20             # top symbols by 24h USDT turnover (used when --top N specified)
MIN_VOLUME_USD: float = 0.0           # 0 = no volume filter — scan ALL USDT-M symbols
MAX_CONCURRENT: int = 10    # parallel symbol workers in live scan
SIMILARITY_MIN: float = 0.7           # Pearson correlation threshold vs BTC pivot vector
SIMILARITY_PIVOTS: int = 6            # number of consecutive pivot-to-pivot % changes to compare

# Network
RETRY_ATTEMPTS: int = 3
RETRY_DELAYS: list[int] = [5, 10, 30]
