"""All configuration constants. No magic numbers anywhere else in the codebase."""

EXECUTION_TF: str = "1h"
ATR_PERIOD: int = 14
MIN_LB: int = 3
MAX_LB: int = 20
SWEEP_MAX_CANDLES: int = 2
SL_BUFFER: float = 0.10       # × ATR behind sweep low

MIN_RR: float = 2.0

# Costs (mandatory in backtest)
TAKER_FEE: float = 0.0006     # per side
SLIPPAGE_ATR_MULT: float = 0.05
FUNDING_RATE: float = 0.0001

# Statistical validity
MIN_TRADES: int = 100
FOLDS: int = 5
WARMUP: int = 50              # warm-up candles before first signal

# Network
RETRY_ATTEMPTS: int = 3
RETRY_DELAYS: list[int] = [5, 10, 30]
