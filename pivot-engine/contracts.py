"""Data contracts — locked interfaces between all modules."""

from dataclasses import dataclass
from typing import Literal

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class Pivot:
    index: int            # candle index where pivot occurred
    confirm_index: int    # candle index where pivot was confirmed (always <= as_of)
    price: float
    kind: Literal["high", "low"]


@dataclass(frozen=True)
class Setup:
    entry_low: float      # 101 zone bottom (sweep low)
    entry_high: float     # 101 zone top (pivot_100 price)
    target: float         # 102 structural target
    stop: float           # SL behind sweep
    rr: float
    direction: Direction


@dataclass(frozen=True)
class TradeResult:
    won: bool
    r_multiple: float     # net, after all costs
    bars_held: int


@dataclass(frozen=True)
class Metrics:
    trades: int
    wr: float
    wr_ci_low: float      # Wilson 95% lower bound
    wr_ci_high: float     # Wilson 95% upper bound
    expectancy_r: float   # primary metric: mean net R
    folds_positive: int
    fold_count: int
    flag: Literal["valid", "unstable", "unreliable"]
