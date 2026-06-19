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
class Zone:
    zone_type: Literal["ob", "fvg"]
    direction: Direction
    zone_high: float
    zone_low: float
    origin_index: int
    timeframe: str


@dataclass(frozen=True)
class RejectionCandle:
    candle_index: int
    grade: Literal["A+", "B"]
    zone: Zone
    entry: float


@dataclass(frozen=True)
class Setup:
    entry_low: float      # zone bottom (or refined 5m entry)
    entry_high: float     # zone top (entry trigger)
    target: float
    stop: float
    rr: float
    direction: Direction
    zone: "Zone | None" = None
    grade: Literal["A+", "B"] = "B"


@dataclass(frozen=True)
class TradeResult:
    won: bool
    r_multiple: float     # net, after all costs
    bars_held: int
    grade: Literal["A+", "B"] = "B"


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
