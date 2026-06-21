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
    shadow_extreme: float   # candle low (long) or high (short) — SL anchor


@dataclass
class Gate2Result:
    """Result of the hierarchical Gate 2 (4H sweep → LTF OB/FVG)."""
    sweep_bar_idx: int        # index in df_4h of the confirmed sweep bar
    swept_level: float        # price level swept (buy-side HIGH for SHORT, sell-side LOW for LONG)
    sweep_zone: "Zone"        # zone built from the 4H sweep
    primary_zone: "Zone"      # best LTF OB/FVG zone for entry/SL (prefer 1H)
    primary_tf: str           # "1h", "30m", or "15m"
    entry: float              # entry price = zone edge of primary_zone
    shadow_extreme: float     # for SL (last LTF bar's high for SHORT, low for LONG)
    grade: str                # "A+" or "B" from OB rejection quality
    ltf_confirmed: list       # list of confirmed TFs e.g. ["1h", "30m"]
    confluence_stars: int     # 1, 2, or 3
    sweep_tf: str = "4h"      # "4h" (strong) or "1h" (fallback)


@dataclass(frozen=True)
class MicroEntry:
    entry: float
    shadow_extreme: float   # LTF candle low (long) or high (short) — refined SL anchor
    score: float            # 0–12 composite quality score
    zone: "Zone"            # the LTF micro OB used


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
