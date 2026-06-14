"""Backtest metrics: expectancy, Wilson CI, walk-forward folds."""

import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Metrics, TradeResult


def compute(results: list[TradeResult]) -> Metrics:
    """Compute full metrics suite from trade results."""
    n = len(results)

    if n == 0:
        return Metrics(
            trades=0, wr=0.0, wr_ci_low=0.0, wr_ci_high=0.0,
            expectancy_r=0.0, folds_positive=0, fold_count=config.FOLDS,
            flag="unreliable",
        )

    wins = sum(1 for t in results if t.won)
    wr = wins / n
    expectancy = sum(t.r_multiple for t in results) / n

    ci_low, ci_high = _wilson_ci(wins, n)

    folds_positive = _walk_forward(results, config.FOLDS)

    if n < config.MIN_TRADES:
        flag: str = "unreliable"
    elif ci_low < 0.5 or _single_fold_dominance(results, config.FOLDS):
        flag = "unstable"
    else:
        flag = "valid"

    return Metrics(
        trades=n,
        wr=wr,
        wr_ci_low=ci_low,
        wr_ci_high=ci_high,
        expectancy_r=expectancy,
        folds_positive=folds_positive,
        fold_count=config.FOLDS,
        flag=flag,  # type: ignore[arg-type]
    )


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _walk_forward(results: list[TradeResult], folds: int) -> int:
    """Anchored walk-forward: fold k uses results[0..k*chunk]. Count folds with positive expectancy."""
    n = len(results)
    chunk = max(1, n // folds)
    positive = 0
    for k in range(1, folds + 1):
        subset = results[: k * chunk]
        if subset and sum(t.r_multiple for t in subset) / len(subset) > 0:
            positive += 1
    return positive


def _single_fold_dominance(results: list[TradeResult], folds: int) -> bool:
    """Return True if removing any single fold turns expectancy negative (fragile result)."""
    n = len(results)
    chunk = max(1, n // folds)
    total_exp = sum(t.r_multiple for t in results) / n if n else 0
    if total_exp <= 0:
        return True
    for k in range(folds):
        fold_start = k * chunk
        fold_end = (k + 1) * chunk
        # Use index exclusion — value-based equality fails when results have duplicates
        rest = results[:fold_start] + results[fold_end:]
        if rest and sum(t.r_multiple for t in rest) / len(rest) <= 0:
            return True
    return False
