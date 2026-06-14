"""Honest report generator. Format matches Section 6 of the build spec."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from contracts import Metrics, Setup


def generate(
    symbol: str,
    direction: str,
    setup: Setup | None,
    metrics: Metrics,
    audit_result: tuple[bool, str],
    signal: dict | None,
) -> str:
    """Return formatted honest report string."""
    audit_passed, audit_detail = audit_result

    lines: list[str] = ["=" * 60]

    if setup is not None:
        lines += [
            f"SYMBOL    : {symbol}",
            f"DIRECTION : {direction.upper()}",
            f"ENTRY ZONE: {setup.entry_low:.4f} – {setup.entry_high:.4f}  (net of costs)",
            f"STOP      : {setup.stop:.4f}   TARGET(102): {setup.target:.4f}",
            f"R:R       : {setup.rr:.2f}",
        ]
    else:
        lines += [
            f"SYMBOL    : {symbol}",
            f"DIRECTION : {direction.upper()}",
            "ENTRY ZONE: — (no valid setup found)",
            "STOP      : —   TARGET(102): —",
            "R:R       : —",
        ]

    wr_pct = f"{metrics.wr * 100:.1f}%"
    ci_low_pct = f"{metrics.wr_ci_low * 100:.1f}%"
    ci_high_pct = f"{metrics.wr_ci_high * 100:.1f}%"
    unreliable_warn = "  ⚠ UNRELIABLE" if metrics.trades < config.MIN_TRADES else ""

    lines += [
        f"EXPECTANCY: {metrics.expectancy_r:+.4f} R",
        f"WR (net)  : {wr_pct}  [Wilson 95%: {ci_low_pct} – {ci_high_pct}]",
        f"TRADES    : {metrics.trades}{unreliable_warn}",
        f"WALK-FWD  : {metrics.folds_positive} / {metrics.fold_count} folds positive",
        f"FLAG      : {metrics.flag.upper()}",
    ]

    if signal and signal.get("signal") is None:
        lines.append(f"\n⛔ SIGNAL BLOCKED: {signal.get('blocked_reason', 'unknown')}")
    elif signal and signal.get("signal") == "active":
        lines.append("\n✅ SIGNAL: ACTIVE — all 6 gates passed")

    lines += [
        "",
        "HONEST ASSESSMENT",
        "-" * 40,
        _reliability_text(metrics),
        _biggest_assumption(),
        f"repaint-audit: {'PASS' if audit_passed else 'FAIL'} — {audit_detail}",
        "=" * 60,
    ]

    return "\n".join(lines)


def _reliability_text(m: Metrics) -> str:
    if m.flag == "unreliable":
        return f"⚠ Only {m.trades} trades — sample too small to trust any edge claim."
    if m.flag == "unstable":
        return (
            f"⚠ WR lower CI ({m.wr_ci_low:.1%}) below 50%, or result driven by single fold. "
            "Edge not confirmed across walk-forward splits."
        )
    return (
        f"✓ {m.trades} trades, expectancy {m.expectancy_r:+.3f} R, "
        f"{m.folds_positive}/{m.fold_count} folds positive. "
        "Consistent across anchored walk-forward — reasonable confidence."
    )


def _biggest_assumption() -> str:
    return (
        "Biggest assumption: sweep+reclaim behaviour (101 zone) remains mean-reverting "
        "in future regimes. If market structure shifts to trend-following with no reclaims, "
        "win rate collapses and SL hits dominate."
    )
