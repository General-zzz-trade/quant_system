"""PnL comparison tool — compare backtest vs live performance."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass(frozen=True, slots=True)
class PnLPoint:
    """Single equity point for comparison."""
    ts: datetime
    equity: Decimal
    realized: Decimal
    unrealized: Decimal


@dataclass(frozen=True, slots=True)
class PnLComparisonResult:
    """Result of comparing two equity curves."""
    backtest_final_equity: Decimal
    live_final_equity: Decimal
    backtest_return_pct: float
    live_return_pct: float
    return_divergence_pct: float
    backtest_max_dd_pct: float
    live_max_dd_pct: float
    correlation: float
    tracking_error_pct: float
    aligned_points: int
    warnings: tuple[str, ...]


def load_equity_csv(path: Path) -> List[PnLPoint]:
    """Load equity curve from backtest output CSV."""
    points: List[PnLPoint] = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("ts", "")
            try:
                ts = datetime.fromisoformat(ts_str)
            except (ValueError, TypeError):
                continue
            points.append(PnLPoint(
                ts=ts,
                equity=Decimal(row.get("equity", "0")),
                realized=Decimal(row.get("realized", "0")),
                unrealized=Decimal(row.get("unrealized", "0")),
            ))
    return points


def _max_drawdown(equities: Sequence[Decimal]) -> float:
    """Compute max drawdown percentage."""
    if len(equities) < 2:
        return 0.0
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = float((peak - eq) / peak) * 100
            max_dd = max(max_dd, dd)
    return max_dd


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation between two return series."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx < 1e-15 or sy < 1e-15:
        return 0.0
    return cov / (sx * sy)


def compare_pnl(
    backtest: Sequence[PnLPoint],
    live: Sequence[PnLPoint],
) -> PnLComparisonResult:
    """Compare backtest and live equity curves.

    Aligns by timestamp (hour-level) and computes divergence metrics.
    """
    warnings: list[str] = []

    if not backtest or not live:
        return PnLComparisonResult(
            backtest_final_equity=backtest[-1].equity if backtest else Decimal("0"),
            live_final_equity=live[-1].equity if live else Decimal("0"),
            backtest_return_pct=0.0, live_return_pct=0.0,
            return_divergence_pct=0.0,
            backtest_max_dd_pct=0.0, live_max_dd_pct=0.0,
            correlation=0.0, tracking_error_pct=0.0,
            aligned_points=0, warnings=("empty input",),
        )

    # Build timestamp index for alignment (round to hour)
    bt_by_hour: Dict[str, PnLPoint] = {}
    for p in backtest:
        key = p.ts.strftime("%Y-%m-%d %H")
        bt_by_hour[key] = p

    live_by_hour: Dict[str, PnLPoint] = {}
    for p in live:
        key = p.ts.strftime("%Y-%m-%d %H")
        live_by_hour[key] = p

    # Align
    common_keys = sorted(set(bt_by_hour.keys()) & set(live_by_hour.keys()))
    if len(common_keys) < 2:
        warnings.append(f"only {len(common_keys)} aligned points")

    bt_aligned = [bt_by_hour[k] for k in common_keys]
    live_aligned = [live_by_hour[k] for k in common_keys]

    # Returns
    bt_returns: list[float] = []
    live_returns: list[float] = []
    for i in range(1, len(bt_aligned)):
        prev_bt = float(bt_aligned[i - 1].equity)
        prev_live = float(live_aligned[i - 1].equity)
        if prev_bt > 0:
            bt_returns.append((float(bt_aligned[i].equity) - prev_bt) / prev_bt)
        else:
            bt_returns.append(0.0)
        if prev_live > 0:
            live_returns.append((float(live_aligned[i].equity) - prev_live) / prev_live)
        else:
            live_returns.append(0.0)

    # Metrics
    bt_start = float(backtest[0].equity)
    bt_end = float(backtest[-1].equity)
    live_start = float(live[0].equity)
    live_end = float(live[-1].equity)

    bt_ret = ((bt_end - bt_start) / bt_start * 100) if bt_start > 0 else 0.0
    live_ret = ((live_end - live_start) / live_start * 100) if live_start > 0 else 0.0

    # Tracking error: std of return differences
    ret_diffs = [b - l for b, l in zip(bt_returns, live_returns)]
    te = 0.0
    if len(ret_diffs) >= 2:
        mean_diff = sum(ret_diffs) / len(ret_diffs)
        var_diff = sum((d - mean_diff) ** 2 for d in ret_diffs) / len(ret_diffs)
        te = math.sqrt(var_diff) * 100  # annualize? keep as per-period for now

    # Divergence checks
    if abs(bt_ret - live_ret) > 5.0:
        warnings.append(f"return divergence {abs(bt_ret - live_ret):.1f}% exceeds 5%")

    corr = _correlation(bt_returns, live_returns) if bt_returns else 0.0
    if corr < 0.9 and len(bt_returns) >= 10:
        warnings.append(f"low return correlation {corr:.3f}")

    return PnLComparisonResult(
        backtest_final_equity=backtest[-1].equity,
        live_final_equity=live[-1].equity,
        backtest_return_pct=bt_ret,
        live_return_pct=live_ret,
        return_divergence_pct=abs(bt_ret - live_ret),
        backtest_max_dd_pct=_max_drawdown([p.equity for p in backtest]),
        live_max_dd_pct=_max_drawdown([p.equity for p in live]),
        correlation=corr,
        tracking_error_pct=te,
        aligned_points=len(common_keys),
        warnings=tuple(warnings),
    )


def compare_from_files(
    backtest_csv: Path,
    live_csv: Path,
) -> PnLComparisonResult:
    """Load and compare two equity curve CSVs."""
    bt = load_equity_csv(backtest_csv)
    live = load_equity_csv(live_csv)
    return compare_pnl(bt, live)
