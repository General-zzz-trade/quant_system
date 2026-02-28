"""Walk-Forward validation for multi-factor strategy.

Usage:
    python3 -m scripts.run_walk_forward_validation
    python3 -m scripts.run_walk_forward_validation --optimize
    python3 -m scripts.run_walk_forward_validation --train-months 6 --test-months 3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from pathlib import Path
from statistics import stdev
from typing import Any, Dict, List, Optional, Tuple

from runner.backtest.csv_io import OhlcvBar, iter_ohlcv_csv
from runner.backtest.metrics import EquityPoint, _build_trades_from_fills, _max_drawdown


BARS_PER_MONTH = 730  # ~1h bars per month (30.4 days * 24)


@dataclass
class FoldResult:
    fold_idx: int
    train_bars: int
    test_bars: int
    train_start_ts: str
    train_end_ts: str
    test_start_ts: str
    test_end_ts: str
    sharpe: float
    total_return: float
    max_drawdown: float
    trades: int
    params: Optional[Dict[str, Any]] = None


def _bars_to_temp_csv(bars: List[OhlcvBar]) -> Path:
    """Write bars to a temporary CSV file, return path."""
    fd, path = tempfile.mkstemp(suffix=".csv", prefix="wf_")
    import os
    os.close(fd)
    p = Path(path)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([
                b.ts.isoformat(),
                str(b.o), str(b.h), str(b.l), str(b.c),
                str(b.v) if b.v is not None else "0",
            ])
    return p


def _compute_sharpe(equity: List[EquityPoint]) -> float:
    """Compute annualized Sharpe from equity points."""
    if len(equity) < 2:
        return 0.0
    eq_vals = [float(ep.equity) for ep in equity]
    rets = []
    for i in range(1, len(eq_vals)):
        if eq_vals[i - 1] != 0:
            rets.append((eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1])
    if len(rets) < 2:
        return 0.0
    mean_r = sum(rets) / len(rets)
    try:
        std_r = stdev(rets)
    except Exception:
        return 0.0
    if std_r <= 0:
        return 0.0
    # Estimate bars per year from timestamps
    total_sec = (equity[-1].ts - equity[0].ts).total_seconds()
    years = total_sec / (365.0 * 24 * 3600) if total_sec > 0 else 0
    bars_per_year = len(rets) / years if years > 0 else 8760.0
    return mean_r / std_r * sqrt(bars_per_year)


def _run_fold(
    bars: List[OhlcvBar],
    train_end: int,
    test_end: int,
    config_kwargs: Dict[str, Any],
    symbol: str,
    starting_balance: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal,
) -> FoldResult:
    """Run a single fold: backtest on bars[0:test_end], extract test metrics."""
    from strategies.multi_factor.decision_module import MultiFactorConfig, MultiFactorDecisionModule
    from runner.backtest_runner import run_backtest

    fold_bars = bars[0:test_end]
    csv_path = _bars_to_temp_csv(fold_bars)

    try:
        cfg = MultiFactorConfig(symbol=symbol, **config_kwargs)
        module = MultiFactorDecisionModule(config=cfg)

        equity, fills = run_backtest(
            csv_path=csv_path,
            symbol=symbol,
            starting_balance=starting_balance,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            decision_modules=[module],
        )
    finally:
        csv_path.unlink(missing_ok=True)

    # Extract test-period equity (equity has one point per bar)
    test_equity = equity[train_end:] if len(equity) > train_end else []

    # Compute test metrics
    sharpe = _compute_sharpe(test_equity)

    if test_equity:
        start_eq = float(test_equity[0].equity)
        end_eq = float(test_equity[-1].equity)
        total_return = (end_eq - start_eq) / start_eq if start_eq != 0 else 0.0
        mdd = float(_max_drawdown([ep.equity for ep in test_equity]))
    else:
        total_return = 0.0
        mdd = 0.0

    # Count trades in test period
    test_start_ts = bars[train_end].ts if train_end < len(bars) else None
    test_fills = []
    if test_start_ts is not None:
        from runner.backtest.metrics import _parse_fill_ts
        for f in fills:
            ft = _parse_fill_ts(f.get("ts"))
            if ft is not None and ft >= test_start_ts:
                test_fills.append(f)
    test_trades = _build_trades_from_fills(test_fills)

    return FoldResult(
        fold_idx=0,  # set by caller
        train_bars=train_end,
        test_bars=test_end - train_end,
        train_start_ts=bars[0].ts.strftime("%Y-%m"),
        train_end_ts=bars[train_end - 1].ts.strftime("%Y-%m") if train_end > 0 else "",
        test_start_ts=bars[train_end].ts.strftime("%Y-%m") if train_end < len(bars) else "",
        test_end_ts=bars[test_end - 1].ts.strftime("%Y-%m") if test_end <= len(bars) else "",
        sharpe=round(sharpe, 4),
        total_return=total_return,
        max_drawdown=mdd,
        trades=len(test_trades),
    )


def _generate_folds(
    total_bars: int,
    train_size: int,
    test_size: int,
) -> List[Tuple[int, int]]:
    """Generate (train_end, test_end) pairs for expanding window WF."""
    folds = []
    fold_idx = 0
    while True:
        train_end = train_size + fold_idx * test_size
        test_end = train_end + test_size
        if test_end > total_bars:
            # Use remaining bars if enough for at least half a test window
            remaining = total_bars - train_end
            if remaining >= test_size // 2:
                folds.append((train_end, total_bars))
            break
        folds.append((train_end, test_end))
        fold_idx += 1
    return folds


def _run_fixed_param_validation(
    bars: List[OhlcvBar],
    args: argparse.Namespace,
) -> List[FoldResult]:
    """Mode 1: fixed params, same config for all folds."""
    train_size = args.train_months * BARS_PER_MONTH
    test_size = args.test_months * BARS_PER_MONTH
    folds = _generate_folds(len(bars), train_size, test_size)

    config_kwargs: Dict[str, Any] = {}  # use defaults

    results = []
    for i, (train_end, test_end) in enumerate(folds):
        print(f"  Running fold {i}/{len(folds)}...", end=" ", flush=True)
        r = _run_fold(
            bars=bars,
            train_end=train_end,
            test_end=test_end,
            config_kwargs=config_kwargs,
            symbol=args.symbol,
            starting_balance=Decimal(str(args.starting_balance)),
            fee_bps=Decimal(str(args.fee_bps)),
            slippage_bps=Decimal(str(args.slippage_bps)),
        )
        r.fold_idx = i
        results.append(r)
        print(
            f"Sharpe={r.sharpe:6.2f}  Return={r.total_return:7.1%}  "
            f"DD={r.max_drawdown:6.1%}  Trades={r.trades:3d}"
        )

    return results


def _run_param_optimization(
    bars: List[OhlcvBar],
    args: argparse.Namespace,
) -> List[FoldResult]:
    """Mode 2: parameter grid search per fold using walk_forward_optimize."""
    from research.walk_forward_optimizer import walk_forward_optimize

    train_size = args.train_months * BARS_PER_MONTH
    test_size = args.test_months * BARS_PER_MONTH

    param_grid = [
        {
            "atr_stop_multiple": s,
            "trailing_atr_multiple": t,
            "trend_threshold": th,
            "max_position_pct": mp,
        }
        for s in [2.0, 3.0, 4.0]
        for t in [4.0, 6.0, 8.0]
        for th in [0.20, 0.30, 0.40]
        for mp in [0.50, 0.80]
    ]

    folds_spec = _generate_folds(len(bars), train_size, test_size)
    n_folds = len(folds_spec)

    def evaluate_fn(params: Dict[str, Any], start: int, end: int) -> float:
        """Evaluate params on bars[start:end], return Sharpe."""
        # We always start from 0 for warmup, but measure only [start:end]
        actual_end = min(end, len(bars))
        if actual_end <= start:
            return 0.0
        r = _run_fold(
            bars=bars,
            train_end=start,
            test_end=actual_end,
            config_kwargs=params,
            symbol=args.symbol,
            starting_balance=Decimal(str(args.starting_balance)),
            fee_bps=Decimal(str(args.fee_bps)),
            slippage_bps=Decimal(str(args.slippage_bps)),
        )
        return r.sharpe

    print(f"  Grid size: {len(param_grid)} combinations x {n_folds} folds")

    wf_result = walk_forward_optimize(
        data_length=len(bars),
        param_grid=param_grid,
        evaluate_fn=evaluate_fn,
        n_folds=n_folds,
        train_ratio=train_size / (train_size + test_size),
        expanding=True,
        metric_higher_is_better=True,
    )

    # Convert WalkForwardFold results to our FoldResult format
    results = []
    for f in wf_result.folds:
        te = min(f.test_end, len(bars))
        ts_ = min(f.test_start, len(bars) - 1)
        results.append(FoldResult(
            fold_idx=f.fold_idx,
            train_bars=f.train_end,
            test_bars=te - f.test_start,
            train_start_ts=bars[f.train_start].ts.strftime("%Y-%m"),
            train_end_ts=bars[f.train_end - 1].ts.strftime("%Y-%m") if f.train_end > 0 else "",
            test_start_ts=bars[ts_].ts.strftime("%Y-%m"),
            test_end_ts=bars[te - 1].ts.strftime("%Y-%m") if te > 0 else "",
            sharpe=round(f.test_metric, 4),
            total_return=0.0,  # not tracked by WF optimizer
            max_drawdown=0.0,
            trades=0,
            params=f.best_params,
        ))

    print(f"\n  Best params frequency: {wf_result.best_params_frequency}")
    print(f"  Avg train Sharpe: {wf_result.avg_train_metric:.4f}")
    print(f"  Avg test Sharpe:  {wf_result.avg_test_metric:.4f}")
    print(f"  Overfit: {'YES' if wf_result.is_overfit else 'NO'}")

    return results


def _print_report(
    results: List[FoldResult],
    full_sample_sharpe: Optional[float],
    out_dir: Optional[Path],
) -> None:
    """Print aggregate report and save to disk."""
    if not results:
        print("No fold results.")
        return

    sharpes = [r.sharpe for r in results]
    returns = [r.total_return for r in results]
    dds = [r.max_drawdown for r in results]
    trades = [r.trades for r in results]

    avg_sharpe = sum(sharpes) / len(sharpes)
    avg_return = sum(returns) / len(returns)
    avg_dd = sum(dds) / len(dds)
    avg_trades = sum(trades) / len(trades)

    std_sharpe = stdev(sharpes) if len(sharpes) >= 2 else 0.0
    std_return = stdev(returns) if len(returns) >= 2 else 0.0
    std_dd = stdev(dds) if len(dds) >= 2 else 0.0
    std_trades = stdev(trades) if len(trades) >= 2 else 0.0

    win_folds = sum(1 for r in results if r.total_return > 0)

    print(f"\n{'=' * 50}")
    print(f"{'Aggregate':^50}")
    print(f"{'=' * 50}")
    print(f"Avg Sharpe:    {avg_sharpe:.2f} +/- {std_sharpe:.2f}")
    print(f"Avg Return:   {avg_return:.1%} +/- {std_return:.1%}")
    print(f"Avg Max DD:   {avg_dd:.1%} +/- {std_dd:.1%}")
    print(f"Avg Trades:    {avg_trades:.0f} +/- {std_trades:.0f}")
    print(f"Win folds:    {win_folds}/{len(results)} ({win_folds/len(results):.0%})")

    if full_sample_sharpe is not None:
        degradation = 1.0 - avg_sharpe / full_sample_sharpe if full_sample_sharpe != 0 else 0.0
        print(f"\nFull-sample Sharpe: {full_sample_sharpe:.4f}")
        print(f"Avg OOS Sharpe:     {avg_sharpe:.4f}")
        print(f"Degradation:        {degradation:.1%}")
        overfit = degradation > 0.50
        print(f"Overfit:            {'YES' if overfit else 'NO'} (degradation {'>' if overfit else '<'} 50%)")

    # Save results
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "folds": [
                {
                    "fold_idx": r.fold_idx,
                    "train_bars": r.train_bars,
                    "test_bars": r.test_bars,
                    "train_period": f"{r.train_start_ts} ~ {r.train_end_ts}",
                    "test_period": f"{r.test_start_ts} ~ {r.test_end_ts}",
                    "sharpe": r.sharpe,
                    "total_return": round(r.total_return, 6),
                    "max_drawdown": round(r.max_drawdown, 6),
                    "trades": r.trades,
                    "params": r.params,
                }
                for r in results
            ],
            "aggregate": {
                "avg_sharpe": round(avg_sharpe, 4),
                "std_sharpe": round(std_sharpe, 4),
                "avg_return": round(avg_return, 6),
                "std_return": round(std_return, 6),
                "avg_max_drawdown": round(avg_dd, 6),
                "std_max_drawdown": round(std_dd, 6),
                "avg_trades": round(avg_trades, 1),
                "win_folds": win_folds,
                "total_folds": len(results),
                "full_sample_sharpe": round(full_sample_sharpe, 4) if full_sample_sharpe is not None else None,
            },
        }

        report_path = out_dir / "walk_forward_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {report_path}")


def _run_full_sample_sharpe(
    bars: List[OhlcvBar],
    symbol: str,
    starting_balance: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal,
) -> float:
    """Run full-sample backtest and return Sharpe for overfit comparison."""
    from strategies.multi_factor.decision_module import MultiFactorConfig, MultiFactorDecisionModule
    from runner.backtest_runner import run_backtest

    csv_path = _bars_to_temp_csv(bars)
    try:
        module = MultiFactorDecisionModule(config=MultiFactorConfig(symbol=symbol))
        equity, _ = run_backtest(
            csv_path=csv_path,
            symbol=symbol,
            starting_balance=starting_balance,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            decision_modules=[module],
        )
    finally:
        csv_path.unlink(missing_ok=True)
    return _compute_sharpe(equity)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward validation for multi-factor strategy")
    p.add_argument("--csv", default="data_files/BTCUSDT_1h.csv", help="OHLCV CSV path")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--starting-balance", type=float, default=10000)
    p.add_argument("--fee-bps", type=float, default=4)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--train-months", type=int, default=6, help="Training window in months")
    p.add_argument("--test-months", type=int, default=3, help="Test window in months")
    p.add_argument("--optimize", action="store_true", help="Enable parameter optimization mode")
    p.add_argument("--out", default="output/walk_forward", help="Output directory")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    print("Loading data...")
    bars = list(iter_ohlcv_csv(csv_path))

    train_size = args.train_months * BARS_PER_MONTH
    test_size = args.test_months * BARS_PER_MONTH
    folds = _generate_folds(len(bars), train_size, test_size)

    print(f"\n=== Walk-Forward Validation ===")
    print(f"Data: {len(bars):,} bars ({bars[0].ts.strftime('%Y-%m')} ~ {bars[-1].ts.strftime('%Y-%m')})")
    print(f"Windows: train={train_size} bars ({args.train_months} months), test={test_size} bars ({args.test_months} months)")
    print(f"Folds: {len(folds)}")
    print(f"Mode: {'Parameter Optimization' if args.optimize else 'Fixed Parameters'}")
    print()

    if args.optimize:
        results = _run_param_optimization(bars, args)
        full_sharpe = None  # optimization mode tracks its own overfit metric
    else:
        results = _run_fixed_param_validation(bars, args)
        print("\nRunning full-sample backtest for overfit comparison...")
        full_sharpe = _run_full_sample_sharpe(
            bars, args.symbol,
            Decimal(str(args.starting_balance)),
            Decimal(str(args.fee_bps)),
            Decimal(str(args.slippage_bps)),
        )

    _print_report(results, full_sharpe, out_dir)


if __name__ == "__main__":
    main()
