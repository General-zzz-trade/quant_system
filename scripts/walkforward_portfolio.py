#!/usr/bin/env python3
"""Portfolio Walk-Forward Validation — expanding window across BTC+ETH+SOL.

Each fold: train per-symbol models → generate signals → combine portfolio → evaluate.
Tests whether the portfolio strategy is robust across different time periods.

Usage:
    python3 -m scripts.walkforward_portfolio
    python3 -m scripts.walkforward_portfolio --alloc inverse_vol
    python3 -m scripts.walkforward_portfolio --no-hpo  # faster
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.walkforward_validate import (
    Fold,
    FoldResult,
    generate_wf_folds,
    run_fold_strategy_f,
    _compute_regime_labels,
    WARMUP,
    HORIZON,
    TARGET_MODE,
    DEADZONE,
    MIN_HOLD,
    MIN_TRAIN_BARS,
    TEST_BARS,
    STEP_BARS,
)
from scripts.train_v7_alpha import (
    _load_and_compute_features,
    BLACKLIST,
)
from scripts.backtest_alpha_v8 import COST_PER_TRADE

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ── Data loading (per symbol) ────────────────────────────────

def _load_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:
    """Load CSV + compute features for a symbol."""
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  [{symbol}] Data not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df = df.rename(columns={ts_col: "timestamp"})

    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        print(f"  [{symbol}] Feature computation failed")
        return None

    # Load model config for feature names / settings
    config_path = Path(f"models_v8/{symbol}_gate_v2/config.json")
    if not config_path.exists():
        print(f"  [{symbol}] Config not found: {config_path}")
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    closes = feat_df["close"].values.astype(np.float64)
    timestamps = df["timestamp"].values.astype(np.int64)

    # Ensure all features exist in feat_df
    feature_names = cfg["features"]
    for fn in feature_names:
        if fn not in feat_df.columns:
            feat_df[fn] = np.nan

    print(f"  [{symbol}] {len(df)} bars, {len(feature_names)} features")

    return {
        "symbol": symbol,
        "feat_df": feat_df,
        "closes": closes,
        "timestamps": timestamps,
        "cfg": cfg,
        "feature_names": feature_names,
    }


# ── Portfolio fold execution ─────────────────────────────────

@dataclass
class PortfolioFoldResult:
    idx: int
    period: str
    sharpe: float
    total_return: float
    max_dd: float
    n_test: int
    per_symbol: Dict[str, Dict[str, float]]


def run_portfolio_fold(
    fold: Fold,
    symbol_data_list: List[Dict[str, Any]],
    alloc_method: str = "equal",
    use_hpo: bool = False,
    selector: str = "stable_icir",
) -> PortfolioFoldResult:
    """Run a single portfolio fold: train each symbol's model, combine signals."""
    from datetime import datetime, timezone

    n_sym = len(symbol_data_list)
    fold_results = {}
    fold_signals = {}
    fold_closes = {}

    # 1. Train each symbol's model and get OOS signal
    for sd in symbol_data_list:
        sym = sd["symbol"]
        cfg = sd["cfg"]
        feature_names = sd["feature_names"]
        feat_df = sd["feat_df"]
        closes = sd["closes"]

        fixed_features = cfg.get("fixed_features")
        candidate_pool = cfg.get("candidate_pool")
        n_flexible = cfg.get("n_flexible", 4)
        pm = cfg.get("position_management", {})
        bear_thresholds = None
        if pm.get("bear_thresholds"):
            bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]
        dd_limit = pm.get("dd_limit")
        if dd_limit is not None and dd_limit > 0:
            dd_limit = -dd_limit

        result = run_fold_strategy_f(
            fold, feat_df, closes, feature_names,
            fixed_features=fixed_features,
            candidate_pool=candidate_pool,
            n_flexible=n_flexible,
            use_hpo=use_hpo,
            deadzone=cfg.get("deadzone", DEADZONE),
            min_hold=cfg.get("min_hold", MIN_HOLD),
            monthly_gate_window=cfg.get("monthly_gate_window", 480),
            bear_thresholds=bear_thresholds,
            dd_limit=dd_limit,
            dd_cooldown=pm.get("dd_cooldown", 48),
            selector=selector,
        )
        fold_results[sym] = result

        # Extract OOS signal from fold result
        # run_fold_strategy_f stores signal in result.returns (bar-level PnL)
        # We need to reconstruct signal from the fold's equity/returns
        # Since we can't easily extract raw signal, compute portfolio from returns
        fold_signals[sym] = result
        fold_closes[sym] = closes[fold.test_start:fold.test_end]

    # 2. Compute portfolio metrics from per-symbol results
    n_test = fold.test_end - fold.test_start

    # Combine per-symbol returns with equal/inverse_vol weights
    per_sym_returns = {}
    for sd in symbol_data_list:
        sym = sd["symbol"]
        r = fold_results[sym]
        # FoldResult total_return is the cumulative; sharpe is annualized
        per_sym_returns[sym] = {
            "sharpe": r.sharpe,
            "return": r.total_return,
            "ic": r.ic,
        }

    # Compute portfolio return (simple weighted average)
    if alloc_method == "equal":
        weights = {sd["symbol"]: 1.0 / n_sym for sd in symbol_data_list}
    elif alloc_method == "inverse_vol":
        # Use per-symbol vol from fold's test period
        vols = {}
        for sd in symbol_data_list:
            sym = sd["symbol"]
            test_closes = sd["closes"][fold.test_start:fold.test_end]
            if len(test_closes) > 1:
                rets = np.diff(test_closes) / test_closes[:-1]
                vols[sym] = max(np.std(rets), 1e-8)
            else:
                vols[sym] = 0.02
        inv = {s: 1.0 / v for s, v in vols.items()}
        total_inv = sum(inv.values())
        weights = {s: v / total_inv for s, v in inv.items()}
    else:
        weights = {sd["symbol"]: 1.0 / n_sym for sd in symbol_data_list}

    # Weighted portfolio return and Sharpe
    portfolio_return = sum(weights[s] * per_sym_returns[s]["return"] for s in weights)
    portfolio_sharpe_parts = [weights[s] * per_sym_returns[s]["sharpe"] for s in weights]
    portfolio_sharpe = sum(portfolio_sharpe_parts)

    # Max DD estimate (weighted combination of individual DDs is approximate)
    # Use the worst individual DD scaled by weight as lower bound
    max_dd = -0.01  # placeholder; we'll compute properly from returns

    # Period label
    try:
        ts = symbol_data_list[0]["timestamps"]
        t_start = datetime.fromtimestamp(int(ts[fold.test_start]) / 1000, tz=timezone.utc)
        t_end = datetime.fromtimestamp(int(ts[fold.test_end - 1]) / 1000, tz=timezone.utc)
        period = f"{t_start.strftime('%Y-%m')} → {t_end.strftime('%Y-%m')}"
    except Exception:
        period = f"fold_{fold.idx}"

    per_symbol_summary = {}
    for sd in symbol_data_list:
        sym = sd["symbol"]
        r = per_sym_returns[sym]
        per_symbol_summary[sym] = {
            "sharpe": r["sharpe"],
            "return": r["return"],
            "ic": r["ic"],
            "weight": weights[sym],
        }

    return PortfolioFoldResult(
        idx=fold.idx,
        period=period,
        sharpe=portfolio_sharpe,
        total_return=portfolio_return,
        max_dd=max_dd,
        n_test=n_test,
        per_symbol=per_symbol_summary,
    )


# ── Main WF engine ───────────────────────────────────────────

def run_portfolio_walkforward(
    symbols: List[str],
    alloc_method: str = "equal",
    use_hpo: bool = False,
    selector: str = "stable_icir",
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run portfolio walk-forward validation."""
    print(f"\n{'='*75}")
    print(f"  PORTFOLIO WALK-FORWARD VALIDATION")
    print(f"  Symbols: {' + '.join(symbols)}")
    print(f"  Allocation: {alloc_method}")
    print(f"  Selector: {selector}, HPO: {use_hpo}")
    print(f"  Min train: {MIN_TRAIN_BARS} bars, Test: {TEST_BARS} bars, Step: {STEP_BARS} bars")
    print(f"{'='*75}")

    # Load all symbol data
    symbol_data_list = []
    for sym in symbols:
        sd = _load_symbol_data(sym)
        if sd is not None:
            symbol_data_list.append(sd)

    if len(symbol_data_list) < 2:
        print("  Need at least 2 symbols. Exiting.")
        return {}

    # Find common bar count (use minimum across symbols)
    n_bars = min(len(sd["feat_df"]) for sd in symbol_data_list)
    print(f"\n  Common bars: {n_bars}")

    # Generate folds
    folds = generate_wf_folds(n_bars)
    print(f"  Folds: {len(folds)}")

    if not folds:
        print("  Not enough data for walk-forward. Exiting.")
        return {}

    # Run each fold
    fold_results: List[PortfolioFoldResult] = []
    for fold in folds:
        t0 = time.time()
        print(f"\n  ── Fold {fold.idx} (train:{fold.train_start}-{fold.train_end}, "
              f"test:{fold.test_start}-{fold.test_end}) ──")

        result = run_portfolio_fold(
            fold, symbol_data_list,
            alloc_method=alloc_method,
            use_hpo=use_hpo,
            selector=selector,
        )
        elapsed = time.time() - t0
        fold_results.append(result)

        # Per-symbol detail
        sym_details = "  ".join(
            f"{s}:{d['sharpe']:.2f}/{d['return']*100:+.1f}%"
            for s, d in result.per_symbol.items()
        )
        print(f"  {result.period}: portfolio Sharpe={result.sharpe:.2f}, "
              f"Return={result.total_return*100:+.2f}% ({elapsed:.1f}s)")
        print(f"    {sym_details}")

    # ── Aggregate results ──
    n_folds = len(fold_results)
    pos_sharpe = sum(1 for r in fold_results if r.sharpe > 0)
    sharpes = [r.sharpe for r in fold_results]
    returns = [r.total_return for r in fold_results]
    avg_sharpe = float(np.mean(sharpes))
    avg_return = float(np.mean(returns))
    total_return = float(np.prod([1 + r for r in returns]) - 1)
    pass_threshold = int(np.ceil(n_folds * 2 / 3))
    passed = pos_sharpe >= pass_threshold

    # Per-symbol aggregate
    per_sym_agg = {}
    for sd in symbol_data_list:
        sym = sd["symbol"]
        sym_sharpes = [r.per_symbol[sym]["sharpe"] for r in fold_results]
        sym_returns = [r.per_symbol[sym]["return"] for r in fold_results]
        sym_pos = sum(1 for s in sym_sharpes if s > 0)
        per_sym_agg[sym] = {
            "avg_sharpe": float(np.mean(sym_sharpes)),
            "avg_return": float(np.mean(sym_returns)),
            "positive_folds": sym_pos,
            "total_folds": n_folds,
        }

    # ── Print report ──
    print(f"\n{'='*75}")
    print(f"  PORTFOLIO WALK-FORWARD REPORT")
    print(f"{'='*75}")
    print(f"  {'Fold':>4s}  {'Period':>25s}  {'Sharpe':>7s}  {'Return':>8s}  Per-Symbol Sharpe")
    print(f"  {'-'*73}")

    for r in fold_results:
        sym_str = "  ".join(f"{s}:{d['sharpe']:.2f}" for s, d in r.per_symbol.items())
        marker = "+" if r.sharpe > 0 else "-"
        print(f"  {r.idx:>4d}  {r.period:>25s}  {r.sharpe:>+7.2f}  {r.total_return*100:>+7.2f}%  {sym_str}  [{marker}]")

    print(f"  {'-'*73}")
    print(f"\n  VERDICT: {pos_sharpe}/{n_folds} positive Sharpe "
          f"(need >= {pass_threshold}) → {'PASS' if passed else 'FAIL'}")
    print(f"\n  Portfolio avg Sharpe: {avg_sharpe:.2f}")
    print(f"  Portfolio avg return: {avg_return*100:+.2f}%")
    print(f"  Portfolio total return (stitched): {total_return*100:+.2f}%")

    print(f"\n  --- Per-Symbol Summary ---")
    print(f"  {'Symbol':<12s} {'Avg Sharpe':>10s} {'Avg Return':>10s} {'Pos Folds':>10s}")
    print(f"  {'-'*44}")
    for sym, agg in per_sym_agg.items():
        print(f"  {sym:<12s} {agg['avg_sharpe']:>+10.2f} {agg['avg_return']*100:>+9.2f}% "
              f"{agg['positive_folds']:>5d}/{agg['total_folds']}")

    # ── Save results ──
    if out_dir is None:
        out_dir = Path(f"results/wf_portfolio/{'_'.join(symbols)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "symbols": symbols,
        "allocation": alloc_method,
        "selector": selector,
        "n_folds": n_folds,
        "positive_sharpe": pos_sharpe,
        "pass_threshold": pass_threshold,
        "passed": passed,
        "avg_sharpe": avg_sharpe,
        "avg_return": avg_return,
        "total_return": total_return,
        "per_symbol": per_sym_agg,
        "folds": [
            {
                "idx": r.idx,
                "period": r.period,
                "sharpe": r.sharpe,
                "return": r.total_return,
                "per_symbol": r.per_symbol,
            }
            for r in fold_results
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Fold-level CSV
    fold_rows = []
    for r in fold_results:
        row = {"fold": r.idx, "period": r.period, "sharpe": r.sharpe, "return": r.total_return}
        for sym, d in r.per_symbol.items():
            row[f"{sym}_sharpe"] = d["sharpe"]
            row[f"{sym}_return"] = d["return"]
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(out_dir / "folds.csv", index=False)

    print(f"\n  Results saved to {out_dir}/")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio walk-forward validation")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--alloc", choices=["equal", "inverse_vol", "risk_parity"],
                        default="equal")
    parser.add_argument("--selector", choices=["greedy", "stable_icir"],
                        default="stable_icir")
    parser.add_argument("--no-hpo", action="store_true", help="Skip HPO (faster)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    run_portfolio_walkforward(
        symbols=[s.upper() for s in args.symbols],
        alloc_method=args.alloc,
        use_hpo=not args.no_hpo,
        selector=args.selector,
        out_dir=Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
