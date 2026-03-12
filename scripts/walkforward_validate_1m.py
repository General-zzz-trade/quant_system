#!/usr/bin/env python3
"""Walk-Forward Validation for 1-minute alpha models.

Expanding-window walk-forward with 3-month test steps on 1m bar data.
Each fold: feature selection -> train -> OOS evaluation with realistic costs.

PASS criterion: >50% of folds have positive Sharpe (net of costs).

Usage:
    python3 -m scripts.walkforward_validate_1m --symbol BTCUSDT --no-hpo
    python3 -m scripts.walkforward_validate_1m --symbol BTCUSDT --horizon 5
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.train_1m_alpha import (
    V1M_DEFAULT_PARAMS,
    _compute_target,
    _fast_spearman_ic,
    COST_PER_TRADE,
    DEADZONE,
    MIN_HOLD,
    ZSCORE_WINDOW,
    MONTHLY_GATE_WINDOW,
    MIN_TRAIN_BARS,
    TEST_BARS,
    STEP_BARS,
    WARMUP,
    HORIZON_DEFAULT,
    TARGET_MODE,
    load_and_compute_features,
    run_ic_analysis,
    check_go_nogo,
)
from alpha.signal_transform import pred_to_signal as _pred_to_signal
from features.dynamic_selector import greedy_ic_select, stable_icir_select
from scripts.signal_postprocess import _apply_monthly_gate, _compute_bear_mask

logger = logging.getLogger(__name__)


# ── Fold data structures ─────────────────────────────────────

@dataclass
class Fold:
    idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class FoldResult:
    idx: int
    period: str
    ic: float
    sharpe: float
    total_return: float
    features: List[str]
    n_train: int
    n_test: int


def generate_wf_folds(n_bars: int) -> List[Fold]:
    """Generate expanding-window walk-forward folds."""
    folds = []
    fold_idx = 0
    test_start = MIN_TRAIN_BARS
    while test_start + TEST_BARS <= n_bars:
        folds.append(Fold(
            idx=fold_idx,
            train_start=0,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + TEST_BARS,
        ))
        fold_idx += 1
        test_start += STEP_BARS
    return folds


def _select_features_dispatch(selector: str):
    if selector == "stable_icir":
        return lambda X, y, names, top_k: stable_icir_select(X, y, names, top_k=top_k)
    return lambda X, y, names, top_k: greedy_ic_select(X, y, names, top_k=top_k)

# ── Single fold execution ────────────────────────────────────

def run_fold(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    *,
    horizon: int = HORIZON_DEFAULT,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    monthly_gate: bool = True,
    long_only: bool = True,
    selector: str = "greedy",
    fixed_features: Optional[List[str]] = None,
    candidate_pool: Optional[List[str]] = None,
    n_flexible: int = 4,
) -> FoldResult:
    """Train and evaluate a single WF fold on 1m data."""
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]

    y_train_full = _compute_target(train_closes, horizon, TARGET_MODE)
    y_test_full = _compute_target(test_closes, horizon, TARGET_MODE)

    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)
    y_test = y_test_full

    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]
    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]

    test_valid = ~np.isnan(y_test)

    if len(X_train) < 10000:
        return FoldResult(idx=fold.idx, period="", ic=0.0, sharpe=0.0,
                          total_return=0.0, features=[], n_train=len(X_train), n_test=0)

    # Feature selection (subsample for speed on large 1m datasets)
    _select = _select_features_dispatch(selector)
    FSEL_MAX = 200_000
    if len(X_train) > FSEL_MAX:
        rng = np.random.RandomState(fold.idx)
        fsel_idx = rng.choice(len(X_train), FSEL_MAX, replace=False)
        X_fsel, y_fsel = X_train[fsel_idx], y_train[fsel_idx]
    else:
        X_fsel, y_fsel = X_train, y_train

    if fixed_features:
        selected = list(fixed_features)
        if n_flexible > 0:
            pool = candidate_pool or [f for f in feature_names if f not in fixed_features]
            pool_in_data = [f for f in pool if f in feature_names]
            if pool_in_data:
                pool_idx = [feature_names.index(f) for f in pool_in_data]
                X_pool = X_fsel[:, pool_idx]
                flex = _select(X_pool, y_fsel, pool_in_data, top_k=n_flexible)
                selected.extend(flex)
    else:
        selected = _select(X_fsel, y_fsel, feature_names, top_k=15)

    if not selected:
        return FoldResult(idx=fold.idx, period="", ic=0.0, sharpe=0.0,
                          total_return=0.0, features=[], n_train=len(X_train), n_test=0)

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # Train
    params = dict(V1M_DEFAULT_PARAMS)
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 300),
        callbacks=[lgb.log_evaluation(0)],
    )

    y_pred = bst.predict(X_test_sel)

    # IC on test
    ic = 0.0
    if test_valid.sum() > 10:
        y_pred_v = y_pred[test_valid]
        y_test_v = y_test[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = _fast_spearman_ic(y_pred_v, y_test_v)

    # Signal generation
    signal = _pred_to_signal(y_pred, target_mode=TARGET_MODE,
                              deadzone=deadzone, min_hold=min_hold,
                              zscore_window=ZSCORE_WINDOW)

    # Long-only
    if long_only:
        np.clip(signal, 0.0, 1.0, out=signal)

    # Monthly gate
    if monthly_gate:
        signal = _apply_monthly_gate(signal, test_closes, MONTHLY_GATE_WINDOW)

    # Trade simulation
    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    signal_for_trade = signal[:len(ret_1bar)]
    gross_pnl = signal_for_trade * ret_1bar

    turnover = np.abs(np.diff(signal_for_trade, prepend=0))
    cost = turnover * COST_PER_TRADE

    # Funding cost: every 480 bars (= 8 hours at 1m)
    funding_cost = np.zeros(len(signal_for_trade))
    if "slow_funding_zscore_24" in test_df.columns:
        # Approximate: use funding_rate from slow features if available
        # For simplicity, assume ~0.01% per 8h settlement
        for i in range(0, len(signal_for_trade), 480):
            end = min(i + 480, len(signal_for_trade))
            funding_cost[i:end] = signal_for_trade[i:end] * 1e-4 / 480

    net_pnl = gross_pnl - cost - funding_cost

    active = signal_for_trade != 0
    n_active = int(active.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            # Annualize: sqrt(525600) for 1m bars
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(525_600)

    total_return = float(np.sum(net_pnl))

    return FoldResult(
        idx=fold.idx, period="", ic=ic, sharpe=sharpe,
        total_return=total_return, features=selected,
        n_train=len(X_train), n_test=len(X_test),
    )


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation for 1m alpha")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data", help="Path to 1m CSV")
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--selector", default="greedy", choices=["greedy", "stable_icir"])
    parser.add_argument("--fixed-features", nargs="+")
    parser.add_argument("--candidate-pool", nargs="+")
    parser.add_argument("--n-flexible", type=int, default=4)
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--no-monthly-gate", action="store_true")
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD)
    parser.add_argument("--deadzone", type=float, default=DEADZONE)
    parser.add_argument("--out-dir", default="results/wf_1m")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys

    # Load and compute features
    df, feat_df = load_and_compute_features(args.symbol, args.data)
    closes = df["close"].values.astype(np.float64)
    feature_names = [c for c in feat_df.columns if c != "close"]

    # IC analysis gate
    ic_df = run_ic_analysis(feat_df, closes, feature_names)
    go = check_go_nogo(ic_df, args.horizon)
    if not go:
        print("\nAborting: IC analysis did not pass GO/NO-GO gate.")
        return

    # Generate folds
    folds = generate_wf_folds(len(df))
    print(f"\n{len(folds)} walk-forward folds generated")
    if not folds:
        print("Not enough data for walk-forward validation")
        return

    # Run folds
    results: List[FoldResult] = []
    for fold in folds:
        t0 = time.time()
        print(f"\n--- Fold {fold.idx} ---", flush=True)
        print(f"  Train: bars 0-{fold.train_end:,} ({fold.train_end:,})")
        print(f"  Test:  bars {fold.test_start:,}-{fold.test_end:,} ({TEST_BARS:,})", flush=True)

        r = run_fold(
            fold, feat_df, closes, feature_names,
            horizon=args.horizon,
            monthly_gate=not args.no_monthly_gate,
            long_only=args.long_only,
            selector=args.selector,
            fixed_features=args.fixed_features,
            candidate_pool=args.candidate_pool,
            n_flexible=args.n_flexible,
            deadzone=args.deadzone,
            min_hold=args.min_hold,
        )

        elapsed = time.time() - t0
        r.period = f"fold_{fold.idx}"
        results.append(r)
        print(f"  IC={r.ic:.4f}  Sharpe={r.sharpe:.2f}  Return={r.total_return:.4f}  "
              f"[{elapsed:.1f}s]  features={r.features[:5]}...", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY")
    print("=" * 70)

    sharpes = [r.sharpe for r in results]
    returns = [r.total_return for r in results]
    pos_sharpe = sum(1 for s in sharpes if s > 0)
    n_folds = len(results)

    print(f"Folds: {n_folds}")
    print(f"Positive Sharpe: {pos_sharpe}/{n_folds} ({100*pos_sharpe/n_folds:.0f}%)")
    print(f"Avg Sharpe: {np.mean(sharpes):.2f}")
    print(f"Med Sharpe: {np.median(sharpes):.2f}")
    print(f"Total Return: {sum(returns):.4f} ({100*sum(returns):.1f}%)")

    pass_pct = pos_sharpe / n_folds if n_folds > 0 else 0
    if pass_pct > 0.5:
        print(f"\n>>> PASS — {pos_sharpe}/{n_folds} folds positive Sharpe")
    else:
        print(f"\n>>> FAIL — only {pos_sharpe}/{n_folds} folds positive Sharpe (need >50%)")

    # Save results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol": args.symbol,
        "horizon": args.horizon,
        "n_folds": n_folds,
        "pos_sharpe": pos_sharpe,
        "avg_sharpe": float(np.mean(sharpes)),
        "total_return": float(sum(returns)),
        "pass": pass_pct > 0.5,
        "folds": [
            {"idx": r.idx, "ic": r.ic, "sharpe": r.sharpe,
             "return": r.total_return, "features": r.features}
            for r in results
        ],
    }
    with open(out_dir / f"wf_1m_{args.symbol}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_dir / f'wf_1m_{args.symbol}.json'}")


if __name__ == "__main__":
    main()
