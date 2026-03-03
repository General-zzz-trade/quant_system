#!/usr/bin/env python3
"""Walk-Forward Validation — prove alpha is reproducible across rolling windows.

Expanding-window walk-forward with 3-month test steps over ~6 years of data.
Each fold: feature selection → optional HPO → train → OOS evaluation.

Usage:
    # Quick (no HPO, ~10 min)
    python3 -m scripts.walkforward_validate --symbol BTCUSDT --no-hpo

    # Full (HPO, ~2 hours)
    python3 -m scripts.walkforward_validate --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS,
    V7_SEARCH_SPACE,
    _compute_target,
    _load_and_compute_features,
    INTERACTION_FEATURES,
    BLACKLIST,
)
from scripts.backtest_alpha_v8 import _pred_to_signal, _apply_monthly_gate, COST_PER_TRADE
from features.dynamic_selector import greedy_ic_select
from alpha.models.lgbm_alpha import LGBMAlphaModel

logger = logging.getLogger(__name__)

# ── Regime detection ─────────────────────────────────────────

# Thresholds for regime classification
_TREND_THRESHOLD = 0.015   # |close_vs_ma| > 1.5% = directional
_VOL_EXPAND_THRESHOLD = 1.2  # vol_5/vol_20 > 1.2 = expanding vol


def _compute_regime_labels(feat_df: pd.DataFrame) -> np.ndarray:
    """Per-bar regime labels from features already in feat_df.

    Uses:
      - vol_regime: vol_5/vol_20 ratio (>1 = expanding)
      - close_vs_ma20: close/ma20 - 1 (pct deviation)
      - close_vs_ma50: close/ma50 - 1 (pct deviation)

    Returns array of strings: "trending" / "ranging" / "high_vol_flat"
    """
    n = len(feat_df)
    labels = np.full(n, "trending", dtype=object)

    has_vr = "vol_regime" in feat_df.columns
    has_ma20 = "close_vs_ma20" in feat_df.columns
    has_ma50 = "close_vs_ma50" in feat_df.columns

    # Without at least MA columns, can't detect regime → all trending
    if not has_ma20 and not has_ma50:
        return labels

    vr = feat_df["vol_regime"].values if has_vr else np.ones(n)
    ma20 = feat_df["close_vs_ma20"].values if has_ma20 else np.full(n, np.nan)
    ma50 = feat_df["close_vs_ma50"].values if has_ma50 else np.full(n, np.nan)

    for i in range(n):
        v, m20, m50 = vr[i], ma20[i], ma50[i]
        # NaN → default trending (warmup period)
        if np.isnan(v) or np.isnan(m20) or np.isnan(m50):
            continue
        # Trending: at least one MA shows strong direction
        is_trending = (abs(m20) > _TREND_THRESHOLD or abs(m50) > _TREND_THRESHOLD)
        is_high_vol = v > _VOL_EXPAND_THRESHOLD
        if is_high_vol and not is_trending:
            labels[i] = "high_vol_flat"
        elif not is_trending:
            labels[i] = "ranging"
        # else: already "trending"
    return labels


def _apply_signal_filters(
    signal: np.ndarray,
    regime_labels: Optional[np.ndarray],
    long_only: bool,
    regime_gate: bool,
    adaptive_sizing: Optional[Dict[str, float]],
) -> np.ndarray:
    """Post-process signal with long-only, regime gating, adaptive sizing."""
    sig = signal.copy()
    if long_only:
        np.clip(sig, 0.0, 1.0, out=sig)
    if regime_labels is not None and (regime_gate or adaptive_sizing):
        for i in range(len(sig)):
            if i >= len(regime_labels):
                break
            r = regime_labels[i]
            if regime_gate and r in ("ranging", "high_vol_flat"):
                sig[i] = 0.0
            elif adaptive_sizing and r in adaptive_sizing:
                sig[i] *= adaptive_sizing[r]
    return sig


def _apply_trend_hold(
    signal: np.ndarray,
    trend_vals: np.ndarray,
    trend_threshold: float,
    max_hold: int,
) -> np.ndarray:
    """Extend long positions when trend is intact.

    When signal transitions from positive to 0 (model says exit), check if the
    trend indicator is still above threshold. If so, keep holding instead of
    exiting, up to max_hold total bars in position.
    """
    out = signal.copy()
    hold_count = 0
    for i in range(len(out)):
        if out[i] > 0:
            hold_count += 1
        elif i > 0 and out[i - 1] > 0 and out[i] == 0:
            tv = trend_vals[i] if i < len(trend_vals) else float("nan")
            if (not np.isnan(tv)
                    and tv > trend_threshold
                    and hold_count < max_hold):
                out[i] = out[i - 1]
                hold_count += 1
            else:
                hold_count = 0
        else:
            hold_count = 0
    return out


# ── Constants ────────────────────────────────────────────────

MIN_TRAIN_BARS = 8760      # 12 months
TEST_BARS = 2190           # 3 months
STEP_BARS = 2190           # 3-month step
WARMUP = 65                # feature warmup bars
TOP_K = 15
HORIZON = 24
TARGET_MODE = "clipped"
MIN_HOLD = 24
DEADZONE = 0.5
HPO_TRIALS = 10


# ── Fold generation ──────────────────────────────────────────

@dataclass
class Fold:
    idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_wf_folds(
    n_bars: int,
    min_train_bars: int = MIN_TRAIN_BARS,
    test_bars: int = TEST_BARS,
    step_bars: int = STEP_BARS,
) -> List[Fold]:
    """Generate expanding-window walk-forward folds.

    Train always starts from bar 0 (expanding window).
    Test window slides by step_bars each fold.
    """
    folds = []
    fold_idx = 0
    test_start = min_train_bars

    while test_start + test_bars <= n_bars:
        folds.append(Fold(
            idx=fold_idx,
            train_start=0,
            train_end=test_start,
            test_start=test_start,
            test_end=test_start + test_bars,
        ))
        fold_idx += 1
        test_start += step_bars

    return folds


# ── Single fold execution ────────────────────────────────────

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


def run_fold(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    use_hpo: bool = True,
    fixed_features: Optional[List[str]] = None,
    candidate_pool: Optional[List[str]] = None,
    n_flexible: int = 4,
    regime_gate: bool = False,
    long_only: bool = False,
    adaptive_sizing: Optional[Dict[str, float]] = None,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    continuous_sizing: bool = False,
    hpo_trials: int = HPO_TRIALS,
    ensemble: bool = False,
    target_mode: str = TARGET_MODE,
    trend_follow: bool = False,
    trend_indicator: str = "tf4h_close_vs_ma20",
    trend_threshold: float = 0.0,
    max_hold: int = 120,
    monthly_gate: bool = False,
    monthly_gate_window: int = 480,
) -> FoldResult:
    """Train and evaluate a single fold."""
    # Prepare train/test data
    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]

    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]

    # Compute target
    y_train_full = _compute_target(train_closes, HORIZON, target_mode)
    y_test_full = _compute_target(test_closes, HORIZON, target_mode)

    # Build X matrices (skip warmup in train; test warmup handled by full feature computation)
    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)
    y_test = y_test_full

    # Skip warmup for training
    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]

    # Remove NaN target rows
    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]

    test_valid = ~np.isnan(y_test)

    if len(X_train) < 1000:
        return FoldResult(
            idx=fold.idx, period="", ic=0.0, sharpe=0.0,
            total_return=0.0, features=[], n_train=len(X_train), n_test=0,
        )

    # Feature selection on train data
    if fixed_features:
        # Fixed mode: lock stable features + greedy-select flexible from candidate pool
        selected = list(fixed_features)
        if n_flexible > 0:
            pool = candidate_pool if candidate_pool else [
                f for f in feature_names if f not in fixed_features]
            pool_in_data = [f for f in pool if f in feature_names]
            if pool_in_data:
                pool_idx = [feature_names.index(f) for f in pool_in_data]
                X_pool = X_train[:, pool_idx]
                flex = greedy_ic_select(
                    X_pool, y_train, pool_in_data, top_k=n_flexible)
                selected.extend(flex)
    else:
        selected = greedy_ic_select(X_train, y_train, feature_names, top_k=TOP_K)
    if not selected:
        return FoldResult(
            idx=fold.idx, period="", ic=0.0, sharpe=0.0,
            total_return=0.0, features=[], n_train=len(X_train), n_test=0,
        )

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # HPO or default params
    params = dict(V7_DEFAULT_PARAMS)
    if use_hpo:
        try:
            from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
            import lightgbm as lgb

            n_tr = len(X_train_sel)
            val_size = min(n_tr // 4, TEST_BARS)
            X_hpo_train = X_train_sel[:-val_size]
            y_hpo_train = y_train[:-val_size]
            X_hpo_val = X_train_sel[-val_size:]
            y_hpo_val = y_train[-val_size:]

            def objective(trial_params):
                p = {**V7_DEFAULT_PARAMS, **trial_params}
                dtrain = lgb.Dataset(X_hpo_train, label=y_hpo_train)
                dval = lgb.Dataset(X_hpo_val, label=y_hpo_val, reference=dtrain)
                bst = lgb.train(
                    p, dtrain,
                    num_boost_round=p["n_estimators"],
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)],
                )
                y_hat = bst.predict(X_hpo_val)
                valid_mask = ~np.isnan(y_hpo_val)
                if valid_mask.sum() < 10:
                    return 0.0
                from features.dynamic_selector import _rankdata, _spearman_ic
                return float(_spearman_ic(
                    _rankdata(y_hat[valid_mask]),
                    _rankdata(y_hpo_val[valid_mask]),
                ))

            opt = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=objective,
                config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"),
            )
            result = opt.optimize()
            params = {**V7_DEFAULT_PARAMS, **result.best_params}
        except Exception as e:
            logger.warning("HPO failed for fold %d: %s, using defaults", fold.idx, e)

    # Train model
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[lgb.log_evaluation(0)],
    )

    # Predict on test
    y_pred = bst.predict(X_test_sel)

    # Ensemble: average LGBM + XGB predictions
    if ensemble:
        try:
            import xgboost as xgb
            xgb_params = {
                "max_depth": params.get("max_depth", 6),
                "learning_rate": params.get("learning_rate", 0.05),
                "objective": "reg:squarederror",
                "verbosity": 0,
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
            }
            dtrain_xgb = xgb.DMatrix(X_train_sel, label=y_train)
            dtest_xgb = xgb.DMatrix(X_test_sel)
            xgb_bst = xgb.train(
                xgb_params, dtrain_xgb,
                num_boost_round=params.get("n_estimators", 500),
            )
            xgb_pred = xgb_bst.predict(dtest_xgb)
            y_pred = 0.5 * y_pred + 0.5 * xgb_pred
        except Exception as e:
            logger.warning("XGB ensemble failed for fold %d: %s", fold.idx, e)

    # IC on test
    ic = 0.0
    if test_valid.sum() > 10:
        from features.dynamic_selector import _rankdata, _spearman_ic
        y_pred_v = y_pred[test_valid]
        y_test_v = y_test[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(y_pred_v), _rankdata(y_test_v)))

    # Trade simulation on test
    signal = _pred_to_signal(y_pred, target_mode=target_mode,
                             deadzone=deadzone, min_hold=min_hold)

    # Continuous sizing: replace binary magnitude with z-score-based sizing
    if continuous_sizing:
        mu = np.mean(y_pred)
        std = np.std(y_pred)
        if std > 1e-12:
            z = (y_pred - mu) / std
            continuous = np.clip(z / 2.0, 0.0, 1.0)
            signal = np.where(signal != 0, continuous, 0.0)

    # Apply regime gating / long-only / adaptive sizing
    if long_only or regime_gate or adaptive_sizing:
        test_regimes = _compute_regime_labels(test_df) if (regime_gate or adaptive_sizing) else None
        signal = _apply_signal_filters(
            signal, test_regimes, long_only, regime_gate, adaptive_sizing)

    # Trend-following hold extension
    if trend_follow and trend_indicator in test_df.columns:
        trend_vals = test_df[trend_indicator].values.astype(np.float64)
        signal = _apply_trend_hold(signal, trend_vals, trend_threshold, max_hold)

    # Monthly gate: zero signal when close <= SMA(window)
    if monthly_gate:
        signal = _apply_monthly_gate(signal, test_closes, monthly_gate_window)

    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    signal_for_trade = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(signal_for_trade, prepend=0))
    gross_pnl = signal_for_trade * ret_1bar
    cost = turnover * COST_PER_TRADE
    net_pnl = gross_pnl - cost

    active = signal_for_trade != 0
    n_active = int(active.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))

    return FoldResult(
        idx=fold.idx,
        period="",  # filled by caller
        ic=ic,
        sharpe=sharpe,
        total_return=total_return,
        features=selected,
        n_train=len(X_train),
        n_test=len(X_test),
    )


# ── Stitched results & reporting ─────────────────────────────

def stitch_results(
    fold_results: List[FoldResult],
) -> Dict[str, Any]:
    """Aggregate fold results into a verdict."""
    n_folds = len(fold_results)
    pos_sharpe = sum(1 for r in fold_results if r.sharpe > 0)
    ics = [r.ic for r in fold_results]
    sharpes = [r.sharpe for r in fold_results]
    returns = [r.total_return for r in fold_results]

    # Feature stability: count occurrences
    feature_counts: Dict[str, int] = {}
    for r in fold_results:
        for f in r.features:
            feature_counts[f] = feature_counts.get(f, 0) + 1
    stable_features = {k: v for k, v in sorted(
        feature_counts.items(), key=lambda x: -x[1])
        if v >= n_folds * 0.8}

    # Stitched equity (concatenated test returns)
    all_returns = sum(returns)
    avg_sharpe = float(np.mean(sharpes)) if sharpes else 0.0

    passed = pos_sharpe >= n_folds * 2 / 3  # >= 67%

    return {
        "n_folds": n_folds,
        "positive_sharpe": pos_sharpe,
        "pass_threshold": int(np.ceil(n_folds * 2 / 3)),
        "passed": passed,
        "avg_ic": float(np.mean(ics)) if ics else 0.0,
        "avg_sharpe": avg_sharpe,
        "total_return": all_returns,
        "fold_ics": ics,
        "fold_sharpes": sharpes,
        "fold_returns": returns,
        "stable_features": stable_features,
    }


def print_report(fold_results: List[FoldResult], summary: Dict[str, Any]) -> None:
    """Print the walk-forward report table."""
    print(f"\n{'='*75}")
    print(f"  WALK-FORWARD VALIDATION REPORT")
    print(f"{'='*75}")
    print(f"  {'Fold':<6} {'Period':<20} {'IC':>8} {'Sharpe':>8} {'Return':>8}  Features")
    print(f"  {'-'*73}")

    for r in fold_results:
        feat_str = ", ".join(r.features[:3])
        if len(r.features) > 3:
            feat_str += f" +{len(r.features)-3}"
        print(f"  {r.idx:<6} {r.period:<20} {r.ic:>8.4f} {r.sharpe:>8.2f} "
              f"{r.total_return*100:>+7.2f}%  [{feat_str}]")

    print(f"  {'-'*73}")
    print(f"\n  VERDICT: {summary['positive_sharpe']}/{summary['n_folds']} positive Sharpe "
          f"(need >= {summary['pass_threshold']}) "
          f"→ {'PASS' if summary['passed'] else 'FAIL'}")
    print(f"\n  Average IC:     {summary['avg_ic']:.4f}")
    print(f"  Average Sharpe: {summary['avg_sharpe']:.2f}")
    print(f"  Total return:   {summary['total_return']*100:+.2f}%")

    if summary["stable_features"]:
        print(f"\n  Feature stability (>= 80% folds):")
        for fname, count in summary["stable_features"].items():
            print(f"    {fname}: {count}/{summary['n_folds']}")


# ── Main ─────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation for alpha models")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--no-hpo", action="store_true",
                        help="Skip HPO, use default params (faster)")
    parser.add_argument("--out-dir", default="results/walkforward")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--fixed-features", nargs="*", default=None,
                        help="Locked features included in every fold")
    parser.add_argument("--candidate-pool", nargs="*", default=None,
                        help="Candidate pool for flexible feature selection")
    parser.add_argument("--n-flexible", type=int, default=4,
                        help="Number of flexible features to select per fold")
    parser.add_argument("--regime-gate", action="store_true",
                        help="Zero signal in ranging/high_vol_flat regimes")
    parser.add_argument("--long-only", action="store_true",
                        help="Clip short signals (long-only mode)")
    parser.add_argument("--adaptive-sizing", type=str, default=None,
                        help='JSON regime→multiplier, e.g. \'{"trending":1.0,"ranging":0.3,"high_vol_flat":0.0}\'')
    parser.add_argument("--deadzone", type=float, default=DEADZONE,
                        help=f"Z-score threshold for signal entry (default: {DEADZONE})")
    parser.add_argument("--min-hold", type=int, default=MIN_HOLD,
                        help=f"Minimum bars to hold position (default: {MIN_HOLD})")
    parser.add_argument("--continuous-sizing", action="store_true",
                        help="Use z-score-based position sizing instead of binary {0,1}")
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS,
                        help=f"Number of HPO trials per fold (default: {HPO_TRIALS})")
    parser.add_argument("--ensemble", action="store_true",
                        help="Average LGBM + XGB predictions")
    parser.add_argument("--target-mode", type=str, default=TARGET_MODE,
                        help=f"Target computation mode (default: {TARGET_MODE})")
    parser.add_argument("--trend-follow", action="store_true",
                        help="Extend holds when trend is intact")
    parser.add_argument("--trend-indicator", type=str, default="tf4h_close_vs_ma20",
                        help="Feature for trend detection (default: tf4h_close_vs_ma20)")
    parser.add_argument("--trend-threshold", type=float, default=0.0,
                        help="Min trend value to extend hold (0 = above MA)")
    parser.add_argument("--max-hold", type=int, default=120,
                        help="Max total bars for trend-extended hold (default: 120)")
    parser.add_argument("--monthly-gate", action="store_true",
                        help="Gate signal when close <= SMA(window)")
    parser.add_argument("--monthly-gate-window", type=int, default=480,
                        help="SMA window for monthly gate (default: 480)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    symbol = args.symbol.upper()
    use_hpo = not args.no_hpo
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_features = args.fixed_features
    candidate_pool = args.candidate_pool
    n_flexible = args.n_flexible
    regime_gate = args.regime_gate
    long_only = args.long_only
    adaptive_sizing = json.loads(args.adaptive_sizing) if args.adaptive_sizing else None
    deadzone = args.deadzone
    min_hold = args.min_hold
    continuous_sizing = args.continuous_sizing
    hpo_trials = args.hpo_trials
    ensemble = args.ensemble
    target_mode = args.target_mode
    trend_follow = args.trend_follow
    trend_indicator = args.trend_indicator
    trend_threshold = args.trend_threshold
    max_hold_bars = args.max_hold
    monthly_gate = args.monthly_gate
    monthly_gate_window = args.monthly_gate_window

    print(f"\n  Walk-Forward Validation: {symbol}")
    print(f"  HPO: {'ON' if use_hpo else 'OFF (default params)'}"
          f"{f' ({hpo_trials} trials)' if use_hpo else ''}")
    print(f"  Signal: deadzone={deadzone}, min_hold={min_hold}"
          f"{', continuous_sizing' if continuous_sizing else ''}")
    if ensemble:
        print(f"  Ensemble: LGBM + XGB (averaged)")
    if target_mode != TARGET_MODE:
        print(f"  Target mode: {target_mode}")
    if trend_follow:
        print(f"  Trend hold: {trend_indicator} > {trend_threshold}, max_hold={max_hold_bars}")
    if monthly_gate:
        print(f"  Monthly gate: SMA({monthly_gate_window})")
    if fixed_features:
        print(f"  Fixed features: {len(fixed_features)} locked + {n_flexible} flexible")
    filters = []
    if regime_gate:
        filters.append("regime-gate")
    if long_only:
        filters.append("long-only")
    if adaptive_sizing:
        filters.append(f"adaptive-sizing={adaptive_sizing}")
    if filters:
        print(f"  Filters: {', '.join(filters)}")
    print(f"  Min train: {MIN_TRAIN_BARS} bars ({MIN_TRAIN_BARS//24:.0f} days)")
    print(f"  Test window: {TEST_BARS} bars ({TEST_BARS//24:.0f} days)")

    # Load data
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_bars = len(df)
    print(f"  Total bars: {n_bars:,}")

    # Generate folds
    folds = generate_wf_folds(n_bars)
    print(f"  Folds: {len(folds)}")

    if not folds:
        print("  Not enough data for walk-forward validation")
        return

    # Compute features for entire dataset once (expanding window means
    # we always train from bar 0, so features are computed up-front)
    print("  Computing features for full dataset...")
    t0 = time.time()
    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        print("  Feature computation failed")
        return
    closes = feat_df["close"].values.astype(np.float64) if "close" in feat_df.columns else df["close"].values.astype(np.float64)
    print(f"  Features computed in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")

    # Available feature names (exclude close, non-feature columns)
    all_feature_names = [c for c in feat_df.columns
                         if c not in ("close", "timestamp", "open_time")
                         and c not in BLACKLIST]

    # Get timestamps for period labels
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    # Run folds
    fold_results: List[FoldResult] = []
    for fold in folds:
        from datetime import datetime, timezone
        try:
            ts_start = datetime.fromtimestamp(timestamps[fold.test_start] / 1000, tz=timezone.utc)
            ts_end = datetime.fromtimestamp(
                timestamps[min(fold.test_end - 1, n_bars - 1)] / 1000, tz=timezone.utc)
            period = f"{ts_start:%Y-%m}→{ts_end:%Y-%m}"
        except (ValueError, OSError, IndexError):
            period = f"fold_{fold.idx}"

        print(f"\n  Fold {fold.idx}: {period} "
              f"(train={fold.train_end - fold.train_start}, test={fold.test_end - fold.test_start})")

        t1 = time.time()
        result = run_fold(fold, feat_df, closes, all_feature_names,
                          use_hpo=use_hpo, fixed_features=fixed_features,
                          candidate_pool=candidate_pool, n_flexible=n_flexible,
                          regime_gate=regime_gate, long_only=long_only,
                          adaptive_sizing=adaptive_sizing,
                          deadzone=deadzone, min_hold=min_hold,
                          continuous_sizing=continuous_sizing,
                          hpo_trials=hpo_trials,
                          ensemble=ensemble, target_mode=target_mode,
                          trend_follow=trend_follow,
                          trend_indicator=trend_indicator,
                          trend_threshold=trend_threshold,
                          max_hold=max_hold_bars,
                          monthly_gate=monthly_gate,
                          monthly_gate_window=monthly_gate_window)
        result.period = period
        fold_results.append(result)

        elapsed = time.time() - t1
        print(f"    IC={result.ic:.4f}  Sharpe={result.sharpe:.2f}  "
              f"Return={result.total_return*100:+.2f}%  "
              f"Features={len(result.features)}  ({elapsed:.1f}s)")

    # Aggregate
    summary = stitch_results(fold_results)
    print_report(fold_results, summary)

    # Save results
    results_dict = {
        "symbol": symbol,
        "use_hpo": use_hpo,
        "hpo_trials": hpo_trials,
        "fixed_features": fixed_features,
        "candidate_pool": candidate_pool,
        "n_flexible": n_flexible,
        "regime_gate": regime_gate,
        "long_only": long_only,
        "adaptive_sizing": adaptive_sizing,
        "horizon": HORIZON,
        "target_mode": target_mode,
        "deadzone": deadzone,
        "min_hold": min_hold,
        "continuous_sizing": continuous_sizing,
        "ensemble": ensemble,
        "trend_follow": trend_follow,
        "trend_indicator": trend_indicator,
        "trend_threshold": trend_threshold,
        "max_hold": max_hold_bars,
        "monthly_gate": monthly_gate,
        "monthly_gate_window": monthly_gate_window,
        "folds": [
            {
                "idx": r.idx,
                "period": r.period,
                "ic": r.ic,
                "sharpe": r.sharpe,
                "total_return": r.total_return,
                "features": r.features,
                "n_train": r.n_train,
                "n_test": r.n_test,
            }
            for r in fold_results
        ],
        "summary": summary,
    }

    out_path = out_dir / f"wf_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
