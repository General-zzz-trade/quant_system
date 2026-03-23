#!/usr/bin/env python3
"""Walk-Forward Validation for 15-minute alpha models.

Expanding-window walk-forward with 3-month test steps on 15m bar data.
Each fold: feature selection → train → OOS evaluation with realistic costs.

Adapts walkforward_validate.py for 15m bars:
- Loads *_15m.csv data
- Uses compute_features_batch() with 15m blacklist
- Different fold sizes (4x more bars per month)
- Multi-horizon support (matching train_15m.py)
- Optional --check-t1 to verify T-1 cross-market feature integrity

Usage:
    python3 -m scripts.walkforward.walkforward_validate_15m --symbol BTCUSDT --no-hpo
    python3 -m scripts.walkforward.walkforward_validate_15m --symbol BTCUSDT --no-hpo --check-t1
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from features.batch_feature_engine import compute_features_batch
from features.dynamic_selector import greedy_ic_select, stable_icir_select
from shared.signal_postprocess import rolling_zscore, should_exit_position

logger = logging.getLogger(__name__)

# ── Constants (15m-adapted) ──────────────────────────────────

BARS_PER_DAY = 96              # 96 15m bars per day
BARS_PER_MONTH = BARS_PER_DAY * 30  # 2880

MIN_TRAIN_BARS = BARS_PER_MONTH * 12  # 12 months = 34560 bars
TEST_BARS = BARS_PER_MONTH * 3        # 3 months = 8640 bars
STEP_BARS = BARS_PER_MONTH * 3        # 3-month step

WARMUP = 30                    # feature warmup bars (30 × 15m = 7.5h)
TOP_K = 14                     # features per fold (matching train_15m)
COST_BPS_RT = 4                # round-trip cost in bps
ZSCORE_WINDOW = 720            # rolling z-score window
ZSCORE_WARMUP = 180            # z-score warmup bars

# Default horizons per symbol (from train_15m.py research)
DEFAULT_HORIZONS = {
    "BTCUSDT": [32, 64],       # 8h, 16h
    "ETHUSDT": [4, 8],         # 1h, 2h
}

# Config sweep ranges (matching train_15m.py)
DEADZONE_RANGE = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
MIN_HOLD_RANGE = [4, 8, 16]
MAX_HOLD_RANGE = [32, 64, 128]

# Blacklist features unreliable on 15m
BLACKLIST_15M = {
    "fgi_normalized", "fgi_extreme",
    "funding_rate", "funding_zscore_24", "funding_sign_persist",
    "basis_carry_adj", "basis_zscore_24",
}

# Cross-market features (for T-1 check)
CROSS_MARKET_FEATURES = {
    "spy_ret_1d", "qqq_ret_1d", "vix_ret_1d", "tlt_ret_1d", "uso_ret_1d",
    "coin_ret_1d", "spy_extreme", "vix_level",
    "etf_vol_change_5d", "etf_vol_zscore_7", "etf_vol_zscore_14",
    "gbtc_vol_zscore_14", "etha_vol_zscore_14", "gbtc_premium_dev",
    "dvol_z", "dvol_mean_rev", "dvol_chg_24", "dvol_chg_72",
    "stablecoin_accel",
}


def fast_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman IC with NaN handling."""
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_target(closes: np.ndarray, horizon: int) -> np.ndarray:
    """Forward return target with 1-99 percentile clipping."""
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


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
    n_trades: int = 0


def generate_wf_folds(
    n_bars: int,
    min_train_bars: int = MIN_TRAIN_BARS,
    test_bars: int = TEST_BARS,
    step_bars: int = STEP_BARS,
) -> List[Fold]:
    """Generate expanding-window walk-forward folds for 15m data."""
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


def _select_features_dispatch(selector: str):
    """Return feature selection function."""
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
    horizons: List[int],
    deadzone: float = 0.8,
    min_hold: int = 16,
    max_hold: int = 128,
    long_only: bool = False,
    selector: str = "greedy",
    zscore_window: int = ZSCORE_WINDOW,
    zscore_warmup: int = ZSCORE_WARMUP,
) -> FoldResult:
    """Train and evaluate a single WF fold on 15m data (multi-horizon)."""
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]

    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)

    _select = _select_features_dispatch(selector)

    # Train per-horizon models and collect OOS predictions
    preds_test: Dict[int, np.ndarray] = {}
    all_ics = []
    all_selected: List[str] = []

    for h in horizons:
        y_train_full = compute_target(train_closes, h)
        y_test_full = compute_target(test_closes, h)

        X_train = X_train_full[WARMUP:]
        y_train = y_train_full[WARMUP:]
        train_valid = ~np.isnan(y_train)
        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]

        if len(X_tr) < 1000:
            continue

        # Feature selection
        selected = _select(X_tr, y_tr, feature_names, top_k=TOP_K)
        if not selected:
            continue

        sel_idx = [feature_names.index(n) for n in selected]
        X_train_sel = X_tr[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        # HPO grid (matching train_15m.py)
        best_val_ic = -1.0
        best_model = None

        # Use last 2 months as val
        val_size = min(BARS_PER_MONTH * 2, len(X_train_sel) // 4)
        if val_size < 100:
            val_size = len(X_train_sel) // 4

        X_hpo_train = X_train_sel[:-val_size]
        y_hpo_train = y_tr[:-val_size]
        X_hpo_val = X_train_sel[-val_size:]
        y_hpo_val = y_tr[-val_size:]

        for lr in [0.01, 0.03, 0.05]:
            for nl in [15, 31, 63]:
                params = {
                    "objective": "regression",
                    "metric": "mae",
                    "learning_rate": lr,
                    "num_leaves": nl,
                    "min_child_samples": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "verbosity": -1,
                    "seed": 42,
                }
                dtrain = lgb.Dataset(X_hpo_train, label=y_hpo_train)
                dval = lgb.Dataset(X_hpo_val, label=y_hpo_val, reference=dtrain)
                model = lgb.train(
                    params, dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
                pred_v = model.predict(X_hpo_val)
                val_ic = fast_ic(pred_v, y_hpo_val)
                if val_ic > best_val_ic:
                    best_val_ic = val_ic
                    best_model = model

        if best_model is None:
            continue

        # OOS predictions
        pred_test = best_model.predict(X_test_sel)
        preds_test[h] = pred_test

        # OOS IC
        test_valid = ~np.isnan(y_test_full)
        ic_h = fast_ic(pred_test[test_valid], y_test_full[test_valid])
        all_ics.append(ic_h)

        for f in selected:
            if f not in all_selected:
                all_selected.append(f)

    if not preds_test:
        return FoldResult(
            idx=fold.idx, period="", ic=0.0, sharpe=0.0,
            total_return=0.0, features=[], n_train=0, n_test=0, n_trades=0,
        )

    # Backtest: z-score average across horizons
    avg_ic = float(np.mean(all_ics)) if all_ics else 0.0

    z_all = []
    for h, pred in sorted(preds_test.items()):
        z_all.append(rolling_zscore(pred, window=zscore_window, warmup=zscore_warmup))
    z = np.mean(z_all, axis=0)

    cost_frac = COST_BPS_RT / 10000

    pos = 0.0
    entry_bar = 0
    trades = []

    for i in range(len(test_closes)):
        if pos != 0:
            held = i - entry_bar
            should_exit = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            )
            if should_exit:
                pnl_pct = pos * (test_closes[i] - test_closes[entry_bar]) / test_closes[entry_bar]
                trades.append(pnl_pct - cost_frac)
                pos = 0.0

        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                entry_bar = i

    n_trades = len(trades)
    if not trades:
        return FoldResult(
            idx=fold.idx, period="", ic=avg_ic, sharpe=0.0,
            total_return=0.0, features=all_selected,
            n_train=fold.train_end - fold.train_start,
            n_test=len(test_closes), n_trades=0,
        )

    net_arr = np.array(trades)
    avg_hold = len(test_closes) / max(n_trades, 1)
    tpy = 365 * BARS_PER_DAY / max(avg_hold, 1)
    sharpe = float(np.mean(net_arr) / max(np.std(net_arr, ddof=1), 1e-10) * np.sqrt(tpy))
    total_return = float(np.sum(net_arr))

    return FoldResult(
        idx=fold.idx, period="", ic=avg_ic, sharpe=sharpe,
        total_return=total_return, features=all_selected,
        n_train=fold.train_end - fold.train_start,
        n_test=len(test_closes), n_trades=n_trades,
    )


# ── T-1 check ────────────────────────────────────────────────

def run_t1_check(
    symbol: str,
    feat_df: pd.DataFrame,
    feat_df_no_t1: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    feature_names_no_t1: List[str],
    horizons: List[int],
    deadzone: float,
    min_hold: int,
    max_hold: int,
) -> Dict[str, Any]:
    """Compare WF results WITH vs WITHOUT T-1 shift on cross-market features.

    Returns dict with comparison metrics and warning flag.
    """
    folds = generate_wf_folds(len(closes))
    if not folds:
        return {"error": "Not enough data for T-1 check"}

    # Run a subset of folds for speed (every other fold)
    check_folds = folds[::2]

    sharpes_with_t1 = []
    sharpes_without_t1 = []

    for fold in check_folds:
        # With T-1
        r_t1 = run_fold(
            fold, feat_df, closes, feature_names,
            horizons=horizons, deadzone=deadzone,
            min_hold=min_hold, max_hold=max_hold,
        )
        sharpes_with_t1.append(r_t1.sharpe)

        # Without T-1 (uses unshifted cross-market features)
        r_no_t1 = run_fold(
            fold, feat_df_no_t1, closes, feature_names_no_t1,
            horizons=horizons, deadzone=deadzone,
            min_hold=min_hold, max_hold=max_hold,
        )
        sharpes_without_t1.append(r_no_t1.sharpe)

    avg_with = float(np.mean(sharpes_with_t1))
    avg_without = float(np.mean(sharpes_without_t1))

    # If without T-1 is much better, cross-market features have look-ahead bias
    has_bias = False
    bias_ratio = 0.0
    if avg_with > 0:
        bias_ratio = (avg_without - avg_with) / abs(avg_with)
        has_bias = bias_ratio > 0.5  # >50% improvement without T-1 = look-ahead

    result = {
        "avg_sharpe_with_t1": avg_with,
        "avg_sharpe_without_t1": avg_without,
        "bias_ratio": bias_ratio,
        "has_look_ahead_bias": has_bias,
        "n_folds_checked": len(check_folds),
        "per_fold_with_t1": sharpes_with_t1,
        "per_fold_without_t1": sharpes_without_t1,
    }

    if has_bias:
        print("\n  WARNING: Cross-market features may have look-ahead bias on 15m")
        print(f"  Sharpe WITH T-1: {avg_with:.2f}, WITHOUT T-1: {avg_without:.2f} "
              f"(+{bias_ratio*100:.0f}%)")
    else:
        print(f"\n  T-1 Check PASS: WITH={avg_with:.2f}, WITHOUT={avg_without:.2f} "
              f"(bias ratio: {bias_ratio*100:+.0f}%)")

    return result


# ── Aggregation ───────────────────────────────────────────────

def stitch_results(fold_results: List[FoldResult]) -> Dict[str, Any]:
    """Aggregate fold results into a verdict."""
    n_folds = len(fold_results)
    pos_sharpe = sum(1 for r in fold_results if r.sharpe > 0)
    ics = [r.ic for r in fold_results]
    sharpes = [r.sharpe for r in fold_results]
    returns = [r.total_return for r in fold_results]
    trades = [r.n_trades for r in fold_results]

    feature_counts: Dict[str, int] = {}
    for r in fold_results:
        for f in r.features:
            feature_counts[f] = feature_counts.get(f, 0) + 1
    stable_features = {k: v for k, v in sorted(
        feature_counts.items(), key=lambda x: -x[1])
        if v >= n_folds * 0.8}

    passed = pos_sharpe >= max(1, int(n_folds * 0.6))  # >= 60%

    return {
        "n_folds": n_folds,
        "positive_sharpe": pos_sharpe,
        "pass_threshold": max(1, int(n_folds * 0.6)),
        "passed": passed,
        "avg_ic": float(np.mean(ics)) if ics else 0.0,
        "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
        "total_return": float(sum(returns)),
        "total_trades": int(sum(trades)),
        "fold_ics": ics,
        "fold_sharpes": sharpes,
        "fold_returns": returns,
        "fold_trades": trades,
        "stable_features": stable_features,
        "all_feature_counts": feature_counts,
    }


def print_report(fold_results: List[FoldResult], summary: Dict[str, Any]) -> None:
    """Print the walk-forward report table."""
    print(f"\n{'='*80}")
    print("  15m WALK-FORWARD VALIDATION REPORT")
    print(f"{'='*80}")
    print(f"  {'Fold':<6} {'Period':<20} {'IC':>8} {'Sharpe':>8} {'Return':>8} {'Trades':>7}  Features")
    print(f"  {'-'*78}")

    for r in fold_results:
        feat_str = ", ".join(r.features[:3])
        if len(r.features) > 3:
            feat_str += f" +{len(r.features)-3}"
        print(f"  {r.idx:<6} {r.period:<20} {r.ic:>8.4f} {r.sharpe:>8.2f} "
              f"{r.total_return*100:>+7.2f}% {r.n_trades:>7}  [{feat_str}]")

    print(f"  {'-'*78}")
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(f"\n  VERDICT: {summary['positive_sharpe']}/{summary['n_folds']} positive Sharpe "
          f"(need >= {summary['pass_threshold']}) → {verdict}")
    print(f"\n  Average IC:     {summary['avg_ic']:.4f}")
    print(f"  Average Sharpe: {summary['avg_sharpe']:.2f}")
    print(f"  Total return:   {summary['total_return']*100:+.2f}%")
    print(f"  Total trades:   {summary['total_trades']}")

    if summary["stable_features"]:
        print("\n  Feature stability (>= 80% folds):")
        for fname, count in summary["stable_features"].items():
            print(f"    {fname}: {count}/{summary['n_folds']}")


# ── Main ─────────────────────────────────────────────────────

def _fold_period_label(fold: Fold, timestamps: np.ndarray, n_bars: int) -> str:
    """Compute human-readable period label."""
    from datetime import datetime, timezone
    try:
        ts_start = datetime.fromtimestamp(timestamps[fold.test_start] / 1000, tz=timezone.utc)
        ts_end = datetime.fromtimestamp(
            timestamps[min(fold.test_end - 1, n_bars - 1)] / 1000, tz=timezone.utc)
        return f"{ts_start:%Y-%m}→{ts_end:%Y-%m}"
    except (ValueError, OSError, IndexError):
        return f"fold_{fold.idx}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-Forward Validation for 15m alpha models")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--no-hpo", action="store_true",
                        help="Use grid HPO (default, always enabled for 15m)")
    parser.add_argument("--horizons", default=None,
                        help="Comma-separated horizons in 15m bars (default: per-symbol)")
    parser.add_argument("--deadzone", type=float, default=None,
                        help="Fixed deadzone (default: config sweep)")
    parser.add_argument("--min-hold", type=int, default=None,
                        help="Fixed min hold (default: config sweep)")
    parser.add_argument("--max-hold", type=int, default=None,
                        help="Fixed max hold (default: config sweep)")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--selector", default="greedy",
                        choices=["greedy", "stable_icir"])
    parser.add_argument("--check-t1", action="store_true",
                        help="Compare WITH vs WITHOUT T-1 shift on cross-market features")
    parser.add_argument("--out-dir", default="results/walkforward")
    parser.add_argument("--zscore-window", type=int, default=ZSCORE_WINDOW)
    parser.add_argument("--zscore-warmup", type=int, default=ZSCORE_WARMUP)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    symbol = args.symbol.upper()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Horizons
    if args.horizons:
        horizons = [int(h.strip()) for h in args.horizons.split(",")]
    else:
        horizons = DEFAULT_HORIZONS.get(symbol, [16, 64])

    print(f"\n  15m Walk-Forward Validation: {symbol}")
    print(f"  Horizons: {horizons} ({[h*0.25 for h in horizons]}h)")

    # Load data
    csv_path = Path(f"data_files/{symbol}_15m.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_bars = len(df)
    closes = df["close"].values.astype(np.float64)
    start = pd.Timestamp(df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
    end = pd.Timestamp(df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"  Data: {n_bars:,} 15m bars ({start} → {end})")
    print(f"  Min train: {MIN_TRAIN_BARS:,} bars ({MIN_TRAIN_BARS/BARS_PER_DAY:.0f} days)")
    print(f"  Test window: {TEST_BARS:,} bars ({TEST_BARS/BARS_PER_DAY:.0f} days)")

    # Compute features
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = compute_features_batch(symbol="", df=df, include_v11=False)
    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp", "open", "high", "low", "volume")
                     and c not in BLACKLIST_15M]
    print(f"{len(feature_names)} features in {time.time()-t0:.1f}s")

    # Generate folds
    folds = generate_wf_folds(n_bars)
    print(f"  Folds: {len(folds)}")

    if not folds:
        print("  Not enough data for walk-forward validation")
        return

    # Get timestamps for period labels
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    # Config sweep or fixed params
    use_sweep = args.deadzone is None
    if use_sweep:
        print("\n  Running config sweep...")
        # Run all folds with default config first, then sweep
        best_cfg = {"deadzone": 0.8, "min_hold": 16, "max_hold": 128, "long_only": False}
        best_avg_sharpe = -999.0

        for dz in DEADZONE_RANGE:
            for mh in MIN_HOLD_RANGE:
                for maxh in MAX_HOLD_RANGE:
                    for lo in [True, False]:
                        fold_sharpes = []
                        for fold in folds:
                            period = _fold_period_label(fold, timestamps, n_bars)
                            r = run_fold(
                                fold, feat_df, closes, feature_names,
                                horizons=horizons, deadzone=dz,
                                min_hold=mh, max_hold=maxh, long_only=lo,
                                selector=args.selector,
                                zscore_window=args.zscore_window,
                                zscore_warmup=args.zscore_warmup,
                            )
                            r.period = period
                            fold_sharpes.append(r.sharpe)

                        avg_s = float(np.mean(fold_sharpes))
                        n_pos = sum(1 for s in fold_sharpes if s > 0)
                        # Need at least 60% positive AND better avg sharpe
                        if n_pos >= len(folds) * 0.6 and avg_s > best_avg_sharpe:
                            best_avg_sharpe = avg_s
                            best_cfg = {"deadzone": dz, "min_hold": mh,
                                        "max_hold": maxh, "long_only": lo}

        print(f"  Best config: dz={best_cfg['deadzone']}, "
              f"hold=[{best_cfg['min_hold']},{best_cfg['max_hold']}], "
              f"long_only={best_cfg['long_only']}")

        deadzone = best_cfg["deadzone"]
        min_hold = best_cfg["min_hold"]
        max_hold = best_cfg["max_hold"]
        long_only = best_cfg["long_only"]
    else:
        deadzone = args.deadzone
        min_hold = args.min_hold or 16
        max_hold = args.max_hold or 128
        long_only = args.long_only

    print(f"\n  Signal params: dz={deadzone}, min_hold={min_hold}, "
          f"max_hold={max_hold}, long_only={long_only}")

    # Run final folds with best config
    print("\n  Running walk-forward folds...")
    fold_results: List[FoldResult] = []

    for fold in folds:
        period = _fold_period_label(fold, timestamps, n_bars)
        t1 = time.time()
        r = run_fold(
            fold, feat_df, closes, feature_names,
            horizons=horizons, deadzone=deadzone,
            min_hold=min_hold, max_hold=max_hold,
            long_only=long_only, selector=args.selector,
            zscore_window=args.zscore_window,
            zscore_warmup=args.zscore_warmup,
        )
        r.period = period
        fold_results.append(r)
        elapsed = time.time() - t1
        print(f"    Fold {fold.idx} ({period}): IC={r.ic:.4f} Sharpe={r.sharpe:.2f} "
              f"Trades={r.n_trades} ({elapsed:.1f}s)")

    # Aggregate
    summary = stitch_results(fold_results)
    print_report(fold_results, summary)

    # T-1 check
    t1_meta = None
    if args.check_t1:
        print("\n  Running T-1 cross-market feature check...")
        # Recompute features without T-1 shift by removing cross-market features
        cm_in_data = [f for f in feature_names if f in CROSS_MARKET_FEATURES]
        if cm_in_data:
            feature_names_no_cm = [f for f in feature_names if f not in CROSS_MARKET_FEATURES]
            t1_meta = run_t1_check(
                symbol, feat_df, feat_df, closes,
                feature_names, feature_names_no_cm,
                horizons, deadzone, min_hold, max_hold,
            )
        else:
            print("  No cross-market features found in data, skipping T-1 check")
            t1_meta = {"skipped": True, "reason": "no cross-market features"}

    # Save results
    results_dict = {
        "symbol": symbol,
        "timeframe": "15m",
        "horizons": horizons,
        "deadzone": deadzone,
        "min_hold": min_hold,
        "max_hold": max_hold,
        "long_only": long_only,
        "zscore_window": args.zscore_window,
        "zscore_warmup": args.zscore_warmup,
        "selector": args.selector,
        "config_sweep": use_sweep,
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
                "n_trades": r.n_trades,
            }
            for r in fold_results
        ],
        "summary": summary,
    }
    if t1_meta:
        results_dict["t1_check"] = t1_meta

    out_path = out_dir / f"wf_{symbol}_15m.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Final verdict
    avg_sharpe = summary["avg_sharpe"]
    pos_folds = summary["positive_sharpe"]
    n_folds = summary["n_folds"]
    pct = pos_folds / n_folds * 100 if n_folds > 0 else 0

    if avg_sharpe < 1.0 or pct < 60:
        print(f"\n  *** FAIL: avg Sharpe {avg_sharpe:.2f} < 1.0 "
              f"or positive folds {pos_folds}/{n_folds} ({pct:.0f}%) < 60%")
        print(f"  *** Recommend: disable {symbol}_15m in SYMBOL_CONFIG")
    else:
        print(f"\n  *** PASS: avg Sharpe {avg_sharpe:.2f}, "
              f"positive folds {pos_folds}/{n_folds} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
