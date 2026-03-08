#!/usr/bin/env python3
"""Walk-Forward Validation for independent short model.

Expanding-window walk-forward: each fold trains LGBM+XGB ensemble on ALL bars
(with bear sample weighting), evaluates short-only + regime-gated signal.

Usage:
    python3 -m scripts.walkforward_short --symbol BTCUSDT
    python3 -m scripts.walkforward_short --symbol BTCUSDT --no-hpo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS,
    V7_SEARCH_SPACE,
    _compute_target,
    _load_and_compute_features,
    BLACKLIST,
)
from scripts.train_short_production import (
    SHORT_FIXED_FEATURES,
    SHORT_CANDIDATE_POOL,
    DEADZONE,
    MIN_HOLD,
    MA_WINDOW,
    BEAR_WEIGHT,
    N_FLEXIBLE,
    _compute_bear_mask,
)
from alpha.signal_transform import pred_to_signal as _pred_to_signal
from scripts.backtest_alpha_v8 import COST_PER_TRADE
from features.dynamic_selector import (
    stable_icir_select,
    greedy_ic_select,
    _rankdata,
    _spearman_ic,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
MIN_TRAIN_BARS = 8760      # 12 months
TEST_BARS = 2190           # 3 months
STEP_BARS = 2190           # 3-month step
WARMUP = 65
HORIZON = 24
TARGET_MODE = "clipped"
HPO_TRIALS = 15
FEATURE_VERSION = "v11.1"


@dataclass
class Fold:
    idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass
class ShortFoldResult:
    idx: int
    period: str
    ic: float
    bear_ic: float
    sharpe: float
    total_return: float
    short_hit_rate: float
    n_short_entries: int
    features: List[str]
    n_train: int
    n_test: int


def generate_wf_folds(n_bars: int) -> List[Fold]:
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


def run_short_fold(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    *,
    use_hpo: bool = True,
    hpo_trials: int = HPO_TRIALS,
    selector: str = "stable_icir",
) -> ShortFoldResult:
    """Train and evaluate short model for a single WF fold."""
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]
    n_test = len(test_closes)

    # ── Target (ALL bars, symmetric returns) ──
    y_train_full = _compute_target(train_closes, HORIZON, TARGET_MODE)
    y_test_full = _compute_target(test_closes, HORIZON, TARGET_MODE)

    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)

    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]
    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]

    if len(X_train) < 1000:
        return ShortFoldResult(
            idx=fold.idx, period="", ic=0.0, bear_ic=0.0, sharpe=0.0,
            total_return=0.0, short_hit_rate=0.0, n_short_entries=0,
            features=[], n_train=len(X_train), n_test=0,
        )

    # ── Bear sample weighting ──
    bear_mask_train = _compute_bear_mask(train_closes, MA_WINDOW)
    train_bear = bear_mask_train[WARMUP:][train_valid]
    sample_weight = np.ones(len(y_train))
    sample_weight[train_bear] = BEAR_WEIGHT

    # ── Feature selection (stable_icir on negative-return bars) ──
    fixed_in_data = [f for f in SHORT_FIXED_FEATURES if f in feature_names]
    pool_in_data = [f for f in SHORT_CANDIDATE_POOL if f in feature_names]

    selected = list(fixed_in_data)
    if pool_in_data and N_FLEXIBLE > 0:
        neg_mask = y_train < 0
        pool_idx = [feature_names.index(f) for f in pool_in_data]
        _select = stable_icir_select if selector == "stable_icir" else greedy_ic_select
        if neg_mask.sum() > 1000:
            X_pool_neg = X_train[neg_mask][:, pool_idx]
            y_neg = y_train[neg_mask]
            flex = _select(X_pool_neg, y_neg, pool_in_data, top_k=N_FLEXIBLE)
        else:
            X_pool = X_train[:, pool_idx]
            flex = _select(X_pool, y_train, pool_in_data, top_k=N_FLEXIBLE)
        selected.extend(flex)

    if not selected:
        return ShortFoldResult(
            idx=fold.idx, period="", ic=0.0, bear_ic=0.0, sharpe=0.0,
            total_return=0.0, short_hit_rate=0.0, n_short_entries=0,
            features=[], n_train=len(X_train), n_test=0,
        )

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # ── HPO (bear IC objective) ──
    params = dict(V7_DEFAULT_PARAMS)
    if use_hpo:
        try:
            from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig

            n_tr = len(X_train_sel)
            val_size = min(n_tr // 4, TEST_BARS)
            X_hpo_train = X_train_sel[:-val_size]
            y_hpo_train = y_train[:-val_size]
            X_hpo_val = X_train_sel[-val_size:]
            y_hpo_val = y_train[-val_size:]
            neg_mask_val = y_hpo_val < 0

            def objective(trial_params):
                if neg_mask_val.sum() < 10:
                    return 0.0
                p = {**V7_DEFAULT_PARAMS, **trial_params}
                dtrain = lgb.Dataset(X_hpo_train, label=y_hpo_train)
                dval = lgb.Dataset(X_hpo_val, label=y_hpo_val, reference=dtrain)
                bst = lgb.train(
                    p, dtrain, num_boost_round=p["n_estimators"],
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(0)],
                )
                y_hat = bst.predict(X_hpo_val)
                yp_neg = y_hat[neg_mask_val]
                yt_neg = y_hpo_val[neg_mask_val]
                if np.std(yp_neg) > 1e-12 and np.std(yt_neg) > 1e-12:
                    return float(_spearman_ic(_rankdata(yp_neg), _rankdata(yt_neg)))
                return 0.0

            opt = HyperOptimizer(
                search_space=V7_SEARCH_SPACE, objective_fn=objective,
                config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"),
            )
            result = opt.optimize()
            params = {**params, **result.best_params}
        except Exception as e:
            logger.warning("HPO failed fold %d: %s", fold.idx, e)

    # ── Train LGBM (with bear weighting) ──
    sw_sel = sample_weight  # already aligned
    dtrain = lgb.Dataset(X_train_sel, label=y_train, weight=sw_sel)
    lgbm_bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[lgb.log_evaluation(0)],
    )
    lgbm_pred = lgbm_bst.predict(X_test_sel)

    # ── Train XGB ──
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
        dtrain_xgb = xgb.DMatrix(X_train_sel, label=y_train, weight=sw_sel)
        dtest_xgb = xgb.DMatrix(X_test_sel)
        xgb_bst = xgb.train(
            xgb_params, dtrain_xgb,
            num_boost_round=params.get("n_estimators", 500),
        )
        xgb_pred = xgb_bst.predict(dtest_xgb)
        y_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred
    except Exception:
        y_pred = lgbm_pred

    # ── IC metrics ──
    test_valid = ~np.isnan(y_test_full)
    ic = 0.0
    if test_valid.sum() > 10:
        yp_v = y_pred[test_valid]
        yt_v = y_test_full[test_valid]
        if np.std(yp_v) > 1e-12 and np.std(yt_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(yp_v), _rankdata(yt_v)))

    # Bear IC (negative return bars only)
    neg_ret_mask = test_valid & (y_test_full < 0)
    bear_ic = 0.0
    if neg_ret_mask.sum() > 10:
        yp_neg = y_pred[neg_ret_mask]
        yt_neg = y_test_full[neg_ret_mask]
        if np.std(yp_neg) > 1e-12 and np.std(yt_neg) > 1e-12:
            bear_ic = float(_spearman_ic(_rankdata(yp_neg), _rankdata(yt_neg)))

    # ── Short-only signal + regime gate ──
    signal = _pred_to_signal(y_pred, target_mode=TARGET_MODE,
                              deadzone=DEADZONE, min_hold=MIN_HOLD)
    np.clip(signal, -1.0, 0.0, out=signal)

    # Regime gate
    bear_mask_test = _compute_bear_mask(test_closes, MA_WINDOW)
    signal[~bear_mask_test] = 0.0

    # ── PnL simulation ──
    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    sig_trade = signal[:len(ret_1bar)]
    gross_pnl = sig_trade * ret_1bar
    turnover = np.abs(np.diff(sig_trade, prepend=0))
    cost = turnover * COST_PER_TRADE

    # Funding cost (short receives positive funding)
    funding_cost = np.zeros(len(sig_trade))
    if "funding_rate" in test_df.columns:
        fr = test_df["funding_rate"].values[:len(sig_trade)].astype(np.float64)
        fr = np.nan_to_num(fr, 0.0)
        funding_cost = sig_trade * fr / 8.0

    net_pnl = gross_pnl - cost - funding_cost

    active = sig_trade != 0
    n_active = int(active.sum())
    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))

    # Short hit rate
    short_entries = np.where((sig_trade < 0) & (np.roll(sig_trade, 1) == 0))[0]
    n_entries = len(short_entries)
    hits = 0
    for idx in short_entries:
        if idx + HORIZON < len(test_closes):
            fwd = test_closes[idx + HORIZON] / test_closes[idx] - 1.0
            if fwd < 0:
                hits += 1
    hit_rate = hits / max(n_entries, 1)

    return ShortFoldResult(
        idx=fold.idx,
        period="",
        ic=ic,
        bear_ic=bear_ic,
        sharpe=sharpe,
        total_return=total_return,
        short_hit_rate=hit_rate,
        n_short_entries=n_entries,
        features=selected,
        n_train=len(X_train),
        n_test=n_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward for short model")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--selector", choices=["greedy", "stable_icir"],
                        default="stable_icir")
    parser.add_argument("--out-dir", default="results/walkforward_short")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    symbol = args.symbol.upper()
    use_hpo = not args.no_hpo
    hpo_trials = args.hpo_trials
    selector = args.selector
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Walk-Forward Validation: SHORT MODEL — {symbol}")
    print(f"  Selector: {selector}")
    print(f"  HPO: {'ON' if use_hpo else 'OFF'}"
          f"{f' ({hpo_trials} trials, bear IC objective)' if use_hpo else ''}")
    print(f"  Signal: deadzone={DEADZONE}, min_hold={MIN_HOLD}, regime_gate=SMA({MA_WINDOW})")
    print(f"  Features: {len(SHORT_FIXED_FEATURES)} fixed + {N_FLEXIBLE} flexible")
    print(f"  Bear weight: {BEAR_WEIGHT}x, Ensemble: LGBM+XGB")
    print(f"  Min train: {MIN_TRAIN_BARS} bars ({MIN_TRAIN_BARS // 24:.0f} days)")
    print(f"  Test window: {TEST_BARS} bars ({TEST_BARS // 24:.0f} days)")

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
        print("  Not enough data")
        return

    # Load/compute features (uses pandas pickle for cached feature DataFrames —
    # this is safe as the cache is generated locally, not from external sources)
    feat_df = None
    cache_dir = Path(".cache/features")
    if not args.no_cache:
        csv_stat = csv_path.stat()
        cache_key_raw = f"{symbol}_{csv_stat.st_mtime}_{csv_stat.st_size}_{FEATURE_VERSION}"
        cache_key = hashlib.sha256(cache_key_raw.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{symbol}_{cache_key}.pkl"
        if cache_file.exists():
            print(f"  Loading cached features...")
            t0 = time.time()
            feat_df = pd.read_pickle(cache_file)
            print(f"  Loaded in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")

    if feat_df is None:
        print("  Computing features...")
        t0 = time.time()
        feat_df = _load_and_compute_features(symbol, df)
        if feat_df is None:
            print("  Feature computation failed")
            return
        print(f"  Computed in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")
        if not args.no_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            feat_df.to_pickle(cache_file)

    closes = (feat_df["close"].values.astype(np.float64)
              if "close" in feat_df.columns
              else df["close"].values.astype(np.float64))

    all_feature_names = [c for c in feat_df.columns
                         if c not in ("close", "timestamp", "open_time")
                         and c not in BLACKLIST]

    # Timestamps for period labels
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    # ── Run folds ──
    from datetime import datetime, timezone

    results: List[ShortFoldResult] = []
    for fold in folds:
        # Period label
        try:
            ts_s = datetime.fromtimestamp(timestamps[fold.test_start] / 1000, tz=timezone.utc)
            ts_e = datetime.fromtimestamp(
                timestamps[min(fold.test_end - 1, n_bars - 1)] / 1000, tz=timezone.utc)
            period = f"{ts_s:%Y-%m}→{ts_e:%Y-%m}"
        except (ValueError, OSError, IndexError):
            period = f"fold_{fold.idx}"

        print(f"\n  Fold {fold.idx}: {period} "
              f"(train={fold.train_end - fold.train_start:,}, test={fold.test_end - fold.test_start:,})")

        t0 = time.time()
        r = run_short_fold(
            fold, feat_df, closes, all_feature_names,
            use_hpo=use_hpo, hpo_trials=hpo_trials, selector=selector,
        )
        r.period = period
        results.append(r)
        elapsed = time.time() - t0

        marker = "+" if r.sharpe > 0 else " "
        print(f"    IC={r.ic:.4f}  Bear_IC={r.bear_ic:.4f}  "
              f"Sharpe={marker}{r.sharpe:.2f}  Return={r.total_return*100:+.2f}%  "
              f"Hit={r.short_hit_rate:.0%}  Entries={r.n_short_entries}  "
              f"({elapsed:.0f}s)")
        print(f"    Features: {r.features[:5]}{'...' if len(r.features) > 5 else ''}")

    # ── Report ──
    print(f"\n{'='*90}")
    print(f"  SHORT MODEL WALK-FORWARD REPORT — {symbol}")
    print(f"{'='*90}")
    print(f"  {'Fold':<5} {'Period':<17} {'IC':>7} {'BearIC':>7} {'Sharpe':>8} {'Return':>8} {'Hit%':>6} {'Entries':>8}")
    print(f"  {'-'*88}")

    for r in results:
        marker = "+" if r.sharpe > 0 else " "
        print(f"  {r.idx:<5} {r.period:<17} {r.ic:>7.4f} {r.bear_ic:>7.4f} "
              f"{marker}{r.sharpe:>7.2f} {r.total_return*100:>+7.2f}% "
              f"{r.short_hit_rate:>5.0%} {r.n_short_entries:>7}")

    print(f"  {'-'*88}")

    n_folds = len(results)
    pos_sharpe = sum(1 for r in results if r.sharpe > 0)
    pos_bear_ic = sum(1 for r in results if r.bear_ic > 0)
    sharpes = [r.sharpe for r in results]
    bear_ics = [r.bear_ic for r in results]
    returns = [r.total_return for r in results]
    hit_rates = [r.short_hit_rate for r in results]

    pass_threshold = int(np.ceil(n_folds * 2 / 3))
    passed = pos_sharpe >= pass_threshold

    print(f"\n  VERDICT: {pos_sharpe}/{n_folds} positive Sharpe "
          f"(need >= {pass_threshold}) → {'PASS' if passed else 'FAIL'}")
    print(f"\n  Average IC:        {np.mean([r.ic for r in results]):.4f}")
    print(f"  Average Bear IC:   {np.mean(bear_ics):.4f} ({pos_bear_ic}/{n_folds} positive)")
    print(f"  Average Sharpe:    {np.mean(sharpes):.2f}")
    print(f"  Median Sharpe:     {np.median(sharpes):.2f}")
    print(f"  Total return:      {sum(returns)*100:+.2f}%")
    print(f"  Average hit rate:  {np.mean(hit_rates):.1%}")

    # Feature stability
    feature_counts: Dict[str, int] = {}
    for r in results:
        for f in r.features:
            feature_counts[f] = feature_counts.get(f, 0) + 1
    stable = {k: v for k, v in sorted(feature_counts.items(), key=lambda x: -x[1])
              if v >= n_folds * 0.8}
    if stable:
        print(f"\n  Feature stability (>= 80% folds):")
        for fname, count in stable.items():
            print(f"    {fname}: {count}/{n_folds}")

    # Save results
    summary = {
        "symbol": symbol,
        "n_folds": n_folds,
        "positive_sharpe": pos_sharpe,
        "pass_threshold": pass_threshold,
        "passed": passed,
        "avg_ic": float(np.mean([r.ic for r in results])),
        "avg_bear_ic": float(np.mean(bear_ics)),
        "avg_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "total_return": float(sum(returns)),
        "avg_hit_rate": float(np.mean(hit_rates)),
        "fold_sharpes": sharpes,
        "fold_bear_ics": bear_ics,
        "fold_returns": returns,
        "stable_features": stable,
        "config": {
            "fixed_features": SHORT_FIXED_FEATURES,
            "candidate_pool": SHORT_CANDIDATE_POOL,
            "n_flexible": N_FLEXIBLE,
            "deadzone": DEADZONE,
            "min_hold": MIN_HOLD,
            "ma_window": MA_WINDOW,
            "bear_weight": BEAR_WEIGHT,
            "selector": selector,
            "hpo_trials": hpo_trials if use_hpo else 0,
        },
    }
    out_path = out_dir / f"{symbol}_short_wf.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
