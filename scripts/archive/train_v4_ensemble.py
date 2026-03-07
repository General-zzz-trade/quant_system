#!/usr/bin/env python3
"""Train V4 Ensemble — multi-model, multi-timeframe, Ridge meta-learner.

5 diverse base models with nested walk-forward:
  A: reg_h1    — regression, 1-bar horizon, 1h features (short-term reversal)
  B: reg_h5    — regression, 5-bar horizon, 1h features (=V3 medium-term)
  C: cls_h5    — classifier, 5-bar horizon, 1h features (direction prediction)
  D: reg_h5_4h — regression, 5-bar horizon, 4h features only (slow signal)
  E: reg_h5_all— regression, 5-bar horizon, 1h+4h features (full spectrum)

Meta-learner: Ridge regression with positive weight constraint.
Bootstrap validation: block bootstrap P(Sharpe>0) gate.

Usage:
    python3 -m scripts.train_v4_ensemble --all
    python3 -m scripts.train_v4_ensemble --symbol BTCUSDT
    python3 -m scripts.train_v4_ensemble --all --n-outer-folds 5 --top-k 20
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import spearman_ic_select
from features.multi_timeframe import TF4H_FEATURE_NAMES, compute_4h_features
from scripts.train_v3_walkforward import (
    _load_schedule, _compute_features, _build_cross_features,
    vol_normalized_target, expanding_window_folds,
    V3_PARAMS, EARLY_STOPPING_ROUNDS, EMBARGO_BARS, BLACKLIST,
    INTERACTION_FEATURES, get_available_features,
)

logger = logging.getLogger(__name__)

# ── Base model definitions ───────────────────────────────────

BASE_MODELS = [
    {"tag": "reg_h1",     "type": "regression",   "horizon": 1, "features": "1h"},
    {"tag": "reg_h5",     "type": "regression",   "horizon": 5, "features": "1h"},
    {"tag": "cls_h5",     "type": "classifier",   "horizon": 5, "features": "1h"},
    {"tag": "reg_h5_4h",  "type": "regression",   "horizon": 5, "features": "4h"},
    {"tag": "reg_h5_all", "type": "regression",   "horizon": 5, "features": "all"},
]

# Lighter params for base models (less overfitting with ensemble)
BASE_PARAMS = {
    **V3_PARAMS,
    "n_estimators": 300,
    "max_depth": 4,
    "num_leaves": 15,
    "learning_rate": 0.03,
    "min_child_samples": 80,
}


# ── Feature management ───────────────────────────────────────

def _get_feature_set(
    spec: str,
    available_1h: List[str],
    available_4h: List[str],
) -> List[str]:
    """Return feature list for a model spec."""
    if spec == "1h":
        return available_1h
    elif spec == "4h":
        return available_4h
    elif spec == "all":
        return available_1h + available_4h
    raise ValueError(f"Unknown feature spec: {spec}")


# ── Single base model training ───────────────────────────────

def _train_base_model(
    model_def: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    top_k: int,
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    """Train one base model on a fold, return OOS predictions.

    Returns:
        (test_predictions, selected_features, metrics)
    """
    # Feature selection on train set
    selected = spearman_ic_select(X_train, y_train, feature_names, top_k=top_k)
    sel_idx = [feature_names.index(n) for n in selected]

    X_tr_sel = X_train[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]

    model = LGBMAlphaModel(
        name=f"v4_{model_def['tag']}",
        feature_names=tuple(selected),
    )

    if model_def["type"] == "classifier":
        y_binary = (y_train > 0).astype(np.float64)
        metrics = model.fit_classifier(
            X_tr_sel, y_binary,
            params=BASE_PARAMS.copy(),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=EMBARGO_BARS,
        )
        # Predict: prob - 0.5 (centered at 0)
        prob = model._model.predict_proba(X_te_sel)[:, 1]
        test_pred = prob - 0.5
    else:
        metrics = model.fit(
            X_tr_sel, y_train,
            params=BASE_PARAMS.copy(),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=EMBARGO_BARS,
        )
        test_pred = model._model.predict(X_te_sel)

    return test_pred, selected, metrics


# ── Meta-learner ─────────────────────────────────────────────

def fit_ridge_meta(
    X_meta: np.ndarray,
    y_meta: np.ndarray,
    alpha: float = 0.1,
) -> np.ndarray:
    """Ridge regression with positive weight constraint.

    w = (X'X + alpha*I)^{-1} X'y
    w = clip(w, 0)
    w /= sum(w)
    """
    n_models = X_meta.shape[1]
    XtX = X_meta.T @ X_meta + alpha * np.eye(n_models)
    Xty = X_meta.T @ y_meta

    try:
        w = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        w = np.full(n_models, 1.0 / n_models)

    # Positive weight constraint
    w = np.clip(w, 0, None)

    # Normalize
    w_sum = w.sum()
    if w_sum > 1e-10:
        w /= w_sum
    else:
        w = np.full(n_models, 1.0 / n_models)

    return w


# ── Bootstrap validation ─────────────────────────────────────

try:
    from features._quant_rolling import cpp_bootstrap_sharpe_ci as _cpp_bootstrap
    _BOOTSTRAP_CPP = True
except ImportError:
    _BOOTSTRAP_CPP = False


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    block_size: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Block bootstrap confidence interval for Sharpe ratio."""
    n = len(returns)
    if n < 10:
        return {"sharpe_mean": 0, "sharpe_95ci_lo": 0, "sharpe_95ci_hi": 0,
                "p_sharpe_gt_0": 0, "p_sharpe_gt_05": 0}

    if _BOOTSTRAP_CPP:
        r = _cpp_bootstrap(returns.tolist(), n_bootstrap, block_size, seed)
        return {
            "sharpe_mean": r.sharpe_mean,
            "sharpe_95ci_lo": r.sharpe_95ci_lo,
            "sharpe_95ci_hi": r.sharpe_95ci_hi,
            "p_sharpe_gt_0": r.p_sharpe_gt_0,
            "p_sharpe_gt_05": r.p_sharpe_gt_05,
        }

    # Python fallback
    rng = np.random.RandomState(seed)
    annualize = np.sqrt(8760)
    sharpes = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        sample = np.empty(n)
        pos = 0
        while pos < n:
            start = rng.randint(0, n)
            for j in range(block_size):
                if pos >= n:
                    break
                sample[pos] = returns[(start + j) % n]
                pos += 1

        mu = np.mean(sample)
        std = np.std(sample, ddof=1)
        sharpes[b] = (mu / std * annualize) if std > 1e-15 else 0.0

    sharpes_sorted = np.sort(sharpes)
    return {
        "sharpe_mean": float(np.mean(sharpes)),
        "sharpe_95ci_lo": float(np.percentile(sharpes_sorted, 2.5)),
        "sharpe_95ci_hi": float(np.percentile(sharpes_sorted, 97.5)),
        "p_sharpe_gt_0": float(np.mean(sharpes > 0)),
        "p_sharpe_gt_05": float(np.mean(sharpes > 0.5)),
    }


# ── Nested walk-forward ensemble ─────────────────────────────

def run_ensemble(
    symbol: str,
    feat_df: pd.DataFrame,
    available_1h: List[str],
    available_4h: List[str],
    out_dir: Path,
    *,
    top_k: int = 20,
    n_outer_folds: int = 5,
) -> Optional[dict]:
    """Run nested walk-forward ensemble training for one symbol."""
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    closes = feat_df["close"].values.astype(np.float64)

    # Pre-compute targets for all horizons used
    target_h1 = vol_normalized_target(closes, horizon=1)
    target_h5 = vol_normalized_target(closes, horizon=5)

    def get_target(horizon: int) -> np.ndarray:
        return target_h1 if horizon == 1 else target_h5

    # Build feature matrices
    feat_sets: Dict[str, Tuple[np.ndarray, List[str]]] = {}

    # 1h features
    feat_1h_names = [n for n in available_1h if n in feat_df.columns]
    X_1h = feat_df[feat_1h_names].values.astype(np.float64)
    feat_sets["1h"] = (X_1h, feat_1h_names)

    # 4h features
    feat_4h_names = [n for n in available_4h if n in feat_df.columns]
    if feat_4h_names:
        X_4h = feat_df[feat_4h_names].values.astype(np.float64)
        feat_sets["4h"] = (X_4h, feat_4h_names)
    else:
        feat_sets["4h"] = (np.empty((len(feat_df), 0)), [])

    # all = 1h + 4h
    all_names = feat_1h_names + feat_4h_names
    X_all = np.hstack([X_1h, feat_sets["4h"][0]]) if feat_4h_names else X_1h
    feat_sets["all"] = (X_all, all_names)

    # Build validity mask (must have valid target and core features)
    SPARSE = {"oi_change_pct", "oi_change_ma8", "oi_close_divergence",
              "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
              "btc_ret1_x_beta30"}
    core_1h = [c for c in feat_1h_names if c not in SPARSE]
    core_idx = [feat_1h_names.index(c) for c in core_1h]

    valid_mask = ~np.isnan(target_h5) & ~np.isnan(target_h1)
    for ci in core_idx:
        valid_mask &= ~np.isnan(X_1h[:, ci])
    # 4h features: require non-NaN for the ones we have
    for j in range(len(feat_4h_names)):
        valid_mask &= ~np.isnan(feat_sets["4h"][0][:, j])

    valid_indices = np.where(valid_mask)[0]
    print(f"  Valid samples: {len(valid_indices)} / {len(feat_df)}")
    if len(valid_indices) < 1500:
        print("  ERROR: Not enough valid samples for ensemble walk-forward")
        return None

    # ── Outer walk-forward folds ──
    folds = expanding_window_folds(len(valid_indices), n_folds=n_outer_folds)
    if not folds:
        print("  ERROR: Could not create folds")
        return None

    n_base = len(BASE_MODELS)
    print(f"  Ensemble: {n_base} base models x {len(folds)} outer folds")

    # Collect OOS predictions for meta-learner
    meta_preds = np.full((len(valid_indices), n_base), np.nan)
    meta_y = np.full(len(valid_indices), np.nan)
    fold_metrics: List[Dict] = []

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        print(f"\n  --- Outer Fold {fi} (train={tr_start}:{tr_end}, test={te_start}:{te_end}) ---")

        fold_model_metrics = {}

        for mi, mdef in enumerate(BASE_MODELS):
            tag = mdef["tag"]
            horizon = mdef["horizon"]
            feat_spec = mdef["features"]

            # Get appropriate feature matrix and target
            X_feat, feat_names = feat_sets[feat_spec]
            target = get_target(horizon)

            if len(feat_names) == 0:
                print(f"    {tag}: SKIP (no features)")
                meta_preds[te_start:te_end, mi] = 0.0
                continue

            # Extract clean train/test from valid indices
            train_vi = valid_indices[tr_start:tr_end]
            test_vi = valid_indices[te_start:te_end]

            X_train = X_feat[train_vi]
            y_train = target[train_vi]
            X_test = X_feat[test_vi]

            # Target for this specific horizon/fold
            y_train_valid = ~np.isnan(y_train)
            if y_train_valid.sum() < 500:
                print(f"    {tag}: SKIP (insufficient target, {y_train_valid.sum()} valid)")
                meta_preds[te_start:te_end, mi] = 0.0
                continue

            # Train on valid-target subset
            X_tr = X_train[y_train_valid]
            y_tr = y_train[y_train_valid]

            try:
                pred, sel_feats, metrics = _train_base_model(
                    mdef, X_tr, y_tr, X_test, feat_names, top_k,
                )
            except Exception as e:
                print(f"    {tag}: ERROR {e}")
                meta_preds[te_start:te_end, mi] = 0.0
                continue

            meta_preds[te_start:te_end, mi] = pred

            # Compute fold IC for this model
            y_test_h5 = target_h5[test_vi]
            valid_test = ~np.isnan(y_test_h5)
            if valid_test.sum() > 10:
                ic = float(np.corrcoef(pred[valid_test], y_test_h5[valid_test])[0, 1])
                dir_acc = float(np.mean(np.sign(pred[valid_test]) == np.sign(y_test_h5[valid_test])))
            else:
                ic, dir_acc = 0.0, 0.5

            fold_model_metrics[tag] = {
                "ic": ic, "dir_acc": dir_acc,
                "best_iter": metrics.get("best_iteration", -1),
                "n_features": len(sel_feats),
            }
            print(f"    {tag}: IC={ic:.4f} dir={dir_acc:.3f} feats={len(sel_feats)}")

        # Store meta-learner target (h5 target for test set)
        meta_y[te_start:te_end] = target_h5[valid_indices[te_start:te_end]]

        fold_metrics.append({
            "fold": fi,
            "train_size": tr_end - tr_start,
            "test_size": te_end - te_start,
            "models": fold_model_metrics,
        })

    # ── Fit meta-learner on all OOS predictions ──
    print("\n  Fitting Ridge meta-learner...")

    # Use only rows where all base models have predictions and target is valid
    meta_valid = ~np.isnan(meta_y)
    for mi in range(n_base):
        meta_valid &= ~np.isnan(meta_preds[:, mi])

    meta_idx = np.where(meta_valid)[0]
    if len(meta_idx) < 100:
        print("  ERROR: Not enough meta-learner samples")
        return None

    X_meta = meta_preds[meta_idx]
    y_meta = meta_y[meta_idx]

    ensemble_weights = fit_ridge_meta(X_meta, y_meta)
    ensemble_pred = X_meta @ ensemble_weights

    # Evaluate ensemble OOS
    ensemble_ic = float(np.corrcoef(ensemble_pred, y_meta)[0, 1])
    ensemble_dir = float(np.mean(np.sign(ensemble_pred) == np.sign(y_meta)))

    print(f"\n  Ensemble weights: {dict(zip([m['tag'] for m in BASE_MODELS], ensemble_weights.round(4)))}")
    print(f"  Ensemble OOS: IC={ensemble_ic:.4f} dir={ensemble_dir:.3f}")

    # Per-model OOS IC for comparison
    print("\n  Per-model OOS IC (all folds combined):")
    per_model_ic = {}
    for mi, mdef in enumerate(BASE_MODELS):
        valid_mi = meta_valid & ~np.isnan(meta_preds[:, mi])
        idx_mi = np.where(valid_mi)[0]
        if len(idx_mi) > 10:
            ic_mi = float(np.corrcoef(meta_preds[idx_mi, mi], meta_y[idx_mi])[0, 1])
        else:
            ic_mi = 0.0
        per_model_ic[mdef["tag"]] = ic_mi
        print(f"    {mdef['tag']:<15s} IC={ic_mi:.4f}")

    # Model diversity: pairwise correlation of predictions
    print("\n  Prediction diversity (pairwise correlation):")
    corr_matrix = np.corrcoef(X_meta.T)
    max_corr = 0.0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = corr_matrix[i, j]
            if abs(c) > max_corr:
                max_corr = abs(c)
            print(f"    {BASE_MODELS[i]['tag']:>12s} x {BASE_MODELS[j]['tag']:<12s}: {c:.3f}")
    print(f"  Max pairwise correlation: {max_corr:.3f} {'OK' if max_corr < 0.8 else 'HIGH'}")

    # ── 1-bar trading PnL for ensemble ──
    test_orig = valid_indices[meta_idx]
    ret_1bar = compute_1bar_returns(closes, test_orig)
    eval_result = evaluate_oos(ensemble_pred, y_meta, ret_1bar)
    print_evaluation(eval_result, label="V4 Ensemble OOS")

    # ── Bootstrap validation ──
    print("\n  Running bootstrap Sharpe CI (10000 iterations)...")
    from scripts.oos_eval import apply_threshold, compute_signal_costs
    best_thr = eval_result["best_threshold"]
    signal = apply_threshold(ensemble_pred, best_thr)
    valid_ret = ~np.isnan(ret_1bar)
    costs = compute_signal_costs(signal[valid_ret])
    net_pnl = signal[valid_ret] * ret_1bar[valid_ret] - costs
    active = signal[valid_ret] != 0
    active_returns = net_pnl[active] if active.sum() > 10 else net_pnl

    bootstrap = bootstrap_sharpe_ci(active_returns, n_bootstrap=10000)
    print(f"  Sharpe 95% CI: [{bootstrap['sharpe_95ci_lo']:.2f}, {bootstrap['sharpe_95ci_hi']:.2f}]")
    print(f"  P(Sharpe>0):   {bootstrap['p_sharpe_gt_0']:.3f}")
    print(f"  P(Sharpe>0.5): {bootstrap['p_sharpe_gt_05']:.3f}")

    # ── DSR gate ──
    from research.overfit_detection import deflated_sharpe_ratio
    n_trials = n_outer_folds * n_base * 5
    obs_sharpe = eval_result["threshold_scan"][0]["sharpe_annual"]
    dsr = deflated_sharpe_ratio(
        observed_sharpe=obs_sharpe / np.sqrt(8760) if obs_sharpe != 0 else 0.0,
        n_trials=n_trials,
        n_observations=len(meta_idx),
        significance=0.05,
    )
    print(f"\n  Deflated Sharpe: p={dsr.p_value:.4f} "
          f"{'PASS' if dsr.is_significant else 'FAIL'} (n_trials={n_trials})")

    # ── Train final base models on all data ──
    print("\n  Training final base models on all data...")
    final_models: Dict[str, Dict] = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    for mi, mdef in enumerate(BASE_MODELS):
        tag = mdef["tag"]
        horizon = mdef["horizon"]
        feat_spec = mdef["features"]

        X_feat, feat_names = feat_sets[feat_spec]
        target = get_target(horizon)

        if len(feat_names) == 0:
            continue

        X_all_valid = X_feat[valid_indices]
        y_all_valid = target[valid_indices]
        y_valid_mask = ~np.isnan(y_all_valid)
        X_final = X_all_valid[y_valid_mask]
        y_final = y_all_valid[y_valid_mask]

        selected = spearman_ic_select(X_final, y_final, feat_names, top_k=top_k)
        sel_idx = [feat_names.index(n) for n in selected]

        model = LGBMAlphaModel(name=f"v4_{tag}", feature_names=tuple(selected))

        if mdef["type"] == "classifier":
            y_bin = (y_final > 0).astype(np.float64)
            model.fit_classifier(
                X_final[:, sel_idx], y_bin,
                params=BASE_PARAMS.copy(),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                embargo_bars=EMBARGO_BARS,
            )
        else:
            model.fit(
                X_final[:, sel_idx], y_final,
                params=BASE_PARAMS.copy(),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                embargo_bars=EMBARGO_BARS,
            )

        model_path = out_dir / f"v4_{tag}.pkl"
        model.save(model_path)
        final_models[tag] = {"features": selected, "path": str(model_path)}
        print(f"    {tag}: saved ({len(selected)} features)")

    # Save ensemble weights and config
    ensemble_config = {
        "version": "v4",
        "weights": {m["tag"]: float(w) for m, w in zip(BASE_MODELS, ensemble_weights)},
        "models": final_models,
        "base_params": BASE_PARAMS,
    }
    with open(out_dir / "v4_ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    # ── Model Registry ──
    try:
        from research.model_registry.registry import ModelRegistry
        registry = ModelRegistry(str(out_dir / "model_registry.db"))
        mv = registry.register(
            name=f"lgbm_v4_ensemble_{symbol.lower()}",
            params={
                **BASE_PARAMS,
                "n_outer_folds": n_outer_folds,
                "top_k": top_k,
                "n_base_models": n_base,
                "ensemble_weights": ensemble_config["weights"],
            },
            features=[f for m in final_models.values() for f in m["features"]],
            metrics={
                "ensemble_oos_ic": ensemble_ic,
                "ensemble_oos_dir_acc": ensemble_dir,
                "best_single_ic": max(per_model_ic.values()),
                "max_pairwise_corr": max_corr,
                "dsr_p_value": dsr.p_value,
                "bootstrap_p_sharpe_gt_0": bootstrap["p_sharpe_gt_0"],
                "bootstrap_p_sharpe_gt_05": bootstrap["p_sharpe_gt_05"],
                "bootstrap_sharpe_mean": bootstrap["sharpe_mean"],
                **{f"{k}_ic": v for k, v in per_model_ic.items()},
            },
            tags=["v4", "ensemble", symbol.lower()],
        )
        print(f"\n  Registered: {mv.name} v{mv.version} (id={mv.model_id[:8]})")
    except Exception as e:
        print(f"  Registry warning: {e}")

    # ── Build result dict ──
    result = {
        "symbol": symbol,
        "version": "v4",
        "n_outer_folds": n_outer_folds,
        "top_k": top_k,
        "n_base_models": n_base,
        "ensemble_weights": ensemble_config["weights"],
        "ensemble_oos_ic": ensemble_ic,
        "ensemble_oos_dir_acc": ensemble_dir,
        "per_model_ic": per_model_ic,
        "max_pairwise_corr": max_corr,
        "dsr_p_value": dsr.p_value,
        "dsr_significant": dsr.is_significant,
        "bootstrap": bootstrap,
        "fold_metrics": fold_metrics,
        "eval_result": {
            "prediction_quality": eval_result["prediction_quality"],
            "best_threshold": eval_result["best_threshold"],
            "threshold_scan": eval_result["threshold_scan"],
        },
    }

    with open(out_dir / "v4_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ── Data loading & feature computation ───────────────────────

def _prepare_data(
    symbol: str,
    cross_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[Tuple[pd.DataFrame, List[str], List[str]]]:
    """Load data and compute all features (1h + 4h).

    Returns:
        (feat_df, available_1h, available_4h) or None if data missing.
    """
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    df = pd.read_csv(csv_path)
    funding = _load_schedule(Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")

    cross_df = cross_features.get(symbol) if cross_features else None
    feat_df = _compute_features(symbol, df, funding, oi, ls, cross_df)

    # Compute 4h features
    tf4h_df = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h_df[col].values

    # 1h available features (same as V3)
    available_1h = get_available_features(symbol)
    available_1h = [n for n in available_1h if n in feat_df.columns]

    # 4h available features
    available_4h = [n for n in TF4H_FEATURE_NAMES if n in feat_df.columns]

    print(f"  Bars: {len(feat_df)}, 1h features: {len(available_1h)}, 4h features: {len(available_4h)}")
    return feat_df, available_1h, available_4h


# ── Entrypoint ───────────────────────────────────────────────

def run_one(
    symbol: str,
    out_base: Path,
    *,
    top_k: int = 20,
    n_outer_folds: int = 5,
    cross_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[dict]:
    print(f"\n{'='*70}")
    print(f"  {symbol} — V4 Ensemble Training")
    print(f"{'='*70}")

    data = _prepare_data(symbol, cross_features)
    if data is None:
        return None

    feat_df, available_1h, available_4h = data
    out_dir = out_base / symbol
    return run_ensemble(
        symbol, feat_df, available_1h, available_4h, out_dir,
        top_k=top_k, n_outer_folds=n_outer_folds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V4 Ensemble — Multi-Model Multi-Timeframe")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K features per model")
    parser.add_argument("--n-outer-folds", type=int, default=5, help="Outer walk-forward folds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = []
    if args.all:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.print_help()
        return

    out_base = Path(args.out)

    cross_features = None
    alt_symbols = [s for s in symbols if s != "BTCUSDT"]
    if alt_symbols:
        print("\n  Building cross-asset features...")
        cross_features = _build_cross_features(symbols)
        if cross_features:
            print(f"  Cross-asset features built for: {list(cross_features.keys())}")

    results = {}
    for sym in symbols:
        r = run_one(sym, out_base, top_k=args.top_k,
                    n_outer_folds=args.n_outer_folds,
                    cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*110}")
        print(f"  V4 Ensemble Summary")
        print(f"{'='*110}")
        print(f"{'Symbol':<10} {'Ens IC':>8} {'Ens Dir':>8} {'Best1 IC':>8} "
              f"{'MaxCorr':>8} {'DSR p':>8} {'P(S>0)':>8} {'P(S>.5)':>8}")
        print(f"{'-'*110}")
        for sym, r in results.items():
            print(f"{sym:<10} "
                  f"{r['ensemble_oos_ic']:>8.4f} "
                  f"{r['ensemble_oos_dir_acc']*100:>7.1f}% "
                  f"{max(r['per_model_ic'].values()):>8.4f} "
                  f"{r['max_pairwise_corr']:>8.3f} "
                  f"{r['dsr_p_value']:>8.4f} "
                  f"{r['bootstrap']['p_sharpe_gt_0']:>8.3f} "
                  f"{r['bootstrap']['p_sharpe_gt_05']:>8.3f}")

        # Weights summary
        print(f"\n  Ensemble Weights:")
        for sym, r in results.items():
            w_str = " | ".join(f"{k}={v:.3f}" for k, v in r["ensemble_weights"].items())
            print(f"    {sym}: {w_str}")
        print(f"{'='*110}")


if __name__ == "__main__":
    main()
