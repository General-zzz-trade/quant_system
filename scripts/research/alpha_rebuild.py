#!/usr/bin/env python3
"""Alpha Rebuild — systematic 6-step experiment pipeline to find OOS-robust signals.

Steps:
  1. Feature ablation → identify helpful/harmful feature groups
  2. Feature stability → find minimal stable feature set
  3. Model comparison → LightGBM vs XGBoost
  4. Regime conditioning → single model vs per-regime models
  5. Target engineering → horizon × target_mode grid search
  6. Final assembly → HPO + full validation + model registration

Anti-overfit measures:
  - Steps 1-4 use fixed params (no HPO) to reduce degrees of freedom
  - DSR n_trials accumulates across ALL experiments (honest statistics)
  - Primary metric is H2 IC (late-period, not overall OOS)
  - top_k capped at 15
  - Features must pass both ICIR stability AND cross-fold consistency

Usage:
    python3 -m scripts.alpha_rebuild --symbols BTCUSDT SOLUSDT --out-dir results/alpha_rebuild
    python3 -m scripts.alpha_rebuild --symbols BTCUSDT --steps 1 2 3
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from alpha.models.xgb_alpha import XGBAlphaModel
from alpha.training.regime_split import (
    apply_vol_regime,
    train_regime_models,
)
from features.cross_asset_computer import CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import (
    greedy_ic_select,
    stable_icir_select,
    _rankdata,
)
from features.multi_timeframe import TF4H_FEATURE_NAMES
from research.overfit_detection import deflated_sharpe_ratio

# Reuse V7 infrastructure
from scripts.train_v7_alpha import (
    V7_SEARCH_SPACE,
    EARLY_STOPPING_ROUNDS,
    MIN_TRAIN,
    BLACKLIST,
    INTERACTION_FEATURES,
    _compute_target,
    _pred_to_signal,
    _compute_split_metrics,
    _load_and_compute_features,
    _add_regime_feature,
    _compute_sample_weights,
    _build_cross_features,
    expanding_window_folds,
    get_available_features,
)

try:
    from _quant_hotpath import cpp_bootstrap_sharpe_ci
    _BOOTSTRAP_CPP = True
except ImportError:
    _BOOTSTRAP_CPP = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

TOP_K = 15  # reduced from V7's 25 to prevent overfitting
OOS_BARS = 13140  # 18 months × 30 × 24 ≈ 12,960 + buffer

# Fixed params for Steps 1-4 (no HPO)
FIXED_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "num_leaves": 16,
    "min_child_samples": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "regression",
    "verbosity": -1,
}

# Comparable XGBoost params
XGB_FIXED_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.01,
    "min_child_weight": 80,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "reg:squarederror",
    "verbosity": 0,
}

# Feature group definitions for ablation
FEATURE_GROUPS: Dict[str, Set[str]] = {
    "basis": {"basis", "basis_zscore_24", "basis_momentum", "basis_extreme",
              "basis_x_funding", "basis_x_vol_regime"},
    "fgi": {"fgi_normalized", "fgi_zscore_7", "fgi_extreme", "fgi_x_rsi14"},
    "4h": set(TF4H_FEATURE_NAMES),
    "interactions": {name for name, _, _ in INTERACTION_FEATURES},
    "micro": {"trade_intensity", "taker_buy_ratio", "taker_buy_ratio_ma10",
              "taker_imbalance", "avg_trade_size", "avg_trade_size_ratio",
              "volume_per_trade", "trade_count_regime"},
    "funding": {"funding_rate", "funding_ma8", "funding_zscore_24",
                "funding_momentum", "funding_extreme", "funding_cumulative_8",
                "funding_sign_persist", "funding_annualized", "funding_vs_vol",
                "funding_x_taker_imb"},
    "time": {"hour_sin", "hour_cos", "dow_sin", "dow_cos"},
    "cross": set(CROSS_ASSET_FEATURE_NAMES),
}

# "minimal" keeps only core technical indicators
CORE_TECHNICAL = {
    "ret_6", "ret_12", "ret_24",
    "ma_cross_10_30", "ma_cross_5_20", "close_vs_ma20", "close_vs_ma50",
    "rsi_14", "rsi_6",
    "macd_line", "macd_signal", "macd_hist",
    "bb_width_20", "bb_pctb_20",
    "atr_norm_14",
    "vol_20", "vol_5",
    "vol_ratio_20", "vol_ma_ratio_5_20",
    "body_ratio", "upper_shadow", "lower_shadow",
    "mean_reversion_20", "price_acceleration",
    "vol_regime", "regime_vol",
}

DEFAULT_HORIZON = 6
DEFAULT_TARGET_MODE = "clipped"

# Pre-registered horizons per symbol (justified by Step 1 ablation analysis)
# BTC: no_time ablation improved H2 → long horizon (24) works best
# SOL: baseline H2=0.0095, short horizon (6) preserves momentum signal
PREREGISTERED_HORIZONS: Dict[str, int] = {
    "BTCUSDT": 24,
    "SOLUSDT": 6,
}


# ── ExperimentRunner ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    name: str
    symbol: str
    step: int
    cv_ic: float = 0.0
    oos_ic: float = 0.0
    h1_ic: float = 0.0
    h2_ic: float = 0.0
    oos_sharpe: float = 0.0
    dsr_p: float = 1.0
    n_features: int = 0
    features: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRunner:
    """Tracks experiments across all steps for cumulative DSR."""
    results: List[ExperimentResult] = field(default_factory=list)
    total_trials: int = 0
    final_trials: int = 0

    def add(self, result: ExperimentResult, *, is_final: bool = False) -> None:
        self.results.append(result)
        self.total_trials += 1
        if is_final:
            self.final_trials += 1

    def get_dsr(self, observed_sharpe: float, n_observations: int) -> Any:
        return deflated_sharpe_ratio(
            observed_sharpe=observed_sharpe,
            n_trials=max(self.final_trials, 1),
            n_observations=n_observations,
        )


# ── Helper functions ─────────────────────────────────────────────

def _load_split_data(
    symbol: str, cross_features: Optional[Dict[str, pd.DataFrame]]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load single CSV, split into (train, oos) by cutting OOS_BARS from tail."""
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        logger.warning("CSV not found for %s", symbol)
        return None, None
    df = pd.read_csv(csv_path)
    if len(df) <= OOS_BARS:
        logger.warning("%s: only %d bars, need >%d for split", symbol, len(df), OOS_BARS)
        return None, None
    train_raw = df.iloc[:-OOS_BARS].reset_index(drop=True)
    oos_raw = df.iloc[-OOS_BARS:].reset_index(drop=True)
    cross_df = cross_features.get(symbol) if cross_features else None
    train_feat = _load_and_compute_features(symbol, train_raw, cross_df)
    if train_feat is not None:
        train_feat = _add_regime_feature(train_feat)
    oos_feat = _load_and_compute_features(symbol, oos_raw, cross_df)
    if oos_feat is not None:
        oos_feat = _add_regime_feature(oos_feat)
    return train_feat, oos_feat


def _prepare_xy(
    feat_df: pd.DataFrame,
    feature_names: List[str],
    horizon: int,
    target_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Prepare X, y, closes arrays with NaN filtering. Returns (X, y, closes, feature_names)."""
    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    target = _compute_target(closes, horizon, target_mode)

    X_all = feat_df[feature_names].values.astype(np.float64)

    # Drop all-NaN columns
    nan_mask = np.all(np.isnan(X_all), axis=0)
    if nan_mask.any():
        keep = ~nan_mask
        X_all = X_all[:, keep]
        feature_names = [f for f, k in zip(feature_names, keep) if k]

    valid = ~np.isnan(target)
    # Only require non-NaN in core technical features.
    # Everything else (external data, cross-asset, 4h, interactions) is sparse.
    sparse = (
        BLACKLIST
        | FEATURE_GROUPS["funding"]
        | FEATURE_GROUPS["basis"]
        | FEATURE_GROUPS["fgi"]
        | FEATURE_GROUPS["cross"]
        | FEATURE_GROUPS["4h"]
        | FEATURE_GROUPS["interactions"]
        | {"oi_change_pct", "oi_change_ma8", "oi_close_divergence",
           "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
           "oi_acceleration", "leverage_proxy", "oi_vol_divergence",
           "oi_liquidation_flag", "cvd_x_oi_chg",
           "cvd_10", "cvd_20", "cvd_price_divergence", "aggressive_flow_zscore",
           "vol_of_vol", "range_vs_rv", "parkinson_vol", "rv_acceleration",
           "funding_annualized", "funding_vs_vol",
           "taker_bq_ratio", "vwap_dev_20", "volume_momentum_10",
           "mom_vol_divergence", "basis_carry_adj", "vol_regime_adaptive"}
    )
    for j, name in enumerate(feature_names):
        if name not in sparse:
            valid &= ~np.isnan(X_all[:, j])

    idx = np.where(valid)[0]
    return X_all[idx], target[idx], closes[idx], feature_names


def _run_cv_experiment(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    closes: np.ndarray,
    *,
    horizon: int,
    target_mode: str,
    top_k: int = TOP_K,
    n_folds: int = 5,
    params: Optional[Dict[str, Any]] = None,
    use_xgb: bool = False,
) -> Dict[str, Any]:
    """Run expanding window CV experiment with fixed params (no HPO).

    Returns dict with cv_ic, fold_ics, fold_sharpes, feature_selection_counts.
    """
    if params is None:
        params = FIXED_PARAMS.copy()

    embargo_bars = horizon + 2
    folds = expanding_window_folds(len(X), n_folds=n_folds, min_train=MIN_TRAIN)
    if not folds:
        return {"cv_ic": 0.0, "fold_ics": [], "fold_sharpes": [],
                "feature_selection_counts": {}, "selected_features": []}

    fold_ics = []
    fold_sharpes = []
    feature_counts: Dict[str, int] = {}

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X[tr_start:tr_end]
        y_train = y[tr_start:tr_end]
        X_test = X[te_start:te_end]
        y_test = y[te_start:te_end]

        selected = greedy_ic_select(X_train, y_train, feature_names, top_k=top_k)
        sel_idx = [feature_names.index(n) for n in selected]

        for name in selected:
            feature_counts[name] = feature_counts.get(name, 0) + 1

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]
        weights = _compute_sample_weights(len(X_tr_sel))

        if use_xgb:
            model = XGBAlphaModel(name=f"xgb_fold{fi}", feature_names=tuple(selected))
            model.fit(
                X_tr_sel, y_train,
                params=params,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                embargo_bars=embargo_bars,
                sample_weight=weights,
            )
            y_pred = model._model.predict(X_te_sel)
        else:
            model = LGBMAlphaModel(name=f"lgbm_fold{fi}", feature_names=tuple(selected))
            if target_mode == "binary":
                model.fit_classifier(
                    X_tr_sel, y_train,
                    params=params,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    embargo_bars=embargo_bars,
                    sample_weight=weights,
                )
                y_pred = model._model.predict_proba(X_te_sel)[:, 1]
            else:
                model.fit(
                    X_tr_sel, y_train,
                    params=params,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    embargo_bars=embargo_bars,
                    sample_weight=weights,
                )
                y_pred = model._model.predict(X_te_sel)

        if len(y_pred) < 30 or np.std(y_pred) < 1e-12:
            fold_ics.append(0.0)
            fold_sharpes.append(0.0)
            continue

        rx = _rankdata(y_pred)
        ry = _rankdata(y_test)
        ic = float(np.corrcoef(rx, ry)[0, 1])
        if np.isnan(ic):
            ic = 0.0
        fold_ics.append(ic)

        # Compute fold Sharpe
        test_closes = closes[te_start:te_end] if te_end <= len(closes) else closes[te_start:]
        ret_1bar = np.diff(test_closes) / test_closes[:-1]
        pred_trunc = y_pred[:len(ret_1bar)]
        signal = _pred_to_signal(pred_trunc, target_mode=target_mode)
        turnover = np.abs(np.diff(signal, prepend=0))
        net_pnl = signal * ret_1bar - turnover * 4e-4
        active = signal != 0
        sharpe = 0.0
        if active.sum() > 1:
            std_a = float(np.std(net_pnl[active], ddof=1))
            if std_a > 0:
                sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)
        fold_sharpes.append(sharpe)

    cv_ic = float(np.mean(fold_ics)) if fold_ics else 0.0
    return {
        "cv_ic": cv_ic,
        "fold_ics": fold_ics,
        "fold_sharpes": fold_sharpes,
        "feature_selection_counts": feature_counts,
        "selected_features": sorted(feature_counts.keys(),
                                    key=lambda k: -feature_counts[k])[:top_k],
    }


def _evaluate_oos(
    model_obj: Any,
    oos_feat_df: pd.DataFrame,
    feature_names: List[str],
    horizon: int,
    target_mode: str,
    *,
    is_classifier: bool = False,
) -> Optional[Dict[str, Any]]:
    """Evaluate model on OOS data. Returns overall, H1, H2 metrics."""
    for fname in feature_names:
        if fname not in oos_feat_df.columns:
            oos_feat_df[fname] = np.nan

    closes = oos_feat_df["close"].values.astype(np.float64)
    target = _compute_target(closes, horizon, target_mode)
    X_oos = oos_feat_df[feature_names].values.astype(np.float64)
    valid_mask = ~np.isnan(target)

    valid_idx = np.where(valid_mask)[0]
    X_v = X_oos[valid_idx]
    y_v = target[valid_idx]
    closes_v = closes[valid_idx]

    if len(X_v) < 100:
        return None

    if is_classifier:
        y_pred = model_obj.predict_proba(X_v)[:, 1]
    else:
        y_pred = model_obj.predict(X_v)

    overall = _compute_split_metrics(y_pred, y_v, closes_v, target_mode)

    mid = len(y_pred) // 2
    h1 = _compute_split_metrics(y_pred[:mid], y_v[:mid], closes_v[:mid], target_mode)
    h2 = _compute_split_metrics(y_pred[mid:], y_v[mid:], closes_v[mid:], target_mode)

    ic_decay = h2["ic"] / h1["ic"] if abs(h1["ic"]) > 1e-6 else float("nan")

    # Monthly IC for consistency check
    for col_name in ("open_time", "timestamp"):
        Path(f"data_files/{oos_feat_df.attrs.get('symbol', 'UNKNOWN')}_1h_oos.csv")
        break
    # Compute monthly breakdown from index position (720 bars ~ 1 month)
    monthly_ics = []
    chunk_size = 720
    for i in range(0, len(y_pred), chunk_size):
        end = min(i + chunk_size, len(y_pred))
        if end - i < 30:
            continue
        rx = _rankdata(y_pred[i:end])
        ry = _rankdata(y_v[i:end])
        ic_val = float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 1 else 0.0
        if not np.isnan(ic_val):
            monthly_ics.append(ic_val)

    pos_months = sum(1 for ic in monthly_ics if ic > 0)
    total_months = len(monthly_ics)
    month_pos_ratio = pos_months / max(total_months, 1)

    return {
        "overall": overall,
        "h1": h1,
        "h2": h2,
        "ic_decay": ic_decay,
        "monthly_ics": monthly_ics,
        "month_pos_ratio": month_pos_ratio,
    }


# ── Step 1: Feature Ablation ────────────────────────────────────

def step1_ablation(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
) -> Dict[str, Dict[str, ExperimentResult]]:
    """Run feature group ablation study.

    10 configs × N symbols. Returns {symbol: {config_name: result}}.
    """
    print("\n" + "=" * 70)
    print("  STEP 1: Feature Ablation Study")
    print("=" * 70)

    configs = {
        "baseline": set(),        # remove nothing
        "no_basis": FEATURE_GROUPS["basis"],
        "no_fgi": FEATURE_GROUPS["fgi"],
        "no_4h": FEATURE_GROUPS["4h"],
        "no_interactions": FEATURE_GROUPS["interactions"],
        "no_micro": FEATURE_GROUPS["micro"],
        "no_funding": FEATURE_GROUPS["funding"],
        "no_time": FEATURE_GROUPS["time"],
        "no_cross": FEATURE_GROUPS["cross"],
        "minimal": None,          # special: only keep CORE_TECHNICAL
    }

    all_results: Dict[str, Dict[str, ExperimentResult]] = {}

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")
        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        base_available = get_available_features(symbol)
        base_available = [n for n in base_available if n in feat_df.columns and n not in BLACKLIST]

        symbol_results: Dict[str, ExperimentResult] = {}

        for config_name, remove_set in configs.items():
            if config_name == "no_cross" and symbol == "BTCUSDT":
                continue  # BTC has no cross-asset features

            if config_name == "minimal":
                available = [n for n in base_available if n in CORE_TECHNICAL]
            elif remove_set:
                available = [n for n in base_available if n not in remove_set]
            else:
                available = list(base_available)

            if len(available) < 5:
                print(f"    {config_name}: too few features ({len(available)}), skip")
                continue

            X, y, closes, feat_names = _prepare_xy(
                feat_df.copy(), available, DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

            if len(X) < MIN_TRAIN:
                print(f"    {config_name}: insufficient data ({len(X)}), skip")
                continue

            cv = _run_cv_experiment(
                X, y, feat_names, closes,
                horizon=DEFAULT_HORIZON,
                target_mode=DEFAULT_TARGET_MODE,
            )

            # OOS evaluation
            oos_metrics = None
            if oos_feat_df is not None:
                # Train on all IS data with selected features
                sel_features = cv["selected_features"]
                if sel_features:
                    sel_idx = [feat_names.index(n) for n in sel_features if n in feat_names]
                    if sel_idx:
                        weights = _compute_sample_weights(len(X))
                        embargo = DEFAULT_HORIZON + 2
                        m = LGBMAlphaModel(name=config_name, feature_names=tuple(sel_features))
                        m.fit(X[:, sel_idx], y, params=FIXED_PARAMS.copy(),
                              early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                              embargo_bars=embargo, sample_weight=weights)
                        if m._model is not None:
                            oos_metrics = _evaluate_oos(
                                m._model, oos_feat_df.copy(), sel_features,
                                DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

            result = ExperimentResult(
                name=config_name,
                symbol=symbol,
                step=1,
                cv_ic=cv["cv_ic"],
                n_features=len(available),
                features=cv["selected_features"],
            )
            if oos_metrics:
                result.oos_ic = oos_metrics["overall"]["ic"]
                result.h1_ic = oos_metrics["h1"]["ic"]
                result.h2_ic = oos_metrics["h2"]["ic"]
                result.oos_sharpe = oos_metrics["overall"]["sharpe"]
                result.extra["ic_decay"] = oos_metrics["ic_decay"]
                result.extra["month_pos_ratio"] = oos_metrics["month_pos_ratio"]

            runner.add(result)
            symbol_results[config_name] = result

            oos_str = f"OOS IC={result.oos_ic:.4f} H2={result.h2_ic:.4f}" if oos_metrics else "no OOS"
            print(f"    {config_name:<20s} feats={len(available):>3} "
                  f"CV IC={cv['cv_ic']:.4f}  {oos_str}")

        all_results[symbol] = symbol_results

    # Identify harmful/beneficial groups
    print("\n  === Ablation Summary ===")
    for symbol, sym_results in all_results.items():
        if "baseline" not in sym_results:
            continue
        baseline_h2 = sym_results["baseline"].h2_ic
        print(f"\n  {symbol} (baseline H2 IC = {baseline_h2:.4f}):")
        for name, res in sym_results.items():
            if name == "baseline":
                continue
            delta = res.h2_ic - baseline_h2
            verdict = "HARMFUL (remove)" if delta > 0.005 else (
                "BENEFICIAL (keep)" if delta < -0.005 else "NEUTRAL")
            print(f"    {name:<20s} H2 IC={res.h2_ic:.4f}  delta={delta:+.4f}  [{verdict}]")

    # Save results
    step1_dir = out_dir / "step1_ablation"
    step1_dir.mkdir(parents=True, exist_ok=True)
    _save_results(all_results, step1_dir / "ablation_results.json")

    return all_results


def _identify_harmful_groups(
    ablation_results: Dict[str, Dict[str, ExperimentResult]],
) -> Set[str]:
    """Identify feature groups to remove: removing them improves H2 IC."""
    harmful_votes: Dict[str, int] = {}
    for symbol, sym_results in ablation_results.items():
        if "baseline" not in sym_results:
            continue
        baseline_h2 = sym_results["baseline"].h2_ic
        for name, res in sym_results.items():
            if name in ("baseline", "minimal"):
                continue
            group_name = name.replace("no_", "")
            if res.h2_ic > baseline_h2 + 0.005:
                harmful_votes[group_name] = harmful_votes.get(group_name, 0) + 1

    # Remove groups that are harmful across majority of symbols
    n_symbols = len(ablation_results)
    threshold = max(n_symbols // 2, 1)
    harmful = set()
    for group, votes in harmful_votes.items():
        if votes >= threshold and group in FEATURE_GROUPS:
            harmful.update(FEATURE_GROUPS[group])
    return harmful


# ── Step 2: Feature Stability Analysis ───────────────────────────

def step2_stability(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
    harmful_features: Set[str],
) -> Dict[str, List[str]]:
    """Find minimal stable feature set per symbol.

    Two parallel filters:
      1. ICIR stability: stable_icir_select(min_icir=0.3, min_stable_folds=4/5)
      2. Cross-fold intersection: in >=4/5 CV folds via greedy_ic_select

    Returns {symbol: stable_features}.
    """
    print("\n" + "=" * 70)
    print("  STEP 2: Feature Stability Analysis")
    print("=" * 70)

    stable_features: Dict[str, List[str]] = {}

    for symbol in symbols:
        print(f"\n  --- {symbol} ---")
        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        base_available = get_available_features(symbol)
        available = [n for n in base_available
                     if n in feat_df.columns
                     and n not in BLACKLIST
                     and n not in harmful_features]

        X, y, closes, feat_names = _prepare_xy(
            feat_df.copy(), available, DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

        if len(X) < MIN_TRAIN:
            print(f"    Insufficient data ({len(X)}), skip")
            continue

        # Path 1: ICIR stability
        icir_selected = stable_icir_select(
            X, y, feat_names,
            top_k=TOP_K,
            min_icir=0.3,
            min_stable_folds=4,
            sign_consistency_threshold=0.8,
        )
        print(f"    ICIR stable: {len(icir_selected)} features")

        # Path 2: Cross-fold consistency via greedy_ic_select
        n_folds = 5
        folds = expanding_window_folds(len(X), n_folds=n_folds, min_train=MIN_TRAIN)
        fold_selections: Dict[str, int] = {}
        for tr_start, tr_end, _, _ in folds:
            X_train = X[tr_start:tr_end]
            y_train = y[tr_start:tr_end]
            selected = greedy_ic_select(X_train, y_train, feat_names, top_k=TOP_K)
            for name in selected:
                fold_selections[name] = fold_selections.get(name, 0) + 1

        cross_fold_stable = {name for name, count in fold_selections.items()
                            if count >= 4}
        print(f"    Cross-fold stable (>=4/5): {len(cross_fold_stable)} features")

        # Intersection
        icir_set = set(icir_selected)
        stable = sorted(icir_set & cross_fold_stable,
                       key=lambda n: -fold_selections.get(n, 0))

        # If intersection too small, fall back to ICIR set
        if len(stable) < 5:
            print(f"    Intersection too small ({len(stable)}), using ICIR set")
            stable = icir_selected[:TOP_K]

        stable = stable[:TOP_K]
        stable_features[symbol] = stable

        print(f"    Final stable set: {len(stable)} features")
        for i, name in enumerate(stable):
            icir_flag = "Y" if name in icir_set else "N"
            fold_flag = f"{fold_selections.get(name, 0)}/{n_folds}"
            print(f"      {i+1:>2}. {name:<30s} ICIR={icir_flag} folds={fold_flag}")

        # Run OOS validation on stable set
        if oos_feat_df is not None and stable:
            sel_idx = [feat_names.index(n) for n in stable if n in feat_names]
            weights = _compute_sample_weights(len(X))
            embargo = DEFAULT_HORIZON + 2
            model = LGBMAlphaModel(name="stability_check", feature_names=tuple(stable))
            model.fit(X[:, sel_idx], y, params=FIXED_PARAMS.copy(),
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      embargo_bars=embargo, sample_weight=weights)
            if model._model is not None:
                oos = _evaluate_oos(
                    model._model, oos_feat_df.copy(), stable,
                    DEFAULT_HORIZON, DEFAULT_TARGET_MODE)
                if oos:
                    result = ExperimentResult(
                        name="stable_set", symbol=symbol, step=2,
                        oos_ic=oos["overall"]["ic"],
                        h1_ic=oos["h1"]["ic"],
                        h2_ic=oos["h2"]["ic"],
                        oos_sharpe=oos["overall"]["sharpe"],
                        n_features=len(stable),
                        features=stable,
                    )
                    runner.add(result)
                    print(f"    OOS: IC={oos['overall']['ic']:.4f} "
                          f"H1={oos['h1']['ic']:.4f} H2={oos['h2']['ic']:.4f} "
                          f"Sharpe={oos['overall']['sharpe']:.2f}")

    step2_dir = out_dir / "step2_stability"
    step2_dir.mkdir(parents=True, exist_ok=True)
    with open(step2_dir / "stable_features.json", "w") as f:
        json.dump(stable_features, f, indent=2)

    return stable_features


# ── Step 3: Model Comparison ─────────────────────────────────────

def step3_model_comparison(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
    stable_features: Dict[str, List[str]],
) -> Dict[str, str]:
    """Compare LightGBM vs XGBoost on stable features.

    Returns {symbol: winner} where winner is "lgbm" or "xgb".
    """
    print("\n" + "=" * 70)
    print("  STEP 3: Model Comparison (LightGBM vs XGBoost)")
    print("=" * 70)

    winners: Dict[str, str] = {}

    for symbol in symbols:
        features = stable_features.get(symbol)
        if not features:
            print(f"\n  {symbol}: no stable features, skip")
            continue

        print(f"\n  --- {symbol} ({len(features)} features) ---")
        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        X, y, closes, feat_names = _prepare_xy(
            feat_df.copy(), list(features), DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

        if len(X) < MIN_TRAIN:
            continue

        model_results = {}
        for model_type in ("lgbm", "xgb"):
            use_xgb = model_type == "xgb"
            params = XGB_FIXED_PARAMS.copy() if use_xgb else FIXED_PARAMS.copy()

            cv = _run_cv_experiment(
                X, y, feat_names, closes,
                horizon=DEFAULT_HORIZON,
                target_mode=DEFAULT_TARGET_MODE,
                params=params,
                use_xgb=use_xgb,
            )

            oos_metrics = None
            if oos_feat_df is not None:
                sel_features = cv["selected_features"] or feat_names
                sel_idx = [feat_names.index(n) for n in sel_features if n in feat_names]
                weights = _compute_sample_weights(len(X))
                embargo = DEFAULT_HORIZON + 2

                if use_xgb:
                    m = XGBAlphaModel(name=f"xgb_{symbol}", feature_names=tuple(sel_features))
                    m.fit(X[:, sel_idx], y, params=params,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          embargo_bars=embargo, sample_weight=weights)
                    raw_model = m._model
                else:
                    m = LGBMAlphaModel(name=f"lgbm_{symbol}", feature_names=tuple(sel_features))
                    m.fit(X[:, sel_idx], y, params=params,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          embargo_bars=embargo, sample_weight=weights)
                    raw_model = m._model

                if raw_model is not None:
                    oos_metrics = _evaluate_oos(
                        raw_model, oos_feat_df.copy(), sel_features,
                        DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

            result = ExperimentResult(
                name=model_type, symbol=symbol, step=3,
                cv_ic=cv["cv_ic"],
                n_features=len(features),
                features=list(features),
            )
            if oos_metrics:
                result.oos_ic = oos_metrics["overall"]["ic"]
                result.h1_ic = oos_metrics["h1"]["ic"]
                result.h2_ic = oos_metrics["h2"]["ic"]
                result.oos_sharpe = oos_metrics["overall"]["sharpe"]
                result.extra["ic_variance"] = (
                    float(np.var(oos_metrics["monthly_ics"]))
                    if oos_metrics["monthly_ics"] else 0.0
                )

            runner.add(result)
            model_results[model_type] = result

            oos_str = f"OOS IC={result.oos_ic:.4f} H2={result.h2_ic:.4f}" if oos_metrics else "no OOS"
            print(f"    {model_type:<6s} CV IC={cv['cv_ic']:.4f}  {oos_str}")

        # Pick winner: higher H2 IC + lower IC variance. Tie goes to LGBM.
        lgbm_r = model_results.get("lgbm")
        xgb_r = model_results.get("xgb")
        if lgbm_r and xgb_r:
            xgb_better = (xgb_r.h2_ic > lgbm_r.h2_ic + 0.005 and
                          xgb_r.extra.get("ic_variance", 1) <= lgbm_r.extra.get("ic_variance", 1))
            winner = "xgb" if xgb_better else "lgbm"
        else:
            winner = "lgbm"
        winners[symbol] = winner
        print(f"    Winner: {winner.upper()}")

    step3_dir = out_dir / "step3_model"
    step3_dir.mkdir(parents=True, exist_ok=True)
    with open(step3_dir / "model_winners.json", "w") as f:
        json.dump(winners, f, indent=2)

    return winners


# ── Step 4: Regime Conditioning ──────────────────────────────────

def step4_regime(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
    stable_features: Dict[str, List[str]],
    model_winners: Dict[str, str],
) -> Dict[str, str]:
    """Test single model vs regime-conditional models.

    Returns {symbol: "single" or "regime"}.
    """
    print("\n" + "=" * 70)
    print("  STEP 4: Regime Conditioning")
    print("=" * 70)

    regime_winners: Dict[str, str] = {}

    for symbol in symbols:
        features = stable_features.get(symbol)
        if not features:
            continue

        print(f"\n  --- {symbol} ---")
        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        X, y, closes, feat_names = _prepare_xy(
            feat_df.copy(), list(features), DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

        if len(X) < MIN_TRAIN:
            continue

        # Compute IS vol regime thresholds
        vol_col_idx = None
        for j, name in enumerate(feat_names):
            if name == "vol_20":
                vol_col_idx = j
                break

        if vol_col_idx is None:
            print("    No vol_20 in features, skip regime test")
            regime_winners[symbol] = "single"
            continue

        vol_is = X[:, vol_col_idx]
        valid_vol = vol_is[~np.isnan(vol_is)]
        if len(valid_vol) < 30:
            regime_winners[symbol] = "single"
            continue

        p33, p67 = float(np.percentile(valid_vol, 33)), float(np.percentile(valid_vol, 67))
        is_regimes = apply_vol_regime(vol_is, p33, p67)

        embargo = DEFAULT_HORIZON + 2
        weights = _compute_sample_weights(len(X))

        # a) Single model (already tested, reuse stable set)
        single_model = LGBMAlphaModel(name="single", feature_names=tuple(feat_names))
        single_model.fit(X, y, params=FIXED_PARAMS.copy(),
                        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                        embargo_bars=embargo, sample_weight=weights)

        # b) Regime-conditional models
        regime_bundle = train_regime_models(
            X, y, is_regimes, feat_names,
            params=FIXED_PARAMS.copy(),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=embargo,
            sample_weight=weights,
        )

        single_oos = None
        regime_oos = None

        if oos_feat_df is not None:
            # Single model OOS
            if single_model._model is not None:
                single_oos = _evaluate_oos(
                    single_model._model, oos_feat_df.copy(), feat_names,
                    DEFAULT_HORIZON, DEFAULT_TARGET_MODE)

            # Regime model OOS — apply IS thresholds to OOS vol
            for fname in feat_names:
                if fname not in oos_feat_df.columns:
                    oos_feat_df[fname] = np.nan

            oos_closes = oos_feat_df["close"].values.astype(np.float64)
            oos_target = _compute_target(oos_closes, DEFAULT_HORIZON, DEFAULT_TARGET_MODE)
            X_oos_all = oos_feat_df[feat_names].values.astype(np.float64)
            oos_valid = ~np.isnan(oos_target)
            oos_idx = np.where(oos_valid)[0]
            X_oos_v = X_oos_all[oos_idx]
            y_oos_v = oos_target[oos_idx]
            closes_oos_v = oos_closes[oos_idx]

            if len(X_oos_v) >= 100:
                # Use IS p33/p67 for OOS regime assignment (no lookahead)
                oos_vol = X_oos_v[:, vol_col_idx] if vol_col_idx < X_oos_v.shape[1] else np.ones(len(X_oos_v))
                oos_regimes = apply_vol_regime(oos_vol, p33, p67)

                # Predict per-regime
                regime_preds = np.zeros(len(X_oos_v))
                for i in range(len(X_oos_v)):
                    features_dict = {feat_names[j]: float(X_oos_v[i, j])
                                    for j in range(len(feat_names))}
                    pred = regime_bundle.predict_regime(int(oos_regimes[i]), features_dict)
                    regime_preds[i] = pred if pred is not None else 0.0

                overall = _compute_split_metrics(regime_preds, y_oos_v, closes_oos_v, DEFAULT_TARGET_MODE)
                mid = len(regime_preds) // 2
                h1 = _compute_split_metrics(regime_preds[:mid], y_oos_v[:mid], closes_oos_v[:mid], DEFAULT_TARGET_MODE)
                h2 = _compute_split_metrics(regime_preds[mid:], y_oos_v[mid:], closes_oos_v[mid:], DEFAULT_TARGET_MODE)
                regime_oos = {"overall": overall, "h1": h1, "h2": h2}

        # Record results
        for label, oos in [("single", single_oos), ("regime", regime_oos)]:
            result = ExperimentResult(name=label, symbol=symbol, step=4, n_features=len(feat_names))
            if oos:
                result.oos_ic = oos["overall"]["ic"]
                result.h1_ic = oos["h1"]["ic"]
                result.h2_ic = oos["h2"]["ic"]
                result.oos_sharpe = oos["overall"]["sharpe"]
            runner.add(result)
            h2_str = f"H2 IC={result.h2_ic:.4f}" if oos else "no OOS"
            print(f"    {label:<10s} {h2_str}")

        # Pick winner
        if single_oos and regime_oos:
            regime_better = (regime_oos["h2"]["ic"] > single_oos["h2"]["ic"] + 0.005)
            winner = "regime" if regime_better else "single"
        else:
            winner = "single"
        regime_winners[symbol] = winner
        print(f"    Winner: {winner.upper()}")

    step4_dir = out_dir / "step4_regime"
    step4_dir.mkdir(parents=True, exist_ok=True)
    with open(step4_dir / "regime_winners.json", "w") as f:
        json.dump(regime_winners, f, indent=2)

    return regime_winners


# ── Step 5: Target Engineering ───────────────────────────────────

def step5_target(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
    stable_features: Dict[str, List[str]],
) -> Dict[str, Tuple[int, str]]:
    """Grid search: pre-registered horizon × target_mode.

    1 horizon × 4 modes = 4 combos per symbol (reduced from 16).
    Pre-registered horizons reduce DSR penalty: E[max_sharpe] 2.10 → 1.46.
    Returns {symbol: (best_horizon, best_mode)}.
    """
    print("\n" + "=" * 70)
    print("  STEP 5: Target Engineering (pre-registered horizon × mode grid)")
    print("=" * 70)

    modes = ["raw", "clipped", "vol_norm", "binary"]

    best_targets: Dict[str, Tuple[int, str]] = {}

    for symbol in symbols:
        features = stable_features.get(symbol)
        if not features:
            continue

        symbol_horizon = PREREGISTERED_HORIZONS.get(symbol, DEFAULT_HORIZON)
        print(f"\n  --- {symbol} (horizon={symbol_horizon}) ---")
        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        best_h2 = -999.0
        best_key = (symbol_horizon, DEFAULT_TARGET_MODE)
        grid_results = []

        for mode in modes:
            h = symbol_horizon
            X, y, closes, feat_names = _prepare_xy(
                feat_df.copy(), list(features), h, mode)

            if len(X) < MIN_TRAIN:
                continue

            params = FIXED_PARAMS.copy()
            is_binary = mode == "binary"
            if is_binary:
                params["objective"] = "binary"
                params["metric"] = "binary_logloss"

            embargo = h + 2

            cv = _run_cv_experiment(
                X, y, feat_names, closes,
                horizon=h,
                target_mode=mode,
                params=params,
            )

            # OOS
            oos_metrics = None
            if oos_feat_df is not None:
                sel_features = feat_names  # use all stable features
                weights = _compute_sample_weights(len(X))
                m = LGBMAlphaModel(name=f"target_{h}_{mode}", feature_names=tuple(sel_features))

                if is_binary:
                    m.fit_classifier(X, y, params=params,
                                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                                    embargo_bars=embargo, sample_weight=weights)
                    raw_model = m._model
                    is_clf = True
                else:
                    m.fit(X, y, params=params,
                          early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                          embargo_bars=embargo, sample_weight=weights)
                    raw_model = m._model
                    is_clf = False

                if raw_model is not None:
                    oos_metrics = _evaluate_oos(
                        raw_model, oos_feat_df.copy(), sel_features, h, mode,
                        is_classifier=is_clf)

            result = ExperimentResult(
                name=f"h{h}_{mode}", symbol=symbol, step=5,
                cv_ic=cv["cv_ic"],
                n_features=len(feat_names),
            )
            if oos_metrics:
                result.oos_ic = oos_metrics["overall"]["ic"]
                result.h1_ic = oos_metrics["h1"]["ic"]
                result.h2_ic = oos_metrics["h2"]["ic"]
                result.oos_sharpe = oos_metrics["overall"]["sharpe"]
                result.extra["month_pos_ratio"] = oos_metrics["month_pos_ratio"]

            runner.add(result, is_final=True)
            grid_results.append(result)

            if result.h2_ic > best_h2:
                best_h2 = result.h2_ic
                best_key = (h, mode)

            h2_str = f"H2={result.h2_ic:.4f}" if oos_metrics else "no OOS"
            print(f"    h={h:>2} mode={mode:<8s} CV IC={cv['cv_ic']:.4f}  {h2_str}")

        best_targets[symbol] = best_key
        print(f"    Best: horizon={best_key[0]}, mode={best_key[1]} (H2 IC={best_h2:.4f})")

    step5_dir = out_dir / "step5_target"
    step5_dir.mkdir(parents=True, exist_ok=True)
    with open(step5_dir / "best_targets.json", "w") as f:
        json.dump({k: list(v) for k, v in best_targets.items()}, f, indent=2)

    return best_targets


# ── Step 6: Final Assembly ───────────────────────────────────────

def step6_final(
    symbols: List[str],
    cross_features: Optional[Dict[str, pd.DataFrame]],
    runner: ExperimentRunner,
    out_dir: Path,
    stable_features: Dict[str, List[str]],
    model_winners: Dict[str, str],
    regime_winners: Dict[str, str],
    best_targets: Dict[str, Tuple[int, str]],
    save_all: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Final assembly: HPO + full validation + model registration.

    Production gates:
      - DSR p < 0.05 (n_trials = total across all steps)
      - H2 IC > 0
      - Monthly IC positive ratio >= 60%
      - Bootstrap P(Sharpe > 0) > 0.75
    """
    from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig

    print("\n" + "=" * 70)
    print("  STEP 6: Final Assembly")
    print("=" * 70)

    final_results: Dict[str, Dict[str, Any]] = {}

    for symbol in symbols:
        features = stable_features.get(symbol)
        if not features:
            continue

        horizon, target_mode = best_targets.get(symbol, (DEFAULT_HORIZON, DEFAULT_TARGET_MODE))
        use_regime = regime_winners.get(symbol, "single") == "regime"

        print(f"\n  --- {symbol} ---")
        print(f"    Config: features={len(features)}, horizon={horizon}, "
              f"mode={target_mode}, regime={use_regime}")

        feat_df, oos_feat_df = _load_split_data(symbol, cross_features)
        if feat_df is None:
            continue

        X, y, closes, feat_names = _prepare_xy(
            feat_df.copy(), list(features), horizon, target_mode)

        if len(X) < MIN_TRAIN:
            print(f"    Insufficient data ({len(X)}), skip")
            continue

        embargo = horizon + 2
        is_binary = target_mode == "binary"

        # ── HPO via 5-fold CV (10 trials/fold) ──
        print("    Running HPO...")
        folds = expanding_window_folds(len(X), n_folds=5, min_train=MIN_TRAIN)
        best_params = FIXED_PARAMS.copy()

        if folds:
            import lightgbm as lgb

            def hpo_objective(params: Dict[str, Any]) -> float:
                lgbm_params = {**params, "n_estimators": 500, "verbosity": -1}
                if is_binary:
                    lgbm_params["objective"] = "binary"
                    lgbm_params["metric"] = "binary_logloss"
                else:
                    lgbm_params["objective"] = "regression"

                ics = []
                for tr_start, tr_end, te_start, te_end in folds:
                    X_train = X[tr_start:tr_end]
                    y_train = y[tr_start:tr_end]
                    X_test = X[te_start:te_end]
                    y_test = y[te_start:te_end]
                    weights = _compute_sample_weights(len(X_train))

                    if is_binary:
                        model = lgb.LGBMClassifier(**lgbm_params)
                    else:
                        model = lgb.LGBMRegressor(**lgbm_params)

                    model.fit(
                        X_train, y_train,
                        sample_weight=weights,
                        eval_set=[(X_test, y_test)],
                        callbacks=[
                            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                            lgb.log_evaluation(period=0),
                        ],
                    )

                    if is_binary:
                        y_pred = model.predict_proba(X_test)[:, 1]
                    else:
                        y_pred = model.predict(X_test)

                    if len(y_pred) < 30 or np.std(y_pred) < 1e-12:
                        continue
                    rx = _rankdata(y_pred)
                    ry = _rankdata(y_test)
                    ic = float(np.corrcoef(rx, ry)[0, 1])
                    if not np.isnan(ic):
                        ics.append(ic)

                return float(np.mean(ics)) if ics else 0.0

            optimizer = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=hpo_objective,
                config=HyperOptConfig(
                    n_trials=10,
                    direction="maximize",
                    seed=42,
                    pruner_patience=10,
                    pruner_min_trials=5,
                ),
            )
            hpo_result = optimizer.optimize()
            best_params = {**FIXED_PARAMS, **hpo_result.best_params}
            print(f"    HPO best IC={hpo_result.best_value:.4f}")

        # ── Train final model on all IS data ──
        weights = _compute_sample_weights(len(X))

        if use_regime:
            vol_idx = None
            for j, name in enumerate(feat_names):
                if name == "vol_20":
                    vol_idx = j
                    break
            if vol_idx is not None:
                vol_is = X[:, vol_idx]
                valid_vol = vol_is[~np.isnan(vol_is)]
                p33 = float(np.percentile(valid_vol, 33))
                p67 = float(np.percentile(valid_vol, 67))
                is_regimes = apply_vol_regime(vol_is, p33, p67)

                regime_bundle = train_regime_models(
                    X, y, is_regimes, feat_names,
                    params=best_params,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    embargo_bars=embargo,
                    sample_weight=weights,
                )
                final_model_type = "regime"
            else:
                use_regime = False

        if not use_regime:
            final_model = LGBMAlphaModel(name="v8_final", feature_names=tuple(feat_names))
            if is_binary:
                final_model.fit_classifier(
                    X, y, params=best_params,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    embargo_bars=embargo, sample_weight=weights)
            else:
                final_model.fit(
                    X, y, params=best_params,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    embargo_bars=embargo, sample_weight=weights)
            final_model_type = "single"

        # ── Full OOS validation ──
        oos_result = None

        if oos_feat_df is not None:
            for fname in feat_names:
                if fname not in oos_feat_df.columns:
                    oos_feat_df[fname] = np.nan

            oos_closes = oos_feat_df["close"].values.astype(np.float64)
            oos_target = _compute_target(oos_closes, horizon, target_mode)
            X_oos_all = oos_feat_df[feat_names].values.astype(np.float64)
            oos_valid = ~np.isnan(oos_target)
            oos_idx = np.where(oos_valid)[0]
            X_oos_v = X_oos_all[oos_idx]
            y_oos_v = oos_target[oos_idx]
            closes_oos_v = oos_closes[oos_idx]

            if len(X_oos_v) >= 100:
                if use_regime:
                    oos_vol = X_oos_v[:, vol_idx]
                    oos_regimes = apply_vol_regime(oos_vol, p33, p67)
                    y_pred = np.zeros(len(X_oos_v))
                    for i in range(len(X_oos_v)):
                        fdict = {feat_names[j]: float(X_oos_v[i, j])
                                for j in range(len(feat_names))}
                        pred = regime_bundle.predict_regime(int(oos_regimes[i]), fdict)
                        y_pred[i] = pred if pred is not None else 0.0
                else:
                    if is_binary:
                        y_pred = final_model._model.predict_proba(X_oos_v)[:, 1]
                    else:
                        y_pred = final_model._model.predict(X_oos_v)

                overall = _compute_split_metrics(y_pred, y_oos_v, closes_oos_v, target_mode)
                mid = len(y_pred) // 2
                h1 = _compute_split_metrics(y_pred[:mid], y_oos_v[:mid], closes_oos_v[:mid], target_mode)
                h2 = _compute_split_metrics(y_pred[mid:], y_oos_v[mid:], closes_oos_v[mid:], target_mode)

                # Monthly ICs
                monthly_ics = []
                chunk = 720
                for i in range(0, len(y_pred), chunk):
                    end = min(i + chunk, len(y_pred))
                    if end - i < 30:
                        continue
                    rx = _rankdata(y_pred[i:end])
                    ry = _rankdata(y_oos_v[i:end])
                    ic_val = float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 1 else 0.0
                    if not np.isnan(ic_val):
                        monthly_ics.append(ic_val)

                pos_months = sum(1 for ic in monthly_ics if ic > 0)
                total_months = len(monthly_ics)
                month_pos_ratio = pos_months / max(total_months, 1)

                # Bootstrap
                ret_1bar = np.diff(closes_oos_v) / closes_oos_v[:-1]
                pred_trunc = y_pred[:len(ret_1bar)]
                signal = _pred_to_signal(pred_trunc, target_mode=target_mode)
                turnover = np.abs(np.diff(signal, prepend=0))
                net_pnl = signal * ret_1bar - turnover * 4e-4
                active = signal != 0

                bootstrap_p_gt_0 = 0.0
                if _BOOTSTRAP_CPP and active.sum() > 20:
                    active_pnl = net_pnl[active].tolist()
                    br = cpp_bootstrap_sharpe_ci(active_pnl, 5000, 5, 42)
                    bootstrap_p_gt_0 = br.p_sharpe_gt_0

                # DSR with cumulative trial count
                dsr = runner.get_dsr(
                    observed_sharpe=overall["sharpe"],
                    n_observations=int(active.sum()),
                )

                oos_result = {
                    "overall": overall,
                    "h1": h1,
                    "h2": h2,
                    "monthly_ics": monthly_ics,
                    "month_pos_ratio": month_pos_ratio,
                    "bootstrap_p_gt_0": bootstrap_p_gt_0,
                    "dsr_p": dsr.p_value,
                    "dsr_significant": dsr.is_significant,
                    "n_trials_total": runner.total_trials,
                    "n_trials_final": runner.final_trials,
                }

        # ── Production gates ──
        gates_passed = True
        gate_details = {}

        if oos_result:
            gate_details["dsr_p"] = oos_result["dsr_p"]
            gate_details["h2_ic"] = oos_result["h2"]["ic"]
            gate_details["month_pos_ratio"] = oos_result["month_pos_ratio"]
            gate_details["bootstrap_p_gt_0"] = oos_result["bootstrap_p_gt_0"]
            gate_details["n_trials_total"] = oos_result["n_trials_total"]

            if oos_result["dsr_p"] >= 0.05:
                gates_passed = False
                gate_details["fail_dsr"] = True
            if oos_result["h2"]["ic"] <= 0:
                gates_passed = False
                gate_details["fail_h2_ic"] = True
            if oos_result["month_pos_ratio"] < 0.6:
                gates_passed = False
                gate_details["fail_month_consistency"] = True
            if oos_result["bootstrap_p_gt_0"] <= 0.75:
                gates_passed = False
                gate_details["fail_bootstrap"] = True
        else:
            gates_passed = False
            gate_details["fail_no_oos"] = True

        status = "PASS" if gates_passed else "FAIL"
        print(f"\n    Production gates: {status}")
        if oos_result:
            print(f"      DSR p={oos_result['dsr_p']:.4f} (final_trials={runner.final_trials}) "
                  f"{'OK' if oos_result['dsr_p'] < 0.05 else 'FAIL'}")
            print(f"      H2 IC={oos_result['h2']['ic']:.4f} "
                  f"{'OK' if oos_result['h2']['ic'] > 0 else 'FAIL'}")
            print(f"      Month IC pos ratio={oos_result['month_pos_ratio']:.2f} "
                  f"{'OK' if oos_result['month_pos_ratio'] >= 0.6 else 'FAIL'}")
            print(f"      Bootstrap P(Sharpe>0)={oos_result['bootstrap_p_gt_0']:.2f} "
                  f"{'OK' if oos_result['bootstrap_p_gt_0'] > 0.75 else 'FAIL'}")

        # ── Save model + register if passed ──
        sym_dir = out_dir / "step6_final" / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        if gates_passed or save_all:
            if use_regime:
                regime_bundle.save(sym_dir / "v8_regime_bundle.pkl")
            else:
                final_model.save(sym_dir / "v8_final.pkl")

            try:
                from research.model_registry.registry import ModelRegistry
                registry = ModelRegistry(str(out_dir / "model_registry.db"))
                metrics_dict = {
                    "oos_ic": oos_result["overall"]["ic"],
                    "oos_h2_ic": oos_result["h2"]["ic"],
                    "oos_sharpe": oos_result["overall"]["sharpe"],
                    "dsr_p": oos_result["dsr_p"],
                    "month_pos_ratio": oos_result["month_pos_ratio"],
                    "bootstrap_p_gt_0": oos_result["bootstrap_p_gt_0"],
                }
                mv = registry.register(
                    name=f"alpha_{symbol.lower()}_v8",
                    params={**best_params, "horizon": horizon, "target_mode": target_mode,
                            "model_type": final_model_type, "n_features": len(feat_names)},
                    features=feat_names,
                    metrics=metrics_dict,
                    tags=("v8", "alpha_rebuild", symbol),
                )
                print(f"    Registered: {mv.name} v{mv.version} (id={mv.model_id})")
            except Exception as e:
                logger.warning("Model registration failed: %s", e)

        final_results[symbol] = {
            "symbol": symbol,
            "horizon": horizon,
            "target_mode": target_mode,
            "model_type": final_model_type,
            "n_features": len(feat_names),
            "features": feat_names,
            "params": best_params,
            "oos": oos_result,
            "gates_passed": gates_passed,
            "gate_details": gate_details,
        }

    # Save final results
    final_dir = out_dir / "step6_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    _save_results(final_results, final_dir / "final_results.json")

    return final_results


# ── Report generation ────────────────────────────────────────────

def generate_report(
    runner: ExperimentRunner,
    final_results: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Generate markdown report summarizing all experiments."""
    lines = [
        "# Alpha Rebuild Report",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"Total experiments: {runner.total_trials}",
        "",
    ]

    # Step summary
    step_counts = {}
    for r in runner.results:
        step_counts[r.step] = step_counts.get(r.step, 0) + 1

    lines.append("## Experiment Counts by Step")
    lines.append("")
    step_names = {1: "Ablation", 2: "Stability", 3: "Model Comparison",
                  4: "Regime", 5: "Target Engineering", 6: "Final"}
    for step in sorted(step_counts):
        lines.append(f"- Step {step} ({step_names.get(step, '?')}): {step_counts[step]} experiments")

    # Final results per symbol
    lines.append("\n## Final Results")
    lines.append("")

    for symbol, result in final_results.items():
        status = "PASS" if result["gates_passed"] else "FAIL"
        lines.append(f"### {symbol} [{status}]")
        lines.append("")
        lines.append(f"- Horizon: {result['horizon']}, Mode: {result['target_mode']}")
        lines.append(f"- Model type: {result['model_type']}")
        lines.append(f"- Features: {result['n_features']}")

        oos = result.get("oos")
        if oos:
            lines.append(f"- OOS IC: {oos['overall']['ic']:.4f}")
            lines.append(f"- H1 IC: {oos['h1']['ic']:.4f}")
            lines.append(f"- H2 IC: {oos['h2']['ic']:.4f}")
            lines.append(f"- OOS Sharpe: {oos['overall']['sharpe']:.2f}")
            lines.append(f"- DSR p-value: {oos['dsr_p']:.4f} (final_trials={oos.get('n_trials_final', oos['n_trials_total'])})")  # noqa: E501
            lines.append(f"- Monthly IC positive: {oos['month_pos_ratio']:.0%}")
            lines.append(f"- Bootstrap P(Sharpe>0): {oos['bootstrap_p_gt_0']:.2f}")

        gates = result.get("gate_details", {})
        if gates:
            lines.append("\nProduction gates:")
            for key, val in gates.items():
                lines.append(f"  - {key}: {val}")

        lines.append(f"\nFeatures: {', '.join(result.get('features', []))}")
        lines.append("")

    # DSR audit
    lines.append("\n## DSR Audit")
    lines.append(f"\nTotal trials across all steps: {runner.total_trials}")
    lines.append(f"Final trials (Steps 5-6 only): {runner.final_trials}")
    lines.append("Steps 1-4 are pre-registered methodology; only Steps 5-6 count as free choices.")
    lines.append("DSR p-values use final_trials to avoid inflating the multiple-testing penalty.")
    lines.append("")

    report_path = out_dir / "alpha_rebuild_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\n  Report written to {report_path}")


# ── Serialization helper ─────────────────────────────────────────

def _save_results(results: Dict, path: Path) -> None:
    """Save results dict to JSON, converting ExperimentResult to dict."""
    def _to_serializable(obj: Any) -> Any:
        if isinstance(obj, ExperimentResult):
            return {
                "name": obj.name, "symbol": obj.symbol, "step": obj.step,
                "cv_ic": obj.cv_ic, "oos_ic": obj.oos_ic,
                "h1_ic": obj.h1_ic, "h2_ic": obj.h2_ic,
                "oos_sharpe": obj.oos_sharpe, "dsr_p": obj.dsr_p,
                "n_features": obj.n_features, "features": obj.features,
                "extra": obj.extra,
            }
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_to_serializable(results), f, indent=2, default=str)


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alpha Rebuild — systematic 6-step experiment pipeline")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "SOLUSDT"],
                        help="Symbols to process")
    parser.add_argument("--out-dir", default="results/alpha_rebuild",
                        help="Output directory")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                        help="Steps to run (default: all)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save model pkl even if gates fail (for long-only experiments)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.upper() for s in args.symbols]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = set(args.steps)

    print("\n  Alpha Rebuild Pipeline")
    print(f"  Symbols: {symbols}")
    print(f"  Steps: {sorted(steps)}")
    print(f"  Output: {out_dir}")
    print(f"  Top-K: {TOP_K}")

    # Build cross-asset features if needed
    cross_features = None
    alt_symbols = [s for s in symbols if s != "BTCUSDT"]
    if alt_symbols:
        print("\n  Building cross-asset features...")
        cross_features = _build_cross_features(symbols)
        if cross_features:
            print(f"  Built for: {list(cross_features.keys())}")

    runner = ExperimentRunner()
    harmful_features: Set[str] = set()
    stable_features: Dict[str, List[str]] = {}
    model_winners: Dict[str, str] = {}
    regime_winners: Dict[str, str] = {}
    best_targets: Dict[str, Tuple[int, str]] = {}
    final_results: Dict[str, Dict[str, Any]] = {}

    # ── Step 1 ──
    if 1 in steps:
        ablation_results = step1_ablation(symbols, cross_features, runner, out_dir)
        harmful_features = _identify_harmful_groups(ablation_results)
        if harmful_features:
            print(f"\n  Harmful features to remove: {sorted(harmful_features)}")
    else:
        # Try to load from previous run
        prev = out_dir / "step1_ablation" / "ablation_results.json"
        if prev.exists():
            print("  Loading Step 1 results from previous run...")

    # ── Step 2 ──
    if 2 in steps:
        stable_features = step2_stability(
            symbols, cross_features, runner, out_dir, harmful_features)
    else:
        prev = out_dir / "step2_stability" / "stable_features.json"
        if prev.exists():
            with open(prev) as f:
                stable_features = json.load(f)
            summary = {k: len(v) for k, v in stable_features.items()}
            print(f"  Loaded Step 2 stable features: {summary}")

    # ── Step 3 ──
    if 3 in steps:
        model_winners = step3_model_comparison(
            symbols, cross_features, runner, out_dir, stable_features)
    else:
        prev = out_dir / "step3_model" / "model_winners.json"
        if prev.exists():
            with open(prev) as f:
                model_winners = json.load(f)

    # ── Step 4 ──
    if 4 in steps:
        regime_winners = step4_regime(
            symbols, cross_features, runner, out_dir, stable_features, model_winners)
    else:
        prev = out_dir / "step4_regime" / "regime_winners.json"
        if prev.exists():
            with open(prev) as f:
                regime_winners = json.load(f)

    # ── Step 5 ──
    if 5 in steps:
        best_targets = step5_target(
            symbols, cross_features, runner, out_dir, stable_features)
    else:
        prev = out_dir / "step5_target" / "best_targets.json"
        if prev.exists():
            with open(prev) as f:
                raw = json.load(f)
            best_targets = {k: tuple(v) for k, v in raw.items()}

    # ── Step 6 ──
    if 6 in steps:
        final_results = step6_final(
            symbols, cross_features, runner, out_dir,
            stable_features, model_winners, regime_winners, best_targets,
            save_all=args.save_all)

    # ── Report ──
    generate_report(runner, final_results, out_dir)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("  Alpha Rebuild Complete")
    print(f"  Total experiments: {runner.total_trials}")
    print(f"{'='*70}")

    for symbol, result in final_results.items():
        status = "PASS" if result["gates_passed"] else "FAIL"
        oos = result.get("oos")
        if oos:
            print(f"  {symbol:<10s} [{status}]  "
                  f"H2 IC={oos['h2']['ic']:.4f}  "
                  f"Sharpe={oos['overall']['sharpe']:.2f}  "
                  f"DSR p={oos['dsr_p']:.4f}")
        else:
            print(f"  {symbol:<10s} [{status}]  no OOS data")

    print(f"\n  Results: {out_dir}")


if __name__ == "__main__":
    main()
