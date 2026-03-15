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
import hashlib
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from alpha.signal_transform import pred_to_signal as _pred_to_signal
from scripts.backtest_alpha_v8 import COST_PER_TRADE
from scripts.signal_postprocess import (
    _apply_monthly_gate,
    _apply_trend_hold,
    _apply_vol_target,
    _compute_bear_mask,
    _enforce_min_hold,
)
from features.dynamic_selector import greedy_ic_select, stable_icir_select

from features.batch_backtest import run_backtest_fast

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

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
ZSCORE_WINDOW = 720
ZSCORE_WARMUP = 168
HPO_TRIALS = 10
NN_SEQ_LEN = 20
NN_EPOCHS = 30
FEATURE_VERSION = "v11.1"

_HPO_CACHE_DIR = Path(".cache/hpo")
_HPO_CACHE_ENABLED = True


def _hpo_cache_key(selected_features: List[str], n_train: int) -> str:
    raw = json.dumps(sorted(selected_features)) + f"_{n_train}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_hpo_cache(key: str) -> Optional[dict]:
    if not _HPO_CACHE_ENABLED:
        return None
    path = _HPO_CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_hpo_cache(key: str, params: dict) -> None:
    if not _HPO_CACHE_ENABLED:
        return
    _HPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _HPO_CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(params, f)


# ── NN Ensemble helpers ─────────────────────────────────────

def _compute_ensemble_weights(
    preds_list: List[np.ndarray], y_true: np.ndarray,
) -> List[float]:
    """IC-weighted ensemble: weight_i = max(spearman_ic_i, 0) / sum.

    Falls back to equal weights if all ICs are non-positive.
    """
    from features.dynamic_selector import _rankdata, _spearman_ic

    n = len(preds_list)
    ics = []
    valid = ~np.isnan(y_true)
    y_v = y_true[valid]
    if len(y_v) < 10:
        return [1.0 / n] * n

    yr = _rankdata(y_v)
    for p in preds_list:
        pv = p[valid]
        if np.std(pv) < 1e-12:
            ics.append(0.0)
        else:
            ics.append(float(_spearman_ic(_rankdata(pv), yr)))

    weights = [max(ic, 0.0) for ic in ics]
    total = sum(weights)
    if total < 1e-12:
        return [1.0 / n] * n
    return [w / total for w in weights]


def _apply_nn_ensemble(
    lgbm_pred: np.ndarray,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    X_test_sel: np.ndarray,
    nn_seq_len: int = NN_SEQ_LEN,
    nn_epochs: int = NN_EPOCHS,
) -> np.ndarray:
    """Train LSTM + Transformer on sliding windows, return IC-weighted ensemble.

    Returns y_pred of same length as lgbm_pred. Falls back to lgbm_pred on failure.
    """
    if not _HAS_TORCH:
        logger.warning("torch not available, skipping NN ensemble")
        return lgbm_pred

    from alpha.nn_utils import make_sliding_windows, align_target
    from alpha.models.lstm_alpha import LSTMAlphaModel
    from alpha.models.transformer_alpha import TransformerAlphaModel
    from sklearn.preprocessing import StandardScaler

    n_test = len(lgbm_pred)
    X_train_sel.shape[1]

    # StandardScaler fit on train
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_sel)
    X_test_sc = scaler.transform(X_test_sel)

    # Sliding windows
    if len(X_train_sc) <= nn_seq_len or len(X_test_sc) <= nn_seq_len:
        logger.warning("Not enough data for NN seq_len=%d, skipping", nn_seq_len)
        return lgbm_pred

    X_train_3d = make_sliding_windows(X_train_sc, nn_seq_len)
    y_train_3d = align_target(y_train, nn_seq_len)
    X_test_3d = make_sliding_windows(X_test_sc, nn_seq_len)

    nn_preds = []
    nn_names = []

    # Train LSTM
    try:
        lstm = LSTMAlphaModel(seq_len=nn_seq_len)
        lstm.fit(X_train_3d, y_train_3d, epochs=nn_epochs)
        lstm_pred_short = lstm.predict_batch(X_test_3d)
        nn_preds.append(lstm_pred_short)
        nn_names.append("LSTM")
    except Exception as e:
        logger.warning("LSTM training failed: %s", e)

    # Train Transformer
    try:
        tfm = TransformerAlphaModel(seq_len=nn_seq_len)
        tfm.fit(X_train_3d, y_train_3d, epochs=nn_epochs)
        tfm_pred_short = tfm.predict_batch(X_test_3d)
        nn_preds.append(tfm_pred_short)
        nn_names.append("Transformer")
    except Exception as e:
        logger.warning("Transformer training failed: %s", e)

    if not nn_preds:
        return lgbm_pred

    # Pad NN predictions: first (seq_len-1) bars use LGBM-only
    pad = nn_seq_len - 1
    nn_preds_full = []
    for p in nn_preds:
        padded = np.concatenate([lgbm_pred[:pad], p[:n_test - pad]])
        nn_preds_full.append(padded[:n_test])

    # Compute IC-weighted ensemble weights on train validation portion.
    # Use last 20% of train 3D windows (with embargo already handled inside NN fit).
    val_start = int(len(y_train_3d) * 0.8)
    val_y = y_train_3d[val_start:]
    nn_models = []
    for name in nn_names:
        nn_models.append(lstm if name == "LSTM" else tfm)

    try:
        train_val_3d = X_train_3d[val_start:]
        val_preds_for_ic = []
        for model in nn_models:
            val_preds_for_ic.append(model.predict_batch(train_val_3d)[:len(val_y)])
        # Weight only among NN models, then blend with LGBM at fixed ratio
        nn_weights = _compute_ensemble_weights(val_preds_for_ic, val_y)
        # Blend: 60% LGBM + 40% NN (IC-weighted among NN models)
        lgbm_w = 0.6
        nn_w = 0.4
    except Exception:
        # Equal weight fallback
        n_models = 1 + len(nn_preds)
        lgbm_w = 1.0 / n_models
        nn_w = 1.0 - lgbm_w
        nn_weights = [1.0 / len(nn_preds)] * len(nn_preds)

    # Weighted average
    y_ensemble = lgbm_w * lgbm_pred[:n_test]
    for w, p in zip(nn_weights, nn_preds_full):
        y_ensemble = y_ensemble + (nn_w * w) * p

    model_names = ["LGBM"] + nn_names
    final_weights = [lgbm_w] + [nn_w * w for w in nn_weights]
    logger.info("NN ensemble weights: %s (%s)",
                [f"{w:.3f}" for w in final_weights], model_names)
    return y_ensemble


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


# ── Feature selection dispatch ─────────────────────────────────

def _select_features_dispatch(selector: str):
    """Return a feature selection function based on selector name."""
    if selector == "stable_icir":
        return lambda X, y, names, top_k: stable_icir_select(X, y, names, top_k=top_k)
    # Default: greedy
    return lambda X, y, names, top_k: greedy_ic_select(X, y, names, top_k=top_k)


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
    vol_target: Optional[float] = None,
    vol_feature: str = "atr_norm_14",
    dd_limit: Optional[float] = None,
    dd_cooldown: int = 48,
    cost_model_type: str = "realistic",
    volumes: Optional[np.ndarray] = None,
    nn_ensemble: bool = False,
    nn_seq_len: int = NN_SEQ_LEN,
    nn_epochs: int = NN_EPOCHS,
    selector: str = "greedy",
    lgbm_threads: Optional[int] = None,
    zscore_window: int = ZSCORE_WINDOW,
    zscore_warmup: int = ZSCORE_WARMUP,
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
    _select = _select_features_dispatch(selector)
    if fixed_features:
        # Fixed mode: lock stable features + select flexible from candidate pool
        selected = list(fixed_features)
        if n_flexible > 0:
            pool = candidate_pool if candidate_pool else [
                f for f in feature_names if f not in fixed_features]
            pool_in_data = [f for f in pool if f in feature_names]
            if pool_in_data:
                pool_idx = [feature_names.index(f) for f in pool_in_data]
                X_pool = X_train[:, pool_idx]
                flex = _select(X_pool, y_train, pool_in_data, top_k=n_flexible)
                selected.extend(flex)
    else:
        selected = _select(X_train, y_train, feature_names, top_k=TOP_K)
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
    if lgbm_threads is not None:
        params["num_threads"] = lgbm_threads
    if use_hpo:
        hpo_key = _hpo_cache_key(selected, len(X_train_sel))
        cached_hpo = _load_hpo_cache(hpo_key)
        if cached_hpo is not None:
            params = {**params, **cached_hpo}
        else:
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
                best = result.best_params
                params = {**params, **best}
                _save_hpo_cache(hpo_key, best)
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
            if lgbm_threads is not None:
                xgb_params["nthread"] = lgbm_threads
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

    # NN ensemble: LSTM + Transformer IC-weighted blend
    if nn_ensemble:
        y_pred = _apply_nn_ensemble(
            y_pred, X_train_sel, y_train, X_test_sel,
            nn_seq_len=nn_seq_len, nn_epochs=nn_epochs,
        )

    # IC on test
    ic = 0.0
    if test_valid.sum() > 10:
        from features.dynamic_selector import _rankdata, _spearman_ic
        y_pred_v = y_pred[test_valid]
        y_test_v = y_test[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(y_pred_v), _rankdata(y_test_v)))

    # Trade simulation on test
    # C++ fast path: when no complex Python-only modifiers are active
    _use_cpp = (not continuous_sizing and not regime_gate
                and not adaptive_sizing and not trend_follow)

    if _use_cpp:
        cfg = {
            "deadzone": deadzone, "min_hold": min_hold,
            "zscore_window": zscore_window, "zscore_warmup": zscore_warmup,
            "cost_per_trade": COST_PER_TRADE,
        }
        if monthly_gate:
            cfg["monthly_gate"] = True
            cfg["ma_window"] = monthly_gate_window
        if long_only:
            cfg["long_only"] = True
        if vol_target is not None and vol_feature in test_df.columns:
            cfg["vol_adaptive"] = True
            cfg["vol_target"] = vol_target
        if dd_limit is not None:
            cfg["dd_breaker"] = True
            cfg["dd_limit"] = dd_limit
            cfg["dd_cooldown"] = dd_cooldown

        test_ts = test_df.index.values.astype(np.int64) if hasattr(test_df.index, 'values') else np.arange(len(test_closes), dtype=np.int64) * 3600_000

        vol_vals = (test_df[vol_feature].values.astype(np.float64)
                    if vol_target is not None and vol_feature in test_df.columns
                    else np.empty(0, dtype=np.float64))

        # Funding
        fr_arr = np.empty(0, dtype=np.float64)
        fr_ts_arr = np.empty(0, dtype=np.int64)
        if "funding_rate" in test_df.columns:
            fr_vals = test_df["funding_rate"].values.astype(np.float64)
            fr_vals = np.nan_to_num(fr_vals, 0.0)
            fr_arr = fr_vals
            fr_ts_arr = test_ts[:len(fr_vals)]

        # Volumes for realistic cost
        vo = np.empty(0, dtype=np.float64)
        v20 = np.empty(0, dtype=np.float64)
        if cost_model_type == "realistic" and volumes is not None:
            cfg["realistic_cost"] = True
            vo = volumes[fold.test_start:fold.test_end][:len(test_closes)].astype(np.float64)
            v20 = (test_df["vol_20"].values[:len(test_closes)].astype(np.float64)
                   if "vol_20" in test_df.columns
                   else np.full(len(test_closes), np.nan))

        cpp_result = run_backtest_fast(
            test_ts[:len(test_closes)], test_closes, y_pred,
            volumes=vo if len(vo) > 0 else None,
            vol_20=v20 if len(v20) > 0 else None,
            vol_values=vol_vals if len(vol_vals) > 0 else None,
            funding_rates=fr_arr if len(fr_arr) > 0 else None,
            funding_ts=fr_ts_arr if len(fr_ts_arr) > 0 else None,
            config=cfg,
        )
        sharpe = float(cpp_result["sharpe"])
        total_return = float(cpp_result["total_return"])
    else:
        signal = _pred_to_signal(y_pred, target_mode=target_mode,
                                 deadzone=deadzone, min_hold=min_hold,
                                 zscore_window=zscore_window,
                                 zscore_warmup=zscore_warmup)

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

        # Vol-adaptive sizing
        if vol_target is not None and vol_feature in test_df.columns:
            vol_vals = test_df[vol_feature].values.astype(np.float64)
            signal = _apply_vol_target(signal, vol_vals, vol_target)

        # DD breaker
        if dd_limit is not None:
            from scripts.backtest_alpha_v8 import _apply_dd_breaker
            signal = _apply_dd_breaker(signal, test_closes, dd_limit, dd_cooldown)

        ret_1bar = np.diff(test_closes) / test_closes[:-1]
        signal_for_trade = signal[:len(ret_1bar)]
        gross_pnl = signal_for_trade * ret_1bar

        if cost_model_type == "realistic" and volumes is not None:
            from execution.sim.cost_model import RealisticCostModel
            cm = RealisticCostModel()
            test_vols = volumes[fold.test_start:fold.test_end][:len(signal_for_trade)]
            vol_20 = test_df["vol_20"].values[:len(signal_for_trade)].astype(np.float64) if "vol_20" in test_df.columns else np.full(len(signal_for_trade), np.nan)
            breakdown = cm.compute_costs(signal_for_trade, test_closes[:len(signal_for_trade)], test_vols, vol_20)
            cost = breakdown.total_cost
            signal_for_trade = breakdown.clipped_signal
            gross_pnl = signal_for_trade * ret_1bar
        else:
            turnover = np.abs(np.diff(signal_for_trade, prepend=0))
            cost = turnover * COST_PER_TRADE

        # Funding cost: long pays positive funding, short receives it
        funding_cost = np.zeros(len(signal_for_trade))
        if "funding_rate" in test_df.columns:
            fr = test_df["funding_rate"].values[:len(signal_for_trade)].astype(np.float64)
            fr = np.nan_to_num(fr, 0.0)
            funding_cost = signal_for_trade * fr / 8.0

        net_pnl = gross_pnl - cost - funding_cost

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


# ── Strategy F: per-fold regime-switch WF ────────────────────

# Bear detector constants
_BEAR_DETECTOR_POOL = [
    "funding_zscore_24", "funding_momentum", "funding_extreme",
    "funding_sign_persist", "funding_cumulative_8",
    "basis", "basis_zscore_24", "basis_momentum",
    "vol_20", "vol_regime", "parkinson_vol", "atr_norm_14",
    "fgi_normalized", "fgi_extreme",
    "rsi_14", "bb_pctb_20",
    "oi_acceleration", "leverage_proxy",
]

_BEAR_CLS_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.02,
    "num_leaves": 16,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
}

def _compute_bear_target(closes: np.ndarray, horizon: int = 24,
                          threshold: float = -0.02) -> np.ndarray:
    """Binary target: 1 if forward return < threshold (crash)."""
    n = len(closes)
    raw = np.full(n, np.nan)
    raw[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1.0
    target = np.where(raw < threshold, 1.0, 0.0)
    target[np.isnan(raw)] = np.nan
    return target


def run_fold_strategy_f(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    *,
    fixed_features: Optional[List[str]] = None,
    candidate_pool: Optional[List[str]] = None,
    n_flexible: int = 4,
    use_hpo: bool = False,
    hpo_trials: int = HPO_TRIALS,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    monthly_gate_window: int = 480,
    bear_thresholds: Optional[list] = None,
    vol_target: Optional[float] = None,
    vol_feature: str = "atr_norm_14",
    dd_limit: Optional[float] = None,
    dd_cooldown: int = 48,
    cost_model_type: str = "realistic",
    volumes: Optional[np.ndarray] = None,
    nn_ensemble: bool = False,
    nn_seq_len: int = NN_SEQ_LEN,
    nn_epochs: int = NN_EPOCHS,
    selector: str = "greedy",
    lgbm_threads: Optional[int] = None,
    short_supplement: bool = False,
    zscore_window: int = ZSCORE_WINDOW,
    zscore_warmup: int = ZSCORE_WARMUP,
) -> FoldResult:
    """Per-fold Strategy F: train V8 ensemble + bear C within each fold.

    Bull regime (close > SMA): V8 LGBM+XGB long-only signal.
    Bear regime (close <= SMA): bear C classifier short signal.
    When short_supplement=True, also trains an independent short regression model
    that fills gaps where the combined signal is flat during bear regime.
    Both models trained exclusively on the fold's train window.
    """
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]
    n_test = len(test_closes)

    # ── 1. Train V8 bull model (same as run_fold) ──
    y_train_full = _compute_target(train_closes, HORIZON, TARGET_MODE)
    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)

    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]
    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]

    if len(X_train) < 1000:
        return FoldResult(idx=fold.idx, period="", ic=0.0, sharpe=0.0,
                          total_return=0.0, features=[], n_train=len(X_train), n_test=0)

    # Feature selection
    _select = _select_features_dispatch(selector)
    if fixed_features:
        selected = list(fixed_features)
        if n_flexible > 0:
            pool = candidate_pool or [f for f in feature_names if f not in fixed_features]
            pool_in_data = [f for f in pool if f in feature_names]
            if pool_in_data:
                pool_idx = [feature_names.index(f) for f in pool_in_data]
                X_pool = X_train[:, pool_idx]
                flex = _select(X_pool, y_train, pool_in_data, top_k=n_flexible)
                selected.extend(flex)
    else:
        selected = _select(X_train, y_train, feature_names, top_k=TOP_K)

    if not selected:
        return FoldResult(idx=fold.idx, period="", ic=0.0, sharpe=0.0,
                          total_return=0.0, features=[], n_train=len(X_train), n_test=0)

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # V8 HPO or default
    params = dict(V7_DEFAULT_PARAMS)
    if lgbm_threads is not None:
        params["num_threads"] = lgbm_threads
    if use_hpo:
        hpo_key = _hpo_cache_key(selected, len(X_train_sel))
        cached_hpo = _load_hpo_cache(hpo_key)
        if cached_hpo is not None:
            params = {**params, **cached_hpo}
        else:
            try:
                from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
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
                    bst = lgb.train(p, dtrain, num_boost_round=p["n_estimators"],
                                    valid_sets=[dval],
                                    callbacks=[lgb.early_stopping(50, verbose=False),
                                               lgb.log_evaluation(0)])
                    y_hat = bst.predict(X_hpo_val)
                    vm = ~np.isnan(y_hpo_val)
                    if vm.sum() < 10:
                        return 0.0
                    from features.dynamic_selector import _rankdata, _spearman_ic
                    return float(_spearman_ic(_rankdata(y_hat[vm]), _rankdata(y_hpo_val[vm])))

                opt = HyperOptimizer(search_space=V7_SEARCH_SPACE, objective_fn=objective,
                                     config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"))
                result = opt.optimize()
                best = result.best_params
                params = {**params, **best}
                _save_hpo_cache(hpo_key, best)
            except Exception as e:
                logger.warning("HPO failed for fold %d: %s", fold.idx, e)

    # Train LGBM
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    lgbm_bst = lgb.train(params, dtrain,
                          num_boost_round=params.get("n_estimators", 500),
                          callbacks=[lgb.log_evaluation(0)])
    lgbm_pred = lgbm_bst.predict(X_test_sel)

    # Train XGB (ensemble)
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
        if lgbm_threads is not None:
            xgb_params["nthread"] = lgbm_threads
        dtrain_xgb = xgb.DMatrix(X_train_sel, label=y_train)
        dtest_xgb = xgb.DMatrix(X_test_sel)
        xgb_bst = xgb.train(xgb_params, dtrain_xgb,
                             num_boost_round=params.get("n_estimators", 500))
        xgb_pred = xgb_bst.predict(dtest_xgb)
        y_pred_v8 = 0.5 * lgbm_pred + 0.5 * xgb_pred
    except Exception:
        y_pred_v8 = lgbm_pred

    # NN ensemble on bull model (bear stays LGBM-only — too few samples for NN)
    if nn_ensemble:
        y_pred_v8 = _apply_nn_ensemble(
            y_pred_v8, X_train_sel, y_train, X_test_sel,
            nn_seq_len=nn_seq_len, nn_epochs=nn_epochs,
        )

    # V8 signal: long-only
    sig_v8 = _pred_to_signal(y_pred_v8, target_mode=TARGET_MODE,
                              deadzone=deadzone, min_hold=min_hold)
    sig_v8 = np.clip(sig_v8, 0.0, 1.0)

    # ── 2. Train bear C model on bear bars within train window ──
    bear_mask_train = _compute_bear_mask(train_closes, monthly_gate_window)
    bear_target = _compute_bear_target(train_closes, horizon=HORIZON, threshold=-0.02)

    bear_features_available = [f for f in _BEAR_DETECTOR_POOL if f in feature_names]
    bear_feat_idx = [feature_names.index(f) for f in bear_features_available]
    X_bear_train_full = X_train_full[:, bear_feat_idx]

    # Valid bear samples: in bear regime + valid target + past warmup
    bear_valid = bear_mask_train & ~np.isnan(bear_target)
    bear_valid[:WARMUP] = False
    bear_idx = np.where(bear_valid)[0]

    sig_bear = np.zeros(n_test)
    bear_trained = False

    if len(bear_idx) >= 200:
        X_bear = np.nan_to_num(X_bear_train_full[bear_idx], 0.0)
        y_bear = bear_target[bear_idx]

        n_pos = int(y_bear.sum())
        n_neg = len(y_bear) - n_pos
        scale_pos = n_neg / max(n_pos, 1)

        bear_params = dict(_BEAR_CLS_PARAMS)
        bear_params["scale_pos_weight"] = scale_pos
        if lgbm_threads is not None:
            bear_params["num_threads"] = lgbm_threads

        dtrain_bear = lgb.Dataset(X_bear, label=y_bear)
        bear_bst = lgb.train(bear_params, dtrain_bear,
                              num_boost_round=bear_params["n_estimators"],
                              callbacks=[lgb.log_evaluation(0)])

        # Predict on test
        X_bear_test = np.nan_to_num(
            test_df[bear_features_available].values.astype(np.float64) if all(
                f in test_df.columns for f in bear_features_available)
            else X_test[:, bear_feat_idx],
            0.0)
        bear_prob = bear_bst.predict(X_bear_test)

        # Convert prob to signal
        if bear_thresholds:
            for i in range(n_test):
                for thresh, score in bear_thresholds:
                    if bear_prob[i] > thresh:
                        sig_bear[i] = score
                        break
        else:
            sig_bear = np.where(bear_prob > 0.5, -1.0, 0.0)

        bear_trained = True

    # ── 3. Regime-switch: combine bull + bear ──
    bear_mask_test = _compute_bear_mask(test_closes, monthly_gate_window)
    signal = np.where(bear_mask_test[:n_test], sig_bear[:n_test], sig_v8[:n_test])

    # ── 3b. Short model supplement (fills flat gaps in bear regime) ──
    if short_supplement:
        from scripts.train_short_production import (
            SHORT_FIXED_FEATURES as _SF_FIXED,
            SHORT_CANDIDATE_POOL as _SF_POOL,
            N_FLEXIBLE as _SF_NFLEX,
            DEADZONE as _SF_DZ,
            MIN_HOLD as _SF_MH,
            MA_WINDOW as _SF_MA,
            BEAR_WEIGHT as _SF_BW,
        )
        # Train short regression on ALL train bars with bear weighting
        short_fixed_in = [f for f in _SF_FIXED if f in feature_names]
        short_pool_in = [f for f in _SF_POOL if f in feature_names]
        short_sel = list(short_fixed_in)
        if short_pool_in and _SF_NFLEX > 0:
            neg_mask_s = y_train < 0
            sp_idx = [feature_names.index(f) for f in short_pool_in]
            if neg_mask_s.sum() > 1000:
                X_sp_neg = X_train[neg_mask_s][:, sp_idx]
                y_sp_neg = y_train[neg_mask_s]
                short_flex = stable_icir_select(X_sp_neg, y_sp_neg, short_pool_in, top_k=_SF_NFLEX)
            else:
                short_flex = stable_icir_select(X_train[:, sp_idx], y_train, short_pool_in, top_k=_SF_NFLEX)
            short_sel.extend(short_flex)

        if short_sel:
            ss_idx = [feature_names.index(f) for f in short_sel]
            X_train_ss = X_train[:, ss_idx]
            X_test_ss = X_test[:, ss_idx]

            # Bear sample weighting
            sw_short = np.ones(len(y_train))
            bear_mask_tr_short = _compute_bear_mask(train_closes, _SF_MA)
            tr_bear_s = bear_mask_tr_short[WARMUP:][train_valid]
            sw_short[tr_bear_s] = _SF_BW

            # Train LGBM + XGB short ensemble
            dt_ss = lgb.Dataset(X_train_ss, label=y_train, weight=sw_short)
            ss_params = dict(V7_DEFAULT_PARAMS)
            if lgbm_threads is not None:
                ss_params["num_threads"] = lgbm_threads
            ss_bst = lgb.train(ss_params, dt_ss,
                               num_boost_round=ss_params.get("n_estimators", 500),
                               callbacks=[lgb.log_evaluation(0)])
            ss_pred = ss_bst.predict(X_test_ss)

            try:
                import xgboost as xgb
                xp_ss = {"max_depth": 6, "learning_rate": 0.05, "objective": "reg:squarederror",
                          "verbosity": 0, "subsample": 0.8, "colsample_bytree": 0.8}
                if lgbm_threads is not None:
                    xp_ss["nthread"] = lgbm_threads
                dt_xgb_ss = xgb.DMatrix(X_train_ss, label=y_train, weight=sw_short)
                xgb_ss = xgb.train(xp_ss, dt_xgb_ss, num_boost_round=500)
                ss_pred = 0.5 * ss_pred + 0.5 * xgb_ss.predict(xgb.DMatrix(X_test_ss))
            except Exception:
                pass

            # Short-only signal with regime gate
            sig_short = _pred_to_signal(ss_pred, target_mode=TARGET_MODE,
                                         deadzone=_SF_DZ, min_hold=_SF_MH)
            np.clip(sig_short, -1.0, 0.0, out=sig_short)
            sig_short[~bear_mask_test[:len(sig_short)]] = 0.0

            # Supplementary: only fill flat gaps
            for i in range(min(len(signal), len(sig_short))):
                if sig_short[i] < 0 and signal[i] == 0.0:
                    signal[i] = sig_short[i]

    # Vol-adaptive sizing
    if vol_target is not None and vol_feature in test_df.columns:
        vol_vals = test_df[vol_feature].values.astype(np.float64)
        signal = _apply_vol_target(signal, vol_vals, vol_target)

    # Enforce min_hold (direct — signal is already discrete, not raw predictions)
    signal = _enforce_min_hold(signal, min_hold)

    # DD breaker
    if dd_limit is not None:
        from scripts.backtest_alpha_v8 import _apply_dd_breaker
        signal = _apply_dd_breaker(signal, test_closes, dd_limit, dd_cooldown)

    # ── 4. Evaluate ──
    y_test_full = _compute_target(test_closes, HORIZON, TARGET_MODE)
    test_valid = ~np.isnan(y_test_full)

    # IC (use V8 predictions for IC, as bear is classifier)
    ic = 0.0
    if test_valid.sum() > 10:
        from features.dynamic_selector import _rankdata, _spearman_ic
        y_pred_v = y_pred_v8[test_valid]
        y_test_v = y_test_full[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(y_pred_v), _rankdata(y_test_v)))

    # PnL
    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    sig_trade = signal[:len(ret_1bar)]
    gross_pnl = sig_trade * ret_1bar

    if cost_model_type == "realistic" and volumes is not None:
        from execution.sim.cost_model import RealisticCostModel
        cm = RealisticCostModel()
        test_vols = volumes[fold.test_start:fold.test_end][:len(sig_trade)]
        vol_20 = test_df["vol_20"].values[:len(sig_trade)].astype(np.float64) if "vol_20" in test_df.columns else np.full(len(sig_trade), np.nan)
        breakdown = cm.compute_costs(sig_trade, test_closes[:len(sig_trade)], test_vols, vol_20)
        cost = breakdown.total_cost
        sig_trade = breakdown.clipped_signal
        gross_pnl = sig_trade * ret_1bar
    else:
        turnover = np.abs(np.diff(sig_trade, prepend=0))
        cost = turnover * COST_PER_TRADE

    # Funding cost
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

    return FoldResult(
        idx=fold.idx,
        period="",
        ic=ic,
        sharpe=sharpe,
        total_return=total_return,
        features=selected + (bear_features_available if bear_trained else []),
        n_train=len(X_train),
        n_test=n_test,
    )


# ── Fold helpers for parallel execution ──────────────────────

def _fold_period_label(fold: Fold, timestamps: np.ndarray, n_bars: int) -> str:
    """Compute human-readable period label for a fold."""
    from datetime import datetime, timezone
    try:
        ts_start = datetime.fromtimestamp(timestamps[fold.test_start] / 1000, tz=timezone.utc)
        ts_end = datetime.fromtimestamp(
            timestamps[min(fold.test_end - 1, n_bars - 1)] / 1000, tz=timezone.utc)
        return f"{ts_start:%Y-%m}→{ts_end:%Y-%m}"
    except (ValueError, OSError, IndexError):
        return f"fold_{fold.idx}"


def _run_single_fold(kwargs: dict) -> Tuple[int, FoldResult, float, str]:
    """Module-level wrapper for ProcessPoolExecutor (must be picklable).

    Dispatches to run_fold() or run_fold_strategy_f() based on kwargs.
    Returns (fold_idx, FoldResult, elapsed_seconds, period_label).
    """
    fold = kwargs.pop("fold")
    strategy_f = kwargs.pop("strategy_f", False)
    period = kwargs.pop("period", "")

    t1 = time.time()
    if strategy_f:
        result = run_fold_strategy_f(fold, **kwargs)
    else:
        kwargs.pop("short_supplement", None)  # not used by run_fold
        result = run_fold(fold, **kwargs)
    elapsed = time.time() - t1
    result.period = period
    return (fold.idx, result, elapsed, period)


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
    print("  WALK-FORWARD VALIDATION REPORT")
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
        print("\n  Feature stability (>= 80% folds):")
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
    parser.add_argument("--strategy-f", action="store_true",
                        help="Per-fold Strategy F: V8 bull + bear C regime-switch")
    parser.add_argument("--bear-thresholds", default=None,
                        help='Graded bear thresholds JSON, e.g. \'[[0.7,-1.0],[0.6,-0.5],[0.5,0.0]]\'')
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Target vol for position scaling (e.g. 0.02)")
    parser.add_argument("--vol-feature", default="atr_norm_14",
                        help="Feature for realized vol (default: atr_norm_14)")
    parser.add_argument("--dd-limit", type=float, default=None,
                        help="Max drawdown circuit breaker (e.g. -0.15)")
    parser.add_argument("--dd-cooldown", type=int, default=48,
                        help="Bars to stay flat after DD breach (default: 48)")
    parser.add_argument("--cost-model", choices=["flat", "realistic"], default="realistic",
                        help="Cost model: flat (6bps) or realistic (sqrt-impact + spread)")
    parser.add_argument("--zscore-window", type=int, default=ZSCORE_WINDOW,
                        help=f"Rolling z-score window in bars (default: {ZSCORE_WINDOW})")
    parser.add_argument("--zscore-warmup", type=int, default=ZSCORE_WARMUP,
                        help=f"Z-score warmup bars before signal generation (default: {ZSCORE_WARMUP})")
    parser.add_argument("--nn-ensemble", action="store_true",
                        help="Add LSTM + Transformer to ensemble (IC-weighted)")
    parser.add_argument("--nn-seq-len", type=int, default=NN_SEQ_LEN,
                        help=f"NN sliding window length (default: {NN_SEQ_LEN})")
    parser.add_argument("--nn-epochs", type=int, default=NN_EPOCHS,
                        help=f"NN max training epochs (default: {NN_EPOCHS})")
    parser.add_argument("--selector", choices=["greedy", "stable_icir"],
                        default="greedy",
                        help="Feature selector: greedy (default) or stable_icir")
    parser.add_argument("--short-supplement", action="store_true",
                        help="Add independent short model supplement to Strategy F (fills flat gaps in bear regime)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel fold workers (default: 1 = serial)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip feature/HPO caching, recompute everything")
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
    strategy_f = args.strategy_f
    bear_thresholds = None
    if args.bear_thresholds:
        bear_thresholds = [tuple(x) for x in json.loads(args.bear_thresholds)]
    vol_target = args.vol_target
    vol_feature_name = args.vol_feature
    dd_limit = args.dd_limit
    dd_cooldown = args.dd_cooldown
    cost_model_type = args.cost_model
    zscore_window = args.zscore_window
    zscore_warmup = args.zscore_warmup
    nn_ensemble = args.nn_ensemble
    nn_seq_len = args.nn_seq_len
    nn_epochs = args.nn_epochs
    selector = args.selector
    short_supplement = args.short_supplement
    parallel = max(1, args.parallel)
    no_cache = args.no_cache

    global _HPO_CACHE_ENABLED
    _HPO_CACHE_ENABLED = not no_cache

    print(f"\n  Walk-Forward Validation: {symbol}")
    if strategy_f:
        print("  MODE: Strategy F (per-fold V8 + bear C regime-switch)")
        if short_supplement:
            print("  SHORT SUPPLEMENT: ON (independent short model fills bear flat gaps)")
    if selector != "greedy":
        print(f"  Selector: {selector}")
    print(f"  HPO: {'ON' if use_hpo else 'OFF (default params)'}"
          f"{f' ({hpo_trials} trials)' if use_hpo else ''}")
    print(f"  Signal: deadzone={deadzone}, min_hold={min_hold}"
          f"{', continuous_sizing' if continuous_sizing else ''}")
    if ensemble:
        print("  Ensemble: LGBM + XGB (averaged)")
    if nn_ensemble:
        print(f"  NN Ensemble: LSTM + Transformer (IC-weighted, seq_len={nn_seq_len}, epochs={nn_epochs})")
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
    if parallel > 1:
        print(f"  Parallel: {parallel} workers")
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
    feat_df = None
    cache_dir = Path(".cache/features")
    if not no_cache:
        csv_stat = csv_path.stat()
        cache_key_raw = f"{symbol}_{csv_stat.st_mtime}_{csv_stat.st_size}_{FEATURE_VERSION}"
        cache_key = hashlib.sha256(cache_key_raw.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"{symbol}_{cache_key}.pkl"
        if cache_file.exists():
            print(f"  Loading cached features from {cache_file}...")
            t0 = time.time()
            feat_df = pd.read_pickle(cache_file)
            print(f"  Cached features loaded in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")

    if feat_df is None:
        print("  Computing features for full dataset...")
        t0 = time.time()
        feat_df = _load_and_compute_features(symbol, df)
        if feat_df is None:
            print("  Feature computation failed")
            return
        print(f"  Features computed in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")
        if not no_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            feat_df.to_pickle(cache_file)
            print(f"  Features cached to {cache_file}")

    closes = feat_df["close"].values.astype(np.float64) if "close" in feat_df.columns else df["close"].values.astype(np.float64)
    volumes_all = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(df))

    # Available feature names (exclude close, non-feature columns)
    all_feature_names = [c for c in feat_df.columns
                         if c not in ("close", "timestamp", "open_time")
                         and c not in BLACKLIST]

    # Get timestamps for period labels
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    # Build per-fold kwargs
    lgbm_threads = max(1, os.cpu_count() // parallel) if parallel > 1 else None
    fold_kwargs_list = []
    for fold in folds:
        period = _fold_period_label(fold, timestamps, n_bars)
        if strategy_f:
            kw = dict(
                fold=fold, feat_df=feat_df, closes=closes,
                feature_names=all_feature_names,
                strategy_f=True, period=period,
                fixed_features=fixed_features,
                candidate_pool=candidate_pool, n_flexible=n_flexible,
                use_hpo=use_hpo, hpo_trials=hpo_trials,
                deadzone=deadzone, min_hold=min_hold,
                monthly_gate_window=monthly_gate_window,
                bear_thresholds=bear_thresholds,
                vol_target=vol_target, vol_feature=vol_feature_name,
                dd_limit=dd_limit, dd_cooldown=dd_cooldown,
                cost_model_type=cost_model_type, volumes=volumes_all,
                nn_ensemble=nn_ensemble, nn_seq_len=nn_seq_len, nn_epochs=nn_epochs,
                selector=selector, lgbm_threads=lgbm_threads,
                short_supplement=short_supplement,
                zscore_window=zscore_window, zscore_warmup=zscore_warmup,
            )
        else:
            kw = dict(
                fold=fold, feat_df=feat_df, closes=closes,
                feature_names=all_feature_names,
                strategy_f=False, period=period,
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
                monthly_gate_window=monthly_gate_window,
                vol_target=vol_target, vol_feature=vol_feature_name,
                dd_limit=dd_limit, dd_cooldown=dd_cooldown,
                cost_model_type=cost_model_type, volumes=volumes_all,
                nn_ensemble=nn_ensemble, nn_seq_len=nn_seq_len,
                nn_epochs=nn_epochs, selector=selector,
                lgbm_threads=lgbm_threads,
                zscore_window=zscore_window, zscore_warmup=zscore_warmup,
            )
        fold_kwargs_list.append(kw)

    # Run folds
    fold_results: List[FoldResult] = []

    if parallel <= 1:
        # Serial execution
        for kw in fold_kwargs_list:
            fold = kw["fold"]
            period = kw["period"]
            print(f"\n  Fold {fold.idx}: {period} "
                  f"(train={fold.train_end - fold.train_start}, "
                  f"test={fold.test_end - fold.test_start})")
            fold_idx, result, elapsed, _ = _run_single_fold(dict(kw))
            fold_results.append(result)
            print(f"    IC={result.ic:.4f}  Sharpe={result.sharpe:.2f}  "
                  f"Return={result.total_return*100:+.2f}%  "
                  f"Features={len(result.features)}  ({elapsed:.1f}s)")
    else:
        # Parallel execution (fork = CoW shared feat_df, zero copy)
        mp_ctx = "fork"
        if nn_ensemble:
            mp_ctx = "spawn"
            print("  WARNING: nn_ensemble uses spawn context — high memory usage")
        import multiprocessing as mp
        ctx = mp.get_context(mp_ctx)
        print(f"\n  Running {len(folds)} folds with {parallel} workers "
              f"(lgbm_threads={lgbm_threads})...")
        results_by_idx: Dict[int, FoldResult] = {}
        with ProcessPoolExecutor(max_workers=parallel, mp_context=ctx) as executor:
            futures = {
                executor.submit(_run_single_fold, dict(kw)): kw["fold"].idx
                for kw in fold_kwargs_list
            }
            for future in as_completed(futures):
                fold_idx, result, elapsed, period = future.result()
                results_by_idx[fold_idx] = result
                print(f"  Fold {fold_idx} ({period}): "
                      f"IC={result.ic:.4f}  Sharpe={result.sharpe:.2f}  "
                      f"Return={result.total_return*100:+.2f}%  ({elapsed:.1f}s)")
        # Sort by fold index
        fold_results = [results_by_idx[i] for i in sorted(results_by_idx)]

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
        "strategy_f": strategy_f,
        "bear_thresholds": bear_thresholds,
        "vol_target": vol_target,
        "dd_limit": dd_limit,
        "dd_cooldown": dd_cooldown,
        "nn_ensemble": nn_ensemble,
        "nn_seq_len": nn_seq_len,
        "nn_epochs": nn_epochs,
        "selector": selector,
        "parallel": parallel,
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

    suffix = "_strategy_f" if strategy_f else ""
    if nn_ensemble:
        suffix += "_nn"
    out_path = out_dir / f"wf_{symbol}{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
