#!/usr/bin/env python3
"""Train V7 Alpha — new data dimensions + multi-timeframe.

Key additions over V6:
  1. Spot-futures basis features (4): contango/backwardation signals
  2. Fear & Greed Index features (3): sentiment extremes
  3. 4h multi-timeframe features (10): slow regime signals
  4. New interaction features (3): basis×funding, basis×vol_regime, fgi×rsi

Usage:
    python3 -m scripts.train_v7_alpha --all
    python3 -m scripts.train_v7_alpha --symbol BTCUSDT --n-trials 20
    python3 -m scripts.train_v7_alpha --symbol BTCUSDT --target-mode clipped --horizon 12
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
from features.enriched_computer import ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import greedy_ic_select, _rankdata
from alpha.signal_transform import pred_to_signal as _pred_to_signal
from features.batch_feature_engine import compute_4h_features, TF4H_FEATURE_NAMES

from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
from research.hyperopt.search_space import SearchSpace, ParamRange
from research.overfit_detection import deflated_sharpe_ratio

try:
    from _quant_hotpath import cpp_bootstrap_sharpe_ci
    _BOOTSTRAP_CPP = True
except ImportError:
    _BOOTSTRAP_CPP = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

V7_DEFAULT_PARAMS = {
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

V7_SEARCH_SPACE = SearchSpace(
    name="lgbm_v7",
    int_params=(
        ParamRange("max_depth", 3, 7),
        ParamRange("num_leaves", 8, 32),
        ParamRange("min_child_samples", 50, 150),
    ),
    float_params=(
        ParamRange("learning_rate", 0.005, 0.05, log_scale=True),
        ParamRange("reg_alpha", 0.01, 1.0, log_scale=True),
        ParamRange("reg_lambda", 0.5, 5.0, log_scale=True),
        ParamRange("subsample", 0.5, 0.9),
        ParamRange("colsample_bytree", 0.5, 0.9),
    ),
)

EARLY_STOPPING_ROUNDS = 50
MIN_TRAIN = 2000

BLACKLIST = {
    "ret_1", "ret_3",
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    "avg_trade_size",
    "oi_chg_x_ret1",
    # Removed coins — features always NaN in live (2026-03-23)
    "dom_vs_sui_dev_20", "dom_vs_sui_ret_24",
    "dom_vs_axs_dev_20", "dom_vs_axs_ret_24",
    # Missing ETF data
    "xlf_ret_5d",
}

INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
    ("cvd_x_oi_chg", "cvd_20", "oi_change_pct"),
    ("vol_of_vol_x_range", "vol_of_vol", "range_vs_rv"),
    # V7 new interactions
    ("basis_x_funding", "basis", "funding_rate"),
    ("basis_x_vol_regime", "basis", "vol_regime"),
    ("fgi_x_rsi14", "fgi_normalized", "rsi_14"),
]

TARGET_MODES = ("raw", "clipped", "vol_norm", "binary")
DEFAULT_HORIZONS = (3, 6, 12, 24)


# _pred_to_signal imported from alpha.signal_transform (canonical, rolling z-score)


# ── Target variable (same as V6) ────────────────────────────

def _compute_target(
    closes: np.ndarray,
    horizon: int,
    mode: str,
    vol_window: int = 20,
) -> np.ndarray:
    n = len(closes)
    raw_ret = np.full(n, np.nan)
    for i in range(n - horizon):
        raw_ret[i] = closes[i + horizon] / closes[i] - 1.0

    if mode == "raw":
        return raw_ret

    if mode == "clipped":
        valid = raw_ret[~np.isnan(raw_ret)]
        if len(valid) < 10:
            return raw_ret
        p1, p99 = np.percentile(valid, [1, 99])
        return np.where(np.isnan(raw_ret), np.nan, np.clip(raw_ret, p1, p99))

    if mode == "vol_norm":
        pct = np.full(n, np.nan)
        for i in range(1, n):
            pct[i] = closes[i] / closes[i - 1] - 1.0

        vol = np.full(n, np.nan)
        for i in range(vol_window, n):
            window = pct[i - vol_window + 1:i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= vol_window // 2:
                vol[i] = np.std(valid, ddof=1)

        vol_valid = vol[~np.isnan(vol)]
        if len(vol_valid) == 0:
            return raw_ret
        return np.where(
            (~np.isnan(raw_ret)) & (~np.isnan(vol)) & (vol > 1e-12),
            raw_ret / vol,
            np.nan,
        )

    if mode == "binary":
        return np.where(np.isnan(raw_ret), np.nan,
                        (raw_ret > 0).astype(np.float64))

    raise ValueError(f"Unknown target mode: {mode}")


def _select_best_target(
    X: np.ndarray,
    y_dict: Dict[Tuple[int, str], np.ndarray],
    feature_names: List[str],
) -> Tuple[int, str]:
    n = X.shape[0]
    split = int(n * 0.8)
    X_eval = X[split:]

    best_score = -1.0
    best_key = (6, "clipped")

    for (horizon, mode), y in y_dict.items():
        y_eval = y[split:]
        valid = ~np.isnan(y_eval)
        if valid.sum() < 100:
            continue
        X_v = X_eval[valid]
        y_v = y_eval[valid]
        if np.std(y_v) < 1e-12:
            continue

        selected = greedy_ic_select(X_v, y_v, feature_names, top_k=10)
        sel_idx = [feature_names.index(n) for n in selected]
        if not sel_idx:
            continue

        ic_sum = 0.0
        for idx in sel_idx:
            col = X_v[:, idx]
            if np.std(col) < 1e-12:
                continue
            ic = abs(float(np.corrcoef(col, y_v)[0, 1]))
            if not np.isnan(ic):
                ic_sum += ic
        avg_ic = ic_sum / len(sel_idx)

        if avg_ic > best_score:
            best_score = avg_ic
            best_key = (horizon, mode)

    return best_key


# ── Regime feature ───────────────────────────────────────────

def _add_regime_feature(feat_df: pd.DataFrame) -> pd.DataFrame:
    if "vol_20" not in feat_df.columns:
        feat_df["regime_vol"] = 1
        return feat_df

    vol = feat_df["vol_20"].values
    valid = vol[~np.isnan(vol)]
    if len(valid) < 30:
        feat_df["regime_vol"] = 1
        return feat_df

    p33, p67 = np.percentile(valid, [33, 67])
    regime = np.where(np.isnan(vol), 1,
                      np.where(vol <= p33, 0,
                               np.where(vol <= p67, 1, 2)))
    feat_df["regime_vol"] = regime.astype(int)
    return feat_df


# ── Sample weights ───────────────────────────────────────────

def _compute_sample_weights(n: int, decay: float = 0.5) -> np.ndarray:
    return np.linspace(decay, 1.0, n)


# ── Walk-forward engine ──────────────────────────────────────

def expanding_window_folds(n: int, n_folds: int = 5, min_train: int = MIN_TRAIN):
    test_total = n - min_train
    if test_total <= 0 or n_folds <= 0:
        return []
    fold_size = test_total // n_folds
    if fold_size < 50:
        return []
    folds = []
    for i in range(n_folds):
        test_start = min_train + i * fold_size
        test_end = test_start + fold_size if i < n_folds - 1 else n
        folds.append((0, test_start, test_start, test_end))
    return folds


def _inner_cv_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    embargo_bars: int,
    inner_folds: int = 3,
) -> callable:
    import lightgbm as lgb

    n = len(X_train)
    fold_size = n // inner_folds

    def objective(params: Dict[str, Any]) -> float:
        lgbm_params = {
            **params,
            "n_estimators": 500,
            "objective": "regression",
            "verbosity": -1,
        }
        ics = []
        for fold_i in range(inner_folds):
            val_start = fold_i * fold_size
            val_end = val_start + fold_size if fold_i < inner_folds - 1 else n

            embargo_start = max(0, val_start - embargo_bars)
            embargo_end = min(n, val_end + embargo_bars)

            train_mask = np.ones(n, dtype=bool)
            train_mask[embargo_start:embargo_end] = False
            if train_mask.sum() < 100:
                continue

            X_tr = X_train[train_mask]
            y_tr = y_train[train_mask]
            w_tr = weights[train_mask]
            X_val = X_train[val_start:val_end]
            y_val = y_train[val_start:val_end]

            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            y_pred = model.predict(X_val)
            if len(y_pred) < 30 or np.std(y_pred) < 1e-12:
                continue

            rx = _rankdata(y_pred)
            ry = _rankdata(y_val)
            ic = float(np.corrcoef(rx, ry)[0, 1])
            if not np.isnan(ic):
                ics.append(ic)

        return float(np.mean(ics)) if ics else 0.0

    return objective


# ── OOS validation ───────────────────────────────────────────

def _validate_oos(
    model: LGBMAlphaModel,
    symbol: str,
    feature_names: List[str],
    target_horizon: int,
    target_mode: str,
    n_target_trials: int = 1,
) -> Optional[Dict[str, Any]]:
    oos_path = Path(f"data_files/{symbol}_1h_oos.csv")
    if not oos_path.exists():
        logger.warning("No OOS file for %s", symbol)
        return None

    oos_df = pd.read_csv(oos_path)
    feat_df = _load_and_compute_features(symbol, oos_df)
    if feat_df is None or len(feat_df) < 100:
        return None

    feat_df = _add_regime_feature(feat_df)

    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    target = _compute_target(closes, target_horizon, target_mode)

    X_oos = feat_df[feature_names].values.astype(np.float64)
    valid_mask = ~np.isnan(target)

    X_v = X_oos[valid_mask]
    y_v = target[valid_mask]
    closes_v = closes[np.where(valid_mask)[0]]

    if len(X_v) < 50:
        return None

    y_pred = model._model.predict(X_v)

    rx = _rankdata(y_pred)
    ry = _rankdata(y_v)
    ic = float(np.corrcoef(rx, ry)[0, 1])

    if target_mode == "binary":
        dir_acc = float(np.mean((y_pred > 0.5).astype(int) == y_v.astype(int)))
    else:
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_v)))

    ret_1bar = np.diff(closes_v) / closes_v[:-1]
    if len(ret_1bar) < len(y_pred):
        y_pred = y_pred[:len(ret_1bar)]

    pred_for_signal = y_pred[:len(ret_1bar)]
    signal = _pred_to_signal(pred_for_signal, target_mode=target_mode)
    turnover = np.abs(np.diff(signal, prepend=0))
    cost_per_bar = 6e-4  # 4bps fee + 2bps slippage
    net_pnl = signal * ret_1bar - turnover * cost_per_bar

    active = signal != 0
    sharpe = 0.0
    if active.sum() > 1:
        std_a = float(np.std(net_pnl[active], ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

    bootstrap_result = None
    if _BOOTSTRAP_CPP and active.sum() > 20:
        active_pnl = net_pnl[active].tolist()
        br = cpp_bootstrap_sharpe_ci(active_pnl, 5000, 5, 42)
        bootstrap_result = {
            "sharpe_mean": br.sharpe_mean,
            "ci_lower": br.sharpe_95ci_lo,
            "ci_upper": br.sharpe_95ci_hi,
            "p_gt_0": br.p_sharpe_gt_0,
        }

    dsr = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=max(n_target_trials, 1),
        n_observations=int(active.sum()),
    )

    passed = dir_acc > 0.52 and ic > 0.01

    result = {
        "n_samples": len(X_v),
        "spearman_ic": ic,
        "direction_accuracy": dir_acc,
        "sharpe": sharpe,
        "bootstrap": bootstrap_result,
        "deflated_sharpe": dsr.deflated_sharpe,
        "dsr_p_value": dsr.p_value,
        "dsr_significant": dsr.is_significant,
        "passed": passed,
    }
    return result


def _compute_split_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    closes: np.ndarray,
    target_mode: str,
) -> Dict[str, Any]:
    """Compute IC, dir_acc, sharpe for a slice of OOS data."""
    if len(y_pred) < 30:
        return {"n": len(y_pred), "ic": float("nan"), "dir_acc": float("nan"),
                "sharpe": float("nan")}

    rx = _rankdata(y_pred)
    ry = _rankdata(y_true)
    ic = float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 1 else 0.0

    if target_mode == "binary":
        dir_acc = float(np.mean(
            (y_pred > 0.5).astype(int) == y_true.astype(int)))
    else:
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))

    ret_1bar = np.diff(closes) / closes[:-1]
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

    cum_pnl = float(np.sum(net_pnl))
    active_pct = float(active.mean()) * 100

    return {
        "n": len(y_pred),
        "ic": ic,
        "dir_acc": dir_acc,
        "sharpe": sharpe,
        "cum_pnl": cum_pnl,
        "active_pct": active_pct,
    }


def _validate_oos_extended(
    model: LGBMAlphaModel,
    symbol: str,
    feature_names: List[str],
    target_horizon: int,
    target_mode: str,
    n_target_trials: int = 1,
) -> Optional[Dict[str, Any]]:
    """Extended OOS: H1/H2 split + monthly breakdown + rolling IC."""
    from datetime import datetime, timezone

    oos_path = Path(f"data_files/{symbol}_1h_oos.csv")
    if not oos_path.exists():
        logger.warning("No OOS file for %s", symbol)
        return None

    oos_df = pd.read_csv(oos_path)
    feat_df = _load_and_compute_features(symbol, oos_df)
    if feat_df is None or len(feat_df) < 100:
        return None

    feat_df = _add_regime_feature(feat_df)

    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    ts_col = "open_time" if "open_time" in oos_df.columns else "timestamp"
    timestamps = oos_df[ts_col].values.astype(np.int64)

    closes = feat_df["close"].values.astype(np.float64)
    target = _compute_target(closes, target_horizon, target_mode)
    X_oos = feat_df[feature_names].values.astype(np.float64)
    valid_mask = ~np.isnan(target)

    valid_idx = np.where(valid_mask)[0]
    X_v = X_oos[valid_idx]
    y_v = target[valid_idx]
    closes_v = closes[valid_idx]
    ts_v = timestamps[valid_idx]

    if len(X_v) < 100:
        return None

    y_pred = model._model.predict(X_v)

    # ── Overall metrics (same as _validate_oos) ──
    overall = _compute_split_metrics(y_pred, y_v, closes_v, target_mode)

    # ── H1/H2 split ──
    mid = len(y_pred) // 2
    h1 = _compute_split_metrics(
        y_pred[:mid], y_v[:mid], closes_v[:mid], target_mode)
    h2 = _compute_split_metrics(
        y_pred[mid:], y_v[mid:], closes_v[mid:], target_mode)

    # IC decay ratio
    ic_decay = h2["ic"] / h1["ic"] if abs(h1["ic"]) > 1e-6 else float("nan")

    # ── Monthly breakdown ──
    months_dt = np.array([
        datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in ts_v
    ])
    month_keys = np.array([f"{d.year}-{d.month:02d}" for d in months_dt])
    unique_months = sorted(set(month_keys))

    monthly = []
    for mk in unique_months:
        mask = month_keys == mk
        if mask.sum() < 20:
            continue
        m_metrics = _compute_split_metrics(
            y_pred[mask], y_v[mask], closes_v[mask], target_mode)
        m_metrics["month"] = mk
        monthly.append(m_metrics)

    # ── Rolling IC (720-bar window = ~1 month) ──
    rolling_window = 720
    rolling_ic = []
    for i in range(rolling_window, len(y_pred), rolling_window // 4):
        s = i - rolling_window
        chunk_pred = y_pred[s:i]
        chunk_true = y_v[s:i]
        rx = _rankdata(chunk_pred)
        ry = _rankdata(chunk_true)
        ic_val = float(np.corrcoef(rx, ry)[0, 1]) if len(rx) > 1 else 0.0
        bar_ts = int(ts_v[i - 1])
        rolling_ic.append({"bar_idx": i, "timestamp": bar_ts, "ic": ic_val})

    # ── Bootstrap & DSR on overall ──
    ret_1bar = np.diff(closes_v) / closes_v[:-1]
    pred_trunc = y_pred[:len(ret_1bar)]
    signal = _pred_to_signal(pred_trunc, target_mode=target_mode)
    turnover = np.abs(np.diff(signal, prepend=0))
    net_pnl = signal * ret_1bar - turnover * 4e-4
    active = signal != 0

    bootstrap_result = None
    if _BOOTSTRAP_CPP and active.sum() > 20:
        active_pnl = net_pnl[active].tolist()
        br = cpp_bootstrap_sharpe_ci(active_pnl, 5000, 5, 42)
        bootstrap_result = {
            "sharpe_mean": br.sharpe_mean,
            "ci_lower": br.sharpe_95ci_lo,
            "ci_upper": br.sharpe_95ci_hi,
            "p_gt_0": br.p_sharpe_gt_0,
        }

    dsr = deflated_sharpe_ratio(
        observed_sharpe=overall["sharpe"],
        n_trials=max(n_target_trials, 1),
        n_observations=int(active.sum()),
    )

    # ── Stability score (0-100) ──
    # Penalize: IC decay, H1/H2 divergence, negative months
    sum(1 for m in monthly if m["ic"] < 0)
    pos_months = sum(1 for m in monthly if m["ic"] > 0)
    month_consistency = pos_months / max(len(monthly), 1)

    stability_score = 0.0
    if not np.isnan(ic_decay):
        # ic_decay near 1.0 = stable, <0.5 = decaying, >1.5 = suspicious
        decay_score = max(0, 1.0 - abs(ic_decay - 1.0)) * 40
        stability_score += decay_score
    stability_score += month_consistency * 40
    if overall["ic"] > 0.02:
        stability_score += 20
    stability_score = min(100, stability_score)

    passed = (overall["dir_acc"] > 0.52 and overall["ic"] > 0.01
              and h2["ic"] > 0)

    return {
        "overall": overall,
        "h1": h1,
        "h2": h2,
        "ic_decay_ratio": ic_decay,
        "monthly": monthly,
        "rolling_ic": rolling_ic,
        "bootstrap": bootstrap_result,
        "deflated_sharpe": dsr.deflated_sharpe,
        "dsr_p_value": dsr.p_value,
        "dsr_significant": dsr.is_significant,
        "stability_score": stability_score,
        "passed": passed,
    }


def _print_extended_oos(symbol: str, ext: Dict[str, Any]) -> None:
    """Pretty-print extended OOS results."""

    status = "PASS" if ext["passed"] else "FAIL"
    print(f"\n  {'='*60}")
    print(f"  Extended OOS Validation: {symbol} [{status}]")
    print(f"  {'='*60}")

    o, h1, h2 = ext["overall"], ext["h1"], ext["h2"]
    print(f"\n  {'Period':<12} {'N':>6} {'IC':>8} {'Dir%':>8} "
          f"{'Sharpe':>8} {'Active%':>8} {'CumPnL':>10}")
    print(f"  {'-'*62}")
    for label, m in [("Overall", o), ("H1 (early)", h1), ("H2 (late)", h2)]:
        ic_s = f"{m['ic']:.4f}" if not np.isnan(m["ic"]) else "NaN"
        da_s = f"{m['dir_acc']*100:.1f}" if not np.isnan(m["dir_acc"]) else "NaN"
        sh_s = f"{m['sharpe']:.2f}" if not np.isnan(m["sharpe"]) else "NaN"
        ac_s = f"{m.get('active_pct', 0):.0f}" if not np.isnan(m.get("active_pct", 0)) else "NaN"
        cp_s = f"{m.get('cum_pnl', 0):.6f}" if not np.isnan(m.get("cum_pnl", 0)) else "NaN"
        print(f"  {label:<12} {m['n']:>6} {ic_s:>8} {da_s:>8} "
              f"{sh_s:>8} {ac_s:>8} {cp_s:>10}")

    decay = ext["ic_decay_ratio"]
    decay_s = f"{decay:.2f}" if not np.isnan(decay) else "NaN"
    verdict = ""
    if not np.isnan(decay):
        if decay > 0.8:
            verdict = "STABLE"
        elif decay > 0.5:
            verdict = "MILD DECAY"
        elif decay > 0:
            verdict = "DECAYING"
        else:
            verdict = "REVERSED"
    print(f"\n  IC decay ratio (H2/H1): {decay_s}  [{verdict}]")
    print(f"  Stability score: {ext['stability_score']:.0f}/100")

    # Monthly table
    monthly = ext.get("monthly", [])
    if monthly:
        print("\n  Monthly Breakdown:")
        print(f"  {'Month':<10} {'N':>6} {'IC':>8} {'Dir%':>8} "
              f"{'Sharpe':>8} {'Active%':>8}")
        print(f"  {'-'*50}")
        for m in monthly:
            ic_s = f"{m['ic']:.4f}" if not np.isnan(m["ic"]) else "NaN"
            da_s = f"{m['dir_acc']*100:.1f}" if not np.isnan(m["dir_acc"]) else "NaN"
            sh_s = f"{m['sharpe']:.2f}" if not np.isnan(m["sharpe"]) else "NaN"
            ac_s = f"{m.get('active_pct', 0):.0f}"
            print(f"  {m['month']:<10} {m['n']:>6} {ic_s:>8} {da_s:>8} "
                  f"{sh_s:>8} {ac_s:>8}")
        pos_months = sum(1 for m in monthly if m["ic"] > 0)
        print(f"  IC positive months: {pos_months}/{len(monthly)}")

    # Bootstrap
    if ext.get("bootstrap"):
        bs = ext["bootstrap"]
        print(f"\n  Bootstrap Sharpe: mean={bs['sharpe_mean']:.2f} "
              f"CI=[{bs['ci_lower']:.2f}, {bs['ci_upper']:.2f}] "
              f"P(>0)={bs['p_gt_0']:.2f}")
    print(f"  DSR: {ext['deflated_sharpe']:.4f} "
          f"(p={ext['dsr_p_value']:.4f}, sig={ext['dsr_significant']})")

    # Rolling IC summary
    ric = ext.get("rolling_ic", [])
    if ric:
        ic_vals = [r["ic"] for r in ric]
        print(f"\n  Rolling IC (720-bar window): "
              f"mean={np.mean(ic_vals):.4f} "
              f"std={np.std(ic_vals):.4f} "
              f"min={np.min(ic_vals):.4f} "
              f"max={np.max(ic_vals):.4f}")
        neg_pct = sum(1 for v in ic_vals if v < 0) / len(ic_vals) * 100
        print(f"  Rolling IC negative windows: {neg_pct:.0f}%")


# ── Data loading ─────────────────────────────────────────────

def _load_and_compute_features(
    symbol: str,
    df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Compute features from raw OHLCV dataframe (V7: +basis +FGI +4h)."""
    from features.batch_feature_engine import compute_features_batch

    # V11: use Python path if macro data exists (C++ doesn't have V11 features)
    from pathlib import Path as _Path
    _has_v11 = _Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)

    # Cross-asset features (for non-BTC symbols)
    if cross_df is not None and symbol != "BTCUSDT":
        ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
        timestamps = df[ts_col].values.astype(np.int64)
        # Vectorized merge: reindex cross_df to match bar timestamps
        cross_aligned = cross_df.reindex(timestamps)
        for name in CROSS_ASSET_FEATURE_NAMES:
            if name in cross_aligned.columns:
                feat_df[name] = cross_aligned[name].values
            else:
                feat_df[name] = np.nan
    elif symbol != "BTCUSDT":
        for name in CROSS_ASSET_FEATURE_NAMES:
            feat_df[name] = np.nan

    for int_name, feat_a, feat_b in INTERACTION_FEATURES:
        if feat_a in feat_df.columns and feat_b in feat_df.columns:
            feat_df[int_name] = feat_df[feat_a].astype(float) * feat_df[feat_b].astype(float)
        else:
            feat_df[int_name] = np.nan

    feat_df["close"] = df["close"].values

    # V7: merge 4h multi-timeframe features
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values

    # V13: Merge OI-derived features from downloaded data
    _merge_v13_oi_features(symbol, df, feat_df)

    # V19: Merge DVOL implied volatility features (disabled: hurts OOS Sharpe, see validation below)
    # Walk-forward result: baseline Sharpe 1.12 → with DVOL 0.45 (14→12 positive folds)
    # DVOL features have high in-sample IC but overfit out-of-sample
    _merge_v19_dvol_features(symbol, df, feat_df)

    return feat_df


def _merge_v13_oi_features(symbol: str, df: pd.DataFrame, feat_df: pd.DataFrame) -> None:
    """Add V13 OI/LS/Taker features + V18 OI change rate features.

    Features: oi_pct_4h, ls_deviation, taker_buy_sell_ratio,
    top_retail_divergence, oi_price_divergence_12,
    oi_change_24, oi_change_96, funding_cum_3.
    Safe: fills NaN if OI data not available.
    """
    from pathlib import Path as _P

    oi_path = _P(f"data_files/{symbol}_oi_1h.csv")
    v13_names = ["oi_pct_4h", "ls_deviation", "taker_buy_sell_ratio",
                 "top_retail_divergence", "oi_price_divergence_12",
                 "oi_change_24", "oi_change_96"]

    if not oi_path.exists():
        for name in v13_names:
            feat_df[name] = np.nan
        # V18: funding_cum_3 from funding_rate column (independent of OI data)
        _add_funding_cum_3(feat_df)
        return

    oi_df = pd.read_csv(oi_path)
    oi_df["timestamp"] = pd.to_numeric(oi_df["timestamp"])

    # Align OI data to price bars by timestamp
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    bar_ts = df[ts_col].values.astype(np.int64)

    # Build OI lookup (timestamp -> row)
    oi_lookup = dict(zip(oi_df["timestamp"].values, range(len(oi_df))))

    oi_vals = oi_df["oi"].values if "oi" in oi_df.columns else np.full(len(oi_df), np.nan)
    ls_vals = oi_df["ls_ratio"].values if "ls_ratio" in oi_df.columns else np.full(len(oi_df), np.nan)
    taker_buy = oi_df["taker_buy_vol"].values if "taker_buy_vol" in oi_df.columns else np.zeros(len(oi_df))
    taker_sell = oi_df["taker_sell_vol"].values if "taker_sell_vol" in oi_df.columns else np.zeros(len(oi_df))
    top_ls = oi_df["top_ls_ratio"].values if "top_ls_ratio" in oi_df.columns else np.full(len(oi_df), np.nan)

    n = len(bar_ts)
    closes = df["close"].values.astype(np.float64)

    # Pre-allocate
    f_oi_pct_4h = np.full(n, np.nan)
    f_ls_dev = np.full(n, np.nan)
    f_taker_ratio = np.full(n, np.nan)
    f_top_retail = np.full(n, np.nan)
    f_oi_price_div = np.full(n, np.nan)

    # Map bar timestamps to OI indices
    oi_idx = np.array([oi_lookup.get(ts, -1) for ts in bar_ts])

    for i in range(n):
        idx = oi_idx[i]
        if idx < 0:
            continue

        # ls_deviation
        ls = ls_vals[idx]
        if not np.isnan(ls):
            f_ls_dev[i] = ls - 1.0

        # taker_buy_sell_ratio
        tb = taker_buy[idx]
        ts_val = taker_sell[idx]
        if tb > 0 and ts_val > 0:
            f_taker_ratio[i] = tb / ts_val

        # top_retail_divergence
        tls = top_ls[idx]
        if not np.isnan(tls) and not np.isnan(ls):
            f_top_retail[i] = tls - ls

        # oi_pct_4h (need idx-4)
        if idx >= 4:
            oi_cur = oi_vals[idx]
            oi_prev = oi_vals[idx - 4]
            if not np.isnan(oi_cur) and not np.isnan(oi_prev) and oi_prev > 0:
                f_oi_pct_4h[i] = (oi_cur - oi_prev) / oi_prev

        # oi_price_divergence_12 (need idx-12 and i-12)
        if idx >= 12 and i >= 12:
            oi_cur = oi_vals[idx]
            oi_old = oi_vals[idx - 12]
            if not np.isnan(oi_cur) and not np.isnan(oi_old) and oi_old > 0 and closes[i - 12] > 0:
                oi_chg = (oi_cur - oi_old) / oi_old
                price_chg = (closes[i] - closes[i - 12]) / closes[i - 12]
                f_oi_price_div[i] = oi_chg - price_chg

    feat_df["oi_pct_4h"] = f_oi_pct_4h
    feat_df["ls_deviation"] = f_ls_dev
    feat_df["taker_buy_sell_ratio"] = f_taker_ratio
    feat_df["top_retail_divergence"] = f_top_retail
    feat_df["oi_price_divergence_12"] = f_oi_price_div

    # --- V18: OI change rate (24-bar and 96-bar) ---
    f_oi_change_24 = np.full(n, np.nan)
    f_oi_change_96 = np.full(n, np.nan)
    for i in range(n):
        idx = oi_idx[i]
        if idx < 0:
            continue
        # oi_change_24
        if idx >= 24:
            oi_cur = oi_vals[idx]
            oi_prev = oi_vals[idx - 24]
            if not np.isnan(oi_cur) and not np.isnan(oi_prev) and oi_prev > 0:
                f_oi_change_24[i] = (oi_cur - oi_prev) / oi_prev
        # oi_change_96
        if idx >= 96:
            oi_cur = oi_vals[idx]
            oi_prev = oi_vals[idx - 96]
            if not np.isnan(oi_cur) and not np.isnan(oi_prev) and oi_prev > 0:
                f_oi_change_96[i] = (oi_cur - oi_prev) / oi_prev
    feat_df["oi_change_24"] = f_oi_change_24
    feat_df["oi_change_96"] = f_oi_change_96

    # V18: funding_cum_3
    _add_funding_cum_3(feat_df)


def _add_funding_cum_3(feat_df: pd.DataFrame) -> None:
    """Add 3-period cumulative funding rate feature.

    Uses funding_rate column if available (computed by batch feature engine).
    Rolling sum of last 3 funding rate observations.
    """
    if "funding_rate" in feat_df.columns:
        fr = feat_df["funding_rate"].astype(float)
        feat_df["funding_cum_3"] = fr.rolling(3, min_periods=3).sum()
    else:
        feat_df["funding_cum_3"] = np.nan


def _merge_v19_dvol_features(symbol: str, df: pd.DataFrame, feat_df: pd.DataFrame) -> None:
    """Add V19 DVOL implied volatility features from Deribit DVOL data.

    Features: dvol_chg_72, iv_term_struct, dvol_z, dvol_chg_24, dvol_mean_rev.
    Safe: fills NaN if DVOL data not available.
    """
    from pathlib import Path as _P

    v19_names = ["dvol_chg_72", "iv_term_struct", "dvol_z", "dvol_chg_24", "dvol_mean_rev"]

    # Only BTC has DVOL data from Deribit
    dvol_path = _P(f"data_files/{symbol}_dvol_1h.csv")
    if not dvol_path.exists():
        for name in v19_names:
            feat_df[name] = np.nan
        return

    dvol_df = pd.read_csv(dvol_path)
    dvol_df["timestamp"] = pd.to_numeric(dvol_df["timestamp"])
    # Deduplicate by timestamp (keep last)
    dvol_df = dvol_df.drop_duplicates(subset="timestamp", keep="last")
    dvol_df = dvol_df.sort_values("timestamp").reset_index(drop=True)

    dvol_vals = dvol_df["close"].values.astype(np.float64)
    dvol_ts = dvol_df["timestamp"].values.astype(np.int64)

    # Build lookup: timestamp -> index
    dvol_lookup = dict(zip(dvol_ts, range(len(dvol_ts))))

    # Align to bar timestamps
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    bar_ts = df[ts_col].values.astype(np.int64)
    n = len(bar_ts)

    # Map bar timestamps to DVOL indices
    dvol_idx = np.array([dvol_lookup.get(ts, -1) for ts in bar_ts])

    # Pre-allocate
    f_chg_72 = np.full(n, np.nan)
    f_term_struct = np.full(n, np.nan)
    f_dvol_z = np.full(n, np.nan)
    f_chg_24 = np.full(n, np.nan)
    f_mean_rev = np.full(n, np.nan)

    # Vectorized computation using pandas Series on the full dvol_vals array
    dvol_s = pd.Series(dvol_vals)

    # Pre-compute rolling stats on the full DVOL series
    dvol_pct_72 = dvol_s.pct_change(72).values
    dvol_pct_24 = dvol_s.pct_change(24).values
    dvol_ma_24 = dvol_s.rolling(24, min_periods=24).mean().values
    dvol_ma_168 = dvol_s.rolling(168, min_periods=168).mean().values
    dvol_std_168 = dvol_s.rolling(168, min_periods=168).std().values
    dvol_ma_720 = dvol_s.rolling(720, min_periods=720).mean().values

    for i in range(n):
        idx = dvol_idx[i]
        if idx < 0:
            continue

        # dvol_chg_72
        if idx >= 72:
            f_chg_72[i] = dvol_pct_72[idx]

        # iv_term_struct: MA(24) / MA(168) - 1
        if idx >= 167:
            ma_long = dvol_ma_168[idx]
            ma_short = dvol_ma_24[idx]
            if not np.isnan(ma_long) and ma_long > 0 and not np.isnan(ma_short):
                f_term_struct[i] = ma_short / ma_long - 1

        # dvol_z: z-score over 168 bars
        if idx >= 167:
            mu = dvol_ma_168[idx]
            std = dvol_std_168[idx]
            if not np.isnan(mu) and not np.isnan(std):
                f_dvol_z[i] = (dvol_vals[idx] - mu) / max(std, 0.1)

        # dvol_chg_24
        if idx >= 24:
            f_chg_24[i] = dvol_pct_24[idx]

        # dvol_mean_rev: DVOL / MA(720) - 1
        if idx >= 719:
            ma720 = dvol_ma_720[idx]
            if not np.isnan(ma720) and ma720 > 0:
                f_mean_rev[i] = dvol_vals[idx] / ma720 - 1

    feat_df["dvol_chg_72"] = f_chg_72
    feat_df["iv_term_struct"] = f_term_struct
    feat_df["dvol_z"] = f_dvol_z
    feat_df["dvol_chg_24"] = f_chg_24
    feat_df["dvol_mean_rev"] = f_mean_rev


# ── Cross-asset feature building ─────────────────────────────

def _build_cross_features(symbols: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
    from features.batch_cross_asset import build_cross_features_batch
    return build_cross_features_batch(symbols)


# ── Main walk-forward engine ─────────────────────────────────

def get_available_features(symbol: str) -> List[str]:
    names = [n for n in ENRICHED_FEATURE_NAMES if n not in BLACKLIST]
    # V7: add 4h multi-timeframe features
    names.extend(TF4H_FEATURE_NAMES)
    if symbol != "BTCUSDT":
        names.extend(n for n in CROSS_ASSET_FEATURE_NAMES if n not in BLACKLIST)
    names.extend(name for name, _, _ in INTERACTION_FEATURES if name not in BLACKLIST)
    names.append("regime_vol")
    return names


def run_walkforward(
    symbol: str,
    feat_df: pd.DataFrame,
    available: List[str],
    out_dir: Path,
    *,
    top_k: int = 25,
    horizon: int = 0,
    target_mode: str = "",
    n_folds: int = 5,
    n_trials: int = 20,
) -> Optional[dict]:
    """Run V7 walk-forward training for one symbol."""
    closes = feat_df["close"].values.astype(np.float64)

    available = [n for n in available if n not in BLACKLIST]
    X_all = feat_df[available]
    all_nan_cols = set(X_all.columns[X_all.isna().all()])
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN: {sorted(all_nan_cols)}")
        available = [c for c in available if c not in all_nan_cols]

    SPARSE = {"oi_change_pct", "oi_change_ma8", "oi_close_divergence",
              "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
              "btc_ret1_x_beta30",
              "oi_acceleration", "leverage_proxy", "oi_vol_divergence",
              "oi_liquidation_flag", "cvd_x_oi_chg",
              "funding_rate", "funding_ma8", "funding_zscore_24",
              "funding_momentum", "funding_extreme", "funding_cumulative_8",
              "funding_sign_persist", "funding_annualized", "funding_vs_vol",
              "funding_x_taker_imb",
              # V7: basis/FGI may be sparse if no spot/FGI data
              "basis", "basis_zscore_24", "basis_momentum", "basis_extreme",
              "basis_x_funding", "basis_x_vol_regime",
              "fgi_normalized", "fgi_zscore_7", "fgi_extreme", "fgi_x_rsi14",
              # External data — may not be available
              "spx_overnight_ret", "mempool_size_zscore_24",
              "liquidation_cascade_score",
              # V18: OI change rate (sparse — only available where OI data exists)
              "oi_change_24", "oi_change_96",
              # V18: 3-period cumulative funding
              "funding_cum_3"}
    core = [c for c in available if c not in SPARSE]
    X_full = feat_df[available].values.astype(np.float64)

    core_idx = [available.index(c) for c in core]
    valid_mask = np.ones(len(X_full), dtype=bool)
    for ci in core_idx:
        valid_mask &= ~np.isnan(X_full[:, ci])

    # ── Target selection ──
    n_target_trials = 1
    if horizon > 0 and target_mode:
        target = _compute_target(closes, horizon, target_mode)
        valid_mask &= ~np.isnan(target)
    else:
        print("  Scanning target modes...")
        y_dict: Dict[Tuple[int, str], np.ndarray] = {}
        for h in DEFAULT_HORIZONS:
            for m in TARGET_MODES:
                y_dict[(h, m)] = _compute_target(closes, h, m)

        proxy_y = y_dict[(6, "raw")]
        valid_mask &= ~np.isnan(proxy_y)
        valid_indices_proxy = np.where(valid_mask)[0]
        X_clean_proxy = X_full[valid_indices_proxy]

        y_dict_clean = {}
        for key, y in y_dict.items():
            y_dict_clean[key] = y[valid_indices_proxy]

        best_key = _select_best_target(X_clean_proxy, y_dict_clean, available)
        horizon, target_mode = best_key
        n_target_trials = len(DEFAULT_HORIZONS) * len(TARGET_MODES)
        print(f"  Best target: horizon={horizon}, mode={target_mode}")

        target = _compute_target(closes, horizon, target_mode)
        valid_mask = np.ones(len(X_full), dtype=bool)
        for ci in core_idx:
            valid_mask &= ~np.isnan(X_full[:, ci])
        valid_mask &= ~np.isnan(target)

    valid_indices = np.where(valid_mask)[0]
    X_clean = X_full[valid_indices]
    y_clean = target[valid_indices]

    print(f"  Valid samples: {len(X_clean)} / {len(X_full)}")
    if len(X_clean) < MIN_TRAIN:
        print(f"  ERROR: Not enough valid samples ({len(X_clean)} < {MIN_TRAIN})")
        return None

    embargo_bars = horizon + 2

    folds = expanding_window_folds(len(X_clean), n_folds=n_folds, min_train=MIN_TRAIN)
    if not folds:
        print("  ERROR: Could not create folds")
        return None

    print(f"  Walk-forward: {len(folds)} folds, horizon={horizon}, "
          f"mode={target_mode}, embargo={embargo_bars}")

    fold_results = []
    feature_selection_counts: Dict[str, int] = {}

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_clean[tr_start:tr_end]
        y_train = y_clean[tr_start:tr_end]
        X_test = X_clean[te_start:te_end]
        y_test = y_clean[te_start:te_end]

        selected = greedy_ic_select(X_train, y_train, available, top_k=top_k)
        sel_idx = [available.index(n) for n in selected]

        for name in selected:
            feature_selection_counts[name] = feature_selection_counts.get(name, 0) + 1

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        weights = _compute_sample_weights(len(X_tr_sel))

        if n_trials > 0:
            obj_fn = _inner_cv_objective(
                X_tr_sel, y_train, weights,
                embargo_bars=embargo_bars,
                inner_folds=3,
            )
            optimizer = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=obj_fn,
                config=HyperOptConfig(
                    n_trials=n_trials,
                    direction="maximize",
                    seed=42 + fi,
                    pruner_patience=10,
                    pruner_min_trials=5,
                ),
            )
            hpo_result = optimizer.optimize()
            fold_params = {**V7_DEFAULT_PARAMS, **hpo_result.best_params}
            print(f"  Fold {fi} HPO: best_ic={hpo_result.best_value:.4f} "
                  f"({hpo_result.n_trials} trials)")
        else:
            fold_params = V7_DEFAULT_PARAMS.copy()

        model = LGBMAlphaModel(name=f"v7_fold{fi}", feature_names=tuple(selected))
        metrics = model.fit(
            X_tr_sel, y_train,
            params=fold_params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=embargo_bars,
            sample_weight=weights,
        )

        y_pred = model._model.predict(X_te_sel)
        if target_mode == "binary":
            dir_acc = float(np.mean(
                (y_pred > 0.5).astype(int) == y_test.astype(int)))
        else:
            dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

        rx = _rankdata(y_pred)
        ry = _rankdata(y_test)
        ic = float(np.corrcoef(rx, ry)[0, 1]) if len(y_pred) > 1 else 0.0

        test_orig = valid_indices[te_start:te_end]
        ret_1bar = np.full(len(test_orig), np.nan)
        for i, idx in enumerate(test_orig):
            if idx + 1 < len(closes):
                ret_1bar[i] = closes[idx + 1] / closes[idx] - 1.0

        valid_ret = ~np.isnan(ret_1bar)
        sharpe = 0.0
        if valid_ret.sum() > 1:
            signal = _pred_to_signal(y_pred[valid_ret], target_mode=target_mode)
            turnover = np.abs(np.diff(signal, prepend=0))
            net_pnl = signal * ret_1bar[valid_ret] - turnover * 4e-4
            active = signal != 0
            n_active = int(active.sum())
            if n_active > 1:
                std_a = float(np.std(net_pnl[active], ddof=1))
                if std_a > 0:
                    sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

        best_iter = metrics.get("best_iteration", V7_DEFAULT_PARAMS["n_estimators"])

        fold_result = {
            "fold": fi,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "direction_accuracy": dir_acc,
            "ic": ic,
            "sharpe": sharpe,
            "best_iteration": int(best_iter),
            "n_features": len(selected),
            "selected_features": selected,
            "hpo_params": fold_params if n_trials > 0 else None,
        }
        fold_results.append(fold_result)

        print(f"  Fold {fi}: train={len(X_train):,} test={len(X_test):,} "
              f"IC={ic:.4f} dir={dir_acc:.3f} sharpe={sharpe:.2f} "
              f"best_iter={int(best_iter)}")

    # ── Cross-fold summary ──
    avg_ic = float(np.mean([f["ic"] for f in fold_results]))
    avg_dir = float(np.mean([f["direction_accuracy"] for f in fold_results]))
    avg_sharpe = float(np.mean([f["sharpe"] for f in fold_results]))

    print(f"\n  Cross-fold avg: IC={avg_ic:.4f} dir={avg_dir:.3f} sharpe={avg_sharpe:.2f}")

    stability_threshold = max(n_folds * 4 // 5, 3)
    stable_features = {k: v for k, v in feature_selection_counts.items()
                       if v >= stability_threshold}
    unstable_features = {k: v for k, v in feature_selection_counts.items()
                         if k not in stable_features}
    print(f"\n  Feature stability: {len(stable_features)} stable "
          f"(>={stability_threshold}/{n_folds} folds), "
          f"{len(unstable_features)} unstable")
    for name, count in sorted(stable_features.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:<30s} {count}/{n_folds} folds")

    # ── Final model: train on all data ──
    print("\n  Training final model on all data...")
    final_features = sorted(stable_features.keys(),
                            key=lambda k: -feature_selection_counts[k])[:top_k]
    if len(final_features) < 5:
        final_features = sorted(feature_selection_counts.keys(),
                                key=lambda k: -feature_selection_counts[k])[:top_k]
    final_idx = [available.index(n) for n in final_features]

    final_params = fold_results[-1].get("hpo_params") or V7_DEFAULT_PARAMS.copy()
    final_weights = _compute_sample_weights(len(X_clean))

    final_model = LGBMAlphaModel(name="lgbm_v7_alpha", feature_names=tuple(final_features))
    final_metrics = final_model.fit(
        X_clean[:, final_idx], y_clean,
        params=final_params,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        embargo_bars=embargo_bars,
        sample_weight=final_weights,
    )
    print(f"  Final model: {len(final_features)} features, "
          f"best_iter={int(final_metrics.get('best_iteration', -1))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(out_dir / "lgbm_v7_alpha.pkl")

    # ── OOS validation (basic) ──
    oos_result = _validate_oos(
        final_model, symbol, final_features,
        target_horizon=horizon,
        target_mode=target_mode,
        n_target_trials=n_target_trials,
    )
    if oos_result:
        status = "PASS" if oos_result["passed"] else "FAIL"
        print(f"\n  OOS validation: {status}")
        print(f"    IC={oos_result['spearman_ic']:.4f} "
              f"dir={oos_result['direction_accuracy']:.3f} "
              f"sharpe={oos_result['sharpe']:.2f}")
        if oos_result.get("bootstrap"):
            bs = oos_result["bootstrap"]
            print(f"    Bootstrap: mean={bs['sharpe_mean']:.2f} "
                  f"CI=[{bs['ci_lower']:.2f}, {bs['ci_upper']:.2f}] "
                  f"P(>0)={bs['p_gt_0']:.2f}")
        print(f"    DSR: {oos_result['deflated_sharpe']:.4f} "
              f"(p={oos_result['dsr_p_value']:.4f}, "
              f"sig={oos_result['dsr_significant']})")

    # ── Extended OOS validation ──
    oos_extended = _validate_oos_extended(
        final_model, symbol, final_features,
        target_horizon=horizon,
        target_mode=target_mode,
        n_target_trials=n_target_trials,
    )
    if oos_extended:
        _print_extended_oos(symbol, oos_extended)

    # ── Build result dict ──
    result = {
        "symbol": symbol,
        "version": "v7",
        "horizon": horizon,
        "target_mode": target_mode,
        "n_folds": n_folds,
        "top_k": top_k,
        "n_trials": n_trials,
        "embargo_bars": embargo_bars,
        "min_train": MIN_TRAIN,
        "n_features_available": len(available),
        "n_features_final": len(final_features),
        "final_features": final_features,
        "cv_ic": avg_ic,
        "cv_direction_accuracy": avg_dir,
        "cv_sharpe": avg_sharpe,
        "fold_results": fold_results,
        "feature_stability": {k: v for k, v in sorted(
            feature_selection_counts.items(), key=lambda x: -x[1])},
        "stable_features": list(stable_features.keys()),
        "unstable_features": list(unstable_features.keys()),
        "oos_validation": oos_result,
        "oos_extended": oos_extended,
        "default_params": V7_DEFAULT_PARAMS,
    }

    with open(out_dir / "v7_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Entrypoint ───────────────────────────────────────────────

def run_one(
    symbol: str,
    out_base: Path,
    *,
    top_k: int = 25,
    horizon: int = 0,
    target_mode: str = "",
    n_folds: int = 5,
    n_trials: int = 20,
    cross_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[dict]:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — V7 Alpha Training")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    cross_df = cross_features.get(symbol) if cross_features else None
    feat_df = _load_and_compute_features(symbol, df, cross_df)
    if feat_df is None:
        print("  ERROR: Feature computation failed")
        return None
    print(f"  Bars: {len(feat_df)}")

    feat_df = _add_regime_feature(feat_df)

    available = get_available_features(symbol)
    available = [n for n in available if n in feat_df.columns]
    print(f"  Available features (after blacklist): {len(available)}")

    out_dir = out_base / symbol
    return run_walkforward(
        symbol, feat_df, available, out_dir,
        top_k=top_k, horizon=horizon, target_mode=target_mode,
        n_folds=n_folds, n_trials=n_trials,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V7 Alpha — New Data Dimensions + Multi-Timeframe")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models_v7", help="Output directory")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K features per fold")
    parser.add_argument("--horizon", type=int, default=0,
                        help="Target horizon (0 = auto-select)")
    parser.add_argument("--target-mode", default="",
                        help="Target mode: raw/clipped/vol_norm/binary (empty = auto)")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per fold")
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
        r = run_one(sym, out_base, top_k=args.top_k, horizon=args.horizon,
                    target_mode=args.target_mode,
                    n_folds=args.n_folds, n_trials=args.n_trials,
                    cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*100}")
        print("  V7 Alpha Training Summary")
        print(f"{'='*100}")
        print(f"{'Symbol':<10} {'H':>3} {'Mode':<10} {'Feats':>5} "
              f"{'CV IC':>8} {'CV Dir':>8} {'CV Shrp':>8} {'OOS':>5}")
        print(f"{'-'*65}")
        for sym, r in results.items():
            oos_str = "N/A"
            if r.get("oos_validation"):
                oos_str = "PASS" if r["oos_validation"]["passed"] else "FAIL"
            print(f"{sym:<10} {r['horizon']:>3} {r['target_mode']:<10} "
                  f"{r['n_features_final']:>5} "
                  f"{r['cv_ic']:>8.4f} {r['cv_direction_accuracy']*100:>7.1f}% "
                  f"{r['cv_sharpe']:>8.2f} {oos_str:>5}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
