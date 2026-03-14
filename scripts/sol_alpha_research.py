#!/usr/bin/env python3
"""SOL Alpha Deep Research.

5-phase analysis to understand why SOL's IC-Sharpe correlation ≈ 0,
diagnose disaster folds, and optimize SOL-specific alpha.

Phases:
  1) Per-Fold Feature IC Breakdown
  2) SOL vs BTC Funding/Basis Comparison
  3) Regime Analysis (disaster fold diagnosis)
  4) IC-Return Disconnect Investigation (signal timing, nonlinear, fat tails)
  5) SOL Feature Optimization (greedy, fixed search, candidate expansion, HPO grid)

Usage:
    python3 -m scripts.sol_alpha_research                    # all 5 phases
    python3 -m scripts.sol_alpha_research --phases 1 2 3     # quick diagnostics (~2min)
    python3 -m scripts.sol_alpha_research --skip-phase5      # skip slowest phase
    python3 -m scripts.sol_alpha_research --quick             # 5-fold mini-WF
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS,
    _compute_target,
    _load_and_compute_features,
    BLACKLIST,
)
from scripts.backtest_alpha_v8 import _pred_to_signal, _apply_monthly_gate, COST_PER_TRADE
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

MIN_TRAIN_BARS = 8760   # 12 months
TEST_BARS = 2190        # 3 months
STEP_BARS = 2190
WARMUP = 65
HORIZON = 24
TARGET_MODE = "clipped"
MIN_HOLD = 24
DEADZONE = 0.5
MA_WINDOW = 480

# SOL WF config (from wf_SOLUSDT.json)
SOL_FIXED_FEATURES = [
    "basis", "ret_24", "fgi_normalized", "fgi_extreme",
    "parkinson_vol", "atr_norm_14", "rsi_14",
    "tf4h_atr_norm_14", "basis_zscore_24", "cvd_20",
]
SOL_CANDIDATE_POOL = [
    "funding_zscore_24", "basis_momentum", "vol_ma_ratio_5_20",
    "mean_reversion_20", "funding_sign_persist", "hour_sin",
]
SOL_N_FLEXIBLE = 4

# Disaster folds from WF results (Sharpe < -10)
DISASTER_FOLDS = {1, 2, 4, 16}
GOOD_FOLDS = {3, 7, 8, 9, 11, 12, 13, 14, 15}  # Sharpe > 3


# ── Data loading ─────────────────────────────────────────────

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


def _load_features(symbol: str) -> pd.DataFrame:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} bars from {csv_path}")
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df = df.rename(columns={ts_col: "timestamp"})
    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        raise RuntimeError(f"Feature computation failed for {symbol}")
    print(f"  Computed {len(feat_df.columns)} columns, {len(feat_df)} rows")
    return feat_df


def _get_available_features(feat_df: pd.DataFrame) -> List[str]:
    exclude = {"close", "timestamp", "open_time"} | BLACKLIST
    available = [c for c in feat_df.columns if c not in exclude and not c.startswith("ret_")]
    all_nan = set(feat_df.columns[feat_df.isna().all()])
    return [c for c in available if c not in all_nan]


def _compute_bear_mask(closes: np.ndarray, ma_window: int = MA_WINDOW) -> np.ndarray:
    n = len(closes)
    if n < ma_window:
        return np.zeros(n, dtype=bool)
    cs = np.cumsum(closes)
    ma = np.empty(n)
    ma[:ma_window] = np.nan
    ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
    return (~np.isnan(ma)) & (closes < ma)

def _fold_period_str(feat_df: pd.DataFrame, fold: Fold) -> str:
    ts = feat_df["timestamp"].values if "timestamp" in feat_df.columns else None
    if ts is not None:
        try:
            t0 = pd.Timestamp(ts[fold.test_start], unit="ms")
            t1 = pd.Timestamp(ts[min(fold.test_end - 1, len(ts) - 1)], unit="ms")
            return f"{t0.strftime('%Y-%m')} to {t1.strftime('%Y-%m')}"
        except Exception:
            pass
    return f"bar {fold.test_start}-{fold.test_end}"


def _run_single_fold_wf(
    fold: Fold,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    fixed_features: Optional[List[str]] = None,
    candidate_pool: Optional[List[str]] = None,
    n_flexible: int = 4,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    monthly_gate_window: int = MA_WINDOW,
) -> Dict[str, Any]:
    """Run a single WF fold and return detailed results (IC, Sharpe, signal, etc.)."""
    import lightgbm as lgb

    train_df = feat_df.iloc[fold.train_start:fold.train_end]
    test_df = feat_df.iloc[fold.test_start:fold.test_end]
    train_closes = closes[fold.train_start:fold.train_end]
    test_closes = closes[fold.test_start:fold.test_end]

    y_train_full = _compute_target(train_closes, HORIZON, TARGET_MODE)
    y_test_full = _compute_target(test_closes, HORIZON, TARGET_MODE)

    X_train_full = train_df[feature_names].values.astype(np.float64)
    X_test = test_df[feature_names].values.astype(np.float64)

    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]

    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    test_valid = ~np.isnan(y_test_full)

    if len(X_train) < 1000:
        return {"ic": 0.0, "sharpe": 0.0, "total_return": 0.0, "features": [], "signal": np.zeros(len(test_closes))}

    # Feature selection
    if fixed_features:
        selected = [f for f in fixed_features if f in feature_names]
        if n_flexible > 0:
            pool = candidate_pool if candidate_pool else [f for f in feature_names if f not in selected]
            pool_in_data = [f for f in pool if f in feature_names]
            if pool_in_data:
                pool_idx = [feature_names.index(f) for f in pool_in_data]
                X_pool = X_train[:, pool_idx]
                flex = greedy_ic_select(X_pool, y_train, pool_in_data, top_k=n_flexible)
                selected.extend(flex)
    else:
        selected = greedy_ic_select(X_train, y_train, feature_names, top_k=15)

    if not selected:
        return {"ic": 0.0, "sharpe": 0.0, "total_return": 0.0, "features": [], "signal": np.zeros(len(test_closes))}

    sel_idx = [feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    # Train LGBM + XGB ensemble
    params = dict(V7_DEFAULT_PARAMS)
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    bst = lgb.train(params, dtrain, num_boost_round=params["n_estimators"],
                     callbacks=[lgb.log_evaluation(0)])
    y_pred = bst.predict(X_test_sel)

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
        xgb_bst = xgb.train(xgb_params, dtrain_xgb, num_boost_round=params["n_estimators"])
        y_pred = 0.5 * y_pred + 0.5 * xgb_bst.predict(dtest_xgb)
    except Exception:
        pass

    # IC
    ic = 0.0
    if test_valid.sum() > 10:
        y_pred_v = y_pred[test_valid]
        y_test_v = y_test_full[test_valid]
        if np.std(y_pred_v) > 1e-12 and np.std(y_test_v) > 1e-12:
            ic = float(_spearman_ic(_rankdata(y_pred_v), _rankdata(y_test_v)))

    # Signal
    signal = _pred_to_signal(y_pred, target_mode=TARGET_MODE, deadzone=deadzone, min_hold=min_hold)
    np.clip(signal, 0.0, 1.0, out=signal)  # long-only
    signal = _apply_monthly_gate(signal, test_closes, monthly_gate_window)

    # PnL
    ret_1bar = np.diff(test_closes) / test_closes[:-1]
    sig = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig, prepend=0))
    gross_pnl = sig * ret_1bar
    cost = turnover * COST_PER_TRADE

    funding_cost = np.zeros(len(sig))
    if "funding_rate" in test_df.columns:
        fr = test_df["funding_rate"].values[:len(sig)].astype(np.float64)
        fr = np.nan_to_num(fr, 0.0)
        funding_cost = sig * fr / 8.0

    net_pnl = gross_pnl - cost - funding_cost

    active = sig != 0
    n_active = int(active.sum())
    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))

    return {
        "ic": ic,
        "sharpe": sharpe,
        "total_return": total_return,
        "features": selected,
        "signal": signal,
        "y_pred": y_pred,
        "y_test": y_test_full,
        "net_pnl": net_pnl,
        "test_closes": test_closes,
        "n_active": n_active,
    }


# ══════════════════════════════════════════════════════════════
# Phase 1: Per-Fold Feature IC Breakdown
# ══════════════════════════════════════════════════════════════

def run_phase1_fold_ic_breakdown(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    folds: List[Fold],
    feature_names: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n  Phase 1: Per-Fold Feature IC Breakdown")
    print("  " + "─" * 50)
    t0 = time.time()

    # Load existing WF results for reference
    wf_path = Path("results/walkforward_fixed/wf_SOLUSDT.json")
    wf_results = {}
    if wf_path.exists():
        with open(wf_path) as f:
            wf_data = json.load(f)
        for fr in wf_data.get("folds", []):
            wf_results[fr["idx"]] = fr

    # Per-fold, per-feature IC
    fold_feature_ics: Dict[int, Dict[str, float]] = {}
    fold_sharpes: Dict[int, float] = {}

    for fold in folds:
        test_df = feat_df.iloc[fold.test_start:fold.test_end]
        test_closes = closes[fold.test_start:fold.test_end]
        y_test = _compute_target(test_closes, HORIZON, TARGET_MODE)
        valid = ~np.isnan(y_test)

        if valid.sum() < 50:
            fold_feature_ics[fold.idx] = {}
            fold_sharpes[fold.idx] = 0.0
            continue

        # Get Sharpe from WF results
        if fold.idx in wf_results:
            fold_sharpes[fold.idx] = wf_results[fold.idx].get("sharpe", 0.0)
        else:
            fold_sharpes[fold.idx] = 0.0

        # IC for each feature
        ics = {}
        y_v = y_test[valid]
        ry = _rankdata(y_v)
        for feat in feature_names:
            if feat not in test_df.columns:
                continue
            x = test_df[feat].values.astype(np.float64)[valid]
            x = np.nan_to_num(x, 0.0)
            if np.std(x) < 1e-12:
                ics[feat] = 0.0
                continue
            rx = _rankdata(x)
            ics[feat] = float(np.corrcoef(rx, ry)[0, 1])
        fold_feature_ics[fold.idx] = ics

    # Classify folds
    disaster_idx = [i for i in fold_sharpes if fold_sharpes[i] < -10]
    good_idx = [i for i in fold_sharpes if fold_sharpes[i] > 3]
    medium_idx = [i for i in fold_sharpes if -10 <= fold_sharpes[i] <= 3]

    print(f"  Disaster folds (Sharpe < -10): {disaster_idx}")
    print(f"  Good folds (Sharpe > 3): {good_idx}")
    print(f"  Medium folds: {medium_idx}")

    # Per-feature stats by fold category
    feature_stats = {}
    for feat in feature_names:
        disaster_ics = [fold_feature_ics[i].get(feat, 0.0) for i in disaster_idx]
        good_ics = [fold_feature_ics[i].get(feat, 0.0) for i in good_idx]
        all_ics = [fold_feature_ics[i].get(feat, 0.0) for i in fold_sharpes]

        feature_stats[feat] = {
            "disaster_mean_ic": float(np.mean(disaster_ics)) if disaster_ics else 0.0,
            "good_mean_ic": float(np.mean(good_ics)) if good_ics else 0.0,
            "all_mean_ic": float(np.mean(all_ics)) if all_ics else 0.0,
            "ic_std": float(np.std(all_ics)) if all_ics else 0.0,
            "ic_gap": (float(np.mean(good_ics)) - float(np.mean(disaster_ics)))
                      if (good_ics and disaster_ics) else 0.0,
        }

    # Sort by IC gap (features that differentiate good vs disaster)
    sorted_by_gap = sorted(feature_stats.items(), key=lambda x: abs(x[1]["ic_gap"]), reverse=True)

    print(f"\n  {'Feature':<30} {'Disaster IC':>12} {'Good IC':>10} {'All IC':>10} {'IC Std':>8} {'Gap':>8}")
    print("  " + "─" * 80)
    for feat, stats in sorted_by_gap[:20]:
        print(f"  {feat:<30} {stats['disaster_mean_ic']:>12.4f} {stats['good_mean_ic']:>10.4f} "
              f"{stats['all_mean_ic']:>10.4f} {stats['ic_std']:>8.4f} {stats['ic_gap']:>8.4f}")

    # IC instability ranking
    print("\n  Top 10 most unstable features (highest IC std):")
    sorted_by_std = sorted(feature_stats.items(), key=lambda x: x[1]["ic_std"], reverse=True)
    for feat, stats in sorted_by_std[:10]:
        print(f"    {feat:<30} IC_std={stats['ic_std']:.4f}  mean_IC={stats['all_mean_ic']:.4f}")

    # Features that flip sign between disaster and good
    sign_flippers = [
        (f, s) for f, s in feature_stats.items()
        if s["disaster_mean_ic"] * s["good_mean_ic"] < 0 and abs(s["ic_gap"]) > 0.02
    ]
    if sign_flippers:
        print(f"\n  Sign-flipping features ({len(sign_flippers)}):")
        for feat, stats in sorted(sign_flippers, key=lambda x: abs(x[1]["ic_gap"]), reverse=True)[:10]:
            print(f"    {feat:<30} disaster={stats['disaster_mean_ic']:+.4f}  good={stats['good_mean_ic']:+.4f}")

    result = {
        "fold_sharpes": {str(k): v for k, v in fold_sharpes.items()},
        "disaster_folds": disaster_idx,
        "good_folds": good_idx,
        "feature_stats": {k: v for k, v in sorted_by_gap[:30]},
        "sign_flippers": [(f, s) for f, s in sign_flippers[:10]],
        "top_unstable": [(f, s) for f, s in sorted_by_std[:10]],
    }
    with open(out_dir / "phase1_fold_ic.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Phase 1 done in {time.time() - t0:.1f}s")
    return result


# ══════════════════════════════════════════════════════════════
# Phase 2: SOL vs BTC Funding/Basis Comparison
# ══════════════════════════════════════════════════════════════

def run_phase2_funding_basis_comparison(
    sol_feat_df: pd.DataFrame,
    btc_feat_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n  Phase 2: SOL vs BTC Funding/Basis Comparison")
    print("  " + "─" * 50)
    t0 = time.time()

    funding_features = ["funding_rate", "funding_ma8", "funding_zscore_24",
                        "funding_momentum", "funding_extreme", "funding_cumulative_8",
                        "funding_sign_persist", "funding_annualized", "funding_vs_vol"]
    basis_features = ["basis", "basis_zscore_24", "basis_momentum", "basis_extreme"]

    results = {"distribution": {}, "ic_comparison": {}, "carry_strategy": {}}

    # Distribution comparison
    print(f"\n  {'Feature':<25} {'SOL mean':>10} {'SOL std':>10} {'SOL skew':>10} {'BTC mean':>10} {'BTC std':>10} {'BTC skew':>10}")
    print("  " + "─" * 85)

    for feat in funding_features + basis_features:
        sol_ok = feat in sol_feat_df.columns
        btc_ok = feat in btc_feat_df.columns
        if not sol_ok and not btc_ok:
            continue

        sol_vals = sol_feat_df[feat].dropna().values if sol_ok else np.array([])
        btc_vals = btc_feat_df[feat].dropna().values if btc_ok else np.array([])

        def _stats(v):
            if len(v) < 10:
                return {"mean": 0, "std": 0, "skew": 0, "kurtosis": 0, "extreme_pct": 0}
            from scipy.stats import skew, kurtosis
            return {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "skew": float(skew(v)),
                "kurtosis": float(kurtosis(v)),
                "extreme_pct": float(np.mean(np.abs(v) > 2 * np.std(v)) * 100),
            }

        sol_s = _stats(sol_vals)
        btc_s = _stats(btc_vals)
        results["distribution"][feat] = {"sol": sol_s, "btc": btc_s}

        print(f"  {feat:<25} {sol_s['mean']:>10.6f} {sol_s['std']:>10.6f} {sol_s['skew']:>10.3f} "
              f"{btc_s['mean']:>10.6f} {btc_s['std']:>10.6f} {btc_s['skew']:>10.3f}")

    # IC comparison
    print("\n  IC comparison (Spearman IC with 24h forward return):")
    print(f"  {'Feature':<25} {'SOL IC':>10} {'SOL ICIR':>10} {'BTC IC':>10} {'BTC ICIR':>10} {'Delta':>8}")
    print("  " + "─" * 75)

    for feat in funding_features + basis_features:
        sol_ic, btc_ic = 0.0, 0.0
        sol_icir, btc_icir = 0.0, 0.0

        for label, fdf, ic_ref, icir_ref in [
            ("sol", sol_feat_df, "sol_ic", "sol_icir"),
            ("btc", btc_feat_df, "btc_ic", "btc_icir"),
        ]:
            if feat not in fdf.columns:
                continue
            c = fdf["close"].values.astype(np.float64)
            y = _compute_target(c, HORIZON, TARGET_MODE)
            x = fdf[feat].values.astype(np.float64)
            valid = ~np.isnan(y) & ~np.isnan(x)
            if valid.sum() < 100:
                continue

            # Overall IC
            rx = _rankdata(x[valid])
            ry = _rankdata(y[valid])
            overall_ic = float(np.corrcoef(rx, ry)[0, 1])

            # Rolling IC for ICIR
            window = 720
            n_v = valid.sum()
            xv, yv = x[valid], y[valid]
            rolling_ics = []
            for start in range(0, n_v - window, window // 2):
                end = start + window
                xw = xv[start:end]
                yw = yv[start:end]
                if np.std(xw) > 1e-12 and np.std(yw) > 1e-12:
                    rolling_ics.append(float(_spearman_ic(_rankdata(xw), _rankdata(yw))))

            icir_val = 0.0
            if len(rolling_ics) > 2:
                ic_std = float(np.std(rolling_ics))
                if ic_std > 1e-12:
                    icir_val = float(np.mean(rolling_ics)) / ic_std

            if label == "sol":
                sol_ic, sol_icir = overall_ic, icir_val
            else:
                btc_ic, btc_icir = overall_ic, icir_val

        results["ic_comparison"][feat] = {
            "sol_ic": sol_ic, "sol_icir": sol_icir,
            "btc_ic": btc_ic, "btc_icir": btc_icir,
        }
        delta = sol_ic - btc_ic
        print(f"  {feat:<25} {sol_ic:>10.4f} {sol_icir:>10.3f} {btc_ic:>10.4f} {btc_icir:>10.3f} {delta:>8.4f}")

    # Simple funding carry strategy comparison
    print("\n  Funding carry strategy (long when funding > 0, short when < 0):")
    for label, fdf in [("SOL", sol_feat_df), ("BTC", btc_feat_df)]:
        if "funding_rate" not in fdf.columns:
            continue
        c = fdf["close"].values.astype(np.float64)
        fr = fdf["funding_rate"].values.astype(np.float64)
        fr = np.nan_to_num(fr, 0.0)

        # Signal: opposite of funding (collect funding)
        sig = np.where(fr > 0, -1.0, np.where(fr < 0, 1.0, 0.0))
        ret_1bar = np.diff(c) / c[:-1]
        sig = sig[:len(ret_1bar)]
        gross_pnl = sig * ret_1bar
        turnover = np.abs(np.diff(sig, prepend=0))
        cost = turnover * COST_PER_TRADE
        funding_income = -sig * fr[:len(sig)] / 8.0  # short collects positive funding
        net_pnl = gross_pnl - cost + funding_income

        active = sig != 0
        sharpe = 0.0
        if active.sum() > 1:
            ap = net_pnl[active]
            std_a = float(np.std(ap, ddof=1))
            if std_a > 0:
                sharpe = float(np.mean(ap)) / std_a * np.sqrt(8760)
        total_ret = float(np.sum(net_pnl)) * 100

        results["carry_strategy"][label] = {"sharpe": round(sharpe, 2), "total_return": round(total_ret, 1)}
        print(f"    {label}: Sharpe={sharpe:.2f}, Return={total_ret:+.1f}%")

    with open(out_dir / "phase2_funding_basis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 2 done in {time.time() - t0:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════
# Phase 3: Regime Analysis
# ══════════════════════════════════════════════════════════════

def run_phase3_regime_analysis(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    folds: List[Fold],
    feature_names: List[str],
    out_dir: Path,
    btc_feat_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    print("\n  Phase 3: Regime Analysis & Disaster Fold Diagnosis")
    print("  " + "─" * 50)
    t0 = time.time()

    # Load WF results
    wf_path = Path("results/walkforward_fixed/wf_SOLUSDT.json")
    wf_sharpes = {}
    wf_ics = {}
    if wf_path.exists():
        with open(wf_path) as f:
            wf_data = json.load(f)
        for fr in wf_data.get("folds", []):
            wf_sharpes[fr["idx"]] = fr.get("sharpe", 0.0)
            wf_ics[fr["idx"]] = fr.get("ic", 0.0)

    # Per-fold regime characteristics
    fold_data = []
    for fold in folds:
        test_df = feat_df.iloc[fold.test_start:fold.test_end]
        test_closes = closes[fold.test_start:fold.test_end]

        # Volatility regime
        atr_norm = test_df["atr_norm_14"].values if "atr_norm_14" in test_df.columns else np.array([0])
        vol_mean = float(np.nanmean(atr_norm))

        # Trend strength
        sol_ret = (test_closes[-1] / test_closes[0] - 1.0) * 100 if len(test_closes) > 1 else 0.0

        # Monthly gate activation
        bear_mask = _compute_bear_mask(test_closes, MA_WINDOW)
        gate_active_pct = float(1.0 - np.mean(bear_mask)) * 100  # % of time gate allows trading

        # Funding regime
        funding = test_df["funding_rate"].values if "funding_rate" in test_df.columns else np.array([0])
        funding_mean = float(np.nanmean(funding))

        # BTC correlation (if BTC data available)
        btc_corr = 0.0
        if btc_feat_df is not None and len(btc_feat_df) > fold.test_end:
            btc_test_closes = btc_feat_df["close"].values[fold.test_start:fold.test_end].astype(np.float64)
            if len(btc_test_closes) == len(test_closes) and len(test_closes) > 30:
                sol_rets = np.diff(test_closes) / test_closes[:-1]
                btc_rets = np.diff(btc_test_closes) / btc_test_closes[:-1]
                valid = ~(np.isnan(sol_rets) | np.isnan(btc_rets))
                if valid.sum() > 30:
                    btc_corr = float(np.corrcoef(sol_rets[valid], btc_rets[valid])[0, 1])

        # BTC buy-hold return
        btc_ret = 0.0
        if btc_feat_df is not None and len(btc_feat_df) > fold.test_end:
            btc_c = btc_feat_df["close"].values[fold.test_start:fold.test_end].astype(np.float64)
            if len(btc_c) > 1:
                btc_ret = (btc_c[-1] / btc_c[0] - 1.0) * 100

        period = _fold_period_str(feat_df, fold)
        sharpe = wf_sharpes.get(fold.idx, 0.0)
        ic = wf_ics.get(fold.idx, 0.0)

        fd = {
            "fold": fold.idx,
            "period": period,
            "sharpe": sharpe,
            "ic": ic,
            "vol_mean": round(vol_mean, 4),
            "sol_ret": round(sol_ret, 1),
            "btc_ret": round(btc_ret, 1),
            "btc_corr": round(btc_corr, 3),
            "gate_active_pct": round(gate_active_pct, 1),
            "funding_mean": round(funding_mean, 6),
        }
        fold_data.append(fd)

    # Print table
    print(f"\n  {'Fold':>4} {'Period':<22} {'Sharpe':>8} {'IC':>8} {'Vol':>8} {'SOL%':>8} {'BTC%':>8} "
          f"{'BTC_r':>8} {'Gate%':>8} {'Fund':>10}")
    print("  " + "─" * 100)
    for fd in fold_data:
        marker = " ***" if fd["sharpe"] < -10 else ""
        print(f"  {fd['fold']:>4} {fd['period']:<22} {fd['sharpe']:>8.2f} {fd['ic']:>8.3f} "
              f"{fd['vol_mean']:>8.4f} {fd['sol_ret']:>+8.1f} {fd['btc_ret']:>+8.1f} "
              f"{fd['btc_corr']:>8.3f} {fd['gate_active_pct']:>8.1f} {fd['funding_mean']:>10.6f}{marker}")

    # Disaster fold diagnosis
    disaster = [fd for fd in fold_data if fd["sharpe"] < -10]
    good = [fd for fd in fold_data if fd["sharpe"] > 3]

    if disaster:
        print(f"\n  Disaster fold characteristics (n={len(disaster)}):")
        for key in ["vol_mean", "sol_ret", "btc_ret", "btc_corr", "gate_active_pct", "funding_mean"]:
            d_vals = [fd[key] for fd in disaster]
            g_vals = [fd[key] for fd in good] if good else [0]
            print(f"    {key:<20} disaster_avg={np.mean(d_vals):>+10.4f}  good_avg={np.mean(g_vals):>+10.4f}  "
                  f"diff={np.mean(d_vals)-np.mean(g_vals):>+10.4f}")

    # Correlation matrix
    keys = ["sharpe", "ic", "vol_mean", "sol_ret", "btc_ret", "btc_corr", "gate_active_pct", "funding_mean"]
    matrix = np.array([[fd[k] for k in keys] for fd in fold_data])
    if len(matrix) > 3:
        corr = np.corrcoef(matrix.T)
        print("\n  Correlation with fold Sharpe:")
        sharpe_idx = 0
        for i, k in enumerate(keys):
            if i == sharpe_idx:
                continue
            print(f"    {k:<20} r={corr[sharpe_idx, i]:>+.3f}")

    result = {
        "fold_data": fold_data,
        "disaster_folds": [fd for fd in fold_data if fd["sharpe"] < -10],
        "good_folds": [fd for fd in fold_data if fd["sharpe"] > 3],
    }
    with open(out_dir / "phase3_regime.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Phase 3 done in {time.time() - t0:.1f}s")
    return result


# ══════════════════════════════════════════════════════════════
# Phase 4: IC-Return Disconnect Investigation
# ══════════════════════════════════════════════════════════════

def run_phase4_ic_disconnect(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    folds: List[Fold],
    feature_names: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    print("\n  Phase 4: IC-Return Disconnect Investigation")
    print("  " + "─" * 50)
    t0 = time.time()

    import lightgbm as lgb
    from sklearn.linear_model import LinearRegression

    results = {"4a_signal_timing": [], "4b_nonlinear": [], "4c_fat_tails": []}

    for fold in folds:
        train_df = feat_df.iloc[fold.train_start:fold.train_end]
        test_df = feat_df.iloc[fold.test_start:fold.test_end]
        train_closes = closes[fold.train_start:fold.train_end]
        test_closes = closes[fold.test_start:fold.test_end]

        y_train_full = _compute_target(train_closes, HORIZON, TARGET_MODE)
        y_test_full = _compute_target(test_closes, HORIZON, TARGET_MODE)

        X_train_full = train_df[feature_names].values.astype(np.float64)
        X_test = test_df[feature_names].values.astype(np.float64)

        X_train = X_train_full[WARMUP:]
        y_train = y_train_full[WARMUP:]
        train_valid = ~np.isnan(y_train)
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        test_valid = ~np.isnan(y_test_full)

        if len(X_train) < 1000 or test_valid.sum() < 50:
            results["4a_signal_timing"].append({"fold": fold.idx, "all_ic": 0, "active_ic": 0, "entry_ic": 0})
            results["4b_nonlinear"].append({"fold": fold.idx, "lgbm_ic": 0, "ols_ic": 0})
            results["4c_fat_tails"].append({"fold": fold.idx, "sharpe": 0, "trimmed_sharpe": 0, "skew": 0, "kurtosis": 0})
            continue

        # Feature selection (same as WF)
        selected = [f for f in SOL_FIXED_FEATURES if f in feature_names]
        pool_in_data = [f for f in SOL_CANDIDATE_POOL if f in feature_names]
        if pool_in_data:
            pool_idx = [feature_names.index(f) for f in pool_in_data]
            X_pool = X_train[:, pool_idx]
            flex = greedy_ic_select(X_pool, y_train, pool_in_data, top_k=SOL_N_FLEXIBLE)
            selected.extend(flex)

        sel_in_names = [f for f in selected if f in feature_names]
        if not sel_in_names:
            results["4a_signal_timing"].append({"fold": fold.idx, "all_ic": 0, "active_ic": 0, "entry_ic": 0})
            results["4b_nonlinear"].append({"fold": fold.idx, "lgbm_ic": 0, "ols_ic": 0})
            results["4c_fat_tails"].append({"fold": fold.idx, "sharpe": 0, "trimmed_sharpe": 0, "skew": 0, "kurtosis": 0})
            continue

        sel_idx = [feature_names.index(n) for n in sel_in_names]
        X_train_sel = X_train[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        # ── 4B: Nonlinear comparison (LGBM vs OLS) ──

        # LGBM
        params = dict(V7_DEFAULT_PARAMS)
        dtrain = lgb.Dataset(X_train_sel, label=y_train)
        bst = lgb.train(params, dtrain, num_boost_round=params["n_estimators"],
                         callbacks=[lgb.log_evaluation(0)])
        y_pred_lgbm = bst.predict(X_test_sel)

        # OLS
        X_train_ols = np.nan_to_num(X_train_sel, 0.0)
        X_test_ols = np.nan_to_num(X_test_sel, 0.0)
        ols = LinearRegression()
        ols.fit(X_train_ols, y_train)
        y_pred_ols = ols.predict(X_test_ols)

        lgbm_ic, ols_ic = 0.0, 0.0
        if test_valid.sum() > 10:
            yv = y_test_full[test_valid]
            if np.std(y_pred_lgbm[test_valid]) > 1e-12:
                lgbm_ic = float(_spearman_ic(_rankdata(y_pred_lgbm[test_valid]), _rankdata(yv)))
            if np.std(y_pred_ols[test_valid]) > 1e-12:
                ols_ic = float(_spearman_ic(_rankdata(y_pred_ols[test_valid]), _rankdata(yv)))

        results["4b_nonlinear"].append({
            "fold": fold.idx,
            "lgbm_ic": round(lgbm_ic, 4),
            "ols_ic": round(ols_ic, 4),
            "nonlinear_gain": round(lgbm_ic - ols_ic, 4),
        })

        # ── 4A: Signal timing analysis ──

        signal = _pred_to_signal(y_pred_lgbm, target_mode=TARGET_MODE, deadzone=DEADZONE, min_hold=MIN_HOLD)
        np.clip(signal, 0.0, 1.0, out=signal)
        signal = _apply_monthly_gate(signal, test_closes, MA_WINDOW)

        # All-bar IC (standard)
        all_ic = lgbm_ic

        # Active-bar IC (only bars where signal != 0)
        active_mask = (signal != 0) & test_valid
        active_ic = 0.0
        if active_mask.sum() > 10:
            y_a = y_test_full[active_mask]
            p_a = y_pred_lgbm[active_mask]
            if np.std(p_a) > 1e-12 and np.std(y_a) > 1e-12:
                active_ic = float(_spearman_ic(_rankdata(p_a), _rankdata(y_a)))

        # Entry-bar IC (bars where signal transitions from 0 to non-zero)
        entries = np.zeros(len(signal), dtype=bool)
        for i in range(1, len(signal)):
            if signal[i] != 0 and signal[i - 1] == 0:
                entries[i] = True
        entry_mask = entries & test_valid
        entry_ic = 0.0
        if entry_mask.sum() > 5:
            y_e = y_test_full[entry_mask]
            p_e = y_pred_lgbm[entry_mask]
            if np.std(p_e) > 1e-12 and np.std(y_e) > 1e-12:
                entry_ic = float(_spearman_ic(_rankdata(p_e), _rankdata(y_e)))

        results["4a_signal_timing"].append({
            "fold": fold.idx,
            "all_ic": round(all_ic, 4),
            "active_ic": round(active_ic, 4),
            "entry_ic": round(entry_ic, 4),
            "n_active": int(active_mask.sum()),
            "n_entries": int(entry_mask.sum()),
            "active_pct": round(float(np.mean(signal != 0)) * 100, 1),
        })

        # ── 4C: Fat tail analysis ──

        ret_1bar = np.diff(test_closes) / test_closes[:-1]
        sig = signal[:len(ret_1bar)]
        turnover = np.abs(np.diff(sig, prepend=0))
        gross_pnl = sig * ret_1bar
        cost = turnover * COST_PER_TRADE

        funding_cost = np.zeros(len(sig))
        if "funding_rate" in test_df.columns:
            fr = test_df["funding_rate"].values[:len(sig)].astype(np.float64)
            fr = np.nan_to_num(fr, 0.0)
            funding_cost = sig * fr / 8.0

        net_pnl = gross_pnl - cost - funding_cost
        active = sig != 0

        if active.sum() > 10:
            active_pnl = net_pnl[active]
            std_a = float(np.std(active_pnl, ddof=1))
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760) if std_a > 0 else 0.0

            # Trimmed Sharpe (remove top/bottom 1%)
            p1, p99 = np.percentile(active_pnl, [1, 99])
            trimmed = active_pnl[(active_pnl >= p1) & (active_pnl <= p99)]
            trimmed_sharpe = 0.0
            if len(trimmed) > 10:
                std_t = float(np.std(trimmed, ddof=1))
                if std_t > 0:
                    trimmed_sharpe = float(np.mean(trimmed)) / std_t * np.sqrt(8760)

            from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
            results["4c_fat_tails"].append({
                "fold": fold.idx,
                "sharpe": round(sharpe, 2),
                "trimmed_sharpe": round(trimmed_sharpe, 2),
                "skew": round(float(sp_skew(active_pnl)), 3),
                "kurtosis": round(float(sp_kurt(active_pnl)), 3),
                "max_single_loss": round(float(np.min(active_pnl)) * 100, 3),
                "max_single_gain": round(float(np.max(active_pnl)) * 100, 3),
            })
        else:
            results["4c_fat_tails"].append({
                "fold": fold.idx, "sharpe": 0, "trimmed_sharpe": 0,
                "skew": 0, "kurtosis": 0, "max_single_loss": 0, "max_single_gain": 0,
            })

        print(f"  Fold {fold.idx:>2}: LGBM_IC={lgbm_ic:>+.4f}  OLS_IC={ols_ic:>+.4f}  "
              f"all_IC={all_ic:>+.4f}  active_IC={active_ic:>+.4f}  entry_IC={entry_ic:>+.4f}")

    # Aggregate analysis
    print("\n  ── 4A Summary: Signal Timing ──")
    all_ics = [r["all_ic"] for r in results["4a_signal_timing"]]
    active_ics = [r["active_ic"] for r in results["4a_signal_timing"]]
    entry_ics = [r["entry_ic"] for r in results["4a_signal_timing"]]
    print(f"  Mean all-bar IC:    {np.mean(all_ics):>+.4f}")
    print(f"  Mean active-bar IC: {np.mean(active_ics):>+.4f}")
    print(f"  Mean entry-bar IC:  {np.mean(entry_ics):>+.4f}")
    if abs(np.mean(all_ics)) > 1e-6:
        ratio = np.mean(active_ics) / np.mean(all_ics)
        print(f"  Active/All IC ratio: {ratio:.2f}" +
              (" → Signal filter is DESTROYING alpha" if ratio < 0.5 else " → Signal filter preserves alpha"))

    print("\n  ── 4B Summary: Nonlinear Interactions ──")
    lgbm_ics = [r["lgbm_ic"] for r in results["4b_nonlinear"]]
    ols_ics = [r["ols_ic"] for r in results["4b_nonlinear"]]
    gains = [r["nonlinear_gain"] for r in results["4b_nonlinear"]]
    print(f"  Mean LGBM IC: {np.mean(lgbm_ics):>+.4f}")
    print(f"  Mean OLS IC:  {np.mean(ols_ics):>+.4f}")
    print(f"  Mean gain:    {np.mean(gains):>+.4f}" +
          (" → Strong nonlinear alpha" if np.mean(gains) > 0.02 else " → Limited nonlinear contribution"))

    print("\n  ── 4C Summary: Fat Tails ──")
    sharpes = [r["sharpe"] for r in results["4c_fat_tails"]]
    trimmed_sharpes = [r["trimmed_sharpe"] for r in results["4c_fat_tails"]]
    skews = [r["skew"] for r in results["4c_fat_tails"]]
    kurtoses = [r["kurtosis"] for r in results["4c_fat_tails"]]
    print(f"  Mean Sharpe:         {np.mean(sharpes):>+.2f}")
    print(f"  Mean Trimmed Sharpe: {np.mean(trimmed_sharpes):>+.2f}")
    print(f"  Mean Skew:           {np.mean(skews):>+.3f}")
    print(f"  Mean Kurtosis:       {np.mean(kurtoses):>+.3f}")

    # IC vs Sharpe and IC vs Trimmed Sharpe correlations
    fold_ics = [r["lgbm_ic"] for r in results["4b_nonlinear"]]
    if len(fold_ics) > 3 and len(sharpes) > 3:
        ic_sharpe_corr = float(np.corrcoef(fold_ics, sharpes)[0, 1])
        ic_trimmed_corr = float(np.corrcoef(fold_ics, trimmed_sharpes)[0, 1])
        print(f"\n  IC vs Sharpe correlation:         {ic_sharpe_corr:>+.3f}")
        print(f"  IC vs Trimmed Sharpe correlation:  {ic_trimmed_corr:>+.3f}")
        if ic_trimmed_corr > ic_sharpe_corr + 0.1:
            print("  → Fat tails explain IC-Sharpe disconnect (trimmed corr much higher)")
        results["ic_sharpe_corr"] = round(ic_sharpe_corr, 3)
        results["ic_trimmed_sharpe_corr"] = round(ic_trimmed_corr, 3)

    with open(out_dir / "phase4_ic_disconnect.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 4 done in {time.time() - t0:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════
# Phase 5: SOL Feature Optimization
# ══════════════════════════════════════════════════════════════

def _run_config_wf(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    folds: List[Fold],
    feature_names: List[str],
    fixed_features: Optional[List[str]],
    candidate_pool: Optional[List[str]],
    n_flexible: int,
    deadzone: float = DEADZONE,
    min_hold: int = MIN_HOLD,
    monthly_gate_window: int = MA_WINDOW,
    label: str = "",
) -> Dict[str, Any]:
    """Run full WF with a specific config and return summary."""
    fold_sharpes = []
    fold_ics = []
    fold_returns = []
    all_features_used = []

    for fold in folds:
        result = _run_single_fold_wf(
            fold, feat_df, closes, feature_names,
            fixed_features=fixed_features,
            candidate_pool=candidate_pool,
            n_flexible=n_flexible,
            deadzone=deadzone,
            min_hold=min_hold,
            monthly_gate_window=monthly_gate_window,
        )
        fold_sharpes.append(result["sharpe"])
        fold_ics.append(result["ic"])
        fold_returns.append(result["total_return"])
        all_features_used.append(result.get("features", []))

    n_positive = sum(1 for s in fold_sharpes if s > 0)
    n_total = len(fold_sharpes)
    avg_sharpe = float(np.mean(fold_sharpes))
    total_return = float(np.sum(fold_returns)) * 100
    pass_pct = n_positive / n_total if n_total > 0 else 0

    return {
        "label": label,
        "n_positive": n_positive,
        "n_total": n_total,
        "pass_pct": round(pass_pct * 100, 1),
        "avg_sharpe": round(avg_sharpe, 2),
        "total_return": round(total_return, 1),
        "fold_sharpes": [round(s, 2) for s in fold_sharpes],
        "fold_ics": [round(ic, 4) for ic in fold_ics],
        "features_used": all_features_used,
    }


def run_phase5_sol_feature_optimization(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    folds: List[Fold],
    feature_names: List[str],
    out_dir: Path,
    quick: bool = False,
) -> Dict[str, Any]:
    print("\n  Phase 5: SOL Feature Optimization")
    print("  " + "─" * 50)
    t0 = time.time()

    if quick:
        # Use 5-fold subset
        step = max(1, len(folds) // 5)
        mini_folds = folds[::step][:5]
        print(f"  Quick mode: using {len(mini_folds)} folds (of {len(folds)})")
    else:
        mini_folds = folds

    results = {"5a_unconstrained": {}, "5b_fixed_search": [], "5c_candidate_expansion": [],
               "5d_hyperparam_grid": []}

    # ── 5A: Unconstrained greedy selection ──
    print("\n  5A: Unconstrained greedy feature selection")
    result_5a = _run_config_wf(
        feat_df, closes, mini_folds, feature_names,
        fixed_features=None, candidate_pool=None, n_flexible=0,
        label="unconstrained_greedy",
    )
    results["5a_unconstrained"] = result_5a
    print(f"    {result_5a['n_positive']}/{result_5a['n_total']} positive, "
          f"Sharpe={result_5a['avg_sharpe']:.2f}, Return={result_5a['total_return']:+.1f}%")

    # What features did unconstrained selection naturally pick?
    from collections import Counter
    feat_counter = Counter()
    for fl in result_5a["features_used"]:
        feat_counter.update(fl)
    print("    Most selected features (unconstrained):")
    for feat, count in feat_counter.most_common(15):
        in_fixed = "FIXED" if feat in SOL_FIXED_FEATURES else ""
        in_pool = "POOL" if feat in SOL_CANDIDATE_POOL else ""
        marker = in_fixed or in_pool or "NEW"
        print(f"      {feat:<30} {count}/{result_5a['n_total']} folds  [{marker}]")

    # ── Baseline ──
    print("\n  Baseline (current SOL config):")
    baseline = _run_config_wf(
        feat_df, closes, mini_folds, feature_names,
        fixed_features=SOL_FIXED_FEATURES,
        candidate_pool=SOL_CANDIDATE_POOL,
        n_flexible=SOL_N_FLEXIBLE,
        label="baseline",
    )
    results["baseline"] = baseline
    print(f"    {baseline['n_positive']}/{baseline['n_total']} positive, "
          f"Sharpe={baseline['avg_sharpe']:.2f}, Return={baseline['total_return']:+.1f}%")

    # ── 5B: Fixed feature replacement search ──
    print("\n  5B: Fixed feature replacement search")
    # Find top candidates from unconstrained that aren't in current fixed set
    replacement_candidates = [f for f, _ in feat_counter.most_common(25)
                              if f not in SOL_FIXED_FEATURES and f in feature_names][:8]

    for i, orig_feat in enumerate(SOL_FIXED_FEATURES):
        for replacement in replacement_candidates[:3]:  # Top 3 replacements per slot
            new_fixed = [replacement if f == orig_feat else f for f in SOL_FIXED_FEATURES]
            # Use mini-folds (3-fold) for quick screening
            screen_folds = mini_folds[:3] if not quick else mini_folds[:2]
            result = _run_config_wf(
                feat_df, closes, screen_folds, feature_names,
                fixed_features=new_fixed,
                candidate_pool=SOL_CANDIDATE_POOL,
                n_flexible=SOL_N_FLEXIBLE,
                label=f"replace_{orig_feat}_with_{replacement}",
            )
            delta_sharpe = result["avg_sharpe"] - baseline["avg_sharpe"]
            results["5b_fixed_search"].append(result)
            if delta_sharpe > 0.3:
                print(f"    Replace {orig_feat:<25} → {replacement:<25} "
                      f"Sharpe delta={delta_sharpe:>+.2f}")

    # ── 5C: Candidate pool expansion ──
    print("\n  5C: Candidate pool expansion")
    extra_candidates = [
        "rolling_beta_30", "relative_strength_20", "taker_imbalance",
        "trade_intensity", "aggressive_flow_zscore", "oi_acceleration",
        "leverage_proxy", "vol_of_vol", "range_vs_rv",
        "funding_cumulative_8", "funding_annualized",
        "rsi_6", "bb_pctb_20", "macd_hist",
        "close_vs_ma20", "close_vs_ma50",
    ]
    # Filter to features that actually exist
    extra_candidates = [f for f in extra_candidates if f in feature_names]

    test_configs = [
        ("pool+4extra", SOL_CANDIDATE_POOL + extra_candidates[:4], SOL_N_FLEXIBLE + 2),
        ("pool+8extra", SOL_CANDIDATE_POOL + extra_candidates[:8], SOL_N_FLEXIBLE + 3),
        ("pool_replaced", extra_candidates[:6], SOL_N_FLEXIBLE),
        ("all_extras", extra_candidates, 6),
    ]

    for label, pool, n_flex in test_configs:
        result = _run_config_wf(
            feat_df, closes, mini_folds, feature_names,
            fixed_features=SOL_FIXED_FEATURES,
            candidate_pool=pool,
            n_flexible=n_flex,
            label=label,
        )
        results["5c_candidate_expansion"].append(result)
        delta = result["avg_sharpe"] - baseline["avg_sharpe"]
        print(f"    {label:<20} {result['n_positive']}/{result['n_total']} positive, "
              f"Sharpe={result['avg_sharpe']:.2f} (delta={delta:>+.2f}), "
              f"Return={result['total_return']:+.1f}%")

    # ── 5D: Hyperparameter sensitivity ──
    print("\n  5D: Hyperparameter sensitivity grid")

    hparam_configs = [
        ("dz=0.3", 0.3, MIN_HOLD, MA_WINDOW),
        ("dz=0.5", 0.5, MIN_HOLD, MA_WINDOW),  # baseline
        ("dz=0.7", 0.7, MIN_HOLD, MA_WINDOW),
        ("dz=1.0", 1.0, MIN_HOLD, MA_WINDOW),
        ("mh=12", DEADZONE, 12, MA_WINDOW),
        ("mh=48", DEADZONE, 48, MA_WINDOW),
        ("gate=240", DEADZONE, MIN_HOLD, 240),
        ("gate=720", DEADZONE, MIN_HOLD, 720),
    ]

    print(f"  {'Config':<15} {'Pass':>6} {'Sharpe':>8} {'Return':>10} {'Delta':>8}")
    print("  " + "─" * 50)

    for label, dz, mh, gw in hparam_configs:
        result = _run_config_wf(
            feat_df, closes, mini_folds, feature_names,
            fixed_features=SOL_FIXED_FEATURES,
            candidate_pool=SOL_CANDIDATE_POOL,
            n_flexible=SOL_N_FLEXIBLE,
            deadzone=dz, min_hold=mh, monthly_gate_window=gw,
            label=label,
        )
        results["5d_hyperparam_grid"].append(result)
        delta = result["avg_sharpe"] - baseline["avg_sharpe"]
        print(f"  {label:<15} {result['n_positive']}/{result['n_total']:>2} "
              f"{result['avg_sharpe']:>8.2f} {result['total_return']:>+10.1f}% {delta:>+8.2f}")

    # Find best config
    all_configs = [baseline] + results["5c_candidate_expansion"] + results["5d_hyperparam_grid"]
    best = max(all_configs, key=lambda x: (x["n_positive"], x["avg_sharpe"]))
    results["best_config"] = best
    print(f"\n  Best config: {best['label']} — {best['n_positive']}/{best['n_total']} positive, "
          f"Sharpe={best['avg_sharpe']:.2f}, Return={best['total_return']:+.1f}%")

    with open(out_dir / "phase5_optimization.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Phase 5 done in {time.time() - t0:.1f}s ({(time.time() - t0)/60:.1f} min)")
    return results


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════

def write_summary(all_results: Dict[str, Any], out_dir: Path) -> None:
    findings = []

    # Phase 1 findings
    p1 = all_results.get("phase1")
    if p1:
        n_flippers = len(p1.get("sign_flippers", []))
        findings.append(f"Phase 1: {n_flippers} sign-flipping features identified")

    # Phase 2 findings
    p2 = all_results.get("phase2")
    if p2:
        carry = p2.get("carry_strategy", {})
        sol_carry = carry.get("SOL", {}).get("sharpe", 0)
        btc_carry = carry.get("BTC", {}).get("sharpe", 0)
        findings.append(f"Phase 2: Funding carry — SOL Sharpe={sol_carry}, BTC Sharpe={btc_carry}")

    # Phase 4 findings
    p4 = all_results.get("phase4")
    if p4:
        ic_sharpe_corr = p4.get("ic_sharpe_corr", 0)
        ic_trim_corr = p4.get("ic_trimmed_sharpe_corr", 0)
        findings.append(f"Phase 4: IC-Sharpe corr={ic_sharpe_corr}, IC-TrimmedSharpe corr={ic_trim_corr}")

        lgbm_ics = [r["lgbm_ic"] for r in p4.get("4b_nonlinear", [])]
        ols_ics = [r["ols_ic"] for r in p4.get("4b_nonlinear", [])]
        if lgbm_ics and ols_ics:
            findings.append(f"Phase 4: LGBM mean IC={np.mean(lgbm_ics):.4f}, OLS mean IC={np.mean(ols_ics):.4f}")

    # Phase 5 findings
    p5 = all_results.get("phase5")
    if p5:
        best = p5.get("best_config", {})
        findings.append(f"Phase 5: Best config={best.get('label', 'N/A')}, "
                        f"Sharpe={best.get('avg_sharpe', 0)}, Return={best.get('total_return', 0)}%")

    summary = {
        "findings": findings,
        "phases_completed": list(all_results.keys()),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved → {out_dir}/summary.json")
    for finding in findings:
        print(f"    {finding}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="SOL Alpha Deep Research")
    parser.add_argument("--phases", nargs="+", type=int, default=None,
                        help="Run only specific phases (e.g. --phases 1 2 3)")
    parser.add_argument("--skip-phase5", action="store_true",
                        help="Skip Phase 5 (slowest)")
    parser.add_argument("--quick", action="store_true",
                        help="5-fold mini-WF for faster iteration")
    args = parser.parse_args()

    phases_to_run = set(args.phases) if args.phases else {1, 2, 3, 4, 5}
    if args.skip_phase5:
        phases_to_run.discard(5)

    out_dir = Path("results/sol_alpha_research")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"\n{'#'*70}")
    print("  SOL Alpha Deep Research")
    print(f"  Phases: {sorted(phases_to_run)}")
    print(f"{'#'*70}")

    # Load SOL features
    print("\n  Loading SOL features...")
    sol_feat_df = _load_features("SOLUSDT")
    sol_closes = sol_feat_df["close"].values.astype(np.float64)
    feature_names = _get_available_features(sol_feat_df)
    print(f"  Available features: {len(feature_names)}")

    # Generate folds
    folds = generate_wf_folds(len(sol_feat_df))
    print(f"  Generated {len(folds)} WF folds")

    # Load BTC if needed for Phase 2 or 3
    btc_feat_df = None
    if phases_to_run & {2, 3}:
        print("\n  Loading BTC features...")
        btc_feat_df = _load_features("BTCUSDT")

    all_results: Dict[str, Any] = {}

    # Phase 1
    if 1 in phases_to_run:
        print(f"\n{'#'*70}")
        print("  PHASE 1: Per-Fold Feature IC Breakdown")
        print(f"{'#'*70}")
        all_results["phase1"] = run_phase1_fold_ic_breakdown(
            sol_feat_df, sol_closes, folds, feature_names, out_dir)

    # Phase 2
    if 2 in phases_to_run:
        print(f"\n{'#'*70}")
        print("  PHASE 2: SOL vs BTC Funding/Basis Comparison")
        print(f"{'#'*70}")
        all_results["phase2"] = run_phase2_funding_basis_comparison(
            sol_feat_df, btc_feat_df, out_dir)

    # Phase 3
    if 3 in phases_to_run:
        print(f"\n{'#'*70}")
        print("  PHASE 3: Regime Analysis")
        print(f"{'#'*70}")
        all_results["phase3"] = run_phase3_regime_analysis(
            sol_feat_df, sol_closes, folds, feature_names, out_dir,
            btc_feat_df=btc_feat_df)

    # Phase 4
    if 4 in phases_to_run:
        print(f"\n{'#'*70}")
        print("  PHASE 4: IC-Return Disconnect Investigation")
        print(f"{'#'*70}")
        all_results["phase4"] = run_phase4_ic_disconnect(
            sol_feat_df, sol_closes, folds, feature_names, out_dir)

    # Phase 5
    if 5 in phases_to_run:
        print(f"\n{'#'*70}")
        print("  PHASE 5: SOL Feature Optimization")
        print(f"{'#'*70}")
        all_results["phase5"] = run_phase5_sol_feature_optimization(
            sol_feat_df, sol_closes, folds, feature_names, out_dir,
            quick=args.quick)

    # Summary
    print(f"\n{'#'*70}")
    print("  SUMMARY")
    print(f"{'#'*70}")
    write_summary(all_results, out_dir)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
