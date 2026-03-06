#!/usr/bin/env python3
"""Train ETH production models — Strategy F (bull + bear).

WF-validated: 15/21 PASS, Avg Sharpe=1.19, Return=+189.5%

Trains:
  1. Bull ensemble (LGBM + XGB) on all data
  2. Bear classifier (LGBM) on bear-regime bars
  3. OOS backtest with regime-switch logic

Output: models_v8/ETHUSDT_gate_v2/ + models_v8/ETHUSDT_bear_c/
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS,
    _load_and_compute_features,
    BLACKLIST,
)
from scripts.backtest_alpha_v8 import _pred_to_signal, _apply_monthly_gate
from scripts.bear_alpha_research import _compute_bear_bull_mask, _compute_bear_target
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic

SYMBOL = "ETHUSDT"
OUT_DIR = Path("models_v8/ETHUSDT_gate_v2")
BEAR_OUT_DIR = Path("models_v8/ETHUSDT_bear_c")
HORIZON = 24
TARGET_MODE = "clipped"
WARMUP = 720

FIXED_FEATURES = [
    "ret_24", "atr_norm_14", "parkinson_vol", "bb_width_20", "vol_20",
    "rsi_14", "tf4h_atr_norm_14", "tf4h_ret_6", "basis_zscore_24",
    "basis_momentum", "btc_ret_24", "btc_rsi_14", "btc_mean_reversion_20",
]

CANDIDATE_POOL = [
    "mean_reversion_20", "rsi_6", "ma_cross_10_30", "close_vs_ma50",
    "vol_of_vol", "fgi_normalized", "funding_ma8", "vwap_dev_20",
    "btc_atr_norm_14", "btc_bb_width_20",
]

BEAR_FEATURES = [
    "funding_zscore_24", "funding_momentum", "funding_extreme",
    "funding_sign_persist", "funding_cumulative_8",
    "basis", "basis_zscore_24", "basis_momentum",
    "vol_20", "vol_regime", "parkinson_vol", "atr_norm_14",
    "fgi_normalized", "fgi_extreme",
    "rsi_14", "bb_pctb_20",
    "oi_acceleration", "leverage_proxy",
]

N_FLEXIBLE = 4
DEADZONE = 0.5
MIN_HOLD = 24
MONTHLY_GATE_WINDOW = 480
COST_PER_TRADE = 4e-4
BEAR_THRESHOLDS = [[0.7, -1.0], [0.6, -0.5], [0.5, 0.0]]


def compute_target(closes, horizon, mode):
    fwd = np.roll(closes, -horizon) / closes - 1.0
    fwd[-horizon:] = np.nan
    if mode == "clipped":
        p1, p99 = np.nanpercentile(fwd, [1, 99])
        fwd = np.clip(fwd, p1, p99)
    return fwd


def bootstrap_sharpe_ci(returns, n_boot=5000, ci=0.95):
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return 0.0, (-99, 99)
    sharpes = []
    n = len(returns)
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)
        std = np.std(sample, ddof=1)
        if std > 1e-12:
            sharpes.append(np.mean(sample) / std * np.sqrt(8760))
    sharpes = sorted(sharpes)
    lo = sharpes[int(n_boot * (1 - ci) / 2)]
    hi = sharpes[int(n_boot * (1 + ci) / 2)]
    p_positive = sum(1 for s in sharpes if s > 0) / len(sharpes)
    return p_positive, (lo, hi)


def main():
    import pandas as pd
    from alpha.models.lgbm_alpha import LGBMAlphaModel

    print(f"Training ETH production models (Strategy F)")
    print(f"  Fixed: {len(FIXED_FEATURES)} features")
    print(f"  Candidate pool: {len(CANDIDATE_POOL)} features")
    print(f"  Bear features: {len(BEAR_FEATURES)}")
    print()

    # Load data
    df = pd.read_csv(f"data_files/{SYMBOL}_1h.csv")
    print(f"  Data: {len(df):,} bars")

    feat_df = _load_and_compute_features(SYMBOL, df)
    if feat_df is None:
        print("ERROR: Feature computation failed")
        return

    print(f"  Features: {len(feat_df.columns)} columns")

    closes = df["close"].values.astype(np.float64)
    y_full = compute_target(closes, HORIZON, TARGET_MODE)

    feature_names = [c for c in feat_df.columns if c not in BLACKLIST and c != "close"]

    missing = [f for f in FIXED_FEATURES if f not in feature_names]
    if missing:
        print(f"ERROR: Missing fixed features: {missing}")
        return

    X_full = feat_df[feature_names].values.astype(np.float64)

    # ========== BULL MODEL ==========
    print("\n" + "=" * 60)
    print("  BULL MODEL (LGBM + XGB ensemble)")
    print("=" * 60)

    X = X_full[WARMUP:]
    y = y_full[WARMUP:]
    valid = ~np.isnan(y)
    X_train = X[valid]
    y_train = y[valid]
    print(f"  Training samples: {len(X_train):,}")

    # Feature selection
    selected = list(FIXED_FEATURES)
    pool_in_data = [f for f in CANDIDATE_POOL if f in feature_names]
    if pool_in_data and N_FLEXIBLE > 0:
        pool_idx = [feature_names.index(f) for f in pool_in_data]
        X_pool = X_train[:, pool_idx]
        flex = greedy_ic_select(X_pool, y_train, pool_in_data, top_k=N_FLEXIBLE)
        selected.extend(flex)
        print(f"  Flexible features selected: {flex}")

    print(f"  Total features: {len(selected)}")
    for i, f in enumerate(selected):
        print(f"    {i+1:2d}. {f}")

    sel_idx = [feature_names.index(n) for n in selected]
    X_sel = X_train[:, sel_idx]

    # Train LGBM
    print("\n  Training LGBM...")
    params = dict(V7_DEFAULT_PARAMS)
    dtrain = lgb.Dataset(X_sel, label=y_train)
    lgbm_model = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[lgb.log_evaluation(0)],
    )
    lgbm_pred = lgbm_model.predict(X_sel)
    lgbm_ic = float(_spearman_ic(_rankdata(lgbm_pred), _rankdata(y_train)))
    print(f"    In-sample IC: {lgbm_ic:.4f}")

    # Train XGB
    print("  Training XGB...")
    xgb_params = {
        "max_depth": params.get("max_depth", 6),
        "learning_rate": params.get("learning_rate", 0.05),
        "objective": "reg:squarederror",
        "verbosity": 0,
        "subsample": params.get("subsample", 0.8),
        "colsample_bytree": params.get("colsample_bytree", 0.8),
    }
    dtrain_xgb = xgb.DMatrix(X_sel, label=y_train)
    xgb_model = xgb.train(
        xgb_params, dtrain_xgb,
        num_boost_round=params.get("n_estimators", 500),
    )
    xgb_pred = xgb_model.predict(dtrain_xgb)
    xgb_ic = float(_spearman_ic(_rankdata(xgb_pred), _rankdata(y_train)))
    print(f"    In-sample IC: {xgb_ic:.4f}")

    ensemble_ic = float(_spearman_ic(
        _rankdata(0.5 * lgbm_pred + 0.5 * xgb_pred), _rankdata(y_train)))
    print(f"  Ensemble in-sample IC: {ensemble_ic:.4f}")

    # ========== BEAR MODEL ==========
    print("\n" + "=" * 60)
    print("  BEAR MODEL (binary classifier)")
    print("=" * 60)

    bear_features = [f for f in BEAR_FEATURES if f in feature_names]
    bear_missing = [f for f in BEAR_FEATURES if f not in feature_names]
    if bear_missing:
        print(f"  WARNING: Missing bear features: {bear_missing}")

    bear_mask = _compute_bear_bull_mask(closes)
    target_bin = _compute_bear_target(closes, horizon=24, threshold=-0.02)

    bear_idx = [feature_names.index(f) for f in bear_features]
    X_bear_all = X_full[:, bear_idx]
    X_bear_all = np.nan_to_num(X_bear_all, 0.0)

    valid_bear = bear_mask & ~np.isnan(target_bin)
    valid_bear[:65] = False
    bear_valid_idx = np.where(valid_bear)[0]
    X_bear = X_bear_all[bear_valid_idx]
    y_bear = target_bin[bear_valid_idx]

    n_pos = int(y_bear.sum())
    n_neg = len(y_bear) - n_pos
    print(f"  Bear samples: {len(X_bear)} (pos={n_pos}, neg={n_neg})")

    neg_ratio = n_neg / max(n_pos, 1)
    cls_params = V7_DEFAULT_PARAMS.copy()
    cls_params["objective"] = "binary"
    cls_params["metric"] = "binary_logloss"
    cls_params["scale_pos_weight"] = neg_ratio
    cls_params["n_estimators"] = 300
    cls_params["learning_rate"] = 0.02
    cls_params["min_child_samples"] = 50

    bear_model_obj = LGBMAlphaModel(name="bear_detector", feature_names=tuple(bear_features))
    bear_metrics = bear_model_obj.fit_classifier(
        X_bear, y_bear,
        params=cls_params,
        early_stopping_rounds=0,
        embargo_bars=26,
    )
    print(f"  Bear metrics: {bear_metrics}")

    # ========== OOS BACKTEST ==========
    print("\n" + "=" * 60)
    print("  OOS BACKTEST (last 25%)")
    print("=" * 60)

    n_total = len(X_full)
    oos_start = int(n_total * 0.75)
    X_oos = X_full[oos_start:]
    y_oos = y_full[oos_start:]
    closes_oos = closes[oos_start:]

    # Bull predictions
    X_oos_sel = X_oos[:, sel_idx]
    oos_lgbm = lgbm_model.predict(X_oos_sel)
    oos_xgb = xgb_model.predict(xgb.DMatrix(X_oos_sel))
    oos_pred = 0.5 * oos_lgbm + 0.5 * oos_xgb

    # Bear predictions
    X_oos_bear = np.nan_to_num(X_oos[:, bear_idx], 0.0)
    bear_prob = bear_model_obj._model.predict_proba(X_oos_bear)[:, 1]

    # Strategy F: regime-switch signal
    signal = _pred_to_signal(oos_pred, deadzone=DEADZONE, min_hold=MIN_HOLD)
    signal = np.maximum(signal, 0.0)  # long-only

    # Apply bear suppression
    for thr_prob, thr_signal in BEAR_THRESHOLDS:
        bear_active = bear_prob >= thr_prob
        signal[bear_active] = np.minimum(signal[bear_active], max(thr_signal, 0.0))

    # Monthly gate
    signal = _apply_monthly_gate(signal, closes_oos, ma_window=MONTHLY_GATE_WINDOW)

    # OOS IC
    oos_valid = ~np.isnan(y_oos)
    oos_ic = 0.0
    if oos_valid.sum() > 10:
        oos_ic = float(_spearman_ic(
            _rankdata(oos_pred[oos_valid]),
            _rankdata(y_oos[oos_valid])))

    # PnL
    ret_1bar = np.roll(closes_oos, -1) / closes_oos - 1.0
    ret_1bar[-1] = 0.0
    turnover = np.abs(np.diff(signal, prepend=0))
    net_pnl = signal * ret_1bar - turnover * COST_PER_TRADE

    active = signal != 0
    n_active = int(active.sum())
    n_total_oos = len(signal)

    sharpe = 0.0
    if n_active > 1:
        std_a = float(np.std(net_pnl[active], ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))
    cumulative = np.cumsum(net_pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    bars_per_month = 720
    monthly_returns = []
    for i in range(0, len(net_pnl), bars_per_month):
        chunk = net_pnl[i:i + bars_per_month]
        monthly_returns.append(float(np.sum(chunk)))
    n_months = len(monthly_returns)
    pos_months = sum(1 for m in monthly_returns if m > 0)

    print(f"    OOS IC: {oos_ic:.4f}")
    print(f"    Sharpe: {sharpe:.2f}")
    print(f"    Return: {total_return:+.1%}")
    print(f"    Max DD: {max_dd:.1%}")
    print(f"    Active: {n_active}/{n_total_oos} bars")
    print(f"    Months: {pos_months}/{n_months} positive")

    # Bootstrap
    print("\n  Bootstrap significance test...")
    p_pos, (ci_lo, ci_hi) = bootstrap_sharpe_ci(
        net_pnl[active] if n_active > 0 else net_pnl)
    print(f"    P(Sharpe > 0): {p_pos:.1%}")
    print(f"    95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")

    # H2 IC
    h2_start = len(oos_pred) // 2
    h2_ic = 0.0
    h2_valid = oos_valid[h2_start:]
    if h2_valid.sum() > 10:
        h2_ic = float(_spearman_ic(
            _rankdata(oos_pred[h2_start:][h2_valid]),
            _rankdata(y_oos[h2_start:][h2_valid])))

    # Checks
    checks = {
        "OOS Sharpe > 0": sharpe > 0,
        "H2 IC > 0": h2_ic > 0,
        "Bootstrap P(S>0) > 50%": p_pos > 0.5,
        "Max DD > -30%": max_dd > -0.30,
        f"Positive months >= {n_months // 2}/{n_months}": pos_months >= n_months // 2,
    }
    passed = all(checks.values())

    print(f"\n  Checks:")
    for name, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}: {name}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")

    # ========== SAVE MODELS ==========
    print("\n" + "=" * 60)
    print("  SAVING MODELS")
    print("=" * 60)

    # Bull models
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lgbm_path = OUT_DIR / "lgbm_v8.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_model, "features": tuple(selected)}, f)
    print(f"  Saved LGBM: {lgbm_path}")

    xgb_path = OUT_DIR / "xgb_v8.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_model, "features": tuple(selected)}, f)
    print(f"  Saved XGB: {xgb_path}")

    bull_config = {
        "version": "v8",
        "symbol": SYMBOL,
        "ensemble": True,
        "ensemble_weights": [0.5, 0.5],
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        "features": selected,
        "fixed_features": FIXED_FEATURES,
        "candidate_pool": CANDIDATE_POOL,
        "n_flexible": N_FLEXIBLE,
        "long_only": True,
        "deadzone": DEADZONE,
        "min_hold": MIN_HOLD,
        "monthly_gate_window": MONTHLY_GATE_WINDOW,
        "horizon": HORIZON,
        "target_mode": TARGET_MODE,
        "params": params,
        "xgb_params": xgb_params,
        "bear_model_path": str(BEAR_OUT_DIR),
        "position_management": {
            "bear_thresholds": BEAR_THRESHOLDS,
            "vol_target": None,
            "dd_limit": 0.10,
            "dd_cooldown": 48,
        },
        "metrics": {
            "sharpe": sharpe,
            "total_return": total_return,
            "max_drawdown": max_dd,
            "ic": oos_ic,
            "h2_ic": h2_ic,
            "bootstrap_p_positive": p_pos,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "n_months": n_months,
            "positive_months": pos_months,
            "monthly_returns": monthly_returns,
            "n_active_bars": n_active,
            "n_total_bars": n_total_oos,
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed": passed,
    }

    config_path = OUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(bull_config, f, indent=2)
    print(f"  Saved bull config: {config_path}")

    # Bear model
    BEAR_OUT_DIR.mkdir(parents=True, exist_ok=True)

    bear_pkl_path = BEAR_OUT_DIR / "lgbm_bear.pkl"
    bear_model_obj.save(bear_pkl_path)
    print(f"  Saved bear model: {bear_pkl_path}")

    bear_config = {
        "version": "bear_c_v1",
        "symbol": SYMBOL,
        "is_classifier": True,
        "models": ["lgbm_bear.pkl"],
        "features": bear_features,
        "threshold": -0.02,
        "metrics": {k: round(v, 6) for k, v in bear_metrics.items()},
        "n_train": len(X_bear),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    bear_config_path = BEAR_OUT_DIR / "config.json"
    with open(bear_config_path, "w") as f:
        json.dump(bear_config, f, indent=2)
    print(f"  Saved bear config: {bear_config_path}")

    print(f"\n{'='*60}")
    print(f"  ETH Production Model Training {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")

    return passed


if __name__ == "__main__":
    main()
