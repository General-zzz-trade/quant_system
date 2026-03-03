#!/usr/bin/env python3
"""SOL Production Model Training — ensemble LGBM+XGB with per-symbol config.

Reads config from alpha.strategy_config.SYMBOL_CONFIG["SOLUSDT"].
Produces models_v8/SOLUSDT_gate_v2/ with:
  - lgbm_v8.pkl, xgb_v8.pkl
  - config.json (includes position_management section)

Usage:
    python3 -m scripts.train_sol_production
    python3 -m scripts.train_sol_production --no-hpo
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from alpha.strategy_config import get_config
from scripts.train_v7_alpha import (
    V7_DEFAULT_PARAMS,
    V7_SEARCH_SPACE,
    _compute_target,
    _load_and_compute_features,
    INTERACTION_FEATURES,
    BLACKLIST,
)
from scripts.backtest_alpha_v8 import _pred_to_signal, _apply_dd_breaker, COST_PER_TRADE
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic
from infra.model_signing import sign_file

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
SYMBOL = "SOLUSDT"
OOS_BARS = 13140          # 18 months holdout
WARMUP = 65
HORIZON = 24
TARGET_MODE = "clipped"
HPO_TRIALS = 10


# ── Bootstrap ─────────────────────────────────────────────────

def _bootstrap_sharpe_pvalue(
    pnl: np.ndarray, n_boot: int = 5000, block_size: int = 24,
) -> Tuple[float, float, float]:
    n = len(pnl)
    if n < block_size * 2:
        s = float(np.mean(pnl) / (np.std(pnl, ddof=1) + 1e-12)) * np.sqrt(8760)
        return (1.0 if s > 0 else 0.0, s, s)

    rng = np.random.default_rng(42)
    n_blocks = (n + block_size - 1) // block_size
    sharpes = np.empty(n_boot)

    for b in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        boot_pnl = np.concatenate([pnl[s:s + block_size] for s in block_starts])[:n]
        std = np.std(boot_pnl, ddof=1)
        if std > 1e-12:
            sharpes[b] = float(np.mean(boot_pnl)) / std * np.sqrt(8760)
        else:
            sharpes[b] = 0.0

    p_pos = float(np.mean(sharpes > 0))
    ci_low = float(np.percentile(sharpes, 2.5))
    ci_high = float(np.percentile(sharpes, 97.5))
    return p_pos, ci_low, ci_high


# ── Monthly gate helper ──────────────────────────────────────

def _apply_monthly_gate(signal: np.ndarray, closes: np.ndarray, window: int) -> np.ndarray:
    """Zero signal when close <= SMA(window)."""
    out = signal.copy()
    ma = np.full(len(closes), np.nan)
    cs = np.cumsum(closes)
    ma[window - 1:] = (cs[window - 1:] - np.concatenate([[0], cs[:-window]])) / window
    for i in range(len(out)):
        if np.isnan(ma[i]) or closes[i] <= ma[i]:
            out[i] = 0.0
    return out


# ── OOS Evaluation ────────────────────────────────────────────

def _evaluate_oos(
    y_pred: np.ndarray,
    closes: np.ndarray,
    y_target: np.ndarray,
    test_df: pd.DataFrame,
    cfg,
) -> Dict[str, Any]:
    """Full OOS evaluation with vol_target + dd_limit + monthly_gate."""
    # IC
    valid = ~np.isnan(y_target)
    ic = 0.0
    if valid.sum() > 10:
        yp = y_pred[valid]
        yt = y_target[valid]
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            ic = float(_spearman_ic(_rankdata(yp), _rankdata(yt)))

    # Signal & post-processing
    signal = _pred_to_signal(y_pred, target_mode=TARGET_MODE,
                             deadzone=cfg.deadzone, min_hold=cfg.min_hold)
    np.clip(signal, 0.0, 1.0, out=signal)  # long-only

    # Monthly gate
    signal = _apply_monthly_gate(signal, closes, cfg.monthly_gate_window)

    # Vol-adaptive sizing
    if cfg.vol_target is not None and cfg.vol_feature in test_df.columns:
        vol_vals = test_df[cfg.vol_feature].values.astype(np.float64)
        for i in range(len(signal)):
            if signal[i] != 0.0 and not np.isnan(vol_vals[i]) and vol_vals[i] > 1e-8:
                signal[i] *= min(cfg.vol_target / vol_vals[i], 1.0)

    # DD breaker
    if cfg.dd_limit is not None:
        signal = _apply_dd_breaker(signal, closes, cfg.dd_limit, cfg.dd_cooldown)

    # PnL
    ret_1bar = np.diff(closes) / closes[:-1]
    sig_trade = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig_trade, prepend=0))
    gross_pnl = sig_trade * ret_1bar
    cost = turnover * COST_PER_TRADE

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

    # Max drawdown
    equity = np.cumsum(net_pnl)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Bootstrap
    p_pos, ci_low, ci_high = _bootstrap_sharpe_pvalue(
        net_pnl[active] if n_active > 10 else net_pnl)

    # Monthly breakdown
    bars_per_month = 24 * 30
    n_months = len(net_pnl) // bars_per_month
    monthly_returns = []
    for m in range(n_months):
        s = m * bars_per_month
        e = (m + 1) * bars_per_month
        monthly_returns.append(float(np.sum(net_pnl[s:e])))
    pos_months = sum(1 for r in monthly_returns if r > 0)

    # H2 IC
    half = len(y_pred) // 2
    h2_ic = 0.0
    valid_h2 = ~np.isnan(y_target[half:])
    if valid_h2.sum() > 10:
        yp2 = y_pred[half:][valid_h2]
        yt2 = y_target[half:][valid_h2]
        if np.std(yp2) > 1e-12 and np.std(yt2) > 1e-12:
            h2_ic = float(_spearman_ic(_rankdata(yp2), _rankdata(yt2)))

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "ic": ic,
        "h2_ic": h2_ic,
        "bootstrap_p_positive": p_pos,
        "bootstrap_ci_95": [ci_low, ci_high],
        "n_months": n_months,
        "positive_months": pos_months,
        "monthly_returns": monthly_returns,
        "n_active_bars": n_active,
        "n_total_bars": len(net_pnl),
    }


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train SOL production model")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--no-hpo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    symbol = args.symbol.upper()
    hpo_trials = args.hpo_trials
    use_hpo = not args.no_hpo

    cfg = get_config(symbol)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  SOL Production Training: {symbol}")
    print(f"  Config: ensemble LGBM+XGB, long-only")
    print(f"  Features: {len(cfg.fixed_features)} fixed + {cfg.n_flexible} flexible "
          f"from {len(cfg.candidate_pool)} candidates")
    print(f"  Vol target: {cfg.vol_target}, DD limit: {cfg.dd_limit}, "
          f"DD cooldown: {cfg.dd_cooldown}")
    print(f"  HPO: {'ON' if use_hpo else 'OFF'}"
          f"{f' ({hpo_trials} trials)' if use_hpo else ''}")
    print(f"  OOS holdout: {OOS_BARS} bars ({OOS_BARS // (24*30):.0f} months)")

    # ── Load data ─────────────────────────────────────────
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_bars = len(df)
    if n_bars <= OOS_BARS:
        print(f"  Not enough data: {n_bars} <= {OOS_BARS}")
        return
    print(f"  Total bars: {n_bars:,}")
    print(f"  Train: {n_bars - OOS_BARS:,} bars, OOS: {OOS_BARS:,} bars")

    # ── Compute features ──────────────────────────────────
    print("  Computing features...")
    t0 = time.time()
    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        print("  Feature computation failed")
        return
    closes = (feat_df["close"].values.astype(np.float64)
              if "close" in feat_df.columns
              else df["close"].values.astype(np.float64))
    print(f"  Features computed in {time.time()-t0:.1f}s ({len(feat_df.columns)} columns)")

    all_feature_names = [c for c in feat_df.columns
                         if c not in ("close", "timestamp", "open_time")
                         and c not in BLACKLIST]

    # ── Split train/OOS ───────────────────────────────────
    split = n_bars - OOS_BARS
    train_df = feat_df.iloc[:split]
    oos_df = feat_df.iloc[split:]
    train_closes = closes[:split]
    oos_closes = closes[split:]

    # ── Compute target ────────────────────────────────────
    y_train_full = _compute_target(train_closes, HORIZON, TARGET_MODE)
    y_oos_full = _compute_target(oos_closes, HORIZON, TARGET_MODE)

    X_train_full = train_df[all_feature_names].values.astype(np.float64)
    X_oos = oos_df[all_feature_names].values.astype(np.float64)

    # Skip warmup
    X_train = X_train_full[WARMUP:]
    y_train = y_train_full[WARMUP:]
    train_valid = ~np.isnan(y_train)
    X_train = X_train[train_valid]
    y_train = y_train[train_valid]
    print(f"  Train samples: {len(X_train):,} (after warmup + NaN removal)")

    # ── Feature selection ─────────────────────────────────
    print("  Feature selection...")
    selected = list(cfg.fixed_features)
    # Filter to features that actually exist in data
    pool_in_data = [f for f in cfg.candidate_pool if f in all_feature_names]
    fixed_in_data = [f for f in cfg.fixed_features if f in all_feature_names]
    if len(fixed_in_data) < len(cfg.fixed_features):
        missing = set(cfg.fixed_features) - set(fixed_in_data)
        print(f"  WARNING: Missing fixed features: {missing}")
        selected = list(fixed_in_data)

    if pool_in_data and cfg.n_flexible > 0:
        pool_idx = [all_feature_names.index(f) for f in pool_in_data]
        X_pool = X_train[:, pool_idx]
        flex = greedy_ic_select(X_pool, y_train, pool_in_data, top_k=cfg.n_flexible)
        selected.extend(flex)
    print(f"  Selected ({len(selected)}): {selected}")

    sel_idx = [all_feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_oos_sel = X_oos[:, sel_idx]

    # ── HPO ───────────────────────────────────────────────
    params = dict(V7_DEFAULT_PARAMS)
    if use_hpo:
        print(f"  Running HPO ({hpo_trials} trials)...")
        try:
            from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
            import lightgbm as lgb

            n_tr = len(X_train_sel)
            val_size = min(n_tr // 4, 2190)
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
                vm = ~np.isnan(y_hpo_val)
                if vm.sum() < 10:
                    return 0.0
                return float(_spearman_ic(_rankdata(y_hat[vm]), _rankdata(y_hpo_val[vm])))

            opt = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=objective,
                config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"),
            )
            result = opt.optimize()
            params = {**V7_DEFAULT_PARAMS, **result.best_params}
            print(f"  HPO best IC: {result.best_value:.4f}")
        except Exception as e:
            print(f"  HPO failed: {e}, using defaults")

    # ── Train LGBM ────────────────────────────────────────
    print("  Training LGBM...")
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train_sel, label=y_train)
    lgbm_bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[lgb.log_evaluation(0)],
    )
    lgbm_pred = lgbm_bst.predict(X_oos_sel)

    # ── Train XGB ─────────────────────────────────────────
    print("  Training XGB...")
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
    doos_xgb = xgb.DMatrix(X_oos_sel)
    xgb_bst = xgb.train(
        xgb_params, dtrain_xgb,
        num_boost_round=params.get("n_estimators", 500),
    )
    xgb_pred = xgb_bst.predict(doos_xgb)

    # ── Ensemble ──────────────────────────────────────────
    y_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred
    print("  Ensemble: LGBM + XGB averaged")

    # ── OOS Evaluation ────────────────────────────────────
    print("\n  Evaluating OOS (with vol_target + dd_limit + monthly_gate)...")
    metrics = _evaluate_oos(y_pred, oos_closes, y_oos_full, oos_df, cfg)

    print(f"\n  {'='*60}")
    print(f"  OOS RESULTS ({OOS_BARS // (24*30)} months)")
    print(f"  {'='*60}")
    print(f"  Sharpe:              {metrics['sharpe']:+.2f}")
    print(f"  Total return:        {metrics['total_return']*100:+.2f}%")
    print(f"  Max drawdown:        {metrics['max_drawdown']*100:+.2f}%")
    print(f"  IC:                  {metrics['ic']:.4f}")
    print(f"  H2 IC:               {metrics['h2_ic']:.4f}")
    print(f"  Bootstrap P(S>0):    {metrics['bootstrap_p_positive']:.1%}")
    print(f"  Bootstrap 95% CI:    [{metrics['bootstrap_ci_95'][0]:.2f}, "
          f"{metrics['bootstrap_ci_95'][1]:.2f}]")
    print(f"  Positive months:     {metrics['positive_months']}/{metrics['n_months']}")
    print(f"  Active bars:         {metrics['n_active_bars']}/{metrics['n_total_bars']}")

    if metrics["monthly_returns"]:
        print(f"\n  Monthly returns:")
        for i, r in enumerate(metrics["monthly_returns"]):
            marker = "+" if r > 0 else " "
            print(f"    Month {i+1:2d}: {marker}{r*100:+.2f}%")

    # ── Pass/Fail ─────────────────────────────────────────
    checks = {
        "OOS Sharpe > 0.5": metrics["sharpe"] > 0.5,
        "H2 IC > 0": metrics["h2_ic"] > 0,
        "Bootstrap P(S>0) > 80%": metrics["bootstrap_p_positive"] > 0.80,
        "Max DD > -25%": metrics["max_drawdown"] > -0.25,
        f"Positive months >= {max(1, metrics['n_months'] * 10 // 18)}/{metrics['n_months']}":
            metrics["positive_months"] >= max(1, metrics["n_months"] * 10 // 18),
    }
    all_pass = all(checks.values())

    print(f"\n  {'='*60}")
    print(f"  PRODUCTION READINESS")
    print(f"  {'='*60}")
    for desc, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}: {desc}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ── Save models ───────────────────────────────────────
    print(f"\n  Saving to {out_dir}/...")

    lgbm_path = out_dir / "lgbm_v8.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_bst, "features": selected}, f)
    sign_file(lgbm_path)

    xgb_path = out_dir / "xgb_v8.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_bst, "features": selected}, f)
    sign_file(xgb_path)

    config = {
        "version": "v8",
        "symbol": symbol,
        "ensemble": True,
        "ensemble_weights": [0.5, 0.5],
        "models": ["lgbm_v8.pkl", "xgb_v8.pkl"],
        "features": selected,
        "fixed_features": list(cfg.fixed_features),
        "candidate_pool": list(cfg.candidate_pool),
        "n_flexible": cfg.n_flexible,
        "long_only": True,
        "deadzone": cfg.deadzone,
        "min_hold": cfg.min_hold,
        "monthly_gate_window": cfg.monthly_gate_window,
        "horizon": HORIZON,
        "target_mode": TARGET_MODE,
        "params": params,
        "xgb_params": xgb_params,
        "hpo_trials": hpo_trials,
        "position_management": {
            "vol_target": cfg.vol_target,
            "vol_feature": cfg.vol_feature,
            "dd_limit": cfg.dd_limit,
            "dd_cooldown": cfg.dd_cooldown,
        },
        "metrics": metrics,
        "checks": {k: v for k, v in checks.items()},
        "passed": all_pass,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Saved: lgbm_v8.pkl, xgb_v8.pkl, config.json")

    # ── Register in ModelRegistry ─────────────────────────
    try:
        from research.model_registry.registry import ModelRegistry
        registry = ModelRegistry()
        mv = registry.register(
            name=f"alpha_v8_{symbol}",
            params=params,
            features=selected,
            metrics={
                "oos_sharpe": metrics["sharpe"],
                "oos_return": metrics["total_return"],
                "oos_max_dd": metrics["max_drawdown"],
                "oos_ic": metrics["ic"],
                "h2_ic": metrics["h2_ic"],
                "bootstrap_p": metrics["bootstrap_p_positive"],
                "positive_months": float(metrics["positive_months"]),
                "n_months": float(metrics["n_months"]),
            },
            tags=["v8", "ensemble", "sol", "production" if all_pass else "candidate"],
        )
        if all_pass:
            registry.promote(mv.model_id)
            print(f"  Registered & promoted: {mv.name} v{mv.version} (id={mv.model_id})")
        else:
            print(f"  Registered (not promoted): {mv.name} v{mv.version}")
    except Exception as e:
        print(f"  Registry failed: {e} (models saved locally)")

    if all_pass:
        print(f"\n  Production model ready at {out_dir}/")
    else:
        print(f"\n  Model did not pass all checks. Review metrics before proceeding.")


if __name__ == "__main__":
    main()
