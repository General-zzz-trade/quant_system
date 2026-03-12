#!/usr/bin/env python3
"""Independent Short Model Training — ensemble LGBM+XGB, short-only evaluation.

Regression model trained on ALL bars (not bear-gated), predicting 24-bar
forward returns.  OOS evaluation clips to short-only: np.clip(signal, -1, 0).

Designed to run in parallel with V8 long model via separate score_key
in LiveInferenceBridge.

Usage:
    python3 -m scripts.train_short_production --symbol BTCUSDT
    python3 -m scripts.train_short_production --symbol BTCUSDT --no-hpo
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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
from scripts.backtest_alpha_v8 import _pred_to_signal, COST_PER_TRADE
from scripts.signal_postprocess import _compute_bear_mask as _shared_compute_bear_mask
from features.dynamic_selector import greedy_ic_select, stable_icir_select, _rankdata, _spearman_ic
from infra.model_signing import sign_file

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
OOS_BARS = 13140          # 18 months holdout
WARMUP = 65
HORIZON = 24
TARGET_MODE = "clipped"
DEADZONE = 1.5            # much higher than long model (0.5) — only very strong short signals
MIN_HOLD = 24
HPO_TRIALS = 30
N_FLEXIBLE = 4            # V1 proven config
MA_WINDOW = 480           # SMA window for bear regime detection
BEAR_WEIGHT = 3.0         # sample weight multiplier for bear-regime bars in training

# Features selected for bear/short IC ranking:
# funding/basis dominate bearish prediction; vol/leverage capture crash dynamics
# V1 proven set (Sharpe +1.12). V2 mempool-as-fixed FAILED (-2.20).
SHORT_FIXED_FEATURES = [
    "funding_zscore_24", "funding_momentum", "basis", "basis_zscore_24",
    "basis_momentum", "parkinson_vol", "atr_norm_14", "leverage_proxy",
    "oi_acceleration", "fgi_normalized",
]

SHORT_CANDIDATE_POOL = [
    # V1 proven candidates (Sharpe +1.12 with HPO-30)
    "funding_sign_persist", "funding_extreme", "funding_cumulative_8",
    "vol_regime", "vol_20", "rsi_14", "bb_pctb_20",
    "liquidation_cascade_score", "cvd_20", "vol_of_vol",
]

# Extended pool for future experiments (not used by default)
SHORT_EXTENDED_POOL = [
    "mempool_size_zscore_24", "mempool_fee_zscore_24",
    "funding_term_slope", "liquidation_volume_ratio",
    "vix_zscore_14", "spx_overnight_ret",
    "liquidation_volume_zscore_24", "fee_urgency_ratio",
    "aggressive_flow_zscore", "rv_acceleration",
]


# ── Bootstrap ─────────────────────────────────────────────────

def _bootstrap_sharpe_pvalue(
    pnl: np.ndarray, n_boot: int = 5000, block_size: int = 24,
) -> Tuple[float, float, float]:
    """Block bootstrap P(Sharpe > 0) and 95% CI."""
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


# ── OOS Evaluation (short-only) ──────────────────────────────

def _compute_bear_mask(closes: np.ndarray, ma_window: int = MA_WINDOW) -> np.ndarray:
    """Return boolean mask where close <= SMA(ma_window)."""
    if len(closes) < ma_window:
        return np.zeros(len(closes), dtype=bool)
    return _shared_compute_bear_mask(closes, ma_window)


def _evaluate_oos_short(
    y_pred: np.ndarray,
    closes: np.ndarray,
    y_target: np.ndarray,
    bear_mask: np.ndarray = None,
) -> Dict[str, Any]:
    """OOS evaluation with short-only signal + regime gating.

    When bear_mask is provided, shorts are only allowed during bear regime
    (close <= SMA480). This prevents shorting during bull pullbacks.
    """
    # IC (negative IC is good for short: model predicts drops correctly)
    valid = ~np.isnan(y_target)
    ic = 0.0
    if valid.sum() > 10:
        yp = y_pred[valid]
        yt = y_target[valid]
        if np.std(yp) > 1e-12 and np.std(yt) > 1e-12:
            ic = float(_spearman_ic(_rankdata(yp), _rankdata(yt)))

    # Signal & PnL — SHORT ONLY + REGIME GATE
    signal = _pred_to_signal(y_pred, target_mode=TARGET_MODE,
                             deadzone=DEADZONE, min_hold=MIN_HOLD)
    np.clip(signal, -1.0, 0.0, out=signal)  # short-only

    # Regime gate: zero out shorts during bull regime
    if bear_mask is not None:
        signal[~bear_mask[:len(signal)]] = 0.0

    ret_1bar = np.diff(closes) / closes[:-1]
    sig_trade = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig_trade, prepend=0))
    gross_pnl = sig_trade * ret_1bar  # negative signal * negative return = positive PnL
    cost = turnover * COST_PER_TRADE
    net_pnl = gross_pnl - cost

    active = sig_trade != 0
    n_active = int(active.sum())

    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net_pnl))

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

    # H2 IC (second half)
    half = len(y_pred) // 2
    h2_ic = 0.0
    valid_h2 = ~np.isnan(y_target[half:])
    if valid_h2.sum() > 10:
        yp2 = y_pred[half:][valid_h2]
        yt2 = y_target[half:][valid_h2]
        if np.std(yp2) > 1e-12 and np.std(yt2) > 1e-12:
            h2_ic = float(_spearman_ic(_rankdata(yp2), _rankdata(yt2)))

    # Short-specific metrics
    # Bear IC: correlation with negative returns only
    neg_ret_mask = valid & (y_target < 0)
    bear_ic = 0.0
    if neg_ret_mask.sum() > 10:
        yp_bear = y_pred[neg_ret_mask]
        yt_bear = y_target[neg_ret_mask]
        if np.std(yp_bear) > 1e-12 and np.std(yt_bear) > 1e-12:
            bear_ic = float(_spearman_ic(_rankdata(yp_bear), _rankdata(yt_bear)))

    # Short hit rate: % of short entries where next 24 bars were negative
    short_entries = np.where((sig_trade < 0) & (np.roll(sig_trade, 1) == 0))[0]
    n_short_entries = len(short_entries)
    short_hits = 0
    for idx in short_entries:
        if idx + HORIZON < len(closes):
            fwd_ret = closes[idx + HORIZON] / closes[idx] - 1.0
            if fwd_ret < 0:
                short_hits += 1
    short_hit_rate = short_hits / max(n_short_entries, 1)

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "ic": ic,
        "h2_ic": h2_ic,
        "bear_ic": bear_ic,
        "short_hit_rate": short_hit_rate,
        "n_short_entries": n_short_entries,
        "bootstrap_p_positive": p_pos,
        "bootstrap_ci_95": [ci_low, ci_high],
        "n_months": n_months,
        "positive_months": pos_months,
        "monthly_returns": monthly_returns,
        "n_active_bars": n_active,
        "n_total_bars": len(net_pnl),
    }


# ── HPO with bear IC objective ────────────────────────────────

def _bear_ic_objective(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
):
    """Return HPO objective function that maximizes IC on negative-return bars."""
    import lightgbm as lgb

    neg_mask = y_val < 0

    def _objective_fn(trial_params):
        if neg_mask.sum() < 10:
            return 0.0
        p = {**V7_DEFAULT_PARAMS, **trial_params}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        bst = lgb.train(
            p, dtrain,
            num_boost_round=p["n_estimators"],
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        y_hat = bst.predict(X_val)
        # IC on negative-return bars only
        yp_neg = y_hat[neg_mask]
        yt_neg = y_val[neg_mask]
        if np.std(yp_neg) > 1e-12 and np.std(yt_neg) > 1e-12:
            return float(_spearman_ic(_rankdata(yp_neg), _rankdata(yt_neg)))
        return 0.0

    return _objective_fn


# ── Main ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent short model")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--no-hpo", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    symbol = args.symbol.upper()
    hpo_trials = args.hpo_trials
    use_hpo = not args.no_hpo
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"models_v8/{symbol}_short")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Short Model Production Training: {symbol}")
    print(f"  Config: ensemble LGBM+XGB, short-only evaluation")
    print(f"  Features: {len(SHORT_FIXED_FEATURES)} fixed + {N_FLEXIBLE} flexible")
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

    # ── Compute features (ALL bars, no bear gating) ───────
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

    # ── Compute target (full symmetric returns, NOT bear-gated) ──
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

    # Compute bear regime mask for sample weighting + OOS evaluation
    bear_mask_full = _compute_bear_mask(closes, MA_WINDOW)
    train_bear_mask = bear_mask_full[:split][WARMUP:][train_valid]
    oos_bear_mask = bear_mask_full[split:]

    # Show bear bar ratio
    n_neg = int((y_train < 0).sum())
    n_bear = int(train_bear_mask.sum())
    print(f"  Negative return bars: {n_neg:,} / {len(y_train):,} "
          f"({n_neg/len(y_train)*100:.1f}%)")
    print(f"  Bear regime bars:    {n_bear:,} / {len(y_train):,} "
          f"({n_bear/len(y_train)*100:.1f}%)")

    # Sample weights: bear bars get BEAR_WEIGHT multiplier
    sample_weight = np.ones(len(y_train))
    sample_weight[train_bear_mask] = BEAR_WEIGHT
    print(f"  Sample weighting: bear bars {BEAR_WEIGHT}x")

    # ── Feature selection (stable ICIR on bear bars) ──────
    print("  Feature selection (stable ICIR on bear bars)...")
    selected = list(SHORT_FIXED_FEATURES)
    pool_in_data = [f for f in SHORT_CANDIDATE_POOL if f in all_feature_names]
    if pool_in_data and N_FLEXIBLE > 0:
        neg_mask = y_train < 0
        pool_idx = [all_feature_names.index(f) for f in pool_in_data]
        if neg_mask.sum() > 1000:
            # Use stable_icir on negative-return bars (more robust than greedy)
            X_pool_neg = X_train[neg_mask][:, pool_idx]
            y_neg = y_train[neg_mask]
            flex = stable_icir_select(X_pool_neg, y_neg, pool_in_data, top_k=N_FLEXIBLE)
        else:
            X_pool = X_train[:, pool_idx]
            flex = stable_icir_select(X_pool, y_train, pool_in_data, top_k=N_FLEXIBLE)
        selected.extend(flex)
    print(f"  Selected: {selected}")

    sel_idx = [all_feature_names.index(n) for n in selected]
    X_train_sel = X_train[:, sel_idx]
    X_oos_sel = X_oos[:, sel_idx]

    # ── HPO (bear IC objective) ───────────────────────────
    params = dict(V7_DEFAULT_PARAMS)
    if use_hpo:
        print(f"  Running HPO ({hpo_trials} trials, bear IC objective)...")
        try:
            from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig

            n_tr = len(X_train_sel)
            val_size = min(n_tr // 4, 2190)
            X_hpo_train = X_train_sel[:-val_size]
            y_hpo_train = y_train[:-val_size]
            X_hpo_val = X_train_sel[-val_size:]
            y_hpo_val = y_train[-val_size:]

            objective_fn = _bear_ic_objective(
                X_hpo_train, y_hpo_train, X_hpo_val, y_hpo_val)

            opt = HyperOptimizer(
                search_space=V7_SEARCH_SPACE,
                objective_fn=objective_fn,
                config=HyperOptConfig(n_trials=hpo_trials, direction="maximize"),
            )
            result = opt.optimize()
            params = {**V7_DEFAULT_PARAMS, **result.best_params}
            print(f"  HPO best bear IC: {result.best_value:.4f}")
        except Exception as e:
            print(f"  HPO failed: {e}, using defaults")

    # ── Train LGBM (with bear sample weighting) ─────────
    print("  Training LGBM...")
    import lightgbm as lgb
    dtrain = lgb.Dataset(X_train_sel, label=y_train, weight=sample_weight)
    lgbm_bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 500),
        callbacks=[lgb.log_evaluation(0)],
    )
    lgbm_pred = lgbm_bst.predict(X_oos_sel)

    # ── Train XGB (with bear sample weighting) ────────────
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
    dtrain_xgb = xgb.DMatrix(X_train_sel, label=y_train, weight=sample_weight)
    doos_xgb = xgb.DMatrix(X_oos_sel)
    xgb_bst = xgb.train(
        xgb_params, dtrain_xgb,
        num_boost_round=params.get("n_estimators", 500),
    )
    xgb_pred = xgb_bst.predict(doos_xgb)

    # ── Ensemble ──────────────────────────────────────────
    y_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred
    print("  Ensemble: LGBM + XGB averaged")

    # ── OOS Evaluation (short-only, regime-gated) ──────
    print("\n  Evaluating OOS (short-only, bear regime gated)...")
    n_oos_bear = int(oos_bear_mask.sum())
    print(f"  OOS bear bars: {n_oos_bear}/{len(oos_bear_mask)} "
          f"({n_oos_bear/len(oos_bear_mask)*100:.1f}%)")
    metrics = _evaluate_oos_short(y_pred, oos_closes, y_oos_full, bear_mask=oos_bear_mask)

    print(f"\n  {'='*60}")
    print(f"  OOS RESULTS -- SHORT MODEL ({OOS_BARS // (24*30)} months)")
    print(f"  {'='*60}")
    print(f"  Sharpe:              {metrics['sharpe']:+.2f}")
    print(f"  Total return:        {metrics['total_return']*100:+.2f}%")
    print(f"  IC:                  {metrics['ic']:.4f}")
    print(f"  H2 IC:               {metrics['h2_ic']:.4f}")
    print(f"  Bear IC:             {metrics['bear_ic']:.4f}")
    print(f"  Short hit rate:      {metrics['short_hit_rate']:.1%}")
    print(f"  Short entries:       {metrics['n_short_entries']}")
    print(f"  Bootstrap P(S>0):    {metrics['bootstrap_p_positive']:.1%}")
    print(f"  Bootstrap 95% CI:    [{metrics['bootstrap_ci_95'][0]:.2f}, "
          f"{metrics['bootstrap_ci_95'][1]:.2f}]")
    print(f"  Positive months:     {metrics['positive_months']}/{metrics['n_months']}")
    print(f"  Active bars:         {metrics['n_active_bars']}/{metrics['n_total_bars']}")

    # Monthly detail
    if metrics["monthly_returns"]:
        print(f"\n  Monthly returns:")
        for i, r in enumerate(metrics["monthly_returns"]):
            marker = "+" if r > 0 else " "
            print(f"    Month {i+1:2d}: {marker}{r*100:+.2f}%")

    # ── Pass/Fail (relaxed for short model) ───────────────
    checks = {
        "OOS Sharpe > 0.3": metrics["sharpe"] > 0.3,
        "Bear IC > 0": metrics["bear_ic"] > 0,
        "Bootstrap P(S>0) > 60%": metrics["bootstrap_p_positive"] > 0.60,
        "Short hit rate > 45%": metrics["short_hit_rate"] > 0.45,
        f"Positive months >= {max(1, metrics['n_months'] * 6 // 18)}/{metrics['n_months']}":
            metrics["positive_months"] >= max(1, metrics["n_months"] * 6 // 18),
    }
    all_pass = all(checks.values())

    print(f"\n  {'='*60}")
    print(f"  PRODUCTION READINESS")
    print(f"  {'='*60}")
    for desc, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}: {desc}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # ── Save models (pickle required for LightGBM/XGBoost booster objects) ──
    print(f"\n  Saving to {out_dir}/...")

    lgbm_path = out_dir / "lgbm_short.pkl"
    with open(lgbm_path, "wb") as f:
        pickle.dump({"model": lgbm_bst, "features": selected}, f)
    sign_file(lgbm_path)

    xgb_path = out_dir / "xgb_short.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump({"model": xgb_bst, "features": selected}, f)
    sign_file(xgb_path)

    config = {
        "version": "short_v1",
        "symbol": symbol,
        "ensemble": True,
        "ensemble_weights": [0.5, 0.5],
        "models": ["lgbm_short.pkl", "xgb_short.pkl"],
        "features": selected,
        "fixed_features": SHORT_FIXED_FEATURES,
        "candidate_pool": SHORT_CANDIDATE_POOL,
        "n_flexible": N_FLEXIBLE,
        "short_only": True,
        "long_only": False,
        "regime_gated": True,
        "ma_window": MA_WINDOW,
        "bear_weight": BEAR_WEIGHT,
        "deadzone": DEADZONE,
        "min_hold": MIN_HOLD,
        "horizon": HORIZON,
        "target_mode": TARGET_MODE,
        "params": params,
        "xgb_params": xgb_params,
        "hpo_trials": hpo_trials,
        "metrics": metrics,
        "checks": {k: v for k, v in checks.items()},
        "passed": all_pass,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print(f"  Saved: lgbm_short.pkl, xgb_short.pkl, config.json")

    # ── Register in ModelRegistry ─────────────────────────
    try:
        from research.model_registry.registry import ModelRegistry
        registry = ModelRegistry()
        mv = registry.register(
            name=f"alpha_short_{symbol}",
            params=params,
            features=selected,
            metrics={
                "oos_sharpe": metrics["sharpe"],
                "oos_return": metrics["total_return"],
                "oos_ic": metrics["ic"],
                "bear_ic": metrics["bear_ic"],
                "short_hit_rate": metrics["short_hit_rate"],
                "bootstrap_p": metrics["bootstrap_p_positive"],
                "positive_months": float(metrics["positive_months"]),
                "n_months": float(metrics["n_months"]),
            },
            tags=["short", "ensemble", "production" if all_pass else "candidate"],
        )
        if all_pass:
            registry.promote(mv.model_id)
            print(f"  Registered & promoted: {mv.name} v{mv.version} (id={mv.model_id})")
        else:
            print(f"  Registered (not promoted): {mv.name} v{mv.version}")
    except Exception as e:
        print(f"  Registry failed: {e} (models saved locally)")

    if all_pass:
        print(f"\n  Short model ready at {out_dir}/")
        print(f"  Next: integrate with LiveInferenceBridge (short_model param)")
    else:
        print(f"\n  Model did not pass all checks. Review metrics before proceeding.")


if __name__ == "__main__":
    main()
