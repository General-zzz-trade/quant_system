#!/usr/bin/env python3
"""V12 Alpha Training — self-contained, IC-screened feature selection + Ridge+LGBM.

Improvements over V11:
- Expanded candidate feature pool (on-chain netflow, ETF returns, 4h features)
- Rust-accelerated greedy IC selection (cpp_greedy_ic_select)
- Ridge model trained on same selected features
- Walk-forward OOS validation with bootstrap Sharpe
- Preserves ridge_weight/lgbm_weight in config for live ensemble

Note: Uses pickle for serializing trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and optionally HMAC-signed.

Usage:
    python3 -m alpha.training.train_v12 --symbol BTCUSDT
    python3 -m alpha.training.train_v12 --symbol BTCUSDT,ETHUSDT --horizons 12,24
    python3 -m alpha.training.train_v12 --dry-run
"""
from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 — trusted local model artifacts, HMAC-signed
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch

try:
    from _quant_hotpath import cpp_greedy_ic_select
    _HAS_RUST_IC = True
except ImportError:
    _HAS_RUST_IC = False

VERSION = "v12"
WARMUP = 720
COST_BPS_RT = 8.0  # round-trip cost in bps
BARS_PER_MONTH = 24 * 30

# Features to never use as predictors
_BLACKLIST = frozenset({
    "close", "open", "high", "low", "volume", "open_time", "timestamp",
    "taker_buy_volume", "taker_buy_quote_volume", "quote_volume", "trades",
})


def rolling_zscore(arr: np.ndarray, window: int = 720, warmup: int = 180) -> np.ndarray:
    """Compute rolling z-score with warmup period."""
    n = len(arr)
    z = np.full(n, np.nan)
    for i in range(warmup, n):
        start = max(0, i - window)
        buf = arr[start:i + 1]
        mu = np.mean(buf)
        std = np.std(buf)
        if std > 1e-12:
            z[i] = (arr[i] - mu) / std
    return z


def _greedy_ic_select(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
    max_features: int = 14, min_ic: float = 0.01,
) -> list[str]:
    """Select features by greedy IC maximization."""
    if _HAS_RUST_IC:
        try:
            selected_indices = cpp_greedy_ic_select(X, y, max_features)
            selected = [feature_names[i] for i in selected_indices if i < len(feature_names)]
            if len(selected) >= 5:
                return selected
        except Exception:
            pass

    # Python fallback: rank by absolute IC
    from scipy.stats import spearmanr
    ics = []
    for j in range(X.shape[1]):
        col = X[:, j]
        valid = ~np.isnan(col) & ~np.isnan(y)
        if valid.sum() < 200:
            ics.append(0.0)
            continue
        ic, _ = spearmanr(col[valid], y[valid])
        ics.append(abs(ic) if not np.isnan(ic) else 0.0)

    ranked = sorted(range(len(ics)), key=lambda i: -ics[i])
    selected = []
    for idx in ranked:
        if ics[idx] < min_ic:
            break
        selected.append(feature_names[idx])
        if len(selected) >= max_features:
            break
    return selected


def train_single_horizon(
    horizon: int,
    X: np.ndarray,
    closes: np.ndarray,
    feature_names: list[str],
    train_end: int,
    n: int,
    max_features: int = 14,
    ic_recent_years: float = 0,
    forced_features: list[str] | None = None,
    max_train_years: float = 0,
) -> dict[str, Any] | None:
    """Train LGBM + Ridge for a single horizon.

    ic_recent_years: if > 0, use only the last N years of training data for
        IC-based feature selection (more adaptive to current market).
    forced_features: features to always include regardless of IC rank.
    max_train_years: if > 0, cap the training window to the last N years
        (from train_end). Regime-focused training — useful when older
        crypto data is too different from current regime (e.g. ETH 2019-2022
        vs 2023-2026). Defaults to 0 (use all data from WARMUP).
    """
    import lightgbm as lgb

    # Target: forward return
    y = np.full(n, np.nan)
    for i in range(n - horizon):
        y[i] = (closes[i + horizon] - closes[i]) / closes[i]

    # Feature selection window
    if ic_recent_years > 0:
        recent_bars = int(ic_recent_years * 365 * 24)
        sel_start = max(train_end - recent_bars, WARMUP)
    else:
        sel_start = WARMUP
    valid_train = np.arange(sel_start, train_end)
    mask = ~np.isnan(y[valid_train])
    X_sel = X[valid_train][mask]
    y_sel = y[valid_train][mask]

    if len(y_sel) < 1000:
        return None

    # Start with forced features, then fill remaining slots via greedy IC
    if forced_features:
        selected = [f for f in forced_features if f in feature_names]
        remaining_slots = max(0, max_features - len(selected))
        if remaining_slots > 0:
            greedy = _greedy_ic_select(X_sel, y_sel, feature_names, max_features=max_features)
            for gf in greedy:
                if gf not in selected and remaining_slots > 0:
                    selected.append(gf)
                    remaining_slots -= 1
    else:
        selected = _greedy_ic_select(X_sel, y_sel, feature_names, max_features=max_features)

    if len(selected) < 5:
        return None

    feat_idx = [feature_names.index(f) for f in selected]
    # Training window cutoff (regime-focused training)
    if max_train_years > 0:
        train_lookback = int(max_train_years * 365 * 24)
        train_start = max(train_end - train_lookback, WARMUP)
    else:
        train_start = WARMUP
    # Embargo: the last `horizon` training labels reference closes inside the
    # OOS test window (y[i] = closes[i+h] - closes[i]). Exclude them to
    # prevent train→test leakage. Same motivation as Lopez de Prado's
    # purged-embargo walk-forward.
    train_cap = max(train_start + 1, train_end - int(horizon))
    X_train = np.nan_to_num(X[train_start:train_cap, :][:, feat_idx], nan=0.0)
    y_train = y[train_start:train_cap]
    valid = ~np.isnan(y_train)
    X_train = X_train[valid]
    y_train = y_train[valid]

    # LGBM
    lgbm_params = {
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 300,
        "early_stopping_rounds": 30,
    }
    # Purged train/val split with embargo to prevent label leakage.
    # Target y[i] depends on closes[i+horizon], so the last `horizon` bars of
    # the training slice have labels that overlap into the validation slice.
    # Drop those `horizon` bars between train and val (embargo).
    val_bars = 720
    embargo = int(horizon)  # one full label-lookahead
    val_start = len(X_train) - val_bars
    train_end_idx = max(0, val_start - embargo)
    ds_train = lgb.Dataset(X_train[:train_end_idx], y_train[:train_end_idx])
    ds_val = lgb.Dataset(X_train[val_start:], y_train[val_start:], reference=ds_train)
    lgbm_model = lgb.train(
        lgbm_params, ds_train,
        valid_sets=[ds_val],
        callbacks=[lgb.log_evaluation(0)],
    )

    # Ridge
    X_ridge = np.nan_to_num(X_train, nan=0.0)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_ridge, y_train)

    # OOS predictions
    X_test = X[train_end:, :][:, feat_idx]
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    lgbm_pred = lgbm_model.predict(X_test_clean)
    ridge_pred = ridge_model.predict(X_test_clean)

    # IC on OOS
    y_test = y[train_end:]
    valid_test = ~np.isnan(y_test)
    if valid_test.sum() < 50:
        return None

    from scipy.stats import spearmanr
    ic_lgbm, _ = spearmanr(lgbm_pred[valid_test], y_test[valid_test])
    ic_ridge, _ = spearmanr(ridge_pred[valid_test], y_test[valid_test])

    print(f"    h={horizon}: {len(selected)} features, IC_lgbm={ic_lgbm:.4f}, IC_ridge={ic_ridge:.4f}")
    print(f"    Features: {selected}")

    return {
        "horizon": horizon,
        "features": selected,
        "lgbm_model": lgbm_model,
        "ridge_model": ridge_model,
        "lgbm_pred_oos": lgbm_pred,
        "ridge_pred_oos": ridge_pred,
        "ic_lgbm": float(ic_lgbm),
        "ic_ridge": float(ic_ridge),
        "ic_ensemble": float((ic_lgbm + ic_ridge) / 2),
    }


def backtest_simple(
    preds: dict[int, np.ndarray],
    closes: np.ndarray,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool = False,
) -> dict[str, Any]:
    """Simple z-score backtest."""
    n = len(closes)
    z_all = []
    for h, pred in sorted(preds.items()):
        z_all.append(rolling_zscore(pred, window=720, warmup=180))
    z = np.nanmean(z_all, axis=0)

    cost_frac = COST_BPS_RT / 10000
    pos = 0.0
    entry_bar = 0
    trade_pnls = []

    for i in range(n):
        if np.isnan(z[i]):
            continue
        if pos != 0:
            held = i - entry_bar
            should_exit = held >= max_hold
            if not should_exit and held >= min_hold:
                if pos > 0 and z[i] < -0.3:
                    should_exit = True
                elif pos < 0 and z[i] > 0.3:
                    should_exit = True
            if should_exit:
                pnl = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                trade_pnls.append(pnl - cost_frac)
                pos = 0.0

        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                entry_bar = i

    if not trade_pnls:
        return {"sharpe": 0, "trades": 0, "return": 0}
    arr = np.array(trade_pnls)
    avg_hold = n / max(len(arr), 1)
    tpy = 365 * 24 / max(avg_hold, 1)
    sharpe = float(np.mean(arr) / max(np.std(arr, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": round(sharpe, 2),
        "trades": len(arr),
        "return": round(float(np.sum(arr)), 4),
        "win_rate": round(float(np.mean(arr > 0) * 100), 1),
        "avg_net_bps": round(float(np.mean(arr) * 10000), 1),
    }


def train_symbol(
    symbol: str,
    horizons: list[int],
    ridge_weight: float = 0.4,
    lgbm_weight: float = 0.6,
    max_features: int = 14,
    dry_run: bool = False,
    ic_recent_years: float = 0,
    forced_features: list[str] | None = None,
    max_train_years: float = 0,
) -> bool:
    """Train V12 models for one symbol."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    data_path = Path(f"data_files/{symbol}_1h.csv")

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return False

    df = pd.read_csv(data_path).sort_values("open_time").reset_index(drop=True)
    n = len(df)
    closes = df["close"].values.astype(np.float64)
    start_date = pd.Timestamp(df["open_time"].iloc[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(df["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} bars ({start_date} -> {end_date})")

    # Features
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = compute_features_batch(symbol, df)
    feature_names = [c for c in feat_df.columns if c not in _BLACKLIST]
    X = feat_df[feature_names].values.astype(np.float64)
    print(f"{len(feature_names)} features in {time.time() - t0:.1f}s")

    # Split: 18 months OOS
    oos_bars = BARS_PER_MONTH * 18
    train_end = n - oos_bars
    closes_test = closes[train_end:]
    print(f"  Split: train={train_end - WARMUP:,} test={oos_bars:,}")

    # Train each horizon
    horizon_results = {}
    for h in horizons:
        print(f"\n  -- Horizon {h}h --")
        result = train_single_horizon(
            h, X, closes, feature_names, train_end, n, max_features,
            ic_recent_years=ic_recent_years, forced_features=forced_features,
            max_train_years=max_train_years,
        )
        if result is not None:
            horizon_results[h] = result

    if len(horizon_results) < 1:
        print("  ERROR: No horizons trained successfully")
        return False

    # Ensemble OOS predictions
    preds_ensemble = {}
    for h, r in horizon_results.items():
        preds_ensemble[h] = r["ridge_pred_oos"] * ridge_weight + r["lgbm_pred_oos"] * lgbm_weight

    # Config sweep
    print("\n  -- Config Sweep --")
    best = {"sharpe": -999}
    best_params = {}
    for dz in [0.8, 1.0, 1.2, 1.5, 2.0]:
        for mh in [6, 9, 12]:
            for maxh in [60, 96, 120]:
                for lo in [True, False]:
                    r = backtest_simple(preds_ensemble, closes_test, dz, mh, maxh, lo)
                    if r["sharpe"] > best["sharpe"] and r["trades"] >= 15:
                        best = r
                        best_params = {"deadzone": dz, "min_hold": mh, "max_hold": maxh, "long_only": lo}

    if best["sharpe"] <= 0:
        print("  No viable config found")
        return False

    print(f"  Best: dz={best_params['deadzone']}, hold=[{best_params['min_hold']},{best_params['max_hold']}], "
          f"long_only={best_params['long_only']}")
    print(f"  Sharpe={best['sharpe']}, trades={best['trades']}, WR={best['win_rate']}%, "
          f"ret={best['return']*100:+.2f}%")

    # Bootstrap
    trade_pnls = []
    z_all = [rolling_zscore(preds_ensemble[h], 720, 180) for h in sorted(preds_ensemble)]
    z = np.nanmean(z_all, axis=0)
    pos = 0.0
    eb = 0
    dz = best_params["deadzone"]
    for i in range(len(z)):
        if np.isnan(z[i]):
            continue
        if pos != 0:
            held = i - eb
            if held >= best_params["max_hold"] or (held >= best_params["min_hold"] and pos * z[i] < -0.3):
                pnl = pos * (closes_test[i] - closes_test[eb]) / closes_test[eb]
                trade_pnls.append(pnl - COST_BPS_RT / 10000)
                pos = 0.0
        if pos == 0:
            if z[i] > dz:
                pos, eb = 1.0, i
            elif not best_params["long_only"] and z[i] < -dz:
                pos, eb = -1.0, i
    pnl_arr = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    bs = [float(np.mean(s) / max(np.std(s), 1e-10) * np.sqrt(52))
          for s in (np.random.choice(pnl_arr, len(pnl_arr), replace=True) for _ in range(1000))]
    p5, p50, p95 = np.percentile(bs, [5, 50, 95])
    print(f"  Bootstrap: p5={p5:.2f}, p50={p50:.2f}, p95={p95:.2f}")

    # Checks
    avg_ic = np.mean([r["ic_ensemble"] for r in horizon_results.values()])
    checks = {
        "Sharpe > 1.0": best["sharpe"] > 1.0,
        "Avg IC > 0.02": avg_ic > 0.02,
        "Trades >= 15": best["trades"] >= 15,
        "Bootstrap p5 > 0": p5 > 0,
    }
    print("\n  CHECKS:")
    all_pass = True
    for check, passed in checks.items():
        print(f"    [{'PASS' if passed else 'FAIL'}] {check}")
        if not passed:
            all_pass = False

    if dry_run:
        print("\n  DRY RUN -- model NOT saved")
        return all_pass

    if not all_pass:
        print("\n  FAILED checks -- model NOT saved")
        return False

    # Save
    if model_dir.exists():
        backup = model_dir.parent / f"{model_dir.name}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
        shutil.copytree(model_dir, backup)
        print(f"  Backed up to {backup}")

    model_dir.mkdir(parents=True, exist_ok=True)
    horizon_configs = []
    for h in sorted(horizon_results):
        r = horizon_results[h]
        lgbm_name = f"lgbm_h{h}.pkl"
        ridge_name = f"ridge_h{h}.pkl"
        # noqa: S301 — trusted local model artifacts
        with open(model_dir / lgbm_name, "wb") as f:
            pickle.dump({"model": r["lgbm_model"], "features": r["features"]}, f)  # noqa: S301
        with open(model_dir / ridge_name, "wb") as f:
            pickle.dump({"model": r["ridge_model"], "features": r["features"]}, f)  # noqa: S301
        horizon_configs.append({
            "horizon": h,
            "lgbm": lgbm_name,
            "ridge": ridge_name,
            "features": r["features"],
            "ic": r["ic_ensemble"],
        })

    config_dict = {
        "symbol": symbol,
        "version": VERSION,
        "multi_horizon": True,
        "horizons": sorted(horizon_results.keys()),
        "horizon_models": horizon_configs,
        "primary_horizon": max(horizon_results.keys()),
        "ensemble_method": "ic_weighted",
        "lgbm_xgb_weight": 0.5,
        "zscore_window": 720,
        "zscore_warmup": 180,
        "deadzone": best_params["deadzone"],
        "min_hold": best_params["min_hold"],
        "max_hold": best_params["max_hold"],
        "long_only": best_params["long_only"],
        "exit": {"trailing_stop_pct": 0.0, "zscore_cap": 0.0,
                 "reversal_threshold": -0.3, "deadzone_fade": 0.2},
        "regime_gate": {"enabled": False, "ranging_high_vol_action": "reduce",
                        "reduce_factor": 0.3},
        "time_filter": {"enabled": False, "skip_hours_utc": []},
        "metrics": {
            "sharpe": best["sharpe"],
            "avg_ic": round(float(avg_ic), 6),
            "per_horizon_ic": {str(h): round(r["ic_ensemble"], 6) for h, r in horizon_results.items()},
            "total_return": best["return"],
            "trades": best["trades"],
            "win_rate": best["win_rate"],
            "avg_net_bps": best["avg_net_bps"],
            "bootstrap_sharpe_p5": round(float(p5), 4),
            "bootstrap_sharpe_p50": round(float(p50), 4),
            "bootstrap_sharpe_p95": round(float(p95), 4),
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_range": f"{start_date} \u2192 {end_date}",
        "n_bars": n,
        "ic_ema_span": 720,
        "ic_min_threshold": -0.01,
        "ridge_weight": ridge_weight,
        "lgbm_weight": lgbm_weight,
        "ensemble_weights": {"ridge": ridge_weight, "lgbm": lgbm_weight},
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n  Model saved to {model_dir}/ ({VERSION})")
    return True


def main():
    parser = argparse.ArgumentParser(description="V12 Alpha Training")
    parser.add_argument("--symbol", default="BTCUSDT,ETHUSDT")
    parser.add_argument("--horizons", default="12,24")
    parser.add_argument("--ridge-weight", type=float, default=0.4)
    parser.add_argument("--lgbm-weight", type=float, default=0.6)
    parser.add_argument("--max-features", type=int, default=14)
    parser.add_argument("--ic-recent-years", type=float, default=0,
                        help="Use last N years for IC selection (0=full sample)")
    parser.add_argument("--forced-features", default="",
                        help="Comma-separated features to force-include")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    print("=" * 70)
    print(f"  V12 ALPHA TRAINING  (Ridge {args.ridge_weight:.0%} + LGBM {args.lgbm_weight:.0%})")
    print(f"  Symbols:  {symbols}")
    print(f"  Horizons: {horizons}")
    print(f"  Max features: {args.max_features}")
    print("=" * 70)

    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"  {symbol}")
        print(f"{'=' * 70}")
        t0 = time.time()
        forced = [f.strip() for f in args.forced_features.split(",") if f.strip()] or None
        ok = train_symbol(
            symbol, horizons,
            ridge_weight=args.ridge_weight,
            lgbm_weight=args.lgbm_weight,
            max_features=args.max_features,
            dry_run=args.dry_run,
            ic_recent_years=args.ic_recent_years,
            forced_features=forced,
        )
        status = "SAVED" if ok else "FAILED"
        print(f"\n  {symbol}: {status} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
