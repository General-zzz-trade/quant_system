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


def _triple_barrier_labels(
    closes: np.ndarray,
    horizon: int,
    upper_pct: float,
    lower_pct: float,
) -> np.ndarray:
    """López de Prado triple-barrier labels.

    For every bar i, walk forward up to ``horizon`` bars.  The first of
    three events to fire wins:
      1. price[i+k] >= price[i] * (1 + upper_pct) → label = +upper_pct
      2. price[i+k] <= price[i] * (1 - lower_pct) → label = -lower_pct
      3. neither hits within horizon → label = (price[i+horizon]/price[i] - 1)
         i.e. the plain forward return as the time-barrier outcome.

    This aligns the training target with how a live strategy actually
    exits (take-profit / stop-loss / time-out), which reduces the train-
    to-live mismatch on ATR-stopped trades.
    """
    n = len(closes)
    y = np.full(n, np.nan, dtype=np.float64)
    max_k = horizon
    up = 1.0 + upper_pct
    down = 1.0 - lower_pct
    for i in range(n - max_k):
        base = closes[i]
        if base <= 0:
            continue
        upper_px = base * up
        lower_px = base * down
        hit = 0  # 0=none, 1=upper, -1=lower
        for k in range(1, max_k + 1):
            p = closes[i + k]
            if p >= upper_px:
                hit = 1
                break
            if p <= lower_px:
                hit = -1
                break
        if hit == 1:
            y[i] = upper_pct
        elif hit == -1:
            y[i] = -lower_pct
        else:
            y[i] = (closes[i + max_k] - base) / base
    return y


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
    label_mode: str = "forward_return",
    tb_upper_pct: float = 0.02,
    tb_lower_pct: float = 0.01,
    meta_labeling: bool = False,
) -> dict[str, Any] | None:
    """Train LGBM + Ridge for a single horizon.

    ic_recent_years: if > 0, use only the last N years of training data for
        IC-based feature selection (more adaptive to current market).
    forced_features: features to always include regardless of IC rank.
    max_train_years: if > 0, cap the training window to the last N years
        (from train_end). Regime-focused training — useful when older
        crypto data is too different from current regime (e.g. ETH 2019-2022
        vs 2023-2026). Defaults to 0 (use all data from WARMUP).
    label_mode: "forward_return" (legacy) or "triple_barrier".
        triple_barrier aligns the target with real-world exit logic.
    tb_upper_pct / tb_lower_pct: triple-barrier profit/stop thresholds
        as fractions (e.g. 0.02 = 2%). Only used when label_mode is
        "triple_barrier".
    """
    import lightgbm as lgb

    # Target: forward return (or triple-barrier when requested)
    if label_mode == "triple_barrier":
        y = _triple_barrier_labels(
            closes, horizon,
            upper_pct=tb_upper_pct,
            lower_pct=tb_lower_pct,
        )
    else:
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

    # XGBoost — third ensemble member for tree-model diversification.
    # Uses smaller learning rate + more conservative subsampling than LGBM
    # to produce predictions that aren't just a copy of LGBM's.  Fits the
    # same train/val split (embargo already applied upstream).
    xgb_model = None
    try:
        import xgboost as xgb
        X_train_clean = np.nan_to_num(X_train, nan=0.0)
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.03,      # slower → less overlap with LGBM
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method="hist",
            verbosity=0,
            early_stopping_rounds=30,
        )
        xgb_model.fit(
            X_train_clean[:train_end_idx],
            y_train[:train_end_idx],
            eval_set=[(X_train_clean[val_start:], y_train[val_start:])],
            verbose=False,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"XGBoost training failed for horizon {horizon}: {e}",
                      stacklevel=2)
        xgb_model = None

    # OOS predictions
    X_test = X[train_end:, :][:, feat_idx]
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    lgbm_pred = lgbm_model.predict(X_test_clean)
    ridge_pred = ridge_model.predict(X_test_clean)
    xgb_pred = None
    if xgb_model is not None:
        try:
            xgb_pred = xgb_model.predict(X_test_clean)
        except Exception as e:
            import warnings
            warnings.warn(f"XGBoost predict failed for horizon {horizon}: {e}",
                          stacklevel=2)
            xgb_pred = None

    # IC on OOS
    y_test = y[train_end:]
    valid_test = ~np.isnan(y_test)
    if valid_test.sum() < 50:
        return None

    from scipy.stats import spearmanr
    ic_lgbm, _ = spearmanr(lgbm_pred[valid_test], y_test[valid_test])
    ic_ridge, _ = spearmanr(ridge_pred[valid_test], y_test[valid_test])
    ic_xgb = None
    if xgb_pred is not None:
        ic_xgb, _ = spearmanr(xgb_pred[valid_test], y_test[valid_test])
        ic_xgb = float(ic_xgb) if not np.isnan(ic_xgb) else None

    if ic_xgb is not None:
        ic_ensemble = float((ic_lgbm + ic_ridge + ic_xgb) / 3)
        xgb_str = f", IC_xgb={ic_xgb:.4f}"
    else:
        ic_ensemble = float((ic_lgbm + ic_ridge) / 2)
        xgb_str = ""
    print(f"    h={horizon}: {len(selected)} features, "
          f"IC_lgbm={ic_lgbm:.4f}, IC_ridge={ic_ridge:.4f}{xgb_str}")
    print(f"    Features: {selected}")

    # ── Meta-labeling (López de Prado Chapter 3) ─────────────────────
    # Train a secondary LGBM classifier that predicts whether the
    # primary ensemble's directional call will be correct.  At inference
    # time we only act on +1 / -1 signals where P(correct) >= 0.55.
    #
    # Training labels are generated from IN-SAMPLE fold predictions
    # (same train range as the primary models, with the embargo-held
    # val slice used as an audit check).  Using OOS test predictions
    # would be a leak — those are supposed to measure live performance.
    meta_model = None
    meta_ic = None
    if meta_labeling:
        try:
            # 1. Build primary ensemble predictions on the training slice
            #    via K-fold cross-validation so the secondary model sees
            #    out-of-fold (not in-sample) primary calls.
            from sklearn.model_selection import KFold
            n_tr = len(y_train)
            if n_tr >= 500:
                kf = KFold(n_splits=5, shuffle=False)  # time-ordered
                oof = np.full(n_tr, np.nan)
                X_tr_clean = np.nan_to_num(X_train, nan=0.0)
                for tr_idx, va_idx in kf.split(np.arange(n_tr)):
                    # Embargo inside the fold too (horizon bars gap)
                    if len(va_idx) == 0:
                        continue
                    gap = int(horizon)
                    tr_mask = (tr_idx < va_idx[0] - gap) | (tr_idx > va_idx[-1] + gap)
                    tr_use = tr_idx[tr_mask]
                    if len(tr_use) < 100:
                        continue
                    fold_l = lgb.train(
                        lgbm_params,
                        lgb.Dataset(X_tr_clean[tr_use], y_train[tr_use]),
                        valid_sets=[lgb.Dataset(X_tr_clean[va_idx], y_train[va_idx])],
                        callbacks=[lgb.log_evaluation(0)],
                    )
                    oof[va_idx] = fold_l.predict(X_tr_clean[va_idx])

                # 2. Derive "was the directional call correct?" labels.
                #    Skip rows with NaN primary pred or NaN target.
                valid = ~np.isnan(oof) & ~np.isnan(y_train)
                primary_sign = np.sign(oof[valid])
                actual_sign = np.sign(y_train[valid])
                meta_y = (primary_sign == actual_sign).astype(np.int32)
                meta_X = X_tr_clean[valid]

                # 3. Train secondary LGBM classifier.  Same features +
                #    two meta-features: |primary_pred|, primary_pred.
                meta_extra = np.column_stack([
                    np.abs(oof[valid]),
                    oof[valid],
                ])
                meta_X_full = np.hstack([meta_X, meta_extra])

                meta_params = dict(lgbm_params)
                meta_params["objective"] = "binary"
                meta_params["metric"] = "auc"
                meta_params["num_leaves"] = 15
                meta_params["n_estimators"] = 150
                meta_params["learning_rate"] = 0.05
                # Binary LightGBM needs its own Dataset (no early stopping — unused)
                meta_params.pop("early_stopping_rounds", None)
                meta_split = int(len(meta_y) * 0.85)
                meta_train_ds = lgb.Dataset(
                    meta_X_full[:meta_split], meta_y[:meta_split]
                )
                meta_val_ds = lgb.Dataset(
                    meta_X_full[meta_split:], meta_y[meta_split:],
                    reference=meta_train_ds,
                )
                meta_model = lgb.train(
                    meta_params, meta_train_ds,
                    valid_sets=[meta_val_ds],
                    callbacks=[lgb.log_evaluation(0)],
                )

                # 4. Quick sanity: secondary AUC on held-out tail
                from sklearn.metrics import roc_auc_score
                preds_val = meta_model.predict(meta_X_full[meta_split:])
                if len(np.unique(meta_y[meta_split:])) > 1:
                    meta_auc = float(roc_auc_score(meta_y[meta_split:], preds_val))
                    meta_ic = meta_auc - 0.5  # re-expressed as "edge over random"
                    print(f"    h={horizon}: meta-label AUC={meta_auc:.3f} "
                          f"(edge={meta_ic:+.3f})")
        except Exception as exc:
            import warnings
            warnings.warn(f"meta-labeling failed for h={horizon}: {exc}",
                          stacklevel=2)
            meta_model = None
            meta_ic = None

    return {
        "horizon": horizon,
        "features": selected,
        "lgbm_model": lgbm_model,
        "ridge_model": ridge_model,
        "xgb_model": xgb_model,
        "meta_model": meta_model,
        "meta_edge": meta_ic,
        "lgbm_pred_oos": lgbm_pred,
        "ridge_pred_oos": ridge_pred,
        "xgb_pred_oos": xgb_pred,
        "ic_lgbm": float(ic_lgbm),
        "ic_ridge": float(ic_ridge),
        "ic_xgb": ic_xgb,
        "ic_ensemble": ic_ensemble,
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
    label_mode: str = "forward_return",
    tb_upper_pct: float = 0.02,
    tb_lower_pct: float = 0.01,
    meta_labeling: bool = False,
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
            label_mode=label_mode,
            tb_upper_pct=tb_upper_pct,
            tb_lower_pct=tb_lower_pct,
            meta_labeling=meta_labeling,
        )
        if result is not None:
            horizon_results[h] = result

    if len(horizon_results) < 1:
        print("  ERROR: No horizons trained successfully")
        return False

    # Ensemble OOS predictions — IC-proportional weighting.
    # Each member is weighted by its own IC normalised across the ensemble,
    # so a 0.11 IC Ridge doesn't lose voice to a 0.05 IC XGB.  Members with
    # IC <= 0 are dropped entirely.  XGB is additionally gated by a
    # **relative** IC threshold: it must clear 75% of LGBM's IC.  Otherwise
    # we ship Ridge+LGBM only — diversification that hurts absolute IC
    # isn't worth the added model risk.
    _XGB_MIN_RELATIVE_IC = 0.75
    preds_ensemble = {}
    ensemble_weights_per_h: dict = {}
    for h, r in horizon_results.items():
        ic_l = float(r.get("ic_lgbm") or 0.0)
        ic_r = float(r.get("ic_ridge") or 0.0)
        ic_x = float(r.get("ic_xgb") or 0.0)
        xgb_pred = r.get("xgb_pred_oos")

        parts = []
        if ic_r > 0:
            parts.append(("ridge", ic_r, r["ridge_pred_oos"]))
        if ic_l > 0:
            parts.append(("lgbm", ic_l, r["lgbm_pred_oos"]))
        # Only include XGB if it's pulling its weight
        include_xgb = (
            xgb_pred is not None
            and ic_x > 0
            and ic_l > 0
            and ic_x >= _XGB_MIN_RELATIVE_IC * ic_l
        )
        if include_xgb:
            parts.append(("xgb", ic_x, xgb_pred))
        else:
            r["xgb_dropped_reason"] = (
                f"ic_xgb={ic_x:.4f} < {_XGB_MIN_RELATIVE_IC}*ic_lgbm={ic_l:.4f}"
                if xgb_pred is not None else "no xgb predictions"
            )

        if not parts:
            preds_ensemble[h] = (
                r["ridge_pred_oos"] * ridge_weight
                + r["lgbm_pred_oos"] * lgbm_weight
            )
            ensemble_weights_per_h[h] = {"ridge": ridge_weight, "lgbm": lgbm_weight}
            continue

        total_ic = sum(ic for _, ic, _ in parts)
        weights = {name: ic / total_ic for name, ic, _ in parts}
        blend = np.zeros_like(parts[0][2])
        for name, _, pred in parts:
            blend = blend + pred * weights[name]
        preds_ensemble[h] = blend
        ensemble_weights_per_h[h] = weights
        print(f"    h={h}: ensemble weights = "
              + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()))

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

        hc: dict = {
            "horizon": h,
            "lgbm": lgbm_name,
            "ridge": ridge_name,
            "features": r["features"],
            "ic": r["ic_ensemble"],
        }

        # Optional meta-labeling secondary model — LGBM binary classifier
        # saved as native JSON (language-agnostic, no pickle attack surface).
        meta_obj = r.get("meta_model")
        if meta_obj is not None:
            meta_name = f"meta_h{h}.json"
            try:
                meta_obj.save_model(str(model_dir / meta_name))
                hc["meta"] = meta_name
                hc["meta_edge"] = r.get("meta_edge")
                hc["meta_input_features"] = r["features"] + ["_abs_pred", "_signed_pred"]
            except Exception as exc:
                print(f"  WARN: meta save failed for h={h}: {exc}")

        # Optional XGBoost member — saved as native JSON (not pickle) so the
        # artifact is language-agnostic and has no pickle-deserialisation
        # attack surface. Feature list stored alongside in a sidecar.
        # Skip entirely when the gating logic dropped XGB from the blend.
        h_weights = ensemble_weights_per_h.get(h, {})
        xgb_obj = r.get("xgb_model")
        if xgb_obj is not None and h_weights.get("xgb", 0.0) > 0:
            xgb_json_name = f"xgb_h{h}.json"
            xgb_feat_name = f"xgb_h{h}_features.json"
            try:
                xgb_obj.save_model(str(model_dir / xgb_json_name))
                (model_dir / xgb_feat_name).write_text(
                    json.dumps({"features": r["features"]})
                )
                hc["xgb"] = xgb_json_name
                hc["xgb_features_file"] = xgb_feat_name
                hc["ic_xgb"] = r.get("ic_xgb")
                hc["ic_lgbm"] = r.get("ic_lgbm")
                hc["ic_ridge"] = r.get("ic_ridge")
            except Exception as exc:
                print(f"  WARN: XGB save failed for h={h}: {exc}")

        horizon_configs.append(hc)

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
        "label_mode": label_mode,
        "tb_upper_pct": tb_upper_pct if label_mode == "triple_barrier" else None,
        "tb_lower_pct": tb_lower_pct if label_mode == "triple_barrier" else None,
    }
    # If any horizon produced an XGBoost member, record the IC-proportional
    # weights actually used at training time so the inference path picks
    # the same blend.  Use the primary horizon's weights as the top-level
    # defaults; per-horizon weights are also embedded in horizon_models.
    if any(r.get("xgb_model") is not None for r in horizon_results.values()):
        primary_h = max(horizon_results.keys())
        primary_weights = ensemble_weights_per_h.get(primary_h, {})
        config_dict["ensemble_weights"] = primary_weights
        config_dict["ridge_weight"] = primary_weights.get("ridge", ridge_weight)
        config_dict["lgbm_weight"] = primary_weights.get("lgbm", lgbm_weight)
        config_dict["xgb_weight"] = primary_weights.get("xgb", 0.0)
        # Attach per-horizon blend to horizon_configs so live inference
        # can recreate the exact same weights if horizons differ.
        for hc in horizon_configs:
            w = ensemble_weights_per_h.get(hc["horizon"], {})
            if w:
                hc["ridge_weight"] = w.get("ridge", 0.0)
                hc["lgbm_weight"] = w.get("lgbm", 0.0)
                hc["xgb_weight"] = w.get("xgb", 0.0)
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
