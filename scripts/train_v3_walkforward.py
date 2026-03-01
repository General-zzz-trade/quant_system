#!/usr/bin/env python3
"""Train V3 Alpha — walk-forward with early stopping, vol-normalized target, Spearman IC.

Key improvements over V2:
  1. Expanding-window walk-forward (5-fold) instead of single 70/30 split
  2. Early stopping with embargo in each fold
  3. Vol-normalized target to equalize regime contributions
  4. Spearman rank IC feature selection (robust to fat tails)
  5. Blacklisted garbage features (OI/LS with insufficient data)
  6. Deflated Sharpe Ratio gate for statistical significance
  7. True OOS evaluation on held-out _oos.csv files

Usage:
    python3 -m scripts.train_v3_walkforward --all
    python3 -m scripts.train_v3_walkforward --symbol BTCUSDT
    python3 -m scripts.train_v3_walkforward --all --top-k 20 --horizon 5 --n-folds 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import spearman_ic_select

try:
    from features._quant_rolling import cpp_vol_normalized_target as _cpp_vol_target
    _TARGET_CPP = True
except ImportError:
    _TARGET_CPP = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

V3_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.02,
    "num_leaves": 20,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "objective": "regression",
    "verbosity": -1,
}

EARLY_STOPPING_ROUNDS = 50
EMBARGO_BARS = 5

BLACKLIST = {
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    "avg_trade_size",
    "oi_chg_x_ret1",
}

INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
]

# ── Data loading (reuses V2 pattern) ────────────────────────


def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
    """Load a CSV schedule as {timestamp_ms: value}."""
    import csv
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row[ts_col])] = float(row[val_col])
    return schedule


def _compute_features(
    symbol: str,
    df: pd.DataFrame,
    funding_schedule: Dict[int, float],
    oi_schedule: Dict[int, float],
    ls_schedule: Dict[int, float],
    cross_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute all features for one symbol's dataframe."""
    from datetime import datetime, timezone

    funding_times = sorted(funding_schedule.keys())
    oi_times = sorted(oi_schedule.keys())
    ls_times = sorted(ls_schedule.keys())
    f_idx, oi_idx, ls_idx = 0, 0, 0

    comp = EnrichedFeatureComputer()
    records = []

    for _, row in df.iterrows():
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        open_ = float(row.get("open", close))
        trades = float(row.get("trades", 0) or 0)
        taker_buy_volume = float(row.get("taker_buy_volume", 0) or 0)
        quote_volume = float(row.get("quote_volume", 0) or 0)

        ts_raw = row.get("timestamp") or row.get("open_time", "")
        hour, dow, ts_ms = -1, -1, 0
        if ts_raw:
            try:
                ts_ms = int(ts_raw)
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                hour, dow = dt.hour, dt.weekday()
            except (ValueError, OSError):
                pass

        funding_rate = None
        while f_idx < len(funding_times) and funding_times[f_idx] <= ts_ms:
            funding_rate = funding_schedule[funding_times[f_idx]]
            f_idx += 1
        if funding_rate is None and f_idx > 0:
            funding_rate = funding_schedule[funding_times[f_idx - 1]]

        open_interest = None
        while oi_idx < len(oi_times) and oi_times[oi_idx] <= ts_ms:
            open_interest = oi_schedule[oi_times[oi_idx]]
            oi_idx += 1
        if open_interest is None and oi_idx > 0:
            open_interest = oi_schedule[oi_times[oi_idx - 1]]

        ls_ratio = None
        while ls_idx < len(ls_times) and ls_times[ls_idx] <= ts_ms:
            ls_ratio = ls_schedule[ls_times[ls_idx]]
            ls_idx += 1
        if ls_ratio is None and ls_idx > 0:
            ls_ratio = ls_schedule[ls_times[ls_idx - 1]]

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
            open_interest=open_interest, ls_ratio=ls_ratio,
        )

        if cross_df is not None and symbol != "BTCUSDT" and ts_ms in cross_df.index:
            cross_row = cross_df.loc[ts_ms]
            for name in CROSS_ASSET_FEATURE_NAMES:
                val = cross_row.get(name)
                feats[name] = None if pd.isna(val) else float(val)
        elif symbol != "BTCUSDT":
            for name in CROSS_ASSET_FEATURE_NAMES:
                feats[name] = None

        records.append(feats)

    feat_df = pd.DataFrame(records)

    for int_name, feat_a, feat_b in INTERACTION_FEATURES:
        if feat_a in feat_df.columns and feat_b in feat_df.columns:
            feat_df[int_name] = feat_df[feat_a].astype(float) * feat_df[feat_b].astype(float)
        else:
            feat_df[int_name] = np.nan

    feat_df["close"] = df["close"].values
    return feat_df


# ── Target variable ─────────────────────────────────────────

def vol_normalized_target(
    closes: np.ndarray,
    horizon: int = 5,
    vol_window: int = 20,
) -> np.ndarray:
    """Compute vol-normalized forward returns.

    raw_ret = close[t+horizon] / close[t] - 1
    vol = rolling_std(pct_change, vol_window), clipped at 5th percentile
    target = raw_ret / vol
    """
    if _TARGET_CPP:
        result = _cpp_vol_target(closes.tolist(), horizon, vol_window)
        return np.array(result, dtype=np.float64)

    # Python fallback
    n = len(closes)
    raw_ret = np.full(n, np.nan)
    for i in range(n - horizon):
        raw_ret[i] = closes[i + horizon] / closes[i] - 1.0

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
    if len(vol_valid) > 0:
        floor = np.percentile(vol_valid, 5)
        vol = np.where(np.isnan(vol), np.nan, np.maximum(vol, floor))
    else:
        return raw_ret

    target = np.where(
        (~np.isnan(raw_ret)) & (~np.isnan(vol)) & (vol > 0),
        raw_ret / vol,
        np.nan,
    )
    return target


# ── Walk-forward engine ──────────────────────────────────────

def expanding_window_folds(n: int, n_folds: int = 5, min_train: int = 500):
    """Generate expanding-window fold indices.

    Each fold: train = [0, split), test = [split, split + fold_size).
    Train expands; test windows are non-overlapping.
    """
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


def run_walkforward(
    symbol: str,
    feat_df: pd.DataFrame,
    available: List[str],
    out_dir: Path,
    *,
    top_k: int = 20,
    horizon: int = 5,
    n_folds: int = 5,
) -> Optional[dict]:
    """Run walk-forward training for one symbol."""
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    closes = feat_df["close"].values.astype(np.float64)
    target = vol_normalized_target(closes, horizon=horizon)

    # Filter blacklisted and all-NaN features
    available = [n for n in available if n not in BLACKLIST]
    X_all = feat_df[available]
    all_nan_cols = set(X_all.columns[X_all.isna().all()])
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN: {sorted(all_nan_cols)}")
        available = [c for c in available if c not in all_nan_cols]

    # Build clean mask (core features must be non-NaN, sparse features handled by LGBM)
    SPARSE = {"oi_change_pct", "oi_change_ma8", "oi_close_divergence",
              "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
              "btc_ret1_x_beta30"}
    core = [c for c in available if c not in SPARSE]
    X_full = feat_df[available].values.astype(np.float64)
    y_full = target

    core_idx = [available.index(c) for c in core]
    valid_mask = np.ones(len(X_full), dtype=bool)
    for ci in core_idx:
        valid_mask &= ~np.isnan(X_full[:, ci])
    valid_mask &= ~np.isnan(y_full)

    valid_indices = np.where(valid_mask)[0]
    X_clean = X_full[valid_indices]
    y_clean = y_full[valid_indices]

    print(f"  Valid samples: {len(X_clean)} / {len(X_full)}")
    if len(X_clean) < 1000:
        print("  ERROR: Not enough valid samples for walk-forward")
        return None

    # ── Walk-forward folds ──
    folds = expanding_window_folds(len(X_clean), n_folds=n_folds)
    if not folds:
        print("  ERROR: Could not create folds")
        return None

    print(f"  Walk-forward: {len(folds)} folds, horizon={horizon}")

    fold_results = []
    feature_selection_counts: Dict[str, int] = {}
    all_test_preds = []
    all_test_y = []
    all_test_idx = []

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_clean[tr_start:tr_end]
        y_train = y_clean[tr_start:tr_end]
        X_test = X_clean[te_start:te_end]
        y_test = y_clean[te_start:te_end]

        # Spearman IC feature selection on training data
        selected = spearman_ic_select(X_train, y_train, available, top_k=top_k)
        sel_idx = [available.index(n) for n in selected]

        for name in selected:
            feature_selection_counts[name] = feature_selection_counts.get(name, 0) + 1

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        # Train with early stopping + embargo
        model = LGBMAlphaModel(name=f"v3_fold{fi}", feature_names=tuple(selected))
        metrics = model.fit(
            X_tr_sel, y_train,
            params=V3_PARAMS.copy(),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=EMBARGO_BARS,
        )

        # Evaluate fold
        y_pred = model._model.predict(X_te_sel)

        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
        ic = float(np.corrcoef(y_pred, y_test)[0, 1]) if len(y_pred) > 1 else 0.0

        # 1-bar returns for Sharpe
        test_orig = valid_indices[te_start:te_end]
        ret_1bar = compute_1bar_returns(closes, test_orig)
        valid_ret = ~np.isnan(ret_1bar)
        if valid_ret.sum() > 1:
            from scripts.oos_eval import apply_threshold, compute_signal_costs
            signal = apply_threshold(y_pred[valid_ret], 0.001)
            costs = compute_signal_costs(signal)
            net_pnl = signal * ret_1bar[valid_ret] - costs
            active = signal != 0
            n_active = int(active.sum())
            if n_active > 1:
                std_a = float(np.std(net_pnl[active], ddof=1))
                sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760) if std_a > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        best_iter = metrics.get("best_iteration", V3_PARAMS["n_estimators"])

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
        }
        fold_results.append(fold_result)

        all_test_preds.extend(y_pred.tolist())
        all_test_y.extend(y_test.tolist())
        all_test_idx.extend(test_orig.tolist())

        print(f"  Fold {fi}: train={len(X_train):,} test={len(X_test):,} "
              f"IC={ic:.4f} dir={dir_acc:.3f} sharpe={sharpe:.2f} "
              f"best_iter={int(best_iter)}/{V3_PARAMS['n_estimators']}")

    # ── Cross-fold summary ──
    avg_ic = np.mean([f["ic"] for f in fold_results])
    avg_dir = np.mean([f["direction_accuracy"] for f in fold_results])
    avg_sharpe = np.mean([f["sharpe"] for f in fold_results])

    print(f"\n  Cross-fold avg: IC={avg_ic:.4f} dir={avg_dir:.3f} sharpe={avg_sharpe:.2f}")

    # Feature stability
    stable_features = {k: v for k, v in feature_selection_counts.items() if v >= max(n_folds * 3 // 5, 3)}
    unstable_features = {k: v for k, v in feature_selection_counts.items() if k not in stable_features}
    print(f"\n  Feature stability: {len(stable_features)} stable (>={max(n_folds * 3 // 5, 3)}/{n_folds} folds), "
          f"{len(unstable_features)} unstable")
    for name, count in sorted(stable_features.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:<30s} {count}/{n_folds} folds")

    # ── Final model: train on all data ──
    print("\n  Training final model on all data...")
    final_features = sorted(stable_features.keys(), key=lambda k: -feature_selection_counts[k])[:top_k]
    if len(final_features) < 5:
        # Fallback to most-selected features
        final_features = sorted(feature_selection_counts.keys(),
                                key=lambda k: -feature_selection_counts[k])[:top_k]
    final_idx = [available.index(n) for n in final_features]

    final_model = LGBMAlphaModel(name="lgbm_v3_alpha", feature_names=tuple(final_features))
    final_metrics = final_model.fit(
        X_clean[:, final_idx], y_clean,
        params=V3_PARAMS.copy(),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        embargo_bars=EMBARGO_BARS,
    )
    print(f"  Final model: {len(final_features)} features, "
          f"best_iter={int(final_metrics.get('best_iteration', -1))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(out_dir / "lgbm_v3_alpha.pkl")

    # ── Deflated Sharpe Ratio ──
    from research.overfit_detection import deflated_sharpe_ratio

    # We tested n_folds * threshold_count configurations effectively
    n_trials = n_folds * 5  # 5 thresholds in eval scan
    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=avg_sharpe / np.sqrt(8760) if avg_sharpe != 0 else 0.0,
        n_trials=n_trials,
        n_observations=len(all_test_preds),
        significance=0.05,
    )
    print(f"\n  Deflated Sharpe: p={dsr_result.p_value:.4f} "
          f"{'PASS' if dsr_result.is_significant else 'FAIL'} "
          f"(n_trials={n_trials})")

    # ── True OOS evaluation on held-out file ──
    oos_result = None
    oos_path = Path(f"data_files/{symbol}_1h_oos.csv")
    if oos_path.exists():
        print(f"\n  Running true OOS on {oos_path.name}...")
        oos_result = _evaluate_oos_file(
            symbol, final_model, final_features, horizon, oos_path,
        )

    # ── Model Registry ──
    try:
        from research.model_registry.registry import ModelRegistry
        registry = ModelRegistry(str(out_dir / "model_registry.db"))
        mv = registry.register(
            name=f"lgbm_v3_{symbol.lower()}",
            params={**V3_PARAMS, "horizon": horizon, "top_k": top_k,
                    "n_folds": n_folds, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
            features=final_features,
            metrics={
                "cv_ic": avg_ic,
                "cv_direction_accuracy": avg_dir,
                "cv_sharpe": avg_sharpe,
                "dsr_p_value": dsr_result.p_value,
                "final_best_iteration": final_metrics.get("best_iteration", -1),
                **({"oos_ic": oos_result["prediction_quality"]["ic"],
                    "oos_dir_acc": oos_result["prediction_quality"]["direction_accuracy"]}
                   if oos_result else {}),
            },
            tags=["v3", "walkforward", symbol.lower()],
        )
        print(f"  Registered: {mv.name} v{mv.version} (id={mv.model_id[:8]})")
    except Exception as e:
        print(f"  Registry warning: {e}")

    # ── Build result dict ──
    result = {
        "symbol": symbol,
        "version": "v3",
        "horizon": horizon,
        "n_folds": n_folds,
        "top_k": top_k,
        "n_features_available": len(available),
        "n_features_final": len(final_features),
        "final_features": final_features,
        "cv_ic": avg_ic,
        "cv_direction_accuracy": avg_dir,
        "cv_sharpe": avg_sharpe,
        "dsr_p_value": dsr_result.p_value,
        "dsr_significant": dsr_result.is_significant,
        "fold_results": fold_results,
        "feature_stability": {k: v for k, v in sorted(
            feature_selection_counts.items(), key=lambda x: -x[1])},
        "stable_features": list(stable_features.keys()),
        "unstable_features": list(unstable_features.keys()),
    }

    if oos_result:
        pq = oos_result["prediction_quality"]
        best_thr_row = None
        for row in oos_result["threshold_scan"]:
            if row["threshold"] == 0.001:
                best_thr_row = row
                break
        if best_thr_row is None:
            best_thr_row = oos_result["threshold_scan"][0]

        result.update({
            "oos_ic": pq["ic"],
            "oos_direction_accuracy": pq["direction_accuracy"],
            "oos_sharpe": best_thr_row["sharpe_annual"],
            "oos_n_trades": best_thr_row["n_trades"],
            "oos_net_return": best_thr_row["net_return"],
            "oos_threshold_scan": oos_result["threshold_scan"],
        })

    with open(out_dir / "v3_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def _evaluate_oos_file(
    symbol: str,
    model: LGBMAlphaModel,
    features: List[str],
    horizon: int,
    oos_path: Path,
) -> Optional[dict]:
    """Run model on OOS CSV file and evaluate."""
    from scripts.oos_eval import compute_1bar_returns, evaluate_oos, print_evaluation

    oos_df = pd.read_csv(oos_path)
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    oi_path = Path(f"data_files/{symbol}_open_interest.csv")
    ls_path = Path(f"data_files/{symbol}_ls_ratio.csv")

    feat_df = _compute_features(
        symbol, oos_df,
        _load_schedule(funding_path, "timestamp", "funding_rate"),
        _load_schedule(oi_path, "timestamp", "sum_open_interest") if oi_path.exists() else {},
        _load_schedule(ls_path, "timestamp", "long_short_ratio") if ls_path.exists() else {},
    )

    closes = feat_df["close"].values.astype(np.float64)
    target = vol_normalized_target(closes, horizon=horizon)

    feat_idx = [i for i, n in enumerate(features) if n in feat_df.columns]
    feat_names_avail = [features[i] for i in range(len(features)) if features[i] in feat_df.columns]

    if len(feat_names_avail) < len(features) * 0.5:
        print(f"  WARNING: OOS missing {len(features) - len(feat_names_avail)} features")

    X_oos = feat_df[feat_names_avail].values.astype(np.float64)
    y_oos = target

    # Pad missing features with NaN
    if len(feat_names_avail) < len(features):
        X_padded = np.full((len(X_oos), len(features)), np.nan)
        for i, name in enumerate(features):
            if name in feat_names_avail:
                src_idx = feat_names_avail.index(name)
                X_padded[:, i] = X_oos[:, src_idx]
        X_oos = X_padded

    valid = ~np.isnan(y_oos)
    for j in range(X_oos.shape[1]):
        if features[j] not in BLACKLIST:
            valid &= ~np.isnan(X_oos[:, j])
    valid_idx = np.where(valid)[0]

    if len(valid_idx) < 50:
        print(f"  OOS: only {len(valid_idx)} valid samples, skipping")
        return None

    X_v = X_oos[valid_idx]
    y_v = y_oos[valid_idx]

    y_pred = model._model.predict(X_v)
    ret_1bar = compute_1bar_returns(closes, valid_idx)

    eval_result = evaluate_oos(y_pred, y_v, ret_1bar)
    print_evaluation(eval_result, label="V3 True OOS")

    return eval_result


# ── Cross-asset feature building ─────────────────────────────

def _build_cross_features(symbols: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
    """Build cross-asset features for altcoins using BTC as benchmark."""
    btc_path = Path("data_files/BTCUSDT_1h.csv")
    if not btc_path.exists():
        return None

    btc_df = pd.read_csv(btc_path)
    btc_ts_col = "timestamp" if "timestamp" in btc_df.columns else "open_time"
    btc_timestamps = btc_df[btc_ts_col]

    result: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        if sym == "BTCUSDT":
            continue
        sym_path = Path(f"data_files/{sym}_1h.csv")
        if not sym_path.exists():
            continue

        sym_df = pd.read_csv(sym_path)
        sym_ts_col = "timestamp" if "timestamp" in sym_df.columns else "open_time"
        sym_ts = sym_df[sym_ts_col]

        comp = CrossAssetComputer()
        records = []
        timestamps = []

        # Align by timestamp
        btc_map = {int(btc_timestamps.iloc[i]): i for i in range(len(btc_df))}

        # Load funding for both symbols
        btc_funding = _load_schedule(Path("data_files/BTCUSDT_funding.csv"), "timestamp", "funding_rate")
        sym_funding = _load_schedule(Path(f"data_files/{sym}_funding.csv"), "timestamp", "funding_rate")
        btc_f_times = sorted(btc_funding.keys())
        sym_f_times = sorted(sym_funding.keys())
        btc_fi, sym_fi = 0, 0

        for i in range(len(sym_df)):
            ts = int(sym_ts.iloc[i])

            # Advance funding pointers
            btc_fr = None
            while btc_fi < len(btc_f_times) and btc_f_times[btc_fi] <= ts:
                btc_fr = btc_funding[btc_f_times[btc_fi]]
                btc_fi += 1
            if btc_fr is None and btc_fi > 0:
                btc_fr = btc_funding[btc_f_times[btc_fi - 1]]

            sym_fr = None
            while sym_fi < len(sym_f_times) and sym_f_times[sym_fi] <= ts:
                sym_fr = sym_funding[sym_f_times[sym_fi]]
                sym_fi += 1
            if sym_fr is None and sym_fi > 0:
                sym_fr = sym_funding[sym_f_times[sym_fi - 1]]

            # BTC bar first (benchmark must be pushed before altcoin)
            if ts in btc_map:
                bi = btc_map[ts]
                btc_close = float(btc_df.iloc[bi]["close"])
                comp.on_bar("BTCUSDT", close=btc_close, funding_rate=btc_fr)

            # Then altcoin bar
            sym_close = float(sym_df.iloc[i]["close"])
            comp.on_bar(sym, close=sym_close, funding_rate=sym_fr)
            feats = comp.get_features(sym)
            records.append(feats)
            timestamps.append(ts)

        cross_df = pd.DataFrame(records, index=timestamps)
        result[sym] = cross_df

    return result if result else None


# ── Entrypoint ───────────────────────────────────────────────

def get_available_features(symbol: str) -> List[str]:
    """All possible feature names for V3 (excluding blacklist)."""
    names = [n for n in ENRICHED_FEATURE_NAMES if n not in BLACKLIST]
    if symbol != "BTCUSDT":
        names.extend(n for n in CROSS_ASSET_FEATURE_NAMES if n not in BLACKLIST)
    names.extend(name for name, _, _ in INTERACTION_FEATURES if name not in BLACKLIST)
    return names


def run_one(
    symbol: str,
    out_base: Path,
    *,
    top_k: int = 20,
    horizon: int = 5,
    n_folds: int = 5,
    cross_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[dict]:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — V3 Walk-Forward Training")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    funding = _load_schedule(Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")

    cross_df = cross_features.get(symbol) if cross_features else None
    feat_df = _compute_features(symbol, df, funding, oi, ls, cross_df)
    print(f"  Bars: {len(feat_df)}")

    available = get_available_features(symbol)
    available = [n for n in available if n in feat_df.columns]
    print(f"  Available features (after blacklist): {len(available)}")

    out_dir = out_base / symbol
    return run_walkforward(
        symbol, feat_df, available, out_dir,
        top_k=top_k, horizon=horizon, n_folds=n_folds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V3 Alpha — Walk-Forward")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K features per fold")
    parser.add_argument("--horizon", type=int, default=5, help="Target horizon in bars")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds")
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
                    n_folds=args.n_folds, cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*100}")
        print(f"  V3 Walk-Forward Summary")
        print(f"{'='*100}")
        print(f"{'Symbol':<10} {'Feats':>5} {'CV IC':>8} {'CV Dir':>8} {'CV Shrp':>8} "
              f"{'DSR p':>8} {'OOS IC':>8} {'OOS Shrp':>8} {'Trades':>7}")
        print(f"{'-'*100}")
        for sym, r in results.items():
            oos_ic = r.get("oos_ic", float("nan"))
            oos_shrp = r.get("oos_sharpe", float("nan"))
            oos_trades = r.get("oos_n_trades", 0)
            print(f"{sym:<10} {r['n_features_final']:>5} "
                  f"{r['cv_ic']:>8.4f} {r['cv_direction_accuracy']*100:>7.1f}% "
                  f"{r['cv_sharpe']:>8.2f} {r['dsr_p_value']:>8.4f} "
                  f"{oos_ic:>8.4f} {oos_shrp:>8.2f} {oos_trades:>7}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
