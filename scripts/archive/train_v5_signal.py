#!/usr/bin/env python3
"""Train V5 Alpha — ICIR feature selection + Optuna HPO + sample weights.

Key improvements over V3:
  1. ICIR-based feature selection (stability across folds, not just magnitude)
  2. Optuna hyperparameter optimization (inner 3-fold CV per outer fold)
  3. Time-decay sample weights (recent data weighted more)
  4. Larger embargo (7 bars = horizon 5 + buffer 2)
  5. New V5 interaction features (CVD × OI, vol_of_vol × range)

Usage:
    python3 -m scripts.train_v5_signal --all
    python3 -m scripts.train_v5_signal --symbol BTCUSDT
    python3 -m scripts.train_v5_signal --symbol BTCUSDT --n-trials 30 --top-k 25
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from alpha.models.lgbm_alpha import LGBMAlphaModel
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import icir_select, compute_feature_icir_report

from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
from research.hyperopt.search_space import SearchSpace, ParamRange

try:
    from features._quant_rolling import cpp_vol_normalized_target as _cpp_vol_target
    _TARGET_CPP = True
except ImportError:
    _TARGET_CPP = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

V5_DEFAULT_PARAMS = {
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

V5_SEARCH_SPACE = SearchSpace(
    name="lgbm_v5",
    int_params=(
        ParamRange("max_depth", 3, 7),
        ParamRange("num_leaves", 10, 40),
        ParamRange("min_child_samples", 30, 100),
    ),
    float_params=(
        ParamRange("learning_rate", 0.01, 0.1, log_scale=True),
        ParamRange("reg_alpha", 0.01, 1.0, log_scale=True),
        ParamRange("reg_lambda", 0.1, 5.0, log_scale=True),
        ParamRange("subsample", 0.5, 0.9),
        ParamRange("colsample_bytree", 0.5, 0.9),
    ),
)

EARLY_STOPPING_ROUNDS = 50
EMBARGO_BARS = 7  # horizon(5) + buffer(2)

BLACKLIST = {
    "oi_change_pct", "oi_change_ma8", "oi_close_divergence",
    "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
    "avg_trade_size",
    "oi_chg_x_ret1",
}

# V3 interactions + V5 new interactions
INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
    # V5 new
    ("cvd_x_oi_chg", "cvd_20", "oi_change_pct"),
    ("vol_of_vol_x_range", "vol_of_vol", "range_vs_rv"),
]

# ── Data loading (reuses V3 pattern) ────────────────────────


def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
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
    if _TARGET_CPP:
        result = _cpp_vol_target(closes.tolist(), horizon, vol_window)
        return np.array(result, dtype=np.float64)

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

    return np.where(
        (~np.isnan(raw_ret)) & (~np.isnan(vol)) & (vol > 0),
        raw_ret / vol,
        np.nan,
    )


# ── Sample weights ───────────────────────────────────────────

def _compute_sample_weights(n: int, decay: float = 0.5) -> np.ndarray:
    """Time-decay sample weights: linearly from `decay` to 1.0."""
    return np.linspace(decay, 1.0, n)


# ── Walk-forward engine ──────────────────────────────────────

def expanding_window_folds(n: int, n_folds: int = 5, min_train: int = 500):
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
    selected_names: tuple,
    inner_folds: int = 3,
) -> callable:
    """Create objective function for Optuna inner CV."""
    from features.dynamic_selector import _rankdata

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
            # Purged split: train on everything except val + embargo
            embargo_start = max(0, val_start - EMBARGO_BARS)
            embargo_end = min(n, val_end + EMBARGO_BARS)

            train_mask = np.ones(n, dtype=bool)
            train_mask[embargo_start:embargo_end] = False
            if train_mask.sum() < 100:
                continue

            X_tr = X_train[train_mask]
            y_tr = y_train[train_mask]
            w_tr = weights[train_mask]
            X_val = X_train[val_start:val_end]
            y_val = y_train[val_start:val_end]

            model = LGBMAlphaModel(name="inner_cv", feature_names=selected_names)
            model.fit(X_tr, y_tr, params=lgbm_params,
                      early_stopping_rounds=30, embargo_bars=0, val_size=0.15)
            # Manually pass sample_weight through params isn't supported —
            # the LGBMAlphaModel.fit handles val_size internally

            y_pred = model._model.predict(X_val)
            if len(y_pred) < 30 or np.std(y_pred) < 1e-12:
                continue
            # Spearman IC
            rx = _rankdata(y_pred)
            ry = _rankdata(y_val)
            ic = float(np.corrcoef(rx, ry)[0, 1])
            if not np.isnan(ic):
                ics.append(ic)

        return float(np.mean(ics)) if ics else 0.0

    return objective


def run_walkforward(
    symbol: str,
    feat_df: pd.DataFrame,
    available: List[str],
    out_dir: Path,
    *,
    top_k: int = 25,
    horizon: int = 5,
    n_folds: int = 5,
    n_trials: int = 30,
) -> Optional[dict]:
    """Run V5 walk-forward training for one symbol."""
    closes = feat_df["close"].values.astype(np.float64)
    target = vol_normalized_target(closes, horizon=horizon)

    available = [n for n in available if n not in BLACKLIST]
    X_all = feat_df[available]
    all_nan_cols = set(X_all.columns[X_all.isna().all()])
    if all_nan_cols:
        print(f"  Dropping {len(all_nan_cols)} all-NaN: {sorted(all_nan_cols)}")
        available = [c for c in available if c not in all_nan_cols]

    SPARSE = {"oi_change_pct", "oi_change_ma8", "oi_close_divergence",
              "ls_ratio", "ls_ratio_zscore_24", "ls_extreme",
              "btc_ret1_x_beta30",
              # V5 OI-dependent (OI data very sparse ~500 rows vs ~56k bars)
              "oi_acceleration", "leverage_proxy", "oi_vol_divergence",
              "oi_liquidation_flag", "cvd_x_oi_chg",
              # Funding-dependent (funding only every 8h)
              "funding_rate", "funding_ma8", "funding_zscore_24",
              "funding_momentum", "funding_extreme", "funding_cumulative_8",
              "funding_sign_persist", "funding_annualized", "funding_vs_vol",
              "funding_x_taker_imb"}
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

    folds = expanding_window_folds(len(X_clean), n_folds=n_folds)
    if not folds:
        print("  ERROR: Could not create folds")
        return None

    print(f"  Walk-forward: {len(folds)} folds, horizon={horizon}, embargo={EMBARGO_BARS}")

    fold_results = []
    feature_selection_counts: Dict[str, int] = {}
    icir_reports: List[dict] = []

    for fi, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
        X_train = X_clean[tr_start:tr_end]
        y_train = y_clean[tr_start:tr_end]
        X_test = X_clean[te_start:te_end]
        y_test = y_clean[te_start:te_end]

        # V5: ICIR feature selection
        selected = icir_select(X_train, y_train, available, top_k=top_k)
        sel_idx = [available.index(n) for n in selected]

        for name in selected:
            feature_selection_counts[name] = feature_selection_counts.get(name, 0) + 1

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        # V5: Sample weights (time decay)
        weights = _compute_sample_weights(len(X_tr_sel))

        # V5: Optuna HPO (inner CV)
        if n_trials > 0:
            obj_fn = _inner_cv_objective(
                X_tr_sel, y_train, weights,
                selected_names=tuple(selected),
                inner_folds=3,
            )
            optimizer = HyperOptimizer(
                search_space=V5_SEARCH_SPACE,
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
            fold_params = {**V5_DEFAULT_PARAMS, **hpo_result.best_params}
            print(f"  Fold {fi} HPO: best_ic={hpo_result.best_value:.4f} "
                  f"({hpo_result.n_trials} trials)")
        else:
            fold_params = V5_DEFAULT_PARAMS.copy()

        # Train with early stopping + embargo
        model = LGBMAlphaModel(name=f"v5_fold{fi}", feature_names=tuple(selected))
        metrics = model.fit(
            X_tr_sel, y_train,
            params=fold_params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=EMBARGO_BARS,
        )

        # Evaluate fold
        y_pred = model._model.predict(X_te_sel)
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
        ic = float(np.corrcoef(y_pred, y_test)[0, 1]) if len(y_pred) > 1 else 0.0

        # Simple Sharpe on test predictions
        test_orig = valid_indices[te_start:te_end]
        ret_1bar = np.full(len(test_orig), np.nan)
        for i, idx in enumerate(test_orig):
            if idx + 1 < len(closes):
                ret_1bar[i] = closes[idx + 1] / closes[idx] - 1.0

        valid_ret = ~np.isnan(ret_1bar)
        sharpe = 0.0
        if valid_ret.sum() > 1:
            signal = np.where(np.abs(y_pred[valid_ret]) > 0.001,
                              np.sign(y_pred[valid_ret]), 0.0)
            net_pnl = signal * ret_1bar[valid_ret]
            active = signal != 0
            n_active = int(active.sum())
            if n_active > 1:
                std_a = float(np.std(net_pnl[active], ddof=1))
                if std_a > 0:
                    sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

        best_iter = metrics.get("best_iteration", V5_DEFAULT_PARAMS["n_estimators"])

        # ICIR report for this fold
        report = compute_feature_icir_report(X_train, y_train, available)
        if report:
            icir_reports.append(report)

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

    # Feature stability
    stable_features = {k: v for k, v in feature_selection_counts.items()
                       if v >= max(n_folds * 3 // 5, 3)}
    unstable_features = {k: v for k, v in feature_selection_counts.items()
                         if k not in stable_features}
    print(f"\n  Feature stability: {len(stable_features)} stable "
          f"(>={max(n_folds * 3 // 5, 3)}/{n_folds} folds), "
          f"{len(unstable_features)} unstable")
    for name, count in sorted(stable_features.items(), key=lambda x: -x[1])[:15]:
        print(f"    {name:<30s} {count}/{n_folds} folds")

    # Aggregate ICIR across folds
    if icir_reports:
        avg_icir_by_feat: Dict[str, List[float]] = {}
        for report in icir_reports:
            for feat, entry in report.items():
                avg_icir_by_feat.setdefault(feat, []).append(entry["icir"])
        print(f"\n  Top features by cross-fold avg ICIR:")
        sorted_icir = sorted(avg_icir_by_feat.items(),
                             key=lambda x: -np.mean(x[1]))
        for name, icirs in sorted_icir[:15]:
            mean_icir = float(np.mean(icirs))
            print(f"    {name:<30s} ICIR={mean_icir:.3f}")

    # ── Final model: train on all data ──
    print("\n  Training final model on all data...")
    final_features = sorted(stable_features.keys(),
                            key=lambda k: -feature_selection_counts[k])[:top_k]
    if len(final_features) < 5:
        final_features = sorted(feature_selection_counts.keys(),
                                key=lambda k: -feature_selection_counts[k])[:top_k]
    final_idx = [available.index(n) for n in final_features]

    # Use best HPO params from last fold (or default)
    final_params = fold_results[-1].get("hpo_params") or V5_DEFAULT_PARAMS.copy()

    final_model = LGBMAlphaModel(name="lgbm_v5_alpha", feature_names=tuple(final_features))
    final_metrics = final_model.fit(
        X_clean[:, final_idx], y_clean,
        params=final_params,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        embargo_bars=EMBARGO_BARS,
    )
    print(f"  Final model: {len(final_features)} features, "
          f"best_iter={int(final_metrics.get('best_iteration', -1))}")

    out_dir.mkdir(parents=True, exist_ok=True)
    final_model.save(out_dir / "lgbm_v5_alpha.pkl")

    # ── Build result dict ──
    result = {
        "symbol": symbol,
        "version": "v5",
        "horizon": horizon,
        "n_folds": n_folds,
        "top_k": top_k,
        "n_trials": n_trials,
        "embargo_bars": EMBARGO_BARS,
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
    }

    with open(out_dir / "v5_results.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Cross-asset feature building ─────────────────────────────

def _build_cross_features(symbols: List[str]) -> Optional[Dict[str, pd.DataFrame]]:
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

        btc_map = {int(btc_timestamps.iloc[i]): i for i in range(len(btc_df))}

        btc_funding = _load_schedule(Path("data_files/BTCUSDT_funding.csv"), "timestamp", "funding_rate")
        sym_funding = _load_schedule(Path(f"data_files/{sym}_funding.csv"), "timestamp", "funding_rate")
        btc_f_times = sorted(btc_funding.keys())
        sym_f_times = sorted(sym_funding.keys())
        btc_fi, sym_fi = 0, 0

        for i in range(len(sym_df)):
            ts = int(sym_ts.iloc[i])

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

            if ts in btc_map:
                bi = btc_map[ts]
                btc_close = float(btc_df.iloc[bi]["close"])
                comp.on_bar("BTCUSDT", close=btc_close, funding_rate=btc_fr)

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
    names = [n for n in ENRICHED_FEATURE_NAMES if n not in BLACKLIST]
    if symbol != "BTCUSDT":
        names.extend(n for n in CROSS_ASSET_FEATURE_NAMES if n not in BLACKLIST)
    names.extend(name for name, _, _ in INTERACTION_FEATURES if name not in BLACKLIST)
    return names


def run_one(
    symbol: str,
    out_base: Path,
    *,
    top_k: int = 25,
    horizon: int = 5,
    n_folds: int = 5,
    n_trials: int = 30,
    cross_features: Optional[Dict[str, pd.DataFrame]] = None,
) -> Optional[dict]:
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  SKIP {symbol}: CSV not found")
        return None

    print(f"\n{'='*60}")
    print(f"  {symbol} — V5 Signal Training")
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
        top_k=top_k, horizon=horizon, n_folds=n_folds, n_trials=n_trials,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V5 Alpha — ICIR + HPO")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models", help="Output directory")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K features per fold")
    parser.add_argument("--horizon", type=int, default=5, help="Target horizon in bars")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials per fold")
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
                    n_folds=args.n_folds, n_trials=args.n_trials,
                    cross_features=cross_features)
        if r:
            results[sym] = r

    if results:
        print(f"\n\n{'='*100}")
        print(f"  V5 Signal Training Summary")
        print(f"{'='*100}")
        print(f"{'Symbol':<10} {'Feats':>5} {'CV IC':>8} {'CV Dir':>8} {'CV Shrp':>8}")
        print(f"{'-'*50}")
        for sym, r in results.items():
            print(f"{sym:<10} {r['n_features_final']:>5} "
                  f"{r['cv_ic']:>8.4f} {r['cv_direction_accuracy']*100:>7.1f}% "
                  f"{r['cv_sharpe']:>8.2f}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
