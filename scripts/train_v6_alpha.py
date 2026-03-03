#!/usr/bin/env python3
"""Train V6 Alpha — signal quality breakthrough.

Key fixes over V5:
  1. Multi-target scanning (raw/clipped/vol_norm/binary × multiple horizons)
  2. Fixed training bugs: sample_weight propagation, proper embargo, min_train=2000
  3. greedy_ic_select replaces unstable ICIR selection
  4. Blacklist extended: ret_1, ret_3 (intra-bar noise)
  5. Feature stability gate: must appear in >=4/5 folds
  6. Regime feature (vol-based)
  7. OOS validation with bootstrap CI + deflated Sharpe

Usage:
    python3 -m scripts.train_v6_alpha --all
    python3 -m scripts.train_v6_alpha --symbol BTCUSDT --n-trials 20
    python3 -m scripts.train_v6_alpha --symbol BTCUSDT --target-mode clipped --horizon 12
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
from features.enriched_computer import EnrichedFeatureComputer, ENRICHED_FEATURE_NAMES
from features.cross_asset_computer import CrossAssetComputer, CROSS_ASSET_FEATURE_NAMES
from features.dynamic_selector import greedy_ic_select, _rankdata

from research.hyperopt.optimizer import HyperOptimizer, HyperOptConfig
from research.hyperopt.search_space import SearchSpace, ParamRange
from research.overfit_detection import deflated_sharpe_ratio

try:
    from features._quant_rolling import cpp_bootstrap_sharpe_ci
    _BOOTSTRAP_CPP = True
except ImportError:
    _BOOTSTRAP_CPP = False

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

V6_DEFAULT_PARAMS = {
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

V6_SEARCH_SPACE = SearchSpace(
    name="lgbm_v6",
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
}

INTERACTION_FEATURES = [
    ("rsi14_x_vol_regime", "rsi_14", "vol_regime"),
    ("funding_x_taker_imb", "funding_rate", "taker_imbalance"),
    ("btc_ret1_x_beta30", "btc_ret_1", "rolling_beta_30"),
    ("trade_int_x_body", "trade_intensity", "body_ratio"),
    ("cvd_x_oi_chg", "cvd_20", "oi_change_pct"),
    ("vol_of_vol_x_range", "vol_of_vol", "range_vs_rv"),
]

TARGET_MODES = ("raw", "clipped", "vol_norm", "binary")
DEFAULT_HORIZONS = (3, 6, 12, 24)


# ── Target variable ─────────────────────────────────────────

def _compute_target(
    closes: np.ndarray,
    horizon: int,
    mode: str,
    vol_window: int = 20,
) -> np.ndarray:
    """Compute forward return target with different modes.

    Args:
        closes: Close price array.
        horizon: Forward return horizon in bars.
        mode: One of 'raw', 'clipped', 'vol_norm', 'binary'.
        vol_window: Window for volatility calculation (vol_norm mode only).

    Returns:
        Target array with NaN for invalid positions.
    """
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
        # No 5th percentile floor (V5 bug fix)
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
    """Select best horizon × mode combination by greedy IC on held-out 20%."""
    n = X.shape[0]
    split = int(n * 0.8)
    X_eval = X[split:]

    best_score = -1.0
    best_key = (6, "clipped")  # fallback default

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

        # Mean absolute IC of selected features
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
    """Add regime_vol based on vol_20 terciles (0/1/2)."""
    if "vol_20" not in feat_df.columns:
        feat_df["regime_vol"] = 1  # default to medium
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
    """Time-decay sample weights: linearly from `decay` to 1.0."""
    return np.linspace(decay, 1.0, n)


# ── Walk-forward engine ──────────────────────────────────────

def expanding_window_folds(n: int, n_folds: int = 5, min_train: int = MIN_TRAIN):
    """Generate expanding window fold indices.

    Returns list of (train_start, train_end, test_start, test_end).
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


def _inner_cv_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    embargo_bars: int,
    inner_folds: int = 3,
) -> callable:
    """Create objective function for Optuna inner CV.

    Unlike V5, this uses direct lgb.LGBMRegressor.fit() with sample_weight
    instead of nesting through LGBMAlphaModel.fit()'s internal val split.
    """
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

            # Purged split with proper embargo
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

            # Direct lgb fit — no nested val_size split
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
    """Validate on OOS data. Returns metrics dict or None if no OOS data."""
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

    # LightGBM handles NaN natively — only filter on target
    X_oos = feat_df[feature_names].values.astype(np.float64)
    valid_mask = ~np.isnan(target)

    X_v = X_oos[valid_mask]
    y_v = target[valid_mask]
    closes_v = closes[np.where(valid_mask)[0]]

    if len(X_v) < 50:
        return None

    y_pred = model._model.predict(X_v)

    # Spearman IC
    rx = _rankdata(y_pred)
    ry = _rankdata(y_v)
    ic = float(np.corrcoef(rx, ry)[0, 1])

    # Direction accuracy
    if target_mode == "binary":
        dir_acc = float(np.mean((y_pred > 0.5).astype(int) == y_v.astype(int)))
    else:
        dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_v)))

    # Sharpe with transaction costs
    ret_1bar = np.diff(closes_v) / closes_v[:-1]
    if len(ret_1bar) < len(y_pred):
        y_pred = y_pred[:len(ret_1bar)]

    pred_for_signal = y_pred[:len(ret_1bar)]
    if target_mode == "binary":
        pred_for_signal = pred_for_signal - 0.5  # center at 0
    signal = np.sign(pred_for_signal)
    turnover = np.abs(np.diff(signal, prepend=0))
    cost_per_bar = 4e-4  # 4 bps one-way
    net_pnl = signal * ret_1bar - turnover * cost_per_bar

    active = signal != 0
    sharpe = 0.0
    if active.sum() > 1:
        std_a = float(np.std(net_pnl[active], ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

    # Bootstrap CI
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

    # Deflated Sharpe
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=max(n_target_trials, 1),
        n_observations=int(active.sum()),
    )

    # Pass conditions
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


# ── Data loading (reuses V5 pattern) ────────────────────────

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


def _load_and_compute_features(
    symbol: str,
    df: pd.DataFrame,
    cross_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """Compute features from raw OHLCV dataframe."""
    from datetime import datetime, timezone

    funding = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(
        Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(
        Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")

    funding_times = sorted(funding.keys())
    oi_times = sorted(oi.keys())
    ls_times = sorted(ls.keys())
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
            funding_rate = funding[funding_times[f_idx]]
            f_idx += 1
        if funding_rate is None and f_idx > 0:
            funding_rate = funding[funding_times[f_idx - 1]]

        open_interest = None
        while oi_idx < len(oi_times) and oi_times[oi_idx] <= ts_ms:
            open_interest = oi[oi_times[oi_idx]]
            oi_idx += 1
        if open_interest is None and oi_idx > 0:
            open_interest = oi[oi_times[oi_idx - 1]]

        ls_ratio = None
        while ls_idx < len(ls_times) and ls_times[ls_idx] <= ts_ms:
            ls_ratio = ls[ls_times[ls_idx]]
            ls_idx += 1
        if ls_ratio is None and ls_idx > 0:
            ls_ratio = ls[ls_times[ls_idx - 1]]

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


# ── Main walk-forward engine ─────────────────────────────────

def get_available_features(symbol: str) -> List[str]:
    names = [n for n in ENRICHED_FEATURE_NAMES if n not in BLACKLIST]
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
    """Run V6 walk-forward training for one symbol."""
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
              "funding_x_taker_imb"}
    core = [c for c in available if c not in SPARSE]
    X_full = feat_df[available].values.astype(np.float64)

    core_idx = [available.index(c) for c in core]
    valid_mask = np.ones(len(X_full), dtype=bool)
    for ci in core_idx:
        valid_mask &= ~np.isnan(X_full[:, ci])

    # ── Target selection ──
    n_target_trials = 1
    if horizon > 0 and target_mode:
        # User specified — use directly
        target = _compute_target(closes, horizon, target_mode)
        valid_mask &= ~np.isnan(target)
    else:
        # Auto-select best target
        print("  Scanning target modes...")
        y_dict: Dict[Tuple[int, str], np.ndarray] = {}
        for h in DEFAULT_HORIZONS:
            for m in TARGET_MODES:
                y_dict[(h, m)] = _compute_target(closes, h, m)

        # Need valid mask before scanning — use raw horizon=6 as proxy
        proxy_y = y_dict[(6, "raw")]
        valid_mask &= ~np.isnan(proxy_y)
        valid_indices_proxy = np.where(valid_mask)[0]
        X_clean_proxy = X_full[valid_indices_proxy]

        # Filter y_dict to valid samples
        y_dict_clean = {}
        for key, y in y_dict.items():
            y_dict_clean[key] = y[valid_indices_proxy]

        best_key = _select_best_target(X_clean_proxy, y_dict_clean, available)
        horizon, target_mode = best_key
        n_target_trials = len(DEFAULT_HORIZONS) * len(TARGET_MODES)
        print(f"  Best target: horizon={horizon}, mode={target_mode}")

        target = _compute_target(closes, horizon, target_mode)
        # Recompute valid mask with actual target
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

        # V6: greedy_ic_select (C++ 310x accelerated)
        selected = greedy_ic_select(X_train, y_train, available, top_k=top_k)
        sel_idx = [available.index(n) for n in selected]

        for name in selected:
            feature_selection_counts[name] = feature_selection_counts.get(name, 0) + 1

        X_tr_sel = X_train[:, sel_idx]
        X_te_sel = X_test[:, sel_idx]

        # Sample weights
        weights = _compute_sample_weights(len(X_tr_sel))

        # Optuna HPO (inner CV with fixed bugs)
        if n_trials > 0:
            obj_fn = _inner_cv_objective(
                X_tr_sel, y_train, weights,
                embargo_bars=embargo_bars,
                inner_folds=3,
            )
            optimizer = HyperOptimizer(
                search_space=V6_SEARCH_SPACE,
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
            fold_params = {**V6_DEFAULT_PARAMS, **hpo_result.best_params}
            print(f"  Fold {fi} HPO: best_ic={hpo_result.best_value:.4f} "
                  f"({hpo_result.n_trials} trials)")
        else:
            fold_params = V6_DEFAULT_PARAMS.copy()

        # Train final fold model with sample_weight + embargo
        model = LGBMAlphaModel(name=f"v6_fold{fi}", feature_names=tuple(selected))
        metrics = model.fit(
            X_tr_sel, y_train,
            params=fold_params,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            embargo_bars=embargo_bars,
            sample_weight=weights,
        )

        # Evaluate fold
        y_pred = model._model.predict(X_te_sel)
        if target_mode == "binary":
            dir_acc = float(np.mean(
                (y_pred > 0.5).astype(int) == y_test.astype(int)))
        else:
            dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

        rx = _rankdata(y_pred)
        ry = _rankdata(y_test)
        ic = float(np.corrcoef(rx, ry)[0, 1]) if len(y_pred) > 1 else 0.0

        # Sharpe on test predictions
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
            turnover = np.abs(np.diff(signal, prepend=0))
            net_pnl = signal * ret_1bar[valid_ret] - turnover * 4e-4
            active = signal != 0
            n_active = int(active.sum())
            if n_active > 1:
                std_a = float(np.std(net_pnl[active], ddof=1))
                if std_a > 0:
                    sharpe = float(np.mean(net_pnl[active])) / std_a * np.sqrt(8760)

        best_iter = metrics.get("best_iteration", V6_DEFAULT_PARAMS["n_estimators"])

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

    # Feature stability gate: must appear in >= 4/5 folds
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

    final_params = fold_results[-1].get("hpo_params") or V6_DEFAULT_PARAMS.copy()
    final_weights = _compute_sample_weights(len(X_clean))

    final_model = LGBMAlphaModel(name="lgbm_v6_alpha", feature_names=tuple(final_features))
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
    final_model.save(out_dir / "lgbm_v6_alpha.pkl")

    # ── OOS validation ──
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

    # ── Build result dict ──
    result = {
        "symbol": symbol,
        "version": "v6",
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
        "default_params": V6_DEFAULT_PARAMS,
    }

    with open(out_dir / "v6_results.json", "w") as f:
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
    print(f"  {symbol} — V6 Alpha Training")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    cross_df = cross_features.get(symbol) if cross_features else None
    feat_df = _load_and_compute_features(symbol, df, cross_df)
    if feat_df is None:
        print(f"  ERROR: Feature computation failed")
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
    parser = argparse.ArgumentParser(description="Train V6 Alpha — Signal Quality Breakthrough")
    parser.add_argument("--symbol", help="Single symbol")
    parser.add_argument("--all", action="store_true", help="All symbols")
    parser.add_argument("--out", default="models_v6", help="Output directory")
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
        print(f"  V6 Alpha Training Summary")
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
