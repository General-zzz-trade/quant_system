#!/usr/bin/env python3
"""Backtest Alpha V8 — realistic OOS trade simulation with the alpha_rebuild model.

Loads the trained V8 model, replays OOS bars with on-the-fly feature computation,
generates signals via z-score normalization + deadzone, and simulates trading with
realistic costs (fees + slippage + turnover penalty).

Usage:
    python3 -m scripts.backtest_alpha_v8 --symbol BTCUSDT
    python3 -m scripts.backtest_alpha_v8 --symbol BTCUSDT --model results/alpha_rebuild_v3/step6_final/BTCUSDT/v8_final.pkl
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features.enriched_computer import EnrichedFeatureComputer

logger = logging.getLogger(__name__)

# ── C++ acceleration ─────────────────────────────────────────────

try:
    from features.batch_backtest import _BT_CPP, run_backtest_fast, pred_to_signal_fast
except ImportError:
    _BT_CPP = False

# ── Constants ────────────────────────────────────────────────────

FEE_BPS = 4e-4          # 4 bps per trade (maker/taker average)
SLIPPAGE_BPS = 2e-4     # 2 bps slippage
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS  # 6 bps total
INITIAL_CAPITAL = 10000.0


def _compute_bear_mask(closes: np.ndarray, ma_window: int = 480) -> np.ndarray:
    """Return boolean mask: True where close <= SMA(ma_window)."""
    n = len(closes)
    mask = np.zeros(n, dtype=bool)
    if n < ma_window:
        mask[:] = True  # not enough data — conservative
        return mask
    cs = np.cumsum(closes)
    ma = np.empty(n)
    ma[:ma_window] = np.nan
    ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
    mask = np.isnan(ma) | (closes <= ma)
    return mask


def _apply_monthly_gate(
    signal: np.ndarray,
    closes: np.ndarray,
    ma_window: int = 480,
) -> np.ndarray:
    """Zero out signal when close <= SMA(ma_window). Vectorized via cumsum."""
    n = len(signal)
    if n != len(closes):
        raise ValueError("signal and closes must have same length")
    out = signal.copy()
    bear = _compute_bear_mask(closes, ma_window)
    out[bear] = 0.0
    return out


def _prob_to_score(
    prob: float,
    bear_thresholds: Optional[List[Tuple[float, float]]] = None,
) -> float:
    """Convert bear model probability to position score.

    Args:
        prob: Classifier probability of crash (prob[:,1]).
        bear_thresholds: List of (prob_threshold, score) tuples, sorted desc.
            e.g. [(0.7, -1.0), (0.6, -0.5), (0.5, 0.0)]
            If None, uses binary: prob > 0.5 → -1.0 else 0.0.
    """
    if bear_thresholds is None:
        return -1.0 if prob > 0.5 else 0.0
    for thresh, score in bear_thresholds:
        if prob > thresh:
            return score
    return 0.0


def _apply_regime_switch(
    signal: np.ndarray,
    closes: np.ndarray,
    feat_df: "pd.DataFrame",
    bear_model_raw: Any,
    bear_features: list,
    ma_window: int = 480,
    min_hold: int = 24,
    bear_thresholds: Optional[List[Tuple[float, float]]] = None,
    vol_target: Optional[float] = None,
    vol_feature: str = "atr_norm_14",
    dd_limit: Optional[float] = None,
    dd_cooldown: int = 48,
) -> np.ndarray:
    """Regime-switch: bull→signal (long-only), bear→bear model (short).

    Supports graded bear sizing, vol-adaptive scaling, and drawdown circuit breaker.

    Args:
        bear_thresholds: Graded bear thresholds. None = binary (legacy).
        vol_target: Target volatility for position scaling. None = off.
        vol_feature: Feature name for realized vol (must be in feat_df).
        dd_limit: Max drawdown before forced flat (e.g. -0.15). None = off.
        dd_cooldown: Bars to stay flat after drawdown breach.
    """
    n = len(signal)
    if n != len(closes):
        raise ValueError("signal and closes must have same length")

    bear_mask = _compute_bear_mask(closes, ma_window)

    # Build bear feature matrix
    X_bear = np.column_stack([
        np.nan_to_num(feat_df[f].values[-n:].astype(np.float64), nan=0.0)
        if f in feat_df.columns else np.zeros(n)
        for f in bear_features
    ])

    # Bear model predict on all bars (classifier: prob[:,1])
    prob = bear_model_raw.predict_proba(X_bear)[:, 1]

    out = signal.copy()
    for i in range(n):
        if bear_mask[i]:
            out[i] = _prob_to_score(prob[i], bear_thresholds)

    # Vol-adaptive sizing: scale all positions by target_vol / realized_vol
    if vol_target is not None and vol_feature in feat_df.columns:
        vol_vals = feat_df[vol_feature].values[-n:].astype(np.float64)
        for i in range(n):
            if out[i] != 0.0 and not np.isnan(vol_vals[i]) and vol_vals[i] > 1e-8:
                scale = min(vol_target / vol_vals[i], 1.0)
                out[i] *= scale

    # Re-apply min_hold across regime switches
    held = np.zeros_like(out)
    held[0] = out[0]
    hold_count = 1
    for i in range(1, len(out)):
        if hold_count < min_hold:
            held[i] = held[i - 1]
            hold_count += 1
        else:
            held[i] = out[i]
            if out[i] != held[i - 1]:
                hold_count = 1
            else:
                hold_count += 1

    # Drawdown circuit breaker: force flat when rolling DD exceeds limit
    if dd_limit is not None:
        held = _apply_dd_breaker(held, closes, dd_limit, dd_cooldown)

    return held


def _apply_dd_breaker(
    signal: np.ndarray,
    closes: np.ndarray,
    dd_limit: float,
    cooldown: int,
) -> np.ndarray:
    """Force flat when cumulative strategy drawdown exceeds dd_limit.

    Tracks equity curve from signal*returns, triggers cooldown period.
    """
    n = min(len(signal), len(closes) - 1)
    out = signal.copy()
    ret_1bar = np.diff(closes) / closes[:-1]

    equity = 1.0
    peak = 1.0
    cool_remaining = 0

    for i in range(n):
        if cool_remaining > 0:
            out[i] = 0.0
            cool_remaining -= 1
            # Still track equity (flat, no PnL)
            continue

        if i < len(ret_1bar):
            equity *= (1.0 + out[i] * ret_1bar[i])
        peak = max(peak, equity)
        dd = (equity - peak) / peak

        if dd < dd_limit:
            cool_remaining = cooldown
            out[i] = 0.0

    return out


def _pred_to_signal(
    y_pred: np.ndarray,
    target_mode: str = "",
    deadzone: float = 0.5,
    min_hold: int = 24,
    zscore_window: int = 720,
) -> np.ndarray:
    """Convert raw predictions to discrete positions {-1, 0, +1} with min hold.

    Uses rolling-window z-score normalization (causal — no lookahead).
    Each bar's z-score is computed using the last `zscore_window` predictions.

    Args:
        y_pred: Raw model predictions.
        target_mode: "binary" or continuous.
        deadzone: z-score threshold to enter a position.
        min_hold: Minimum bars to hold before allowing signal change.
        zscore_window: Rolling window size for z-score (default: 720 = 30 days).
    """
    # Step 1: raw discrete signal from predictions
    if target_mode == "binary":
        centered = y_pred - 0.5
        raw = np.sign(centered)
        raw = np.where(np.abs(centered) < 0.02, 0.0, raw)
    else:
        # Rolling-window z-score: causal, adapts to recent distribution
        n = len(y_pred)
        raw = np.zeros(n)
        buf = np.empty(zscore_window)
        buf_idx = 0
        buf_count = 0
        for i in range(n):
            buf[buf_idx] = y_pred[i]
            buf_idx = (buf_idx + 1) % zscore_window
            buf_count = min(buf_count + 1, zscore_window)
            if buf_count < min(168, zscore_window):
                continue  # warmup: need at least 168 bars (1 week)
            window = buf[:buf_count] if buf_count < zscore_window else buf
            mu = np.mean(window)
            std = np.std(window)
            if std < 1e-12:
                continue
            z = (y_pred[i] - mu) / std
            if z > deadzone:
                raw[i] = 1.0
            elif z < -deadzone:
                raw[i] = -1.0

    # Step 2: enforce minimum holding period
    signal = np.zeros_like(raw)
    signal[0] = raw[0]
    hold_count = 1
    for i in range(1, len(raw)):
        if hold_count < min_hold:
            signal[i] = signal[i - 1]
            hold_count += 1
        else:
            signal[i] = raw[i]
            if raw[i] != signal[i - 1]:
                hold_count = 1
            else:
                hold_count += 1
    return signal


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


def _load_spot_closes(symbol: str) -> Dict[int, float]:
    import csv
    path = Path(f"data_files/{symbol}_spot_1h.csv")
    closes: Dict[int, float] = {}
    if not path.exists():
        return closes
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_col = "open_time" if "open_time" in row else "timestamp"
            closes[int(row[ts_col])] = float(row["close"])
    return closes


def _load_fgi_schedule() -> Dict[int, float]:
    import csv
    path = Path("data_files/fear_greed_index.csv")
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule[int(row["timestamp"])] = float(row["value"])
    return schedule


def _load_macro_schedule() -> Dict[int, Dict]:
    """Load macro data, merging all values per date."""
    import csv as _csv
    path = Path("data_files/macro_daily.csv")
    if not path.exists():
        return {}
    date_data: Dict[str, Dict] = {}
    date_ts: Dict[str, int] = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            date = row.get("date", "")
            ts_ms = int(row.get("timestamp_ms", 0))
            if not date or ts_ms == 0:
                continue
            if date not in date_data:
                date_data[date] = {"date": date}
                date_ts[date] = ts_ms
            else:
                date_ts[date] = min(date_ts[date], ts_ms)
            for key in ("dxy", "spx", "vix"):
                val = row.get(key, "")
                if val != "":
                    date_data[date][key] = float(val)
    result: Dict[int, Dict] = {}
    for date, d in date_data.items():
        if len(d) > 1:
            result[date_ts[date]] = d
    return result


def _load_mempool_schedule() -> Dict[int, Dict]:
    """Load mempool fee data as schedule."""
    import csv as _csv
    path = Path("data_files/btc_mempool_fees.csv")
    if not path.exists():
        return {}
    result: Dict[int, Dict] = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            ts_ms = int(row.get("timestamp", 0))
            if ts_ms == 0:
                continue
            result[ts_ms] = {
                "fastest_fee": float(row.get("max_fee", row.get("avg_fee", 0))),
                "economy_fee": float(row.get("min_fee", 1)),
                "mempool_size": float(row.get("avg_fee", 0)) * 1000,
            }
    return result


def _load_liq_proxy_schedule(symbol: str) -> Dict[int, Dict]:
    """Load liquidation proxy data as schedule."""
    import csv as _csv
    path = Path(f"data_files/{symbol}_liquidation_proxy.csv")
    if not path.exists():
        return {}
    result: Dict[int, Dict] = {}
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            ts = int(float(row.get("ts", 0)))
            if ts == 0:
                continue
            result[ts] = {
                "liq_total_volume": float(row.get("liq_proxy_volume", 0)),
                "liq_buy_volume": float(row.get("liq_proxy_buy", 0)),
                "liq_sell_volume": float(row.get("liq_proxy_sell", 0)),
                "liq_count": 1.0 if float(row.get("liq_proxy_volume", 0)) > 0 else 0.0,
            }
    return result


def _advance_dict_schedule(times, sched, idx, ts_ms):
    """Advance a dict-valued schedule pointer. Returns (value, new_idx)."""
    val = None
    while idx < len(times) and times[idx] <= ts_ms:
        val = sched[times[idx]]
        idx += 1
    if val is None and idx > 0:
        val = sched[times[idx - 1]]
    return val, idx


def compute_oos_features(
    symbol: str, df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute features from raw OHLCV dataframe (same as train_v7_alpha)."""
    funding = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    oi = _load_schedule(
        Path(f"data_files/{symbol}_open_interest.csv"), "timestamp", "sum_open_interest")
    ls = _load_schedule(
        Path(f"data_files/{symbol}_ls_ratio.csv"), "timestamp", "long_short_ratio")
    spot_closes = _load_spot_closes(symbol)
    fgi_schedule = _load_fgi_schedule()

    # V11: Load new schedules
    macro_schedule = _load_macro_schedule()
    mempool_schedule = _load_mempool_schedule()
    liq_schedule = _load_liq_proxy_schedule(symbol)

    funding_times = sorted(funding.keys())
    oi_times = sorted(oi.keys())
    ls_times = sorted(ls.keys())
    spot_times = sorted(spot_closes.keys())
    fgi_times = sorted(fgi_schedule.keys())
    macro_times = sorted(macro_schedule.keys())
    mempool_times = sorted(mempool_schedule.keys())
    liq_times = sorted(liq_schedule.keys())
    f_idx, oi_idx, ls_idx, spot_idx, fgi_idx = 0, 0, 0, 0, 0
    macro_idx, mempool_idx, liq_idx = 0, 0, 0

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
        taker_buy_quote_volume = float(row.get("taker_buy_quote_volume", 0) or 0)

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

        spot_close = None
        while spot_idx < len(spot_times) and spot_times[spot_idx] <= ts_ms:
            spot_close = spot_closes[spot_times[spot_idx]]
            spot_idx += 1
        if spot_close is None and spot_idx > 0:
            spot_close = spot_closes[spot_times[spot_idx - 1]]

        fear_greed = None
        while fgi_idx < len(fgi_times) and fgi_times[fgi_idx] <= ts_ms:
            fear_greed = fgi_schedule[fgi_times[fgi_idx]]
            fgi_idx += 1
        if fear_greed is None and fgi_idx > 0:
            fear_greed = fgi_schedule[fgi_times[fgi_idx - 1]]

        # V11: Advance new schedule pointers
        macro_val, macro_idx = _advance_dict_schedule(macro_times, macro_schedule, macro_idx, ts_ms)
        mempool_val, mempool_idx = _advance_dict_schedule(mempool_times, mempool_schedule, mempool_idx, ts_ms)
        liq_val, liq_idx = _advance_dict_schedule(liq_times, liq_schedule, liq_idx, ts_ms)

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
            taker_buy_quote_volume=taker_buy_quote_volume,
            open_interest=open_interest, ls_ratio=ls_ratio,
            spot_close=spot_close, fear_greed=fear_greed,
            macro_metrics=macro_val,
            mempool_metrics=mempool_val,
            liquidation_metrics=liq_val,
        )
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feat_df["close"] = df["close"].values
    return feat_df


def _load_models_from_dir(model_dir: Path) -> Tuple[list, list, dict]:
    """Load models from a V8 model directory with config.json.

    Returns (raw_models, weights, config_dict).
    """
    from infra.model_signing import load_verified_pickle

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    raw_models = []
    weights = cfg.get("ensemble_weights", [])
    for fname in cfg.get("models", []):
        pkl_path = model_dir / fname
        data = load_verified_pickle(pkl_path)
        raw_models.append(data["model"])

    if len(weights) < len(raw_models):
        weights = [1.0 / len(raw_models)] * len(raw_models)

    return raw_models, weights, cfg


def run_backtest(
    symbol: str,
    model_path: Path,
    config_path: Path,
    out_dir: Path,
    long_only: bool = False,
    monthly_gate: bool = False,
    monthly_gate_window: int = 480,
    full: bool = False,
    oos_bars: int = 13140,
    bear_model_path: Optional[str] = None,
    bear_thresholds: Optional[List[Tuple[float, float]]] = None,
    vol_target: Optional[float] = None,
    vol_feature: str = "atr_norm_14",
    dd_limit: Optional[float] = None,
    dd_cooldown: int = 48,
    cost_model_type: str = "flat",
) -> Dict[str, Any]:
    """Run realistic backtest on historical data.

    Args:
        full: Use all available data instead of last oos_bars.
        oos_bars: Number of bars for OOS window (default 13140 = ~18 months).
        bear_thresholds: Graded bear prob thresholds [(0.7,-1.0),(0.6,-0.5),(0.5,0.0)].
        vol_target: Target vol for position scaling (e.g. 0.02).
        vol_feature: Feature for realized vol (default: atr_norm_14).
        dd_limit: Max drawdown before circuit breaker (e.g. -0.15). None=off.
        dd_cooldown: Bars to stay flat after DD breach.
    """
    # Load config — support both V8 (top-level) and legacy (nested under symbol) format
    with open(config_path) as f:
        config = json.load(f)

    if "features" in config:
        # V8 format: features at top level
        feature_names = config["features"]
        horizon = config.get("horizon", 24)
        target_mode = config.get("target_mode", "clipped")
    else:
        sym_config = config.get(symbol, config.get(list(config.keys())[0]))
        feature_names = sym_config["features"]
        horizon = sym_config["horizon"]
        target_mode = sym_config["target_mode"]

    # Load model(s) — support model directory (ensemble) or single pkl
    from infra.model_signing import load_verified_pickle
    ensemble_mode = False
    if model_path.is_dir():
        raw_models, weights, dir_cfg = _load_models_from_dir(model_path)
        ensemble_mode = dir_cfg.get("ensemble", False) and len(raw_models) > 1
        model_label = f"{model_path} (ensemble {len(raw_models)} models)"
    else:
        model_dict = load_verified_pickle(model_path)
        raw_models = [model_dict["model"]]
        weights = [1.0]
        model_label = str(model_path)

    # Auto-detect settings from config.json
    if config.get("monthly_gate", False) and not monthly_gate:
        monthly_gate = True
    if config.get("long_only", False) and not long_only:
        long_only = True

    # Auto-detect position management from config
    pm = config.get("position_management", {})
    if bear_thresholds is None and pm.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]
    if vol_target is None and pm.get("vol_target") is not None:
        vol_target = pm["vol_target"]
        vol_feature = pm.get("vol_feature", vol_feature)
    if dd_limit is None and pm.get("dd_limit") is not None:
        dd_limit = pm["dd_limit"]
        # Config stores positive (e.g. 0.10), breaker expects negative (e.g. -0.10)
        if dd_limit > 0:
            dd_limit = -dd_limit
        dd_cooldown = pm.get("dd_cooldown", dd_cooldown)

    print(f"\n{'='*70}")
    print(f"  Alpha V8 Backtest: {symbol}")
    print(f"  Model: {model_label}")
    print(f"  Horizon: {horizon}, Mode: {target_mode}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Long-only: {long_only}, Monthly gate: {monthly_gate}")
    print(f"  Cost: {COST_PER_TRADE*10000:.0f} bps per trade (fee={FEE_BPS*10000:.0f} + slip={SLIPPAGE_BPS*10000:.0f})")
    print(f"{'='*70}")

    # Load data
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    if full:
        oos_df = df.reset_index(drop=True)
    else:
        oos_df = df.iloc[-oos_bars:].reset_index(drop=True)
    print(f"  OOS bars: {len(oos_df)}")

    # Get timestamps for reporting
    ts_col = "timestamp" if "timestamp" in oos_df.columns else "open_time"
    timestamps = oos_df[ts_col].values.astype(np.int64)

    # Compute features
    print("  Computing features...")
    from features.batch_feature_engine import compute_features_batch
    # Enable V11 features (macro/mempool/liquidation) if any are needed
    v11_features = {"spx_overnight_ret", "mempool_size_zscore_24", "mempool_fee_zscore_24",
                    "fee_urgency_ratio", "exchange_supply_zscore_30", "liquidation_cascade_score",
                    "dxy_change_5d", "vix_zscore_14"}
    needs_v11 = bool(set(feature_names) & v11_features)
    feat_df = compute_features_batch(symbol, oos_df, include_v11=needs_v11)

    # Cross-asset features for non-BTC symbols (BTC-lead alpha)
    cross_features = {"btc_ret_1", "btc_ret_3", "btc_ret_6", "btc_ret_12", "btc_ret_24",
                      "btc_rsi_14", "btc_macd_line", "btc_mean_reversion_20",
                      "btc_atr_norm_14", "btc_bb_width_20",
                      "rolling_beta_30", "rolling_beta_60", "relative_strength_20",
                      "rolling_corr_30", "funding_diff", "funding_diff_ma8", "spread_zscore_20"}
    needs_cross = symbol != "BTCUSDT" and bool(set(feature_names) & cross_features)
    if needs_cross:
        print("  Computing cross-asset (BTC-lead) features...")
        from scripts.train_v7_alpha import _build_cross_features
        cross_map = _build_cross_features([symbol])
        if cross_map and symbol in cross_map:
            cross_df = cross_map[symbol]
            ts_col_name = "timestamp" if "timestamp" in oos_df.columns else "open_time"
            oos_ts = oos_df[ts_col_name].values.astype(np.int64)
            for cname in cross_df.columns:
                if cname in feature_names or cname in cross_features:
                    vals = np.full(len(oos_df), np.nan)
                    for i, ts in enumerate(oos_ts):
                        if ts in cross_df.index:
                            v = cross_df.loc[ts].get(cname)
                            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                                vals[i] = float(v)
                    feat_df[cname] = vals
            print(f"    Merged {len([c for c in cross_df.columns if c in feat_df.columns])} cross-asset features")

    # Prepare X matrix
    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    X = feat_df[feature_names].values.astype(np.float64)

    # Extract volumes for realistic cost model
    vol_col = "volume" if "volume" in oos_df.columns else None
    volumes_raw = oos_df[vol_col].values.astype(np.float64) if vol_col else np.ones(len(oos_df))
    vol_20_raw = feat_df["vol_20"].values.astype(np.float64) if "vol_20" in feat_df.columns else np.full(len(feat_df), np.nan)

    # Warmup: skip first 65 bars where features are not fully computed
    warmup = 65
    X = X[warmup:]
    closes = closes[warmup:]
    timestamps = timestamps[warmup:]
    volumes_arr = volumes_raw[warmup:]
    vol_20_arr = vol_20_raw[warmup:]
    n = len(X)

    # Predict (ensemble: weighted average of all models)
    print(f"  Running inference ({len(raw_models)} model{'s' if len(raw_models)>1 else ''})...")
    preds = []
    for i, rm in enumerate(raw_models):
        import xgboost as xgb
        if isinstance(rm, xgb.core.Booster):
            dm = xgb.DMatrix(X)
            p = rm.predict(dm)
        else:
            p = rm.predict(X)
        preds.append(p * weights[i])
    y_pred = np.sum(preds, axis=0) / sum(weights)

    # Generate signals
    signal = _pred_to_signal(y_pred, target_mode=target_mode)
    if long_only:
        signal = np.clip(signal, 0.0, None)
        print(f"  Long-only mode: short signals clipped to 0")

    # Load bear model if available (Strategy F regime-switch)
    bear_model_dir = bear_model_path
    if bear_model_dir is None and model_path.is_dir():
        bear_model_dir = config.get("bear_model_path")
    bear_model_raw = None
    bear_features = []
    if bear_model_dir:
        bear_dir = Path(bear_model_dir)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with open(bear_cfg_path) as f:
                bear_cfg = json.load(f)
            bear_pkl = bear_dir / bear_cfg["models"][0]
            if bear_pkl.exists():
                from infra.model_signing import load_verified_pickle

                bear_data = load_verified_pickle(bear_pkl)
                bear_model_raw = bear_data["model"]
                bear_features = bear_data.get("features", bear_cfg.get("features", []))
                print(f"  Bear model loaded: {bear_pkl} ({len(bear_features)} features)")

    if monthly_gate and bear_model_raw is not None:
        # Strategy F: regime-switch (bull=long-only, bear=short via bear model)
        pre_active = np.mean(signal != 0) * 100
        # Trim feat_df to match warmup-sliced arrays
        feat_df_trimmed = feat_df.iloc[warmup:].reset_index(drop=True)
        signal = _apply_regime_switch(
            signal, closes, feat_df_trimmed, bear_model_raw, bear_features,
            ma_window=monthly_gate_window, min_hold=config.get("min_hold", 24),
            bear_thresholds=bear_thresholds,
            vol_target=vol_target, vol_feature=vol_feature,
            dd_limit=dd_limit, dd_cooldown=dd_cooldown,
        )
        bear_mask = _compute_bear_mask(closes, monthly_gate_window)
        n_bull = int((~bear_mask).sum())
        n_bear = int(bear_mask.sum())
        n_long = int((signal > 0).sum())
        n_short = int((signal < 0).sum())
        print(f"  Strategy F regime-switch (MA{monthly_gate_window}):")
        print(f"    Bull bars: {n_bull}, Bear bars: {n_bear}")
        print(f"    Long positions: {n_long}, Short positions: {n_short}")
    elif monthly_gate:
        pre_active = np.mean(signal != 0) * 100
        signal = _apply_monthly_gate(signal, closes, monthly_gate_window)
        post_active = np.mean(signal != 0) * 100
        print(f"  Monthly gate (MA{monthly_gate_window}): active {pre_active:.1f}% → {post_active:.1f}% "
              f"(gated {pre_active - post_active:.1f}%)")

    # DD breaker for non-regime-switch path
    if dd_limit is not None and bear_model_raw is None:
        signal = _apply_dd_breaker(signal, closes, dd_limit, dd_cooldown)
        print(f"  DD breaker: limit={dd_limit*100:.1f}%, cooldown={dd_cooldown} bars")

    # Position management info
    if bear_thresholds is not None and bear_model_raw is not None:
        print(f"  Bear thresholds: {bear_thresholds}")
    if vol_target is not None:
        print(f"  Vol target: {vol_target} (feature: {vol_feature})")
    if dd_limit is not None:
        print(f"  DD limit: {dd_limit*100:.1f}%, cooldown: {dd_cooldown} bars")

    print(f"  Signal stats: active={np.mean(signal != 0)*100:.1f}%, "
          f"long={np.mean(signal > 0)*100:.1f}%, short={np.mean(signal < 0)*100:.1f}%")

    # ── Simulate trading ──
    print("  Simulating trades...")
    ret_1bar = np.diff(closes) / closes[:-1]
    signal_for_trade = signal[:len(ret_1bar)]

    gross_pnl = signal_for_trade * ret_1bar

    # Cost computation
    if cost_model_type == "realistic":
        from execution.sim.cost_model import RealisticCostModel
        cm = RealisticCostModel()
        breakdown = cm.compute_costs(
            signal_for_trade,
            closes[:len(signal_for_trade)],
            volumes_arr[:len(signal_for_trade)],
            vol_20_arr[:len(signal_for_trade)],
            capital=INITIAL_CAPITAL,
        )
        cost = breakdown.total_cost
        signal_for_trade = breakdown.clipped_signal
        gross_pnl = signal_for_trade * ret_1bar  # Recompute with clipped signal
        print(f"  Cost model: realistic (fee={np.sum(breakdown.fee_cost)*100:.4f}%, "
              f"impact={np.sum(breakdown.impact_cost)*100:.4f}%, "
              f"spread={np.sum(breakdown.spread_cost)*100:.4f}%)")
    else:
        turnover = np.abs(np.diff(signal_for_trade, prepend=0))
        cost = turnover * COST_PER_TRADE

    # Funding rate cost: long pays positive funding, short receives positive funding
    # Funding settles every 8h; distribute across the 8 bars in that window
    funding_schedule = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    funding_cost = np.zeros(len(signal_for_trade))
    if funding_schedule:
        funding_times = sorted(funding_schedule.keys())
        f_idx = 0
        current_rate = 0.0
        trade_timestamps = timestamps[:len(signal_for_trade)]
        for i in range(len(signal_for_trade)):
            ts = trade_timestamps[i]
            while f_idx < len(funding_times) and funding_times[f_idx] <= ts:
                current_rate = funding_schedule[funding_times[f_idx]]
                f_idx += 1
            # Distribute 8h funding across 8 bars (1h each)
            # Long position pays funding_rate, short receives it
            if signal_for_trade[i] != 0.0:
                funding_cost[i] = signal_for_trade[i] * current_rate / 8.0
        total_funding = float(np.sum(np.abs(funding_cost)))
        print(f"  Funding cost: {total_funding*100:.4f}% total")

    net_pnl = gross_pnl - cost - funding_cost

    # Equity curve
    equity = np.ones(len(net_pnl) + 1) * INITIAL_CAPITAL
    for i in range(len(net_pnl)):
        equity[i + 1] = equity[i] * (1 + net_pnl[i])

    # ── Compute metrics ──
    active = signal_for_trade != 0
    n_active = int(active.sum())

    # Sharpe
    sharpe = 0.0
    if n_active > 1:
        active_pnl = net_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Win rate (bar-level)
    if n_active > 0:
        win_rate = float(np.mean(net_pnl[active] > 0))
    else:
        win_rate = 0.0

    # Cumulative return
    total_return = (equity[-1] / equity[0]) - 1.0

    # Annual return
    n_hours = len(ret_1bar)
    annual_return = (1 + total_return) ** (8760 / max(n_hours, 1)) - 1.0

    # Profit factor
    gross_wins = float(np.sum(net_pnl[net_pnl > 0]))
    gross_losses = float(np.abs(np.sum(net_pnl[net_pnl < 0])))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Total turnover and cost
    turnover = np.abs(np.diff(signal_for_trade, prepend=0))
    total_turnover = float(np.sum(turnover))
    total_cost = float(np.sum(cost))

    # Trade count (position changes)
    position_changes = np.where(np.diff(signal_for_trade) != 0)[0]
    n_trades = len(position_changes)

    # Average holding period
    in_position = signal_for_trade != 0
    if n_trades > 0 and n_active > 0:
        avg_holding = n_active / max(n_trades, 1)
    else:
        avg_holding = 0

    # Monthly breakdown
    monthly = []
    dt_list = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in timestamps[:len(net_pnl)]]
    month_keys = [f"{d.year}-{d.month:02d}" for d in dt_list]
    unique_months = sorted(set(month_keys))

    for mk in unique_months:
        mask = np.array([m == mk for m in month_keys])
        if mask.sum() < 10:
            continue
        m_pnl = net_pnl[mask]
        m_active = active[mask]
        m_ret = float(np.sum(m_pnl))
        m_sharpe = 0.0
        if m_active.sum() > 1:
            m_active_pnl = m_pnl[m_active]
            m_std = float(np.std(m_active_pnl, ddof=1))
            if m_std > 0:
                m_sharpe = float(np.mean(m_active_pnl)) / m_std * np.sqrt(8760)
        monthly.append({
            "month": mk,
            "return": m_ret,
            "sharpe": m_sharpe,
            "active_pct": float(m_active.mean()) * 100,
            "bars": int(mask.sum()),
        })

    pos_months = sum(1 for m in monthly if m["return"] > 0)

    # ── Print results ──
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS: {symbol}")
    print(f"{'='*70}")
    print(f"  Period: {dt_list[0].strftime('%Y-%m-%d')} → {dt_list[-1].strftime('%Y-%m-%d')}")
    print(f"  Bars: {n_hours:,}")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Final equity: ${equity[-1]:,.2f}")
    print(f"\n  --- Performance ---")
    print(f"  Total return:    {total_return*100:+.2f}%")
    print(f"  Annual return:   {annual_return*100:+.2f}%")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd*100:.2f}%")
    print(f"  Profit factor:   {profit_factor:.2f}")
    print(f"\n  --- Trading ---")
    print(f"  Position changes: {n_trades}")
    print(f"  Avg holding:     {avg_holding:.1f} bars ({avg_holding:.0f}h)")
    print(f"  Active:          {n_active/n_hours*100:.1f}%")
    print(f"  Win rate (bar):  {win_rate*100:.1f}%")
    print(f"  Total turnover:  {total_turnover:.1f}")
    print(f"  Total cost:      {total_cost*100:.4f}%")
    print(f"\n  --- Monthly Breakdown ---")
    print(f"  {'Month':<10} {'Return':>8} {'Sharpe':>8} {'Active%':>8}")
    print(f"  {'-'*36}")
    for m in monthly:
        print(f"  {m['month']:<10} {m['return']*100:>+7.2f}% {m['sharpe']:>8.2f} {m['active_pct']:>7.1f}%")
    print(f"  {'-'*36}")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # ── H1/H2 split ──
    mid = len(net_pnl) // 2
    for label, s, e in [("H1 (early)", 0, mid), ("H2 (late)", mid, len(net_pnl))]:
        h_pnl = net_pnl[s:e]
        h_active = active[s:e]
        h_ret = float(np.sum(h_pnl))
        h_sharpe = 0.0
        if h_active.sum() > 1:
            h_active_pnl = h_pnl[h_active]
            h_std = float(np.std(h_active_pnl, ddof=1))
            if h_std > 0:
                h_sharpe = float(np.mean(h_active_pnl)) / h_std * np.sqrt(8760)
        print(f"\n  {label}: return={h_ret*100:+.2f}% sharpe={h_sharpe:.2f} active={h_active.mean()*100:.1f}%")

    # ── Per-regime metrics (when bear model used) ──
    regime_metrics = {}
    if bear_model_raw is not None:
        bear_mask_final = _compute_bear_mask(closes, monthly_gate_window)
        for regime_label, regime_mask in [("Bull", ~bear_mask_final[:len(net_pnl)]),
                                           ("Bear", bear_mask_final[:len(net_pnl)])]:
            r_pnl = net_pnl[regime_mask]
            r_active = active[regime_mask]
            r_n = int(regime_mask.sum())
            r_n_active = int(r_active.sum())
            r_ret = float(np.sum(r_pnl))
            r_sharpe = 0.0
            if r_n_active > 1:
                r_active_pnl = r_pnl[r_active]
                r_std = float(np.std(r_active_pnl, ddof=1))
                if r_std > 0:
                    r_sharpe = float(np.mean(r_active_pnl)) / r_std * np.sqrt(8760)
            regime_metrics[regime_label.lower()] = {
                "bars": r_n, "active": r_n_active,
                "return": r_ret, "sharpe": r_sharpe,
            }
            sig_in_regime = signal_for_trade[regime_mask]
            r_long = int((sig_in_regime > 0).sum())
            r_short = int((sig_in_regime < 0).sum())
            print(f"\n  {regime_label} regime: bars={r_n}, active={r_n_active}, "
                  f"long={r_long}, short={r_short}")
            print(f"    Return={r_ret*100:+.2f}%, Sharpe={r_sharpe:.2f}")

    # ── Save results ──
    out_dir.mkdir(parents=True, exist_ok=True)

    # Equity curve CSV
    eq_df = pd.DataFrame({
        "timestamp": timestamps[:len(equity)],
        "equity": equity,
        "close": np.concatenate([[closes[0]], closes[:len(net_pnl)]]),
        "signal": np.concatenate([[0.0], signal_for_trade]),
        "net_pnl": np.concatenate([[0.0], net_pnl]),
    })
    eq_df.to_csv(out_dir / "equity_curve.csv", index=False)

    # Monthly CSV
    pd.DataFrame(monthly).to_csv(out_dir / "monthly.csv", index=False)

    # Summary JSON
    summary = {
        "symbol": symbol,
        "horizon": horizon,
        "target_mode": target_mode,
        "n_features": len(feature_names),
        "features": feature_names,
        "period_start": dt_list[0].isoformat(),
        "period_end": dt_list[-1].isoformat(),
        "n_bars": n_hours,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(equity[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "n_trades": n_trades,
        "avg_holding_bars": avg_holding,
        "active_pct": n_active / n_hours * 100,
        "win_rate": win_rate,
        "total_turnover": total_turnover,
        "total_cost_pct": total_cost * 100,
        "pos_months": pos_months,
        "total_months": len(monthly),
        "cost_bps": COST_PER_TRADE * 10000,
    }
    if regime_metrics:
        summary["regime_metrics"] = regime_metrics
        summary["strategy"] = "F_regime_switch"
    if bear_thresholds is not None:
        summary["bear_thresholds"] = bear_thresholds
    if vol_target is not None:
        summary["vol_target"] = vol_target
        summary["vol_feature"] = vol_feature
    if dd_limit is not None:
        summary["dd_limit"] = dd_limit
        summary["dd_cooldown"] = dd_cooldown
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Alpha V8 model on OOS data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--model", default=None,
                        help="Model .pkl path or model directory (for ensemble)")
    parser.add_argument("--config", default=None, help="Config JSON path")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--long-only", action="store_true",
                        help="Clip short signals to 0 (long-only mode)")
    parser.add_argument("--monthly-gate", action="store_true",
                        help="Gate signal when close <= SMA(window)")
    parser.add_argument("--monthly-gate-window", type=int, default=480,
                        help="SMA window for monthly gate (default: 480)")
    parser.add_argument("--full", action="store_true",
                        help="Use all available historical data (not just OOS window)")
    parser.add_argument("--oos-bars", type=int, default=13140,
                        help="OOS window size in bars (default: 13140 = ~18 months)")
    parser.add_argument("--bear-model", default=None,
                        help="Bear model directory (overrides config.json bear_model_path)")
    parser.add_argument("--bear-thresholds", default=None,
                        help='Graded bear thresholds JSON, e.g. \'[[0.7,-1.0],[0.6,-0.5],[0.5,0.0]]\'')
    parser.add_argument("--vol-target", type=float, default=None,
                        help="Target volatility for position scaling (e.g. 0.02)")
    parser.add_argument("--vol-feature", default="atr_norm_14",
                        help="Feature for realized vol (default: atr_norm_14)")
    parser.add_argument("--dd-limit", type=float, default=None,
                        help="Max drawdown circuit breaker (e.g. -0.15). Negative value.")
    parser.add_argument("--dd-cooldown", type=int, default=48,
                        help="Bars to stay flat after DD breach (default: 48)")
    parser.add_argument("--cost-model", choices=["flat", "realistic"], default="flat",
                        help="Cost model: flat (6bps) or realistic (sqrt-impact + spread)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    symbol = args.symbol.upper()

    # Resolve model path: --model can be a directory (ensemble) or single pkl
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = Path(f"models_v8/{symbol}_gate_v2")
        if not model_path.exists():
            model_path = Path(f"results/alpha_rebuild_v3/step6_final/{symbol}/v8_final.pkl")

    # Resolve config path
    if args.config:
        config_path = Path(args.config)
    elif model_path.is_dir() and (model_path / "config.json").exists():
        config_path = model_path / "config.json"
    else:
        config_path = Path("results/alpha_rebuild_v3/step6_final/final_results.json")

    out_dir = Path(args.out) if args.out else Path(
        f"results/backtest_v8/{symbol}")

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    bear_thresholds = None
    if args.bear_thresholds:
        bear_thresholds = [tuple(x) for x in json.loads(args.bear_thresholds)]

    run_backtest(symbol, model_path, config_path, out_dir,
                 long_only=args.long_only,
                 monthly_gate=args.monthly_gate,
                 monthly_gate_window=args.monthly_gate_window,
                 full=args.full,
                 oos_bars=args.oos_bars,
                 bear_model_path=args.bear_model,
                 bear_thresholds=bear_thresholds,
                 vol_target=args.vol_target,
                 vol_feature=args.vol_feature,
                 dd_limit=args.dd_limit,
                 dd_cooldown=args.dd_cooldown,
                 cost_model_type=args.cost_model)


if __name__ == "__main__":
    main()
