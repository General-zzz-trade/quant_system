#!/usr/bin/env python3
"""Parity gap ablation study — isolates each factor's contribution.

Compares batch backtest vs kernel backtest by progressively removing
differences to identify the dominant factors causing the 3x Sharpe gap.

Factors tested:
  A. Batch ensemble baseline (weighted average → z-score → signal)
  B. Last-model-only (simulates kernel "last model wins" behavior)
  C. First-model-only
  D. Ensemble + strength clip to [0,1]
  E. No monthly gate (same as A)
  F. With monthly gate (zero bear bars, no bear model)
  G. Full batch + bear model (backtest_portfolio equivalent)

Usage:
    python3 -m scripts.parity_ablation --symbol BTCUSDT --oos-bars 13140
    python3 -m scripts.parity_ablation --symbol ALL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from alpha.signal_transform import pred_to_signal as _pred_to_signal

os.environ.setdefault("QUANT_ALLOW_UNSIGNED_MODELS", "1")

FEE_BPS = 4e-4
SLIPPAGE_BPS = 2e-4
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS


def _sharpe(pnl: np.ndarray) -> float:
    active = pnl[pnl != 0]
    if len(active) < 2:
        return 0.0
    s = float(np.std(active, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(active)) / s * np.sqrt(8760)


def _sim_pnl(signal: np.ndarray, closes: np.ndarray, funding_rates: np.ndarray) -> np.ndarray:
    n = len(signal)
    pnl = np.zeros(n)
    for i in range(n - 1):
        ret = (closes[i + 1] - closes[i]) / closes[i]
        sig = signal[i]
        prev_sig = signal[i - 1] if i > 0 else 0.0
        turnover = abs(sig - prev_sig)
        cost = turnover * COST_PER_TRADE
        funding_cost = sig * funding_rates[i] / 8.0
        pnl[i] = sig * ret - cost - abs(funding_cost)
    return pnl


def run_ablation(symbol: str, oos_bars: int, full: bool):
    from infra.model_signing import load_verified_pickle
    from features.batch_feature_engine import compute_features_batch

    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    feature_names = cfg["features"]
    deadzone = cfg.get("deadzone", 0.5)
    min_hold = cfg.get("min_hold", 24)
    target_mode = cfg.get("target_mode", "clipped")

    raw_models = []
    weights = cfg.get("ensemble_weights", [])
    for fname in cfg.get("models", []):
        pkl_path = model_dir / fname
        if pkl_path.exists():
            data = load_verified_pickle(pkl_path)
            raw_models.append(data["model"])
    if len(weights) < len(raw_models):
        weights = [1.0 / len(raw_models)] * len(raw_models)

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    df = pd.read_csv(csv_path)
    if not full:
        df = df.iloc[-oos_bars:].reset_index(drop=True)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)

    v11_features = {"spx_overnight_ret", "dxy_change_5d", "vix_zscore_14",
                    "mempool_size_zscore_24", "fee_urgency_ratio",
                    "exchange_supply_zscore_30", "liquidation_cascade_score"}
    needs_v11 = bool(set(feature_names) & v11_features)
    feat_df = compute_features_batch(symbol, df, include_v11=needs_v11)

    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    warmup = 65
    closes = feat_df["close"].values[warmup:].astype(np.float64)
    X = feat_df[feature_names].values[warmup:].astype(np.float64)
    timestamps = timestamps[warmup:]
    n = len(X)

    # Funding rates
    funding_rates = np.zeros(n)
    funding_path = Path(f"data_files/{symbol}_funding.csv")
    if funding_path.exists():
        fdf = pd.read_csv(funding_path)
        f_ts = fdf["timestamp"].values.astype(np.int64)
        f_rate = fdf["funding_rate"].values.astype(np.float64)
        fi, cur_rate = 0, 0.0
        for i in range(n):
            while fi < len(f_ts) and f_ts[fi] <= timestamps[i]:
                cur_rate = f_rate[fi]
                fi += 1
            funding_rates[i] = cur_rate

    # Per-model predictions
    import xgboost as xgb
    preds = []
    for rm in raw_models:
        if isinstance(rm, xgb.core.Booster):
            p = rm.predict(xgb.DMatrix(X))
        else:
            p = rm.predict(X)
        preds.append(p)

    # ── Experiments ──
    y_ensemble = sum(p * w for p, w in zip(preds, weights)) / sum(weights)

    # A: Batch ensemble baseline
    signal_A = _pred_to_signal(y_ensemble, target_mode=target_mode,
                               deadzone=deadzone, min_hold=min_hold)

    # B: Last model only (simulates kernel behavior)
    signal_B = _pred_to_signal(preds[-1], target_mode=target_mode,
                               deadzone=deadzone, min_hold=min_hold)

    # C: First model only
    signal_C = _pred_to_signal(preds[0], target_mode=target_mode,
                               deadzone=deadzone, min_hold=min_hold)

    # D: Ensemble but clip raw pred to strength range [-1, 1]
    y_clipped = np.clip(y_ensemble, -1.0, 1.0)
    signal_D = _pred_to_signal(y_clipped, target_mode=target_mode,
                               deadzone=deadzone, min_hold=min_hold)

    # F: Ensemble + monthly gate (zero in bear, no bear model)
    signal_F = signal_A.copy()
    ma_window = cfg.get("monthly_gate_window", 480)
    if n > ma_window:
        cs = np.cumsum(closes)
        ma = np.empty(n)
        ma[:ma_window] = np.nan
        ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
        bear_mask = (~np.isnan(ma)) & (closes <= ma)
        signal_F[bear_mask] = 0.0

    # G: Full batch + bear model (backtest_portfolio equivalent)
    signal_G = signal_A.copy()
    is_long_only = cfg.get("long_only", False)
    bear_path = cfg.get("bear_model_path")
    pm = cfg.get("position_management", {})
    bear_thresholds = None
    if pm.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]

    if bear_path:
        bear_dir = Path(bear_path)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with open(bear_cfg_path) as f:
                bear_cfg = json.load(f)
            bear_pkl_path = bear_dir / bear_cfg["models"][0]
            if bear_pkl_path.exists():
                bear_data = load_verified_pickle(bear_pkl_path)
                bear_model = bear_data["model"]
                bear_features = bear_data.get("features", bear_cfg.get("features", []))
                feat_trimmed = feat_df.iloc[warmup:].reset_index(drop=True)
                X_bear = np.column_stack([
                    np.nan_to_num(feat_trimmed[f].values.astype(np.float64), nan=0.0)
                    if f in feat_trimmed.columns else np.zeros(n)
                    for f in bear_features
                ])
                prob = bear_model.predict_proba(X_bear)[:, 1]

                cs = np.cumsum(closes)
                ma = np.empty(n)
                ma[:ma_window] = np.nan
                ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
                bear_mask_full = np.isnan(ma) | (closes <= ma)

                if is_long_only:
                    signal_G = np.clip(signal_G, 0.0, None)

                for i in range(n):
                    if bear_mask_full[i]:
                        if bear_thresholds:
                            score = 0.0
                            for thresh, s in bear_thresholds:
                                if prob[i] > thresh:
                                    score = s
                                    break
                            signal_G[i] = score
                        else:
                            signal_G[i] = -1.0 if prob[i] > 0.5 else 0.0

    # ── Results ──
    experiments = {
        "A: Batch ensemble (baseline)": signal_A,
        "B: Last model only (kernel)": signal_B,
        "C: First model only": signal_C,
        "D: Ensemble + clip [-1,1]": signal_D,
        "F: Ensemble + gate (no bear)": signal_F,
        "G: Full batch + bear model": signal_G,
    }

    print(f"\n{'='*70}")
    print(f"  Parity Gap Ablation: {symbol}")
    print(f"  Bars: {n}, Models: {len(raw_models)}, Weights: {weights}")
    print(f"  Deadzone: {deadzone}, Min hold: {min_hold}")
    print(f"{'='*70}")
    print()
    print(f"  {'Experiment':<40} {'Sharpe':>8} {'Return':>10} {'Active%':>10}")
    print(f"  {'-'*70}")

    for name, sig in experiments.items():
        pnl = _sim_pnl(sig, closes, funding_rates)
        sharpe = _sharpe(pnl)
        total_ret = float(np.sum(pnl)) * 100
        active_pct = float(np.mean(sig != 0)) * 100
        print(f"  {name:<40} {sharpe:>8.2f} {total_ret:>+9.2f}% {active_pct:>9.1f}%")

    # Factor attribution
    print(f"\n  --- Factor Attribution ---")
    pnl_A = _sim_pnl(signal_A, closes, funding_rates)
    pnl_B = _sim_pnl(signal_B, closes, funding_rates)
    pnl_F = _sim_pnl(signal_F, closes, funding_rates)
    pnl_G = _sim_pnl(signal_G, closes, funding_rates)

    s_A = _sharpe(pnl_A)
    s_B = _sharpe(pnl_B)
    s_F = _sharpe(pnl_F)
    s_G = _sharpe(pnl_G)
    print(f"  Ensemble avg vs last-model (A-B): {s_A - s_B:+.2f} Sharpe")
    print(f"  Monthly gate effect (A-F):        {s_A - s_F:+.2f} Sharpe")
    print(f"  Bear model effect (A-G):          {s_A - s_G:+.2f} Sharpe")

    # Prediction stats
    print(f"\n  --- Prediction Distribution ---")
    for i, (name, pred) in enumerate(zip(cfg.get("models", []), preds)):
        print(f"  {name}: mean={np.mean(pred):.6f}, std={np.std(pred):.6f}, "
              f"|max|={np.max(np.abs(pred)):.6f}")
    print(f"  Ensemble: mean={np.mean(y_ensemble):.6f}, std={np.std(y_ensemble):.6f}")
    if len(preds) >= 2:
        corr = np.corrcoef(preds[0], preds[1])[0, 1]
        print(f"  Model correlation: {corr:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="ALL")
    parser.add_argument("--oos-bars", type=int, default=13140)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"] if args.symbol == "ALL" else [args.symbol]
    for sym in symbols:
        run_ablation(sym, args.oos_bars, args.full)


if __name__ == "__main__":
    main()
