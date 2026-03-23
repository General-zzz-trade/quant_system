#!/usr/bin/env python3
"""Shadow A/B testing framework -- run two model versions side-by-side.

Loads two model directories, runs both through the full signal pipeline
on the same historical data, and compares signal agreement, z-score
correlation, divergence analysis, and simulated PnL.  No trades executed.

Usage:
    python3 -m scripts.ops.shadow_compare \
        --model-a models_v8/BTCUSDT_gate_v2 \
        --model-b models_v8/BTCUSDT_4h \
        --symbol BTCUSDT --days 90

    python3 -m scripts.ops.shadow_compare \
        --model-a models_v8/ETHUSDT_gate_v2 \
        --model-b models_v8/ETHUSDT_4h \
        --symbol ETHUSDT --days 60 --json

    # Legacy registry-based comparison (backward compat):
    python3 -m scripts.ops.shadow_compare \
        --symbol BTCUSDT --registry-mode
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data_files")
MODEL_BASE = Path("models_v8")
OUTPUT_DIR = Path("data/runtime")


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class ModelSignals:
    """Signal series produced by one model."""
    name: str
    model_dir: str
    predictions: np.ndarray    # raw ensemble predictions
    z_scores: np.ndarray       # rolling z-scores
    signals: np.ndarray        # discretized signals (+1, 0, -1)
    config: dict = field(default_factory=dict)


@dataclass
class BarDivergence:
    """One bar where models A and B disagree."""
    bar_index: int
    timestamp: str
    close: float
    a_z: float
    b_z: float
    a_signal: int
    b_signal: int
    fwd_return: float          # realized forward return (next-bar)
    a_correct: bool            # model A's signal direction matched return
    b_correct: bool            # model B's signal direction matched return


@dataclass
class ABReport:
    """Full A/B comparison report."""
    symbol: str
    n_bars: int
    n_eval_bars: int           # bars after warmup
    model_a: str
    model_b: str
    # Agreement
    signal_agreement_rate: float
    signal_agreement_long: float
    signal_agreement_short: float
    signal_agreement_flat: float
    # Correlation
    z_score_correlation: float
    prediction_correlation: float
    # Directional accuracy (when models diverge)
    n_divergences: int
    a_correct_on_diverge: int
    b_correct_on_diverge: int
    # Simulated PnL
    a_total_return: float
    b_total_return: float
    a_sharpe: float
    b_sharpe: float
    a_max_drawdown: float
    b_max_drawdown: float
    a_n_trades: int
    b_n_trades: int
    # Signal distribution
    a_signal_dist: Dict[str, int] = field(default_factory=dict)
    b_signal_dist: Dict[str, int] = field(default_factory=dict)
    # Divergence samples
    divergences: List[dict] = field(default_factory=list)
    # Verdict
    verdict: str = ""
    details: str = ""


# ── Model loading and prediction ─────────────────────────────────────

def _load_model_and_predict(
    model_dir: Path,
    feat_df: pd.DataFrame,
    label: str,
) -> Optional[ModelSignals]:
    """Load a model directory and generate predictions + signals.

    Uses the same model_loader as the live AlphaRunner to ensure
    identical ensemble weighting (Ridge 60% + LightGBM 40%).
    """
    from alpha.model_loader_prod import load_model
    from scripts.shared.signal_postprocess import rolling_zscore

    try:
        model_info = load_model(model_dir)
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_dir, e)
        return None

    config = model_info["config"]
    features_list = model_info["features"]
    deadzone = model_info["deadzone"]
    min_hold = model_info["min_hold"]
    max_hold = model_info["max_hold"]
    zscore_window = model_info["zscore_window"]
    zscore_warmup = model_info["zscore_warmup"]
    long_only = model_info.get("long_only", False)

    # Ensure features are available
    available = [f for f in features_list if f in feat_df.columns]
    if len(available) < len(features_list) * 0.3:
        logger.error(
            "%s: only %d/%d features available in data",
            label, len(available), len(features_list),
        )
        return None

    missing = set(features_list) - set(available)
    if missing:
        logger.info("%s: %d missing features (using 0.0): %s",
                     label, len(missing), list(missing)[:5])

    # IC-weighted ensemble prediction (same as signal_reconcile.py)
    horizon_models = model_info.get("horizon_models", [])
    if not horizon_models:
        logger.error("%s: no horizon models found", label)
        return None

    n = len(feat_df)
    predictions = np.zeros(n)
    total_weight = 0.0
    ridge_weight_cfg = config.get("ridge_weight", 0.6)

    for hm in horizon_models:
        ic_weight = hm.get("ic", 0.01)
        hm_features = hm.get("features", features_list)
        hm_available = [f for f in hm_features if f in feat_df.columns]

        if len(hm_available) < len(hm_features) * 0.3:
            continue

        X_hm = feat_df[hm_available].fillna(0.0).values
        hm_pred = np.zeros(n)

        # Ridge prediction
        ridge = hm.get("ridge")
        ridge_features = hm.get("ridge_features")
        ridge_ok = False
        if ridge is not None:
            r_feats = ridge_features or hm_features
            r_available = [f for f in r_feats if f in feat_df.columns]
            expected = getattr(ridge, "n_features_in_", len(r_available))
            if len(r_available) == expected:
                X_ridge = feat_df[r_available].fillna(0.0).values
                try:
                    hm_pred += ridge_weight_cfg * ridge.predict(X_ridge)
                    ridge_ok = True
                except Exception as e:
                    logger.warning("%s: Ridge predict failed: %s", label, e)

        # LightGBM prediction
        lgbm = hm.get("lgbm")
        if lgbm is not None:
            try:
                lgbm_w = (1.0 - ridge_weight_cfg) if ridge_ok else 1.0
                hm_pred += lgbm_w * lgbm.predict(X_hm)
            except Exception as e:
                logger.warning("%s: LightGBM predict failed: %s", label, e)

        predictions += ic_weight * hm_pred
        total_weight += ic_weight

    if total_weight > 0:
        predictions /= total_weight

    # Rolling z-score
    z_scores = rolling_zscore(predictions, window=zscore_window, warmup=zscore_warmup)

    # Discretize + min-hold + max-hold (reuse signal_reconcile logic)
    signals = _apply_constraints(z_scores, deadzone, min_hold, max_hold, long_only)

    return ModelSignals(
        name=label,
        model_dir=str(model_dir),
        predictions=predictions,
        z_scores=z_scores,
        signals=signals,
        config={
            "deadzone": deadzone,
            "min_hold": min_hold,
            "max_hold": max_hold,
            "zscore_window": zscore_window,
            "zscore_warmup": zscore_warmup,
            "long_only": long_only,
            "ridge_weight": ridge_weight_cfg,
            "n_features": len(features_list),
            "n_available": len(available),
            "n_horizons": len(horizon_models),
        },
    )


def _apply_constraints(
    z_scores: np.ndarray,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    long_only: bool,
) -> np.ndarray:
    """Discretize z-scores and apply min-hold / max-hold constraints."""
    n = len(z_scores)
    signals = np.zeros(n)
    hold_count = 1

    for i in range(n):
        # Discretize
        if z_scores[i] > deadzone:
            desired = 1.0
        elif z_scores[i] < -deadzone:
            desired = -1.0
        else:
            desired = 0.0

        if long_only and desired < 0:
            desired = 0.0

        if i == 0:
            signals[i] = desired
            hold_count = 1
            continue

        prev = signals[i - 1]

        # Min-hold lockout
        if hold_count < min_hold and prev != 0:
            signals[i] = prev
            hold_count += 1
            continue

        # Max-hold forced exit
        if hold_count >= max_hold and prev != 0:
            signals[i] = 0.0
            hold_count = 1
            continue

        # Allow change
        if desired != prev:
            signals[i] = desired
            hold_count = 1
        else:
            signals[i] = desired
            hold_count += 1

    return signals


# ── Simulated PnL ────────────────────────────────────────────────────

def _simulate_pnl(
    signals: np.ndarray,
    closes: np.ndarray,
) -> Dict[str, Any]:
    """Simple bar-level PnL simulation.

    Assumes: enter at close of signal bar, exit at close of next bar.
    Position = signal direction. No fees (comparison is relative).
    """
    n = len(signals)
    returns = np.diff(closes) / closes[:-1]  # bar-to-bar returns

    # Strategy returns: signal[i] * return[i] (signal applied, return realized next bar)
    # Shift signal by 1 to avoid look-ahead
    strat_returns = np.zeros(n - 1)
    for i in range(n - 1):
        strat_returns[i] = signals[i] * returns[i]

    cum_returns = np.cumsum(strat_returns)
    total_return = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0

    # Sharpe (annualized, assuming hourly bars)
    if len(strat_returns) > 1 and np.std(strat_returns) > 1e-12:
        sharpe = float(np.mean(strat_returns) / np.std(strat_returns) * np.sqrt(8760))
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns - peak
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Trade count (signal changes from 0 to non-zero, or direction changes)
    n_trades = 0
    for i in range(1, len(signals)):
        if signals[i] != signals[i - 1] and signals[i] != 0:
            n_trades += 1

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "strat_returns": strat_returns,
    }


# ── Comparison engine ─────────────────────────────────────────────────

def _signal_distribution(signals: np.ndarray) -> Dict[str, int]:
    """Count signal values."""
    return {
        "long": int(np.sum(signals == 1)),
        "flat": int(np.sum(signals == 0)),
        "short": int(np.sum(signals == -1)),
    }


def compare_ab(
    model_a_dir: Path,
    model_b_dir: Path,
    symbol: str,
    days: int = 90,
    label_a: str = "A",
    label_b: str = "B",
) -> Optional[ABReport]:
    """Run full A/B comparison between two model directories."""
    from features.batch_feature_engine import compute_features_batch

    # Determine interval from model config
    interval_a = _detect_interval(model_a_dir)
    interval_b = _detect_interval(model_b_dir)

    if interval_a != interval_b:
        logger.warning(
            "Models use different intervals (%s vs %s) -- using %s for data",
            interval_a, interval_b, interval_a,
        )

    interval = interval_a
    suffix_map = {"60": "1h", "15": "15m", "240": "4h", "5": "5m", "1440": "1d"}
    suffix = suffix_map.get(interval, "1h")

    # Load price data
    csv_path = DATA_DIR / f"{symbol}_{suffix}.csv"
    if not csv_path.exists():
        logger.error("Data file not found: %s", csv_path)
        return None

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d bars from %s", len(df), csv_path)

    # Trim to requested window (keep warmup)
    interval_hours = int(interval) / 60
    bars_per_day = 24 / interval_hours
    eval_bars = int(days * bars_per_day)
    warmup_bars = 900  # > zscore_window(720) + zscore_warmup(180)
    total_needed = eval_bars + warmup_bars

    if len(df) > total_needed:
        df = df.iloc[-total_needed:].reset_index(drop=True)
    elif len(df) < warmup_bars + 100:
        logger.error("Insufficient data: %d bars (need %d+ for warmup)", len(df), warmup_bars + 100)
        return None

    # Compute batch features
    try:
        feat_df = compute_features_batch(symbol, df)
    except Exception as e:
        logger.error("Batch feature computation failed: %s", e)
        return None

    logger.info("Computed %d features x %d bars", len(feat_df.columns), len(feat_df))

    # Generate signals from both models
    sig_a = _load_model_and_predict(model_a_dir, feat_df, label_a)
    sig_b = _load_model_and_predict(model_b_dir, feat_df, label_b)

    if sig_a is None or sig_b is None:
        logger.error("Failed to generate signals from one or both models")
        return None

    # Evaluation window: skip warmup
    eval_start = max(warmup_bars, len(df) - eval_bars)
    n_eval = len(df) - eval_start

    sigs_a = sig_a.signals[eval_start:]
    sigs_b = sig_b.signals[eval_start:]
    z_a = sig_a.z_scores[eval_start:]
    z_b = sig_b.z_scores[eval_start:]
    pred_a = sig_a.predictions[eval_start:]
    pred_b = sig_b.predictions[eval_start:]
    closes = df["close"].values.astype(np.float64)[eval_start:]

    # Timestamps for divergence reporting
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values[eval_start:]

    # Signal agreement
    agree_mask = sigs_a == sigs_b
    agreement_rate = float(np.mean(agree_mask))

    both_long = float(np.mean((sigs_a == 1) & (sigs_b == 1)))
    both_short = float(np.mean((sigs_a == -1) & (sigs_b == -1)))
    both_flat = float(np.mean((sigs_a == 0) & (sigs_b == 0)))

    # Z-score and prediction correlation
    valid_z = (z_a != 0) | (z_b != 0)
    if np.sum(valid_z) > 10:
        z_corr = float(np.corrcoef(z_a[valid_z], z_b[valid_z])[0, 1])
    else:
        z_corr = 0.0

    valid_p = (pred_a != 0) | (pred_b != 0)
    if np.sum(valid_p) > 10:
        p_corr = float(np.corrcoef(pred_a[valid_p], pred_b[valid_p])[0, 1])
    else:
        p_corr = 0.0

    # Forward returns for divergence analysis
    fwd_returns = np.zeros(n_eval)
    for i in range(n_eval - 1):
        fwd_returns[i] = (closes[i + 1] - closes[i]) / closes[i]

    # Find divergences
    diverge_mask = ~agree_mask
    diverge_indices = np.where(diverge_mask)[0]
    a_correct_count = 0
    b_correct_count = 0
    divergences: List[dict] = []

    for idx in diverge_indices:
        if idx >= n_eval - 1:
            continue
        fwd_ret = fwd_returns[idx]
        a_sig = int(sigs_a[idx])
        b_sig = int(sigs_b[idx])
        a_ok = (a_sig > 0 and fwd_ret > 0) or (a_sig < 0 and fwd_ret < 0) or a_sig == 0
        b_ok = (b_sig > 0 and fwd_ret > 0) or (b_sig < 0 and fwd_ret < 0) or b_sig == 0

        if a_ok:
            a_correct_count += 1
        if b_ok:
            b_correct_count += 1

        # Store sample divergences (cap at 50)
        if len(divergences) < 50:
            ts_val = timestamps[idx]
            if isinstance(ts_val, (int, float, np.integer, np.floating)):
                ts_str = datetime.fromtimestamp(float(ts_val) / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                )
            else:
                ts_str = str(ts_val)

            divergences.append({
                "bar": int(idx + eval_start),
                "timestamp": ts_str,
                "close": round(float(closes[idx]), 2),
                "a_z": round(float(z_a[idx]), 4),
                "b_z": round(float(z_b[idx]), 4),
                "a_signal": a_sig,
                "b_signal": b_sig,
                "fwd_return_bps": round(fwd_ret * 10000, 2),
                "a_correct": a_ok,
                "b_correct": b_ok,
            })

    # Simulated PnL
    pnl_a = _simulate_pnl(sig_a.signals[eval_start:], closes)
    pnl_b = _simulate_pnl(sig_b.signals[eval_start:], closes)

    # Build report
    report = ABReport(
        symbol=symbol,
        n_bars=len(df),
        n_eval_bars=n_eval,
        model_a=str(model_a_dir),
        model_b=str(model_b_dir),
        signal_agreement_rate=round(agreement_rate, 4),
        signal_agreement_long=round(both_long, 4),
        signal_agreement_short=round(both_short, 4),
        signal_agreement_flat=round(both_flat, 4),
        z_score_correlation=round(z_corr, 4),
        prediction_correlation=round(p_corr, 4),
        n_divergences=int(np.sum(diverge_mask)),
        a_correct_on_diverge=a_correct_count,
        b_correct_on_diverge=b_correct_count,
        a_total_return=round(pnl_a["total_return"], 6),
        b_total_return=round(pnl_b["total_return"], 6),
        a_sharpe=round(pnl_a["sharpe"], 2),
        b_sharpe=round(pnl_b["sharpe"], 2),
        a_max_drawdown=round(pnl_a["max_drawdown"], 6),
        b_max_drawdown=round(pnl_b["max_drawdown"], 6),
        a_n_trades=pnl_a["n_trades"],
        b_n_trades=pnl_b["n_trades"],
        a_signal_dist=_signal_distribution(sigs_a),
        b_signal_dist=_signal_distribution(sigs_b),
        divergences=divergences,
    )

    # Verdict
    if pnl_a["sharpe"] > pnl_b["sharpe"] + 0.5:
        report.verdict = f"Model {label_a} WINS"
        report.details = (
            f"{label_a} Sharpe {pnl_a['sharpe']:.2f} vs {label_b} {pnl_b['sharpe']:.2f} "
            f"(+{pnl_a['sharpe'] - pnl_b['sharpe']:.2f})"
        )
    elif pnl_b["sharpe"] > pnl_a["sharpe"] + 0.5:
        report.verdict = f"Model {label_b} WINS"
        report.details = (
            f"{label_b} Sharpe {pnl_b['sharpe']:.2f} vs {label_a} {pnl_a['sharpe']:.2f} "
            f"(+{pnl_b['sharpe'] - pnl_a['sharpe']:.2f})"
        )
    else:
        report.verdict = "NO SIGNIFICANT DIFFERENCE"
        report.details = (
            f"Sharpe within 0.5: {label_a}={pnl_a['sharpe']:.2f}, "
            f"{label_b}={pnl_b['sharpe']:.2f}"
        )

    return report


def _detect_interval(model_dir: Path) -> str:
    """Detect bar interval from model config or directory name.

    Returns canonical interval string: "60", "15", "240", "5", "1440".
    """
    raw = None
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            if "interval" in cfg:
                raw = str(cfg["interval"])
        except (OSError, json.JSONDecodeError, KeyError):
            raw = ""  # config read failure, use default interval

    # Heuristic from directory name if config didn't provide it
    if raw is None:
        name = model_dir.name.lower()
        if "15m" in name:
            raw = "15m"
        elif "4h" in name:
            raw = "4h"
        elif "5m" in name:
            raw = "5m"
        elif "1d" in name:
            raw = "1d"
        else:
            raw = "60"

    # Normalize human-readable intervals to minutes
    normalize = {"1h": "60", "4h": "240", "15m": "15", "5m": "5", "1d": "1440"}
    return normalize.get(raw, raw)


# ── Output formatting ─────────────────────────────────────────────────

def _report_to_dict(report: ABReport) -> dict:
    """Convert ABReport to JSON-serializable dict."""
    return {
        "symbol": report.symbol,
        "n_bars": report.n_bars,
        "n_eval_bars": report.n_eval_bars,
        "model_a": report.model_a,
        "model_b": report.model_b,
        "agreement": {
            "overall": report.signal_agreement_rate,
            "both_long": report.signal_agreement_long,
            "both_short": report.signal_agreement_short,
            "both_flat": report.signal_agreement_flat,
        },
        "correlation": {
            "z_score": report.z_score_correlation,
            "prediction": report.prediction_correlation,
        },
        "divergences": {
            "count": report.n_divergences,
            "a_correct": report.a_correct_on_diverge,
            "b_correct": report.b_correct_on_diverge,
            "samples": report.divergences[:20],
        },
        "pnl": {
            "a": {
                "total_return": report.a_total_return,
                "sharpe": report.a_sharpe,
                "max_drawdown": report.a_max_drawdown,
                "n_trades": report.a_n_trades,
            },
            "b": {
                "total_return": report.b_total_return,
                "sharpe": report.b_sharpe,
                "max_drawdown": report.b_max_drawdown,
                "n_trades": report.b_n_trades,
            },
        },
        "signal_distribution": {
            "a": report.a_signal_dist,
            "b": report.b_signal_dist,
        },
        "verdict": report.verdict,
        "details": report.details,
    }


def print_ab_report(report: ABReport, label_a: str = "A", label_b: str = "B") -> None:
    """Print human-readable A/B comparison report."""
    W = 72
    print()
    print("=" * W)
    print(f"  SHADOW A/B TEST: {report.symbol}")
    print("=" * W)
    print(f"  Model {label_a}: {report.model_a}")
    print(f"  Model {label_b}: {report.model_b}")
    print(f"  Bars: {report.n_bars} total, {report.n_eval_bars} evaluated")
    print()

    # Signal agreement
    print("  SIGNAL AGREEMENT:")
    pct = report.signal_agreement_rate * 100
    status = "HIGH" if pct >= 80 else ("MED" if pct >= 50 else "LOW")
    print(f"    Overall:     {pct:6.1f}%  [{status}]")
    print(f"    Both long:   {report.signal_agreement_long * 100:6.1f}%")
    print(f"    Both short:  {report.signal_agreement_short * 100:6.1f}%")
    print(f"    Both flat:   {report.signal_agreement_flat * 100:6.1f}%")
    print()

    # Correlation
    print("  CORRELATION:")
    print(f"    Z-score:     {report.z_score_correlation:+.4f}")
    print(f"    Prediction:  {report.prediction_correlation:+.4f}")
    print()

    # Divergence analysis
    print(f"  DIVERGENCES ({report.n_divergences} bars):")
    if report.n_divergences > 0:
        a_pct = report.a_correct_on_diverge / max(1, report.n_divergences) * 100
        b_pct = report.b_correct_on_diverge / max(1, report.n_divergences) * 100
        print(f"    {label_a} correct when diverge: {report.a_correct_on_diverge:>4} ({a_pct:.1f}%)")
        print(f"    {label_b} correct when diverge: {report.b_correct_on_diverge:>4} ({b_pct:.1f}%)")
    print()

    # Simulated PnL
    print("  SIMULATED PnL:")
    print(f"    {'Metric':<20} {label_a:>12} {label_b:>12} {'Delta':>12}")
    print(f"    {'-'*56}")
    metrics = [
        ("Total return", report.a_total_return, report.b_total_return, ".4f"),
        ("Sharpe", report.a_sharpe, report.b_sharpe, ".2f"),
        ("Max drawdown", report.a_max_drawdown, report.b_max_drawdown, ".4f"),
        ("Trades", report.a_n_trades, report.b_n_trades, "d"),
    ]
    for name, a_val, b_val, fmt in metrics:
        delta = b_val - a_val
        if fmt == "d":
            print(f"    {name:<20} {a_val:>12{fmt}} {b_val:>12{fmt}} {delta:>+12{fmt}}")
        else:
            print(f"    {name:<20} {a_val:>12{fmt}} {b_val:>12{fmt}} {delta:>+12{fmt}}")
    print()

    # Signal distribution
    print("  SIGNAL DISTRIBUTION:")
    ad, bd = report.a_signal_dist, report.b_signal_dist
    print(f"    {label_a}: long={ad.get('long', 0):>5}  flat={ad.get('flat', 0):>5}  short={ad.get('short', 0):>5}")
    print(f"    {label_b}: long={bd.get('long', 0):>5}  flat={bd.get('flat', 0):>5}  short={bd.get('short', 0):>5}")
    print()

    # Sample divergences
    if report.divergences:
        n_show = min(10, len(report.divergences))
        print(f"  DIVERGENCE SAMPLES ({n_show}/{report.n_divergences}):")
        for d in report.divergences[:n_show]:
            winner = label_a if d["a_correct"] and not d["b_correct"] else (
                label_b if d["b_correct"] and not d["a_correct"] else "="
            )
            print(
                f"    {d['timestamp']}: ${d['close']:>10.2f} "
                f"{label_a}_sig={d['a_signal']:+d} {label_b}_sig={d['b_signal']:+d} "
                f"fwd={d['fwd_return_bps']:+.1f}bps winner={winner}"
            )
        print()

    # Verdict
    print(f"  VERDICT: {report.verdict}")
    print(f"  {report.details}")
    print("=" * W)
    print()


# ── Legacy registry-based comparison (backward compat) ───────────────

def compare_models_legacy(
    symbol: str,
    registry_db: str = "model_registry.db",
    candidate_id: Optional[str] = None,
    auto_promote: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compare candidate vs production model via model registry.

    This is the original registry-based comparison, preserved for backward
    compatibility.  New code should use compare_ab() instead.
    """
    from research.model_registry.registry import ModelRegistry
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from scripts.train_unified import (
        validate_oos_extended,
        _load_and_compute_features,
        _add_regime_feature,
    )

    registry = ModelRegistry(registry_db)
    reg_name = f"alpha_unified_{symbol}"

    prod_mv = registry.get_production(reg_name)
    if prod_mv is None:
        logger.warning("No production model for %s", reg_name)
        return None

    if candidate_id:
        cand_mv = registry.get(candidate_id)
    else:
        versions = registry.list_versions(reg_name)
        candidates = [v for v in versions if not v.is_production]
        if not candidates:
            logger.info("No candidate models to compare for %s", reg_name)
            return None
        cand_mv = candidates[-1]

    if cand_mv is None:
        logger.warning("Candidate model not found")
        return None

    logger.info("Comparing %s v%d (prod) vs v%d (candidate)",
                reg_name, prod_mv.version, cand_mv.version)

    oos_path = Path(f"data_files/{symbol}_1h_oos.csv")
    if not oos_path.exists():
        logger.warning("No OOS file for %s", symbol)
        return None

    oos_df = pd.read_csv(oos_path)
    feat_df = _load_and_compute_features(symbol, oos_df)
    if feat_df is None or len(feat_df) < 100:
        return None
    feat_df = _add_regime_feature(feat_df)

    ts_col = "open_time" if "open_time" in oos_df.columns else "timestamp"
    timestamps = oos_df[ts_col].values.astype(np.int64)

    prod_features = list(prod_mv.features)
    prod_model = LGBMAlphaModel(name="prod", feature_names=tuple(prod_features))
    prod_model_path = Path(f"models_unified/{symbol}/lgbm_unified.pkl")
    if prod_model_path.exists():
        prod_model.load(prod_model_path)
    else:
        logger.warning("Production model file not found")
        return None

    horizon = 6
    if prod_mv.tags:
        horizon_tags = [t for t in prod_mv.tags if t.startswith("horizon_")]
        if horizon_tags:
            try:
                horizon = int(horizon_tags[0].replace("horizon_", ""))
            except ValueError as e:
                logger.warning("Failed to parse horizon tag '%s': %s", horizon_tags[0], e)
    mode_tag = [t for t in prod_mv.tags if t.startswith("mode_")]
    target_mode = mode_tag[0].replace("mode_", "") if mode_tag else "clipped"

    prod_oos = validate_oos_extended(
        prod_model, symbol, prod_features,
        target_horizon=horizon, target_mode=target_mode,
        feat_df=feat_df.copy(), timestamps=timestamps,
    )

    cand_features = list(cand_mv.features)
    cand_model = LGBMAlphaModel(name="cand", feature_names=tuple(cand_features))
    cand_model_path = Path(f"models_unified/{symbol}/lgbm_unified_v{cand_mv.version}.pkl")
    if not cand_model_path.exists():
        cand_model_path = Path(f"models_unified/{symbol}/lgbm_unified_candidate.pkl")
    if cand_model_path.exists():
        cand_model.load(cand_model_path)
    else:
        logger.warning("Candidate model file not found")
        return None

    cand_oos = validate_oos_extended(
        cand_model, symbol, cand_features,
        target_horizon=horizon, target_mode=target_mode,
        feat_df=feat_df.copy(), timestamps=timestamps,
    )

    if prod_oos is None or cand_oos is None:
        logger.warning("OOS validation failed for one or both models")
        return None

    comparison = {
        "symbol": symbol,
        "production": {
            "model_id": prod_mv.model_id,
            "version": prod_mv.version,
            "oos_ic": prod_oos["overall"]["ic"],
            "h2_ic": prod_oos["h2"]["ic"],
            "oos_sharpe": prod_oos["overall"]["sharpe"],
            "stability": prod_oos["stability_score"],
            "ic_positive_ratio": prod_oos.get("ic_positive_ratio", 0),
        },
        "candidate": {
            "model_id": cand_mv.model_id,
            "version": cand_mv.version,
            "oos_ic": cand_oos["overall"]["ic"],
            "h2_ic": cand_oos["h2"]["ic"],
            "oos_sharpe": cand_oos["overall"]["sharpe"],
            "stability": cand_oos["stability_score"],
            "ic_positive_ratio": cand_oos.get("ic_positive_ratio", 0),
        },
    }

    ic_delta = cand_oos["overall"]["ic"] - prod_oos["overall"]["ic"]
    sharpe_delta = cand_oos["overall"]["sharpe"] - prod_oos["overall"]["sharpe"]
    comparison["deltas"] = {
        "ic": ic_delta,
        "sharpe": sharpe_delta,
        "h2_ic": cand_oos["h2"]["ic"] - prod_oos["h2"]["ic"],
    }

    should_promote = (
        cand_oos["passed"]
        and cand_oos["overall"]["ic"] > prod_oos["overall"]["ic"]
        and cand_oos["h2"]["ic"] > 0
        and cand_oos["deflated_sharpe"] > 0
    )
    comparison["should_promote"] = should_promote
    comparison["reason"] = (
        "Candidate outperforms production" if should_promote
        else "Candidate does not meet promotion criteria"
    )

    print(f"\n{'='*60}")
    print(f"  Shadow Comparison: {symbol}")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<20} {'Production':>12} {'Candidate':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    for metric in ["oos_ic", "h2_ic", "oos_sharpe", "stability", "ic_positive_ratio"]:
        prod_val = comparison["production"][metric]
        cand_val = comparison["candidate"][metric]
        delta = cand_val - prod_val
        print(f"  {metric:<20} {prod_val:>12.4f} {cand_val:>12.4f} {delta:>+12.4f}")

    print(f"\n  Decision: {'PROMOTE' if should_promote else 'KEEP PRODUCTION'}")
    print(f"  Reason: {comparison['reason']}")

    if auto_promote and should_promote:
        registry.promote(cand_mv.model_id)
        comparison["promoted"] = True
        print(f"  AUTO-PROMOTED: {reg_name} v{cand_mv.version}")
    else:
        comparison["promoted"] = False

    out_path = Path(f"models_unified/{symbol}/shadow_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shadow A/B testing: compare two models side-by-side without trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Compare two model directories on 90 days of BTC data:
  python3 -m scripts.ops.shadow_compare \\
      --model-a models_v8/BTCUSDT_gate_v2 \\
      --model-b models_v8/BTCUSDT_4h \\
      --symbol BTCUSDT --days 90

  # JSON output for automation:
  python3 -m scripts.ops.shadow_compare \\
      --model-a models_v8/ETHUSDT_gate_v2 \\
      --model-b models_v8/ETHUSDT_4h \\
      --symbol ETHUSDT --days 60 --json

  # Legacy registry-based comparison:
  python3 -m scripts.ops.shadow_compare --symbol BTCUSDT --registry-mode
""",
    )
    parser.add_argument("--model-a", type=Path, help="Path to model A directory (e.g. current production)")
    parser.add_argument("--model-b", type=Path, help="Path to model B directory (e.g. candidate)")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g. BTCUSDT)")
    parser.add_argument("--days", type=int, default=90, help="Number of days to evaluate (default: 90)")
    parser.add_argument("--label-a", default="A", help="Label for model A (default: A)")
    parser.add_argument("--label-b", default="B", help="Label for model B (default: B)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", type=Path, help="Save JSON report to file")

    # Legacy registry-mode flags
    parser.add_argument("--registry-mode", action="store_true",
                        help="Use legacy registry-based comparison")
    parser.add_argument("--registry-db", default="model_registry.db",
                        help="Model registry database (legacy mode)")
    parser.add_argument("--candidate-id", help="Specific candidate model ID (legacy mode)")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Auto-promote if candidate wins (legacy mode)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Legacy registry mode
    if args.registry_mode:
        result = compare_models_legacy(
            args.symbol.upper(),
            registry_db=args.registry_db,
            candidate_id=args.candidate_id,
            auto_promote=args.auto_promote,
        )
        return 0 if result else 1

    # New A/B mode requires both model dirs
    if not args.model_a or not args.model_b:
        parser.error("--model-a and --model-b are required (or use --registry-mode)")

    if not args.model_a.exists():
        logger.error("Model A directory does not exist: %s", args.model_a)
        return 1
    if not args.model_b.exists():
        logger.error("Model B directory does not exist: %s", args.model_b)
        return 1

    report = compare_ab(
        model_a_dir=args.model_a,
        model_b_dir=args.model_b,
        symbol=args.symbol.upper(),
        days=args.days,
        label_a=args.label_a,
        label_b=args.label_b,
    )

    if report is None:
        logger.error("A/B comparison failed")
        return 1

    report_dict = _report_to_dict(report)
    report_dict["generated_at"] = datetime.now(timezone.utc).isoformat()

    if args.json:
        print(json.dumps(report_dict, indent=2, default=str))
    else:
        print_ab_report(report, label_a=args.label_a, label_b=args.label_b)

    # Save JSON report
    out_path = args.output or (OUTPUT_DIR / f"shadow_ab_{report.symbol}.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info("Report saved to %s", out_path)
    except Exception as e:
        logger.warning("Failed to save report: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
