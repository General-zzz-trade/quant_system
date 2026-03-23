#!/usr/bin/env python3
"""Signal consistency validation: live trading signals vs backtest replay.

Parses live trading logs to extract bar-level signal events, then regenerates
signals from the same price data using the batch feature engine + model
prediction pipeline.  Compares signal direction, z-score, regime, and deadzone
to detect divergences between live (streaming) and backtest (batch) paths.

Key insight: live signals use checkpoint-restored RustFeatureEngine state
(incremental/streaming), while backtest uses batch feature computation from
CSV.  Small z-score differences are expected; signal direction mismatch is
the critical metric.

Usage:
    python3 -m scripts.ops.signal_reconcile --hours 24
    python3 -m scripts.ops.signal_reconcile --hours 48 --symbol ETHUSDT
    python3 -m scripts.ops.signal_reconcile --hours 24 --json
    python3 -m scripts.ops.signal_reconcile --hours 24 --alert
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from monitoring.daily_reconcile import (  # noqa: E402
    BarEntry,
    SymbolSession,
    parse_log,
)

log = logging.getLogger("signal_reconcile")

# ── Paths ─────────────────────────────────────────────────────────────

LOG_FILE = Path("logs/bybit_alpha.log")
RUNTIME_DIR = Path("data/runtime")
OUTPUT_FILE = RUNTIME_DIR / "signal_reconcile.json"
DATA_DIR = Path("data_files")
MODEL_BASE = Path("models_v8")

# ── Thresholds ────────────────────────────────────────────────────────

ZSCORE_TOLERANCE = 0.1       # z-score difference tolerance (streaming vs batch)
MISMATCH_WARN_RATE = 0.10    # 10% mismatch rate triggers WARNING
MISMATCH_CRIT_RATE = 0.20    # 20% mismatch rate triggers CRITICAL


# ── Backtest signal regeneration ──────────────────────────────────────

@dataclass
class BacktestBar:
    """A bar with regenerated backtest signal."""
    timestamp: datetime
    close: float
    pred: float
    z: float
    signal: int
    regime: str
    deadzone: float


def _resolve_symbol(runner_key: str) -> str:
    """Map runner key (e.g. ETHUSDT_15m) to underlying symbol."""
    # SYMBOL_CONFIG maps keys like ETHUSDT_15m -> {"symbol": "ETHUSDT"}
    from runner.strategy_config import SYMBOL_CONFIG
    cfg = SYMBOL_CONFIG.get(runner_key, {})
    return cfg.get("symbol", runner_key)


def _resolve_interval(runner_key: str) -> str:
    """Map runner key to kline interval string."""
    from runner.strategy_config import SYMBOL_CONFIG
    cfg = SYMBOL_CONFIG.get(runner_key, {})
    return cfg.get("interval", "60")


def _interval_suffix(interval: str) -> str:
    """Convert interval to CSV suffix: '60' -> '1h', '15' -> '15m', '240' -> '4h'."""
    mapping = {"60": "1h", "15": "15m", "240": "4h", "5": "5m"}
    return mapping.get(interval, "1h")


def _load_price_data(symbol: str, interval: str, since: datetime) -> Optional[object]:
    """Load price data from CSV for the given symbol and interval."""
    import pandas as pd

    suffix = _interval_suffix(interval)
    csv_path = DATA_DIR / f"{symbol}_{suffix}.csv"
    if not csv_path.exists():
        log.warning("Data file not found: %s", csv_path)
        return None

    df = pd.read_csv(csv_path)
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    # Convert to datetime for filtering
    since_ts = since.timestamp() * 1000  # ms
    # Keep some extra history for feature warmup (800 bars)
    interval_ms = int(interval) * 60 * 1000
    warmup_ms = 800 * interval_ms
    df = df[df[ts_col] >= (since_ts - warmup_ms)].copy()
    df = df.reset_index(drop=True)
    return df


def regenerate_backtest_signals(
    runner_key: str,
    bars: list[BarEntry],
    since: datetime,
) -> list[BacktestBar]:
    """Regenerate signals using batch feature engine + model prediction.

    This mimics the live pipeline: features -> predict -> z-score -> constraints.
    The key difference is batch (full-history) vs streaming (incremental) features.
    """
    symbol = _resolve_symbol(runner_key)
    interval = _resolve_interval(runner_key)

    # Load model
    from runner.strategy_config import SYMBOL_CONFIG
    model_dir_name = SYMBOL_CONFIG.get(runner_key, {}).get("model_dir", runner_key)
    model_dir = MODEL_BASE / model_dir_name

    if not model_dir.exists():
        log.warning("Model directory not found: %s", model_dir)
        return []

    try:
        from alpha.model_loader_prod import load_model
        model_info = load_model(model_dir)
    except Exception as e:
        log.warning("Failed to load model %s: %s", model_dir, e)
        return []

    features_list = model_info["features"]
    deadzone_base = model_info["deadzone"]
    min_hold = model_info["min_hold"]
    max_hold = model_info["max_hold"]
    zscore_window = model_info["zscore_window"]
    zscore_warmup = model_info["zscore_warmup"]
    long_only = model_info.get("long_only", False)

    # Load price data
    df = _load_price_data(symbol, interval, since)
    if df is None or len(df) < zscore_warmup + 100:
        log.warning("Insufficient data for %s (%s bars)", runner_key,
                     len(df) if df is not None else 0)
        return []

    # Compute batch features
    try:
        from features.batch_feature_engine import compute_features_batch
        feat_df = compute_features_batch(symbol, df)
    except Exception as e:
        log.warning("Batch feature computation failed for %s: %s", runner_key, e)
        return []

    # Ensure features are available
    available_features = [f for f in features_list if f in feat_df.columns]
    if len(available_features) < len(features_list) * 0.5:
        log.warning("%s: only %d/%d features available", runner_key,
                     len(available_features), len(features_list))
        return []

    # Generate predictions using primary model (Ridge+LGBM ensemble)
    X = feat_df[available_features].fillna(0.0).values

    # Ensemble predict
    horizon_models = model_info.get("horizon_models", [])
    if not horizon_models:
        log.warning("No horizon models for %s", runner_key)
        return []

    predictions = np.zeros(len(X))
    total_weight = 0.0

    for hm in horizon_models:
        ic_weight = hm.get("ic", 0.01)
        hm_features = hm.get("features", features_list)
        hm_available = [f for f in hm_features if f in feat_df.columns]

        if len(hm_available) < len(hm_features) * 0.5:
            continue

        X_hm = feat_df[hm_available].fillna(0.0).values

        # Ridge prediction (60% weight if available)
        ridge = hm.get("ridge")
        ridge_features = hm.get("ridge_features")
        lgbm = hm.get("lgbm")

        hm_pred = np.zeros(len(X_hm))

        if ridge is not None:
            r_feats = ridge_features or hm_features
            r_available = [f for f in r_feats if f in feat_df.columns]
            if len(r_available) == getattr(ridge, 'n_features_in_', len(r_available)):
                X_ridge = feat_df[r_available].fillna(0.0).values
                try:
                    ridge_pred = ridge.predict(X_ridge)
                    ridge_weight = model_info.get("config", {}).get("ridge_weight", 0.6)
                    hm_pred += ridge_weight * ridge_pred
                except (OSError, KeyError, ValueError):
                    ridge = None  # ridge model not available

        if lgbm is not None:
            try:
                lgbm_pred = lgbm.predict(X_hm)
                lgbm_weight = 1.0 - model_info.get("config", {}).get("ridge_weight", 0.6)
                if ridge is None:
                    lgbm_weight = 1.0
                hm_pred += lgbm_weight * lgbm_pred
            except Exception:
                continue  # model load failure, skip this horizon

        predictions += ic_weight * hm_pred
        total_weight += ic_weight

    if total_weight > 0:
        predictions /= total_weight

    # Rolling z-score
    from scripts.shared.signal_postprocess import rolling_zscore
    z_scores = rolling_zscore(predictions, window=zscore_window, warmup=zscore_warmup)

    # Regime filter: simplified vol/trend check from bar data
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    closes = df["close"].values.astype(np.float64)

    # Regime detection (simplified: vol filter)
    regime_labels = _compute_regime_labels(closes, interval)

    # Vol-adaptive deadzone
    deadzones = _compute_adaptive_deadzone(closes, deadzone_base, interval)

    # Apply constraints: discretize + min-hold + max-hold
    signals = _apply_signal_constraints(
        z_scores, deadzones, regime_labels, min_hold, max_hold, long_only
    )

    # Build result aligned to the analysis window
    since_ts = since.timestamp() * 1000
    timestamps_ms = df[ts_col].values.astype(np.float64)
    results = []

    for i in range(len(df)):
        if timestamps_ms[i] < since_ts:
            continue
        ts = datetime.fromtimestamp(timestamps_ms[i] / 1000)
        results.append(BacktestBar(
            timestamp=ts,
            close=float(closes[i]),
            pred=float(predictions[i]),
            z=float(z_scores[i]),
            signal=int(signals[i]),
            regime=regime_labels[i],
            deadzone=float(deadzones[i]),
        ))

    return results


def _compute_regime_labels(closes: np.ndarray, interval: str) -> list[str]:
    """Compute simplified regime labels (active/filtered) from close prices."""
    n = len(closes)
    labels = ["active"] * n

    # Compute returns
    rets = np.diff(closes) / closes[:-1]
    rets = np.insert(rets, 0, 0.0)

    # Vol filter: if realized vol is extremely low, regime is filtered
    # (matching _check_regime() in alpha_runner.py)
    window = 20
    for i in range(window, n):
        rv = np.std(rets[i - window:i])
        # Threshold depends on timeframe
        if interval == "15":
            threshold = 0.0005  # very low vol for 15m
        elif interval == "240":
            threshold = 0.003   # 4h bars
        else:
            threshold = 0.001   # 1h bars

        if rv < threshold:
            labels[i] = "filtered"

    return labels


def _compute_adaptive_deadzone(
    closes: np.ndarray, deadzone_base: float, interval: str
) -> np.ndarray:
    """Compute vol-adaptive deadzone matching AlphaRunner logic."""
    n = len(closes)
    deadzones = np.full(n, deadzone_base)

    rets = np.diff(closes) / closes[:-1]
    rets = np.insert(rets, 0, 0.0)

    # Median vol by timeframe
    if interval == "15":
        vol_median = 0.003
    elif interval == "240":
        vol_median = 0.013
    else:
        vol_median = 0.0063

    window = 20
    for i in range(window, n):
        rv_20 = np.std(rets[i - window:i])
        vol_ratio = rv_20 / vol_median if vol_median > 0 else 1.0
        vol_ratio = max(0.5, min(2.0, vol_ratio))
        deadzones[i] = deadzone_base * vol_ratio

    return deadzones


def _apply_signal_constraints(
    z_scores: np.ndarray,
    deadzones: np.ndarray,
    regime_labels: list[str],
    min_hold: int,
    max_hold: int,
    long_only: bool,
) -> np.ndarray:
    """Apply discretization + min-hold + max-hold matching constraint pipeline."""
    n = len(z_scores)
    signals = np.zeros(n)

    hold_count = 1

    for i in range(n):
        dz = deadzones[i]

        # Regime filter overrides everything
        if regime_labels[i] != "active":
            dz = 999.0

        # Discretize
        if z_scores[i] > dz:
            desired = 1.0
        elif z_scores[i] < -dz:
            desired = -1.0
        else:
            desired = 0.0

        # Long-only clip
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


# ── Comparison engine ─────────────────────────────────────────────────

@dataclass
class BarComparison:
    """Comparison result for one bar."""
    timestamp: str
    close: float
    # Live
    live_z: float
    live_signal: int
    live_regime: str
    live_dz: float
    # Backtest
    bt_z: float
    bt_signal: int
    bt_regime: str
    bt_dz: float
    # Deltas
    z_delta: float
    signal_match: bool
    regime_match: bool
    dz_match: bool


@dataclass
class SymbolReport:
    """Reconciliation report for one symbol."""
    symbol: str
    status: str = "ok"
    error: str = ""
    n_bars_live: int = 0
    n_bars_matched: int = 0
    # Metrics
    signal_match_rate: float = 0.0
    z_score_mean_delta: float = 0.0
    z_score_max_delta: float = 0.0
    regime_match_rate: float = 0.0
    dz_match_rate: float = 0.0
    # Distribution
    live_signal_dist: dict = field(default_factory=dict)
    bt_signal_dist: dict = field(default_factory=dict)
    # Mismatches
    mismatches: list[dict] = field(default_factory=list)
    # Root cause hints
    root_causes: dict = field(default_factory=dict)


def _match_bars_by_time(
    live_bars: list[BarEntry],
    bt_bars: list[BacktestBar],
    tolerance_s: int = 300,  # 5 min tolerance for time matching
) -> list[tuple[BarEntry, BacktestBar]]:
    """Match live bars to backtest bars by closest timestamp."""
    if not live_bars or not bt_bars:
        return []

    # Build timestamp index for bt bars
    bt_by_ts = {}
    for bt in bt_bars:
        key = bt.timestamp.replace(second=0, microsecond=0)
        bt_by_ts[key] = bt

    pairs = []
    for live in live_bars:
        live_key = live.timestamp.replace(second=0, microsecond=0)
        # Try exact match first
        if live_key in bt_by_ts:
            pairs.append((live, bt_by_ts[live_key]))
            continue
        # Try nearby bars
        for delta_m in [-1, 1, -2, 2]:
            nearby = live_key + timedelta(minutes=delta_m)
            if nearby in bt_by_ts:
                pairs.append((live, bt_by_ts[nearby]))
                break

    return pairs


def _diagnose_root_cause(comp: BarComparison) -> str:
    """Guess root cause for a signal mismatch."""
    if not comp.regime_match:
        return "regime_divergence"
    if abs(comp.z_delta) > 1.0:
        return "z_score_large_divergence"
    if abs(comp.z_delta) > ZSCORE_TOLERANCE:
        return "z_score_small_divergence"
    if not comp.dz_match:
        return "deadzone_mismatch"
    # Signal changed but z and regime match — likely hold-count timing
    return "hold_timing"


def reconcile_symbol_signals(
    runner_key: str,
    session: SymbolSession,
    since: datetime,
) -> SymbolReport:
    """Full signal consistency check for one symbol."""
    report = SymbolReport(symbol=runner_key)

    bars = session.bars
    if not bars:
        report.status = "no_data"
        report.error = "No bar entries found in log"
        return report

    report.n_bars_live = len(bars)

    # Regenerate backtest signals
    log.info("Regenerating backtest signals for %s (%d live bars)...",
             runner_key, len(bars))
    bt_bars = regenerate_backtest_signals(runner_key, bars, since)

    if not bt_bars:
        report.status = "backtest_failed"
        report.error = "Could not regenerate backtest signals (check model/data)"
        return report

    # Match by timestamp
    pairs = _match_bars_by_time(bars, bt_bars)
    report.n_bars_matched = len(pairs)

    if not pairs:
        report.status = "no_matches"
        report.error = "No bars matched by timestamp between live and backtest"
        return report

    # Compare each pair
    comparisons = []
    signal_matches = 0
    regime_matches = 0
    dz_matches = 0
    z_deltas = []
    live_dist = {-1: 0, 0: 0, 1: 0}
    bt_dist = {-1: 0, 0: 0, 1: 0}
    root_causes: dict[str, int] = {}

    for live, bt in pairs:
        z_delta = abs(live.z - bt.z)
        sig_match = live.signal == bt.signal
        reg_match = live.regime == bt.regime
        dz_match_ok = abs(live.dz - bt.deadzone) < 0.1 if live.dz > 0 else True

        comp = BarComparison(
            timestamp=str(live.timestamp),
            close=live.close,
            live_z=live.z,
            live_signal=live.signal,
            live_regime=live.regime,
            live_dz=live.dz,
            bt_z=bt.z,
            bt_signal=bt.signal,
            bt_regime=bt.regime,
            bt_dz=bt.deadzone,
            z_delta=z_delta,
            signal_match=sig_match,
            regime_match=reg_match,
            dz_match=dz_match_ok,
        )
        comparisons.append(comp)

        if sig_match:
            signal_matches += 1
        else:
            cause = _diagnose_root_cause(comp)
            root_causes[cause] = root_causes.get(cause, 0) + 1
            report.mismatches.append({
                "timestamp": comp.timestamp,
                "close": comp.close,
                "live_z": round(comp.live_z, 4),
                "live_signal": comp.live_signal,
                "live_regime": comp.live_regime,
                "live_dz": round(comp.live_dz, 4),
                "bt_z": round(comp.bt_z, 4),
                "bt_signal": comp.bt_signal,
                "bt_regime": comp.bt_regime,
                "bt_dz": round(comp.bt_dz, 4),
                "z_delta": round(comp.z_delta, 4),
                "root_cause": cause,
            })

        if reg_match:
            regime_matches += 1
        if dz_match_ok:
            dz_matches += 1
        z_deltas.append(z_delta)

        live_dist[live.signal] = live_dist.get(live.signal, 0) + 1
        bt_dist[bt.signal] = bt_dist.get(bt.signal, 0) + 1

    n = len(pairs)
    report.signal_match_rate = round(signal_matches / n, 4) if n > 0 else 0.0
    report.z_score_mean_delta = round(float(np.mean(z_deltas)), 4) if z_deltas else 0.0
    report.z_score_max_delta = round(float(np.max(z_deltas)), 4) if z_deltas else 0.0
    report.regime_match_rate = round(regime_matches / n, 4) if n > 0 else 0.0
    report.dz_match_rate = round(dz_matches / n, 4) if n > 0 else 0.0
    report.live_signal_dist = live_dist
    report.bt_signal_dist = bt_dist
    report.root_causes = root_causes

    # Trim mismatches to 30 for readability
    if len(report.mismatches) > 30:
        report.mismatches = report.mismatches[:30]

    return report


# ── Output formatting ─────────────────────────────────────────────────

def _report_to_dict(report: SymbolReport) -> dict:
    """Convert SymbolReport to JSON-serializable dict."""
    return {
        "symbol": report.symbol,
        "status": report.status,
        "error": report.error,
        "bars": {
            "live": report.n_bars_live,
            "matched": report.n_bars_matched,
        },
        "metrics": {
            "signal_match_rate": report.signal_match_rate,
            "z_score_mean_delta": report.z_score_mean_delta,
            "z_score_max_delta": report.z_score_max_delta,
            "regime_match_rate": report.regime_match_rate,
            "dz_match_rate": report.dz_match_rate,
        },
        "signal_distribution": {
            "live": report.live_signal_dist,
            "backtest": report.bt_signal_dist,
        },
        "root_causes": report.root_causes,
        "mismatches": report.mismatches,
    }


def print_report(report: SymbolReport) -> None:
    """Print human-readable signal reconciliation report."""
    print("=" * 72)
    print(f"  SIGNAL RECONCILIATION: {report.symbol}")
    print("=" * 72)

    if report.status != "ok":
        print(f"  Status: {report.status}")
        if report.error:
            print(f"  Error: {report.error}")
        print()
        return

    print(f"  Bars: {report.n_bars_live} live, {report.n_bars_matched} matched")
    print()

    # Key metrics
    print("  KEY METRICS:")
    match_pct = report.signal_match_rate * 100
    if match_pct >= 95:
        status = "PASS"
    elif match_pct >= 90:
        status = "WARN"
    else:
        status = "FAIL"
    print(f"    Signal match rate:     {match_pct:.1f}%  [{status}]")

    z_status = "PASS" if report.z_score_mean_delta < 0.5 else (
        "WARN" if report.z_score_mean_delta < 1.0 else "FAIL")
    print(f"    Z-score mean delta:    {report.z_score_mean_delta:.4f}  [{z_status}]")
    print(f"    Z-score max delta:     {report.z_score_max_delta:.4f}")

    reg_pct = report.regime_match_rate * 100
    reg_status = "PASS" if reg_pct >= 90 else ("WARN" if reg_pct >= 75 else "FAIL")
    print(f"    Regime match rate:     {reg_pct:.1f}%  [{reg_status}]")

    dz_pct = report.dz_match_rate * 100
    dz_status = "PASS" if dz_pct >= 85 else ("WARN" if dz_pct >= 70 else "FAIL")
    print(f"    Deadzone match rate:   {dz_pct:.1f}%  [{dz_status}]")
    print()

    # Signal distribution
    ld = report.live_signal_dist
    bd = report.bt_signal_dist
    print("  SIGNAL DISTRIBUTION:")
    print(f"    Live:     short={ld.get(-1, 0):>4}  flat={ld.get(0, 0):>4}  long={ld.get(1, 0):>4}")
    print(f"    Backtest: short={bd.get(-1, 0):>4}  flat={bd.get(0, 0):>4}  long={bd.get(1, 0):>4}")
    print()

    # Root causes
    if report.root_causes:
        total_mm = sum(report.root_causes.values())
        print(f"  ROOT CAUSES ({total_mm} mismatches):")
        for cause, count in sorted(report.root_causes.items(), key=lambda x: -x[1]):
            print(f"    {cause:30s}  {count:>4}  ({count / total_mm * 100:.0f}%)")
        print()

    # Sample mismatches
    if report.mismatches:
        n_show = min(10, len(report.mismatches))
        print(f"  MISMATCHES (showing {n_show}/{len(report.mismatches)}):")
        for m in report.mismatches[:n_show]:
            print(f"    {m['timestamp']}: close=${m['close']:.2f} "
                  f"live_z={m['live_z']:+.3f} bt_z={m['bt_z']:+.3f} "
                  f"live_sig={m['live_signal']:+d} bt_sig={m['bt_signal']:+d} "
                  f"cause={m['root_cause']}")
    else:
        print("  No signal mismatches found.")
    print()
    print("=" * 72)


def _send_alert_if_needed(reports: list[SymbolReport], send_alert: bool) -> None:
    """Send Telegram alert if mismatch rate exceeds threshold."""
    if not send_alert:
        return

    alert_symbols = []
    for r in reports:
        if r.status != "ok":
            continue
        mismatch_rate = 1.0 - r.signal_match_rate
        if mismatch_rate > MISMATCH_WARN_RATE:
            alert_symbols.append((r.symbol, mismatch_rate))

    if not alert_symbols:
        return

    try:
        from monitoring.notify import send_alert as _send, AlertLevel

        for sym, rate in alert_symbols:
            level = AlertLevel.CRITICAL if rate > MISMATCH_CRIT_RATE else AlertLevel.WARNING
            _send(
                level=level,
                title=f"Signal mismatch: {sym}",
                details={
                    "mismatch_rate": f"{rate * 100:.1f}%",
                    "threshold": f"{MISMATCH_WARN_RATE * 100:.0f}%",
                    "action": "Check streaming vs batch feature divergence",
                },
                source="signal_reconcile",
            )
            log.warning("Alert sent for %s: mismatch rate %.1f%%", sym, rate * 100)

    except Exception as e:
        log.warning("Failed to send alert: %s", e)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Signal consistency validation: live vs backtest signals",
    )
    parser.add_argument(
        "--hours", type=int, default=24,
        help="Number of hours to look back (default: 24)",
    )
    parser.add_argument(
        "--log-file", type=Path, default=LOG_FILE,
        help=f"Path to alpha log file (default: {LOG_FILE})",
    )
    parser.add_argument(
        "--symbol", default=None,
        help="Filter to specific symbol/runner key (e.g. ETHUSDT, ETHUSDT_15m)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of text",
    )
    parser.add_argument(
        "--alert", action="store_true",
        help="Send Telegram alert if mismatch rate > 10%%",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"JSON output path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Validate log file
    if not args.log_file.exists():
        log.error("Log file not found: %s", args.log_file)
        print(f"Error: log file not found: {args.log_file}", file=sys.stderr)
        return 1

    # Compute lookback window
    since = datetime.now() - timedelta(hours=args.hours)
    log.info("Analyzing %s (last %d hours, since %s)", args.log_file, args.hours, since)

    # Parse live signals from log
    sessions = parse_log(
        args.log_file,
        symbol_filter=args.symbol,
        since=since,
    )

    if not sessions:
        msg = f"No trading data found in log for the last {args.hours} hours."
        log.warning(msg)
        print(msg, file=sys.stderr)
        return 1

    # Run reconciliation per symbol
    reports: list[SymbolReport] = []
    for runner_key in sorted(sessions):
        session = sessions[runner_key]
        if not session.bars:
            log.info("Skipping %s: no bars", runner_key)
            continue
        log.info("Reconciling %s: %d live bars", runner_key, len(session.bars))
        report = reconcile_symbol_signals(runner_key, session, since)
        reports.append(report)

    if not reports:
        print("No symbols with bar data found.", file=sys.stderr)
        return 1

    # Build summary
    ok_reports = [r for r in reports if r.status == "ok"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_file": str(args.log_file),
        "lookback_hours": args.hours,
        "since": str(since),
        "symbols": [r.symbol for r in reports],
        "reports": [_report_to_dict(r) for r in reports],
    }

    if ok_reports:
        summary["aggregate"] = {
            "signal_match_rate": round(
                float(np.mean([r.signal_match_rate for r in ok_reports])), 4
            ),
            "z_score_mean_delta": round(
                float(np.mean([r.z_score_mean_delta for r in ok_reports])), 4
            ),
            "regime_match_rate": round(
                float(np.mean([r.regime_match_rate for r in ok_reports])), 4
            ),
            "total_bars_matched": sum(r.n_bars_matched for r in ok_reports),
            "total_mismatches": sum(len(r.mismatches) for r in ok_reports),
        }

    # Output
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        for report in reports:
            print_report(report)

        # Print aggregate summary
        if ok_reports and len(ok_reports) > 1:
            agg = summary["aggregate"]
            print()
            print("=" * 72)
            print("  AGGREGATE SUMMARY")
            print("=" * 72)
            print(f"  Symbols analyzed: {len(ok_reports)}")
            print(f"  Total bars matched: {agg['total_bars_matched']}")
            agg_match = agg['signal_match_rate'] * 100
            agg_status = "PASS" if agg_match >= 95 else ("WARN" if agg_match >= 90 else "FAIL")
            print(f"  Signal match rate: {agg_match:.1f}%  [{agg_status}]")
            print(f"  Z-score mean delta: {agg['z_score_mean_delta']:.4f}")
            print(f"  Total mismatches: {agg['total_mismatches']}")
            print("=" * 72)

    # Save JSON report
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Report saved to %s", args.output)
    except Exception as e:
        log.warning("Failed to save report: %s", e)

    # Health check alert
    _send_alert_if_needed(reports, args.alert)

    # Exit code: 0 if all ok, 1 if any critical mismatch
    for r in ok_reports:
        if (1.0 - r.signal_match_rate) > MISMATCH_CRIT_RATE:
            return 2  # critical mismatch
    return 0


if __name__ == "__main__":
    sys.exit(main())
