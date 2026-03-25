#!/usr/bin/env python3
"""Gate Impact Backtest — measure P&L impact of new gates on existing alpha.

Runs the standard walk-forward signal pipeline, then applies gate scaling
to measure the incremental impact of each new gate on Sharpe/return.

Tests:
  A. Baseline: current production signal (no new gates)
  B. + MultiTF Confluence Gate
  C. + Liquidation Cascade Gate
  D. + Carry Cost Gate
  E. All new gates combined
  F. + Online Ridge (weight drift comparison)

Usage:
    python3 -m scripts.research.backtest_gate_impact --symbol BTCUSDT
    python3 -m scripts.research.backtest_gate_impact --symbol ETHUSDT
    python3 -m scripts.research.backtest_gate_impact --symbol BTCUSDT --realistic
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from alpha.training.train_v7_alpha import (
    _load_and_compute_features,
)

log = logging.getLogger("gate_impact")

# ── Constants ────────────────────────────────────────────────
COST_PER_TRADE = 0.0006  # 6 bps roundtrip
MIN_TRAIN_BARS = 8760    # 12 months
WARMUP = 65
HORIZON = 24


@dataclass
class GateBacktestResult:
    """Result of a single gate configuration backtest."""
    name: str
    sharpe: float
    total_return_pct: float
    n_trades: int
    win_rate: float
    max_dd_pct: float
    avg_scale: float  # average gate scale factor applied


def _load_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load OHLCV + compute features."""
    csv_path = f"data_files/{symbol}_1h.csv"
    if not Path(csv_path).exists():
        log.error("Missing: %s", csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path)
    log.info("Loaded %s: %d bars", symbol, len(df))

    feat_df = _load_and_compute_features(symbol, df)
    log.info("Features computed: %d columns", len(feat_df.columns))
    return df, feat_df


def _simple_backtest(
    closes: np.ndarray,
    signal: np.ndarray,
    cost_per_trade: float = COST_PER_TRADE,
) -> Dict[str, float]:
    """Simple vectorized backtest returning Sharpe, return, trades, win_rate, max_dd."""
    signal = np.asarray(signal, dtype=float)
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    ret_1bar = np.zeros(n)
    ret_1bar[1:] = closes[1:] / closes[:-1] - 1.0

    # Turnover
    turnover = np.zeros(n)
    turnover[1:] = np.abs(signal[1:] - signal[:-1])
    costs = turnover * cost_per_trade

    # PnL
    gross_pnl = signal * ret_1bar
    net_pnl = gross_pnl - costs

    # Sharpe
    active = signal != 0
    n_active = int(np.sum(active))
    if n_active < 50:
        return {"sharpe": 0.0, "total_return_pct": 0.0, "n_trades": 0,
                "win_rate": 0.0, "max_dd_pct": 0.0}

    mean_pnl = float(np.mean(net_pnl[active]))
    std_pnl = float(np.std(net_pnl[active]))
    sharpe = mean_pnl / max(std_pnl, 1e-8) * np.sqrt(8760)

    # Equity curve
    equity = np.cumprod(1.0 + net_pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = float(np.min(drawdown)) * 100

    # Trade count
    n_trades = int(np.sum(turnover > 0))

    # Win rate
    total_return = float(equity[-1] / equity[0] - 1.0)

    # Per-trade wins
    trade_pnls = []
    in_trade = False
    trade_start = 0.0
    cum = 0.0
    for i in range(n):
        cum += net_pnl[i]
        if signal[i] != 0 and not in_trade:
            in_trade = True
            trade_start = cum
        elif signal[i] == 0 and in_trade:
            in_trade = False
            trade_pnls.append(cum - trade_start)

    wins = sum(1 for p in trade_pnls if p > 0)
    wr = wins / max(len(trade_pnls), 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "max_dd_pct": round(max_dd, 2),
    }


def _generate_baseline_signal(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    deadzone: float = 0.5,
    min_hold: int = 18,
    zscore_window: int = 720,
    long_only: bool = True,
) -> np.ndarray:
    """Generate baseline signal using multi-feature momentum prediction proxy.

    Combines multiple features to approximate model output, since we
    don't have the actual trained model for all symbols.
    """
    n = len(feat_df)
    pred = np.zeros(n, dtype=float)

    # Simple ensemble of available features (mimics Ridge linear combination)
    feature_weights = {
        "ret_24": 0.3,
        "rsi_14": -0.2,       # mean reversion (high RSI → lower returns)
        "macd_hist": 0.15,
        "close_vs_ma20": 0.15,
        "funding_zscore_24": -0.1,  # contrarian funding
        "cvd_20": 0.1,
    }

    for feat, weight in feature_weights.items():
        if feat in feat_df.columns:
            vals = feat_df[feat].values.astype(float)
            vals = np.nan_to_num(vals, 0.0)
            # Normalize to z-score
            mean = np.nanmean(vals)
            std = np.nanstd(vals)
            if std > 1e-8:
                pred += weight * (vals - mean) / std

    pred = np.nan_to_num(pred, 0.0)

    # Signal pipeline
    from scripts.shared.signal_postprocess import pred_to_signal
    signal = pred_to_signal(
        pred, deadzone=deadzone, min_hold=min_hold,
        zscore_window=zscore_window, zscore_warmup=180, long_only=long_only,
    )
    return np.asarray(signal, dtype=float)


def _apply_multi_tf_gate(
    signal: np.ndarray,
    feat_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Apply MultiTF Confluence Gate scaling."""
    from runner.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
    gate = MultiTFConfluenceGate()
    scales = np.ones(len(signal))

    for i in range(len(signal)):
        if signal[i] == 0:
            continue
        ev = type("Ev", (), {"metadata": {"signal": int(signal[i])}})()
        ctx = {}
        for col in ["tf4h_close_vs_ma20", "tf4h_rsi_14", "tf4h_macd_hist"]:
            if col in feat_df.columns:
                val = feat_df[col].iloc[i]
                if pd.notna(val):
                    ctx[col] = float(val)
        r = gate.check(ev, ctx)
        scales[i] = r.scale

    scaled_signal = signal * scales
    # Re-discretize
    result = np.zeros(len(signal))
    result[scaled_signal > 0.5] = 1.0
    result[scaled_signal < -0.5] = -1.0
    # For fractional scales, use continuous sizing
    for i in range(len(signal)):
        if signal[i] != 0 and 0 < abs(scales[i]) < 1.0:
            result[i] = signal[i] * scales[i]

    avg_scale = float(np.mean(scales[signal != 0])) if np.any(signal != 0) else 1.0
    return result, avg_scale


def _apply_liquidation_cascade_gate(
    signal: np.ndarray,
    feat_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Apply Liquidation Cascade Gate scaling."""
    from runner.gates.liquidation_cascade_gate import LiquidationCascadeGate
    gate = LiquidationCascadeGate()
    scales = np.ones(len(signal))

    for i in range(len(signal)):
        if signal[i] == 0:
            continue
        ev = type("Ev", (), {"metadata": {"signal": int(signal[i])}})()
        ctx = {}
        for col in ["liquidation_volume_zscore_24", "oi_acceleration",
                     "liquidation_cascade_score", "liquidation_imbalance"]:
            if col in feat_df.columns:
                val = feat_df[col].iloc[i]
                if pd.notna(val):
                    ctx[col] = float(val)
        r = gate.check(ev, ctx)
        if not r.allowed:
            scales[i] = 0.0
        else:
            scales[i] = r.scale

    result = signal * scales
    avg_scale = float(np.mean(scales[signal != 0])) if np.any(signal != 0) else 1.0
    return result, avg_scale


def _apply_carry_cost_gate(
    signal: np.ndarray,
    feat_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Apply Carry Cost Gate scaling."""
    from runner.gates.carry_cost_gate import CarryCostGate
    gate = CarryCostGate()
    scales = np.ones(len(signal))

    for i in range(len(signal)):
        if signal[i] == 0:
            continue
        ev = type("Ev", (), {"metadata": {"signal": int(signal[i])}})()
        ctx = {"signal": int(signal[i])}
        for col in ["funding_rate", "basis"]:
            if col in feat_df.columns:
                val = feat_df[col].iloc[i]
                if pd.notna(val):
                    ctx[col] = float(val)
        r = gate.check(ev, ctx)
        scales[i] = r.scale

    result = signal * scales
    avg_scale = float(np.mean(scales[signal != 0])) if np.any(signal != 0) else 1.0
    return result, avg_scale


def _apply_all_gates(
    signal: np.ndarray,
    feat_df: pd.DataFrame,
) -> tuple[np.ndarray, float]:
    """Apply all new gates in chain order."""
    from runner.gates.liquidation_cascade_gate import LiquidationCascadeGate
    from runner.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
    from runner.gates.carry_cost_gate import CarryCostGate

    liq_gate = LiquidationCascadeGate()
    mtf_gate = MultiTFConfluenceGate()
    carry_gate = CarryCostGate()

    scales = np.ones(len(signal))

    for i in range(len(signal)):
        if signal[i] == 0:
            continue

        sig = int(signal[i])
        ev = type("Ev", (), {"metadata": {"signal": sig}})()

        # Build context from features
        ctx = {"signal": sig}
        feature_cols = [
            "liquidation_volume_zscore_24", "oi_acceleration",
            "liquidation_cascade_score", "liquidation_imbalance",
            "tf4h_close_vs_ma20", "tf4h_rsi_14", "tf4h_macd_hist",
            "funding_rate", "basis",
        ]
        for col in feature_cols:
            if col in feat_df.columns:
                val = feat_df[col].iloc[i]
                if pd.notna(val):
                    ctx[col] = float(val)

        # Chain: liquidation → multi_tf → carry
        cumulative_scale = 1.0

        r = liq_gate.check(ev, ctx)
        if not r.allowed:
            scales[i] = 0.0
            continue
        cumulative_scale *= r.scale

        r = mtf_gate.check(ev, ctx)
        cumulative_scale *= r.scale

        r = carry_gate.check(ev, ctx)
        cumulative_scale *= r.scale

        scales[i] = cumulative_scale

    result = signal * scales
    avg_scale = float(np.mean(scales[signal != 0])) if np.any(signal != 0) else 1.0
    return result, avg_scale


def run_gate_backtest(
    symbol: str,
    realistic: bool = False,
) -> List[GateBacktestResult]:
    """Run complete gate impact backtest for a symbol."""
    df, feat_df = _load_data(symbol)
    closes = df["close"].values.astype(float)

    # Use OOS portion only (skip first 12 months of training)
    oos_start = MIN_TRAIN_BARS + WARMUP
    if oos_start >= len(closes):
        log.error("Not enough data: %d bars, need %d", len(closes), oos_start)
        sys.exit(1)

    oos_closes = closes[oos_start:]
    oos_feat = feat_df.iloc[oos_start:].reset_index(drop=True)

    # Determine params from model config
    config_path = f"models_v8/{symbol}_gate_v2/config.json"
    deadzone = 0.5
    min_hold = 18
    long_only = True
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        deadzone = cfg.get("deadzone", 0.5)
        min_hold = cfg.get("min_hold", 18)
        long_only = cfg.get("long_only", True)
        log.info("Model config: dz=%.1f mh=%d long_only=%s", deadzone, min_hold, long_only)

    # Generate baseline signal
    log.info("Generating baseline signal (dz=%.1f, mh=%d, long_only=%s)...",
             deadzone, min_hold, long_only)
    baseline_signal = _generate_baseline_signal(
        oos_feat, oos_closes, deadzone=deadzone, min_hold=min_hold, long_only=long_only,
    )
    log.info("Baseline: %d active bars / %d total", int(np.sum(baseline_signal != 0)), len(baseline_signal))

    results: List[GateBacktestResult] = []

    # A. Baseline
    log.info("Running: A. Baseline...")
    metrics = _simple_backtest(oos_closes, baseline_signal)
    results.append(GateBacktestResult(
        name="A. Baseline (no gates)",
        avg_scale=1.0,
        **metrics,
    ))

    # B. + MultiTF Confluence
    log.info("Running: B. + MultiTF Confluence...")
    sig_b, avg_b = _apply_multi_tf_gate(baseline_signal.copy(), oos_feat)
    metrics = _simple_backtest(oos_closes, sig_b)
    results.append(GateBacktestResult(name="B. + MultiTF Confluence", avg_scale=avg_b, **metrics))

    # C. + Liquidation Cascade
    log.info("Running: C. + Liquidation Cascade...")
    sig_c, avg_c = _apply_liquidation_cascade_gate(baseline_signal.copy(), oos_feat)
    metrics = _simple_backtest(oos_closes, sig_c)
    results.append(GateBacktestResult(name="C. + Liquidation Cascade", avg_scale=avg_c, **metrics))

    # D. + Carry Cost
    log.info("Running: D. + Carry Cost...")
    sig_d, avg_d = _apply_carry_cost_gate(baseline_signal.copy(), oos_feat)
    metrics = _simple_backtest(oos_closes, sig_d)
    results.append(GateBacktestResult(name="D. + Carry Cost", avg_scale=avg_d, **metrics))

    # E. All gates combined
    log.info("Running: E. All gates combined...")
    sig_e, avg_e = _apply_all_gates(baseline_signal.copy(), oos_feat)
    metrics = _simple_backtest(oos_closes, sig_e)
    results.append(GateBacktestResult(name="E. All gates combined", avg_scale=avg_e, **metrics))

    return results


def print_results(symbol: str, results: List[GateBacktestResult]) -> None:
    """Print formatted results table."""
    print()
    print("=" * 100)
    print(f"  Gate Impact Backtest — {symbol}")
    print("=" * 100)
    print(f"  {'Configuration':<30} {'Sharpe':>8} {'Return%':>9} {'Trades':>7} "
          f"{'WR%':>6} {'MaxDD%':>8} {'AvgScale':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*7} {'-'*6} {'-'*8} {'-'*9}")

    baseline_sharpe = results[0].sharpe if results else 0
    for r in results:
        delta = ""
        if r.name != results[0].name and baseline_sharpe != 0:
            d = r.sharpe - baseline_sharpe
            delta = f" ({d:+.3f})"
        print(f"  {r.name:<30} {r.sharpe:>8.3f}{delta:>0} {r.total_return_pct:>9.2f} "
              f"{r.n_trades:>7} {r.win_rate:>6.1f} {r.max_dd_pct:>8.2f} {r.avg_scale:>9.3f}")

    print("=" * 100)

    # Verdict
    if results:
        best = max(results, key=lambda r: r.sharpe)
        baseline = results[0]
        improvement = best.sharpe - baseline.sharpe
        print(f"\n  Best: {best.name} (Sharpe {best.sharpe:.3f})")
        if improvement > 0:
            print(f"  Improvement over baseline: +{improvement:.3f} Sharpe")
        else:
            print(f"  Baseline is optimal (gates reduce performance by {abs(improvement):.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Gate Impact Backtest")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to test")
    parser.add_argument("--realistic", action="store_true", help="Use realistic engine")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Multiple symbols (overrides --symbol)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    symbols = args.symbols or [args.symbol]

    for symbol in symbols:
        t0 = time.time()
        log.info("Starting gate impact backtest for %s...", symbol)
        results = run_gate_backtest(symbol, realistic=args.realistic)
        print_results(symbol, results)
        log.info("%s completed in %.1fs", symbol, time.time() - t0)


if __name__ == "__main__":
    main()
