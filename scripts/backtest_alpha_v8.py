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

# ── Constants ────────────────────────────────────────────────────

FEE_BPS = 4e-4          # 4 bps per trade (maker/taker average)
SLIPPAGE_BPS = 2e-4     # 2 bps slippage
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS  # 6 bps total
INITIAL_CAPITAL = 10000.0


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
    if n < ma_window:
        return out
    cs = np.cumsum(closes)
    ma = np.empty(n)
    ma[:ma_window] = np.nan
    ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
    # Gate: zero signal where close <= MA or MA not yet available
    gate_off = np.isnan(ma) | (closes <= ma)
    out[gate_off] = 0.0
    return out


def _pred_to_signal(
    y_pred: np.ndarray,
    target_mode: str = "",
    deadzone: float = 0.5,
    min_hold: int = 24,
) -> np.ndarray:
    """Convert raw predictions to discrete positions {-1, 0, +1} with min hold.

    Args:
        y_pred: Raw model predictions.
        target_mode: "binary" or continuous.
        deadzone: z-score threshold to enter a position.
        min_hold: Minimum bars to hold before allowing signal change.
    """
    # Step 1: raw discrete signal from predictions
    if target_mode == "binary":
        centered = y_pred - 0.5
        raw = np.sign(centered)
        raw = np.where(np.abs(centered) < 0.02, 0.0, raw)
    else:
        mu = np.mean(y_pred)
        std = np.std(y_pred)
        if std < 1e-12:
            return np.zeros_like(y_pred)
        z = (y_pred - mu) / std
        raw = np.where(z > deadzone, 1.0, np.where(z < -deadzone, -1.0, 0.0))

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

    funding_times = sorted(funding.keys())
    oi_times = sorted(oi.keys())
    ls_times = sorted(ls.keys())
    spot_times = sorted(spot_closes.keys())
    fgi_times = sorted(fgi_schedule.keys())
    f_idx, oi_idx, ls_idx, spot_idx, fgi_idx = 0, 0, 0, 0, 0

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

        feats = comp.on_bar(
            symbol, close=close, volume=volume, high=high, low=low,
            open_=open_, hour=hour, dow=dow, funding_rate=funding_rate,
            trades=trades, taker_buy_volume=taker_buy_volume,
            quote_volume=quote_volume,
            taker_buy_quote_volume=taker_buy_quote_volume,
            open_interest=open_interest, ls_ratio=ls_ratio,
            spot_close=spot_close, fear_greed=fear_greed,
        )
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feat_df["close"] = df["close"].values
    return feat_df


def _load_models_from_dir(model_dir: Path) -> Tuple[list, list, dict]:
    """Load models from a V8 model directory with config.json.

    Returns (raw_models, weights, config_dict).
    """
    import pickle
    config_path = model_dir / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)

    raw_models = []
    weights = cfg.get("ensemble_weights", [])
    for fname in cfg.get("models", []):
        pkl_path = model_dir / fname
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
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
) -> Dict[str, Any]:
    """Run realistic backtest on historical data.

    Args:
        full: Use all available data instead of last oos_bars.
        oos_bars: Number of bars for OOS window (default 13140 = ~18 months).
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
    import pickle
    ensemble_mode = False
    if model_path.is_dir():
        raw_models, weights, dir_cfg = _load_models_from_dir(model_path)
        ensemble_mode = dir_cfg.get("ensemble", False) and len(raw_models) > 1
        model_label = f"{model_path} (ensemble {len(raw_models)} models)"
    else:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)
        raw_models = [model_dict["model"]]
        weights = [1.0]
        model_label = str(model_path)

    print(f"\n{'='*70}")
    print(f"  Alpha V8 Backtest: {symbol}")
    print(f"  Model: {model_label}")
    print(f"  Horizon: {horizon}, Mode: {target_mode}")
    print(f"  Features: {len(feature_names)}")
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
    feat_df = compute_oos_features(symbol, oos_df)

    # Prepare X matrix
    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    X = feat_df[feature_names].values.astype(np.float64)

    # Warmup: skip first 65 bars where features are not fully computed
    warmup = 65
    X = X[warmup:]
    closes = closes[warmup:]
    timestamps = timestamps[warmup:]
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
    if monthly_gate:
        pre_active = np.mean(signal != 0) * 100
        signal = _apply_monthly_gate(signal, closes, monthly_gate_window)
        post_active = np.mean(signal != 0) * 100
        print(f"  Monthly gate (MA{monthly_gate_window}): active {pre_active:.1f}% → {post_active:.1f}% "
              f"(gated {pre_active - post_active:.1f}%)")
    print(f"  Signal stats: active={np.mean(signal != 0)*100:.1f}%, "
          f"long={np.mean(signal > 0)*100:.1f}%, short={np.mean(signal < 0)*100:.1f}%")

    # ── Simulate trading ──
    print("  Simulating trades...")
    ret_1bar = np.diff(closes) / closes[:-1]
    signal_for_trade = signal[:len(ret_1bar)]

    # Turnover cost
    turnover = np.abs(np.diff(signal_for_trade, prepend=0))
    gross_pnl = signal_for_trade * ret_1bar
    cost = turnover * COST_PER_TRADE
    net_pnl = gross_pnl - cost

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

    run_backtest(symbol, model_path, config_path, out_dir,
                 long_only=args.long_only,
                 monthly_gate=args.monthly_gate,
                 monthly_gate_window=args.monthly_gate_window,
                 full=args.full,
                 oos_bars=args.oos_bars)


if __name__ == "__main__":
    main()
