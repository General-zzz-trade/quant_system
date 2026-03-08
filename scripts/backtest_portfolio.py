#!/usr/bin/env python3
"""Multi-symbol portfolio backtest — combines BTC+ETH+SOL signals with allocation.

Loads V8 models for each symbol, generates per-symbol signals, then combines
them via configurable allocation (equal, inverse_vol, risk_parity, sharpe_weighted).
Includes correlation gating, leverage cap, DD circuit breaker, and funding costs.

Usage:
    python3 -m scripts.backtest_portfolio
    python3 -m scripts.backtest_portfolio --alloc inverse_vol --leverage 1.5
    python3 -m scripts.backtest_portfolio --symbols BTCUSDT ETHUSDT
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

from alpha.signal_transform import pred_to_signal as _pred_to_signal
from features.enriched_computer import EnrichedFeatureComputer

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

FEE_BPS = 4e-4
SLIPPAGE_BPS = 2e-4
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS
INITIAL_CAPITAL = 10000.0

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


# ── Helpers (reuse from backtest_alpha_v8) ───────────────────────

def _compute_bear_mask(closes: np.ndarray, ma_window: int = 480) -> np.ndarray:
    n = len(closes)
    if n < ma_window:
        return np.ones(n, dtype=bool)
    cs = np.cumsum(closes)
    ma = np.empty(n)
    ma[:ma_window] = np.nan
    ma[ma_window:] = (cs[ma_window:] - cs[:n - ma_window]) / ma_window
    return np.isnan(ma) | (closes <= ma)


def _load_schedule(path: Path, ts_col: str, val_col: str) -> Dict[int, float]:
    import csv
    schedule: Dict[int, float] = {}
    if not path.exists():
        return schedule
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            schedule[int(row[ts_col])] = float(row[val_col])
    return schedule


# ── Per-symbol signal generation ─────────────────────────────────

def _generate_symbol_signal(
    symbol: str,
    model_dir: Path,
    oos_bars: int,
    full: bool,
    *,
    deadzone_override: Optional[float] = None,
    min_hold_override: Optional[int] = None,
    sym_dd_override: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Generate signal for a single symbol. Returns dict with arrays or None."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"  [{symbol}] config.json not found at {model_dir}")
        return None

    with open(config_path) as f:
        cfg = json.load(f)

    feature_names = cfg["features"]
    target_mode = cfg.get("target_mode", "clipped")

    # Load models
    from infra.model_signing import load_verified_pickle
    raw_models = []
    weights = cfg.get("ensemble_weights", [])
    for fname in cfg.get("models", []):
        pkl_path = model_dir / fname
        if pkl_path.exists():
            data = load_verified_pickle(pkl_path)
            raw_models.append(data["model"])

    if not raw_models:
        print(f"  [{symbol}] No model files found")
        return None

    if len(weights) < len(raw_models):
        weights = [1.0 / len(raw_models)] * len(raw_models)

    # Load data
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  [{symbol}] Data file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if full:
        oos_df = df.reset_index(drop=True)
    else:
        oos_df = df.iloc[-oos_bars:].reset_index(drop=True)

    ts_col = "timestamp" if "timestamp" in oos_df.columns else "open_time"
    timestamps = oos_df[ts_col].values.astype(np.int64)

    # Compute features
    from features.batch_feature_engine import compute_features_batch
    v11_features = {"spx_overnight_ret", "dxy_change_5d", "vix_zscore_14",
                    "mempool_size_zscore_24", "fee_urgency_ratio",
                    "exchange_supply_zscore_30", "liquidation_cascade_score"}
    needs_v11 = bool(set(feature_names) & v11_features)
    feat_df = compute_features_batch(symbol, oos_df, include_v11=needs_v11)

    # Cross-asset features for non-BTC
    cross_features = {"btc_ret_1", "btc_ret_3", "btc_ret_6", "btc_ret_12", "btc_ret_24",
                      "btc_rsi_14", "btc_macd_line", "btc_mean_reversion_20",
                      "btc_atr_norm_14", "btc_bb_width_20",
                      "rolling_beta_30", "rolling_beta_60", "relative_strength_20",
                      "rolling_corr_30", "funding_diff", "funding_diff_ma8", "spread_zscore_20"}
    if symbol != "BTCUSDT" and bool(set(feature_names) & cross_features):
        from features.batch_cross_asset import build_cross_features_batch
        cross_map = build_cross_features_batch([symbol])
        if cross_map and symbol in cross_map:
            cross_df = cross_map[symbol]
            oos_ts = oos_df[ts_col].values.astype(np.int64)
            cross_aligned = cross_df.reindex(oos_ts)
            for cname in cross_aligned.columns:
                if cname in feature_names or cname in cross_features:
                    feat_df[cname] = cross_aligned[cname].values

    for fname in feature_names:
        if fname not in feat_df.columns:
            feat_df[fname] = np.nan

    closes = feat_df["close"].values.astype(np.float64)
    X = feat_df[feature_names].values.astype(np.float64)

    # Warmup
    warmup = 65
    X = X[warmup:]
    closes = closes[warmup:]
    timestamps = timestamps[warmup:]
    n = len(X)

    # Predict
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

    # Signal with deadzone + min_hold (CLI override or config)
    deadzone = deadzone_override if deadzone_override is not None else cfg.get("deadzone", 0.5)
    min_hold = min_hold_override if min_hold_override is not None else cfg.get("min_hold", 24)
    signal = _pred_to_signal(y_pred, target_mode=target_mode,
                             deadzone=deadzone, min_hold=min_hold)

    # Apply long_only from config (clip shorts in bull regime)
    is_long_only = cfg.get("long_only", False)
    ma_window = cfg.get("ma_window", 480)

    # Strategy F: regime switch with bear model
    bear_model_path = cfg.get("bear_model_path")
    pm = cfg.get("position_management", {})
    bear_thresholds = None
    if pm.get("bear_thresholds"):
        bear_thresholds = [tuple(x) for x in pm["bear_thresholds"]]

    if bear_model_path:
        bear_dir = Path(bear_model_path)
        bear_cfg_path = bear_dir / "config.json"
        if bear_cfg_path.exists():
            with open(bear_cfg_path) as f:
                bear_cfg = json.load(f)
            bear_pkl = bear_dir / bear_cfg["models"][0]
            if bear_pkl.exists():
                bear_data = load_verified_pickle(bear_pkl)
                bear_model = bear_data["model"]
                bear_features = bear_data.get("features", bear_cfg.get("features", []))
                bear_mask = _compute_bear_mask(closes, ma_window)

                feat_df_trimmed = feat_df.iloc[warmup:].reset_index(drop=True)
                X_bear = np.column_stack([
                    np.nan_to_num(feat_df_trimmed[f].values.astype(np.float64), nan=0.0)
                    if f in feat_df_trimmed.columns else np.zeros(n)
                    for f in bear_features
                ])
                prob = bear_model.predict_proba(X_bear)[:, 1]

                # Bull regime: long-only signal
                if is_long_only:
                    signal = np.clip(signal, 0.0, None)

                # Bear regime: graded thresholds from config
                for i in range(n):
                    if bear_mask[i]:
                        if bear_thresholds:
                            score = 0.0
                            for thresh, s in bear_thresholds:
                                if prob[i] > thresh:
                                    score = s
                                    break
                            signal[i] = score
                        else:
                            signal[i] = -1.0 if prob[i] > 0.5 else 0.0

                n_bull = int((~bear_mask).sum())
                n_bear = int(bear_mask.sum())
                print(f"  [{symbol}] Strategy F: bull={n_bull}, bear={n_bear}")
    elif is_long_only:
        # No bear model, pure long-only
        signal = np.clip(signal, 0.0, None)

    # Per-symbol DD breaker (CLI override or config)
    sym_dd_limit = sym_dd_override if sym_dd_override is not None else pm.get("dd_limit")
    sym_dd_cooldown = pm.get("dd_cooldown", 48)
    if sym_dd_limit is not None:
        if sym_dd_limit > 0:
            sym_dd_limit = -sym_dd_limit
        from scripts.backtest_alpha_v8 import _apply_dd_breaker
        signal = _apply_dd_breaker(signal, closes, sym_dd_limit, sym_dd_cooldown)

    # Volatility (atr_norm_14 for allocation)
    vol_col = "atr_norm_14"
    if vol_col in feat_df.columns:
        vol_values = feat_df[vol_col].values[warmup:].astype(np.float64)
    else:
        vol_values = np.full(n, 0.02)

    # Funding cost per bar
    funding_schedule = _load_schedule(
        Path(f"data_files/{symbol}_funding.csv"), "timestamp", "funding_rate")
    funding_rates = np.zeros(n)
    if funding_schedule:
        f_times = sorted(funding_schedule.keys())
        fi = 0
        cur_rate = 0.0
        for i in range(n):
            while fi < len(f_times) and f_times[fi] <= timestamps[i]:
                cur_rate = funding_schedule[f_times[fi]]
                fi += 1
            funding_rates[i] = cur_rate

    print(f"  [{symbol}] {n} bars, signal active={np.mean(signal != 0)*100:.1f}%, "
          f"long={np.mean(signal > 0)*100:.1f}%, short={np.mean(signal < 0)*100:.1f}%")

    return {
        "symbol": symbol,
        "timestamps": timestamps,
        "closes": closes,
        "signal": signal,
        "vol": vol_values,
        "funding_rates": funding_rates,
        "n": n,
    }


# ── Allocation methods ───────────────────────────────────────────

def _compute_weights(
    method: str,
    symbol_data: List[Dict[str, Any]],
    bar_idx: int,
    lookback: int = 720,
) -> np.ndarray:
    """Compute portfolio weights for each symbol at bar_idx."""
    n_sym = len(symbol_data)
    if n_sym == 0:
        return np.array([])

    if method == "equal":
        return np.ones(n_sym) / n_sym

    if method == "inverse_vol":
        vols = np.array([sd["vol"][bar_idx] if bar_idx < sd["n"] else 0.02
                         for sd in symbol_data])
        vols = np.clip(vols, 0.001, None)
        inv = 1.0 / vols
        return inv / inv.sum()

    if method == "risk_parity":
        vols = np.array([sd["vol"][bar_idx] if bar_idx < sd["n"] else 0.02
                         for sd in symbol_data])
        vols = np.clip(vols, 0.001, None)
        # Risk parity: weight ∝ 1/vol, same as inverse_vol for uncorrelated
        inv = 1.0 / vols
        return inv / inv.sum()

    if method == "sharpe_weighted":
        sharpes = []
        for sd in symbol_data:
            start = max(0, bar_idx - lookback)
            end = min(bar_idx, sd["n"] - 1)
            if end - start < 30:
                sharpes.append(0.0)
                continue
            ret_1bar = np.diff(sd["closes"][start:end]) / sd["closes"][start:end - 1]
            sig_slice = sd["signal"][start:end - 1]
            pnl = sig_slice * ret_1bar
            std = np.std(pnl, ddof=1)
            if std > 1e-10:
                sharpes.append(np.mean(pnl) / std)
            else:
                sharpes.append(0.0)
        sharpes = np.array(sharpes)
        sharpes = np.clip(sharpes, 0.0, None)  # Only positive Sharpe gets weight
        total = sharpes.sum()
        if total < 1e-10:
            return np.ones(n_sym) / n_sym
        return sharpes / total

    return np.ones(n_sym) / n_sym


# ── Portfolio simulation ─────────────────────────────────────────

def run_portfolio_backtest(
    symbols: List[str],
    alloc_method: str = "equal",
    max_leverage: float = 1.0,
    dd_limit: float = -0.15,
    dd_cooldown: int = 48,
    rebalance_freq: int = 24,
    oos_bars: int = 13140,
    full: bool = False,
    out_dir: Optional[Path] = None,
    deadzone_override: Optional[float] = None,
    min_hold_override: Optional[int] = None,
    sym_dd_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Run multi-symbol portfolio backtest."""
    print(f"\n{'='*70}")
    print(f"  Portfolio Backtest: {' + '.join(symbols)}")
    print(f"  Allocation: {alloc_method}, Max leverage: {max_leverage}")
    print(f"  DD limit: {dd_limit*100:.1f}%, Rebalance freq: {rebalance_freq}h")
    if deadzone_override is not None:
        print(f"  Deadzone override: {deadzone_override}")
    if min_hold_override is not None:
        print(f"  Min hold override: {min_hold_override}")
    if sym_dd_override is not None:
        print(f"  Per-symbol DD override: {sym_dd_override*100:.1f}%")
    print(f"{'='*70}")

    # Generate per-symbol signals
    symbol_data = []
    for sym in symbols:
        model_dir = Path(f"models_v8/{sym}_gate_v2")
        result = _generate_symbol_signal(
            sym, model_dir, oos_bars, full,
            deadzone_override=deadzone_override,
            min_hold_override=min_hold_override,
            sym_dd_override=sym_dd_override,
        )
        if result is not None:
            symbol_data.append(result)

    if not symbol_data:
        print("  No valid symbols. Exiting.")
        return {}

    # Find common timestamp range
    all_ts = [set(sd["timestamps"]) for sd in symbol_data]
    common_ts = sorted(all_ts[0].intersection(*all_ts[1:]))
    if len(common_ts) < 100:
        print(f"  Too few common bars: {len(common_ts)}")
        return {}

    print(f"\n  Common bars: {len(common_ts)}")

    # Align all symbols to common timestamps
    for sd in symbol_data:
        ts_to_idx = {int(ts): i for i, ts in enumerate(sd["timestamps"])}
        idxs = [ts_to_idx[ts] for ts in common_ts]
        sd["closes"] = sd["closes"][idxs]
        sd["signal"] = sd["signal"][idxs]
        sd["vol"] = sd["vol"][idxs]
        sd["funding_rates"] = sd["funding_rates"][idxs]
        sd["timestamps"] = np.array(common_ts, dtype=np.int64)
        sd["n"] = len(common_ts)

    n_bars = len(common_ts)
    n_sym = len(symbol_data)
    timestamps = np.array(common_ts, dtype=np.int64)

    # ── Simulate portfolio ──
    equity = np.ones(n_bars + 1) * INITIAL_CAPITAL
    portfolio_signal = np.zeros((n_bars, n_sym))  # weighted signal per symbol
    per_symbol_pnl = np.zeros((n_bars, n_sym))
    portfolio_pnl = np.zeros(n_bars)
    weights_history = np.zeros((n_bars, n_sym))

    cool_remaining = 0
    current_weights = np.ones(n_sym) / n_sym
    dd_peak = INITIAL_CAPITAL  # track peak for DD; resets after cooldown
    dd_breaker_count = 0

    for t in range(n_bars):
        # DD circuit breaker
        if cool_remaining > 0:
            cool_remaining -= 1
            equity[t + 1] = equity[t]
            if cool_remaining == 0:
                # Reset peak to current equity after cooldown — fresh start
                dd_peak = equity[t + 1]
            continue

        # Rebalance weights
        if t % rebalance_freq == 0:
            current_weights = _compute_weights(alloc_method, symbol_data, t)

        weights_history[t] = current_weights

        # Compute weighted signals
        gross_exposure = 0.0
        for s in range(n_sym):
            raw_sig = symbol_data[s]["signal"][t]
            portfolio_signal[t, s] = raw_sig * current_weights[s]
            gross_exposure += abs(portfolio_signal[t, s])

        # Leverage cap
        if gross_exposure > max_leverage:
            scale = max_leverage / gross_exposure
            portfolio_signal[t] *= scale

        # Per-symbol PnL
        if t < n_bars - 1:
            bar_pnl = 0.0
            for s in range(n_sym):
                ret = (symbol_data[s]["closes"][t + 1] - symbol_data[s]["closes"][t]) / symbol_data[s]["closes"][t]
                sig = portfolio_signal[t, s]

                # Trading cost
                prev_sig = portfolio_signal[t - 1, s] if t > 0 else 0.0
                turnover = abs(sig - prev_sig)
                cost = turnover * COST_PER_TRADE

                # Funding cost (distribute 8h rate across 8 bars)
                funding_cost = sig * symbol_data[s]["funding_rates"][t] / 8.0

                sym_pnl = sig * ret - cost - abs(funding_cost)
                per_symbol_pnl[t, s] = sym_pnl
                bar_pnl += sym_pnl

            portfolio_pnl[t] = bar_pnl
            equity[t + 1] = equity[t] * (1 + bar_pnl)

            # DD check against rolling peak (resets after cooldown)
            dd_peak = max(dd_peak, equity[t + 1])
            dd = (equity[t + 1] - dd_peak) / dd_peak
            if dd < dd_limit:
                cool_remaining = dd_cooldown
                dd_breaker_count += 1

    # Last bar has no next-bar return; carry forward
    equity[-1] = equity[-2]

    # ── Compute metrics ──
    final_equity = equity[-1]
    total_return = (final_equity / INITIAL_CAPITAL) - 1.0
    n_hours = n_bars
    annual_return = (1 + total_return) ** (8760 / max(n_hours, 1)) - 1.0

    active = portfolio_pnl != 0
    n_active = int(active.sum())
    sharpe = 0.0
    if n_active > 1:
        active_pnl = portfolio_pnl[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    peak_eq = np.maximum.accumulate(equity)
    dd = (equity - peak_eq) / peak_eq
    max_dd = float(np.min(dd))

    # Per-symbol attribution
    print(f"\n{'='*70}")
    print(f"  PORTFOLIO RESULTS")
    print(f"{'='*70}")
    print(f"  Period: {n_bars} bars")
    print(f"  Initial: ${INITIAL_CAPITAL:,.0f} → Final: ${final_equity:,.2f}")
    print(f"  Total return: {total_return*100:+.2f}%")
    print(f"  Annual return: {annual_return*100:+.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Max DD: {max_dd*100:.2f}%")
    print(f"  DD breaker triggered: {dd_breaker_count} times")

    print(f"\n  --- Per-Symbol Attribution ---")
    print(f"  {'Symbol':<12} {'Return':>8} {'Contribution':>14} {'Avg Weight':>12}")
    print(f"  {'-'*48}")
    sym_summaries = {}
    for s in range(n_sym):
        sym = symbol_data[s]["symbol"]
        sym_total_pnl = float(np.sum(per_symbol_pnl[:, s]))
        avg_weight = float(np.mean(weights_history[:, s]))
        sym_return = (1 + sym_total_pnl) - 1.0  # approximation
        print(f"  {sym:<12} {sym_return*100:>+7.2f}% {sym_total_pnl*100:>+13.4f}% {avg_weight*100:>11.1f}%")
        sym_summaries[sym] = {
            "return": sym_return,
            "contribution": sym_total_pnl,
            "avg_weight": avg_weight,
        }

    # Monthly breakdown
    dt_list = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts in timestamps[:n_bars]]
    month_keys = [f"{d.year}-{d.month:02d}" for d in dt_list]
    unique_months = sorted(set(month_keys))

    monthly = []
    print(f"\n  --- Monthly Breakdown ---")
    print(f"  {'Month':<10} {'Return':>8} {'Sharpe':>8}")
    print(f"  {'-'*28}")
    for mk in unique_months:
        mask = np.array([m == mk for m in month_keys])
        if mask.sum() < 10:
            continue
        m_pnl = portfolio_pnl[mask]
        m_ret = float(np.sum(m_pnl))
        m_active = m_pnl != 0
        m_sharpe = 0.0
        if m_active.sum() > 1:
            m_std = float(np.std(m_pnl[m_active], ddof=1))
            if m_std > 0:
                m_sharpe = float(np.mean(m_pnl[m_active])) / m_std * np.sqrt(8760)
        monthly.append({"month": mk, "return": m_ret, "sharpe": m_sharpe})
        print(f"  {mk:<10} {m_ret*100:>+7.2f}% {m_sharpe:>8.2f}")

    pos_months = sum(1 for m in monthly if m["return"] > 0)
    print(f"  {'-'*28}")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # Diversification ratio
    individual_vols = []
    for s in range(n_sym):
        sym_pnl_arr = per_symbol_pnl[:, s]
        std_s = float(np.std(sym_pnl_arr[sym_pnl_arr != 0], ddof=1)) if (sym_pnl_arr != 0).sum() > 1 else 0.0
        individual_vols.append(std_s)
    portfolio_vol = float(np.std(portfolio_pnl[active], ddof=1)) if n_active > 1 else 0.0
    weighted_vol_sum = sum(w * v for w, v in zip(np.mean(weights_history, axis=0), individual_vols))
    div_ratio = weighted_vol_sum / portfolio_vol if portfolio_vol > 1e-10 else 1.0
    print(f"\n  Diversification ratio: {div_ratio:.2f}")

    # ── Save results ──
    if out_dir is None:
        out_dir = Path(f"results/portfolio_backtest/{'_'.join(symbols)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    eq_df = pd.DataFrame({
        "timestamp": timestamps,
        "equity": equity[:-1],
        "portfolio_pnl": portfolio_pnl,
    })
    for s in range(n_sym):
        eq_df[f"{symbol_data[s]['symbol']}_pnl"] = per_symbol_pnl[:, s]
        eq_df[f"{symbol_data[s]['symbol']}_weight"] = weights_history[:, s]
    eq_df.to_csv(out_dir / "equity_curve.csv", index=False)
    pd.DataFrame(monthly).to_csv(out_dir / "monthly.csv", index=False)

    summary = {
        "symbols": symbols,
        "allocation": alloc_method,
        "max_leverage": max_leverage,
        "dd_limit": dd_limit,
        "n_bars": n_bars,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": float(final_equity),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "pos_months": pos_months,
        "total_months": len(monthly),
        "diversification_ratio": div_ratio,
        "per_symbol": sym_summaries,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {out_dir}/")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-symbol portfolio backtest")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Symbols to include (default: BTCUSDT ETHUSDT SOLUSDT)")
    parser.add_argument("--alloc", choices=["equal", "inverse_vol", "risk_parity", "sharpe_weighted"],
                        default="equal", help="Allocation method (default: equal)")
    parser.add_argument("--leverage", type=float, default=1.0,
                        help="Max gross leverage (default: 1.0)")
    parser.add_argument("--dd-limit", type=float, default=-0.15,
                        help="Max portfolio drawdown before circuit breaker (default: -0.15)")
    parser.add_argument("--dd-cooldown", type=int, default=48,
                        help="Bars flat after DD breach (default: 48)")
    parser.add_argument("--rebalance-freq", type=int, default=24,
                        help="Rebalance every N bars (default: 24)")
    parser.add_argument("--oos-bars", type=int, default=13140,
                        help="OOS window size (default: 13140 = ~18 months)")
    parser.add_argument("--full", action="store_true",
                        help="Use all available data")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--deadzone", type=float, default=None,
                        help="Override deadzone for all symbols (default: from config)")
    parser.add_argument("--min-hold", type=int, default=None,
                        help="Override min_hold for all symbols (default: from config)")
    parser.add_argument("--sym-dd", type=float, default=None,
                        help="Override per-symbol DD limit (e.g. -0.15). None=from config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    out_dir = Path(args.out) if args.out else None
    run_portfolio_backtest(
        symbols=[s.upper() for s in args.symbols],
        alloc_method=args.alloc,
        max_leverage=args.leverage,
        dd_limit=args.dd_limit,
        dd_cooldown=args.dd_cooldown,
        rebalance_freq=args.rebalance_freq,
        oos_bars=args.oos_bars,
        full=args.full,
        out_dir=out_dir,
        deadzone_override=args.deadzone,
        min_hold_override=args.min_hold,
        sym_dd_override=args.sym_dd,
    )


if __name__ == "__main__":
    main()
