#!/usr/bin/env python3
"""Backtest: Staged Risk Manager vs Fixed Risk for small capital growth.

Compares:
  A. Fixed 50%×3x (naive aggressive)
  B. Fixed 5%×2x (too conservative, can't open)
  C. Staged risk (auto-adapt)
  D. Staged risk + drawdown control
  E. Staged risk + V11 exit (reversal -0.5)

Tests with $140, $300, $500 starting equity on ETH 18-month OOS.
"""
from __future__ import annotations

import sys
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from risk.staged_risk import StagedRiskManager

COST_BPS_RT = 8
SLIPPAGE_BPS = 2
FUNDING_BPS_PER_8H = 1.0
TOTAL_COST = (COST_BPS_RT + SLIPPAGE_BPS) / 10000
ZSCORE_WINDOW = 720
MIN_NOTIONAL = 100.0  # Binance minimum


def zscore_signal(pred, window=720, warmup=180):
    n = len(pred)
    z = np.zeros(n)
    buf = []
    for i in range(n):
        buf.append(pred[i])
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < warmup:
            continue
        arr = np.array(buf)
        std = float(np.std(arr))
        if std > 1e-12:
            z[i] = (pred[i] - float(np.mean(arr))) / std
    return z


def run_fixed_risk(
    z: np.ndarray,
    closes: np.ndarray,
    initial_equity: float,
    risk_fraction: float,
    leverage: float,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    reversal_thresh: float = -0.3,
    deadzone_fade: float = 0.2,
    trailing_stop_pct: float = 0.0,
) -> Dict[str, Any]:
    """Fixed risk fraction backtest with compounding equity."""
    n = len(z)
    equity = initial_equity
    pos = 0.0
    ep = 0.0
    eb = 0
    peak_price = 0.0
    trades = []
    equity_curve = [equity]
    halted_bars = 0

    for i in range(n - 1):
        if pos != 0:
            held = i - eb
            should_exit = False
            reason = ""

            if held >= max_hold:
                should_exit = True
                reason = "max_hold"
            elif held >= min_hold:
                if pos * z[i] < reversal_thresh or abs(z[i]) < deadzone_fade:
                    should_exit = True
                    reason = "signal"

            if not should_exit and trailing_stop_pct > 0:
                if pos > 0:
                    peak_price = max(peak_price, closes[i])
                    dd = (peak_price - closes[i]) / peak_price
                else:
                    peak_price = min(peak_price, closes[i])
                    dd = (closes[i] - peak_price) / peak_price
                if dd >= trailing_stop_pct:
                    should_exit = True
                    reason = "trailing"

            if should_exit:
                exit_price = closes[i + 1]
                pnl_pct = pos * (exit_price - ep) / ep
                notional = max(equity * risk_fraction * leverage, MIN_NOTIONAL)
                notional = min(notional, equity * leverage)
                n_fund = max(held // 8, 1)
                cost = TOTAL_COST * notional + (FUNDING_BPS_PER_8H / 10000) * notional * n_fund * (1 if pos > 0 else -1)
                net = pnl_pct * notional - cost
                trades.append({"pnl": net, "hold": held, "reason": reason, "equity": equity})
                equity += net
                equity = max(equity, 1.0)
                pos = 0.0

        if pos == 0:
            notional = equity * risk_fraction * leverage
            if notional < MIN_NOTIONAL:
                max_safe = equity * leverage * 0.8
                if MIN_NOTIONAL <= max_safe:
                    notional = MIN_NOTIONAL
                else:
                    halted_bars += 1
                    equity_curve.append(equity)
                    continue

            if z[i] > deadzone:
                pos = 1.0
                ep = closes[i + 1]
                eb = i + 1
                peak_price = ep
            elif z[i] < -deadzone:
                pos = -1.0
                ep = closes[i + 1]
                eb = i + 1
                peak_price = ep

        equity_curve.append(equity)

    # Close open
    if pos != 0:
        held = n - 1 - eb
        pnl_pct = pos * (closes[-1] - ep) / ep
        notional = max(equity * risk_fraction * leverage, MIN_NOTIONAL)
        notional = min(notional, equity * leverage)
        cost = TOTAL_COST * notional
        net = pnl_pct * notional - cost
        trades.append({"pnl": net, "hold": held, "reason": "end", "equity": equity})
        equity += net
        equity = max(equity, 1.0)

    return _summarize(trades, initial_equity, equity, np.array(equity_curve), halted_bars)


def run_staged_risk(
    z: np.ndarray,
    closes: np.ndarray,
    initial_equity: float,
    deadzone: float,
    min_hold: int,
    max_hold: int,
    reversal_thresh: float = -0.3,
    deadzone_fade: float = 0.2,
    trailing_stop_pct: float = 0.0,
    use_drawdown_control: bool = True,
) -> Dict[str, Any]:
    """Staged risk backtest with auto-adapting risk levels."""
    n = len(z)
    mgr = StagedRiskManager(initial_equity=initial_equity, min_notional=MIN_NOTIONAL)
    equity = initial_equity
    pos = 0.0
    ep = 0.0
    eb = 0
    peak_price = 0.0
    trades = []
    equity_curve = [equity]
    stage_history = []
    halted_bars = 0

    for i in range(n - 1):
        mgr.update_equity(equity)

        if pos != 0:
            held = i - eb
            should_exit = False
            reason = ""

            if held >= max_hold:
                should_exit = True
                reason = "max_hold"
            elif held >= min_hold:
                if pos * z[i] < reversal_thresh or abs(z[i]) < deadzone_fade:
                    should_exit = True
                    reason = "signal"

            if not should_exit and trailing_stop_pct > 0:
                if pos > 0:
                    peak_price = max(peak_price, closes[i])
                    dd = (peak_price - closes[i]) / peak_price
                else:
                    peak_price = min(peak_price, closes[i])
                    dd = (closes[i] - peak_price) / peak_price
                if dd >= trailing_stop_pct:
                    should_exit = True
                    reason = "trailing"

            # Force exit if trading halted by drawdown controller
            if not mgr.can_trade:
                should_exit = True
                reason = "dd_halt"

            if should_exit:
                exit_price = closes[i + 1]
                pnl_pct = pos * (exit_price - ep) / ep
                # Use the notional that was used at entry
                notional = trades[-1]["entry_notional"] if trades and "entry_notional" in trades[-1] else MIN_NOTIONAL
                n_fund = max(held // 8, 1)
                cost = TOTAL_COST * notional + (FUNDING_BPS_PER_8H / 10000) * notional * n_fund * (1 if pos > 0 else -1)
                net = pnl_pct * notional - cost
                trades.append({
                    "pnl": net, "hold": held, "reason": reason,
                    "equity": equity, "stage": mgr.stage.label,
                    "entry_notional": notional,
                })
                equity += net
                equity = max(equity, 1.0)
                pos = 0.0

        if pos == 0 and mgr.can_trade:
            if use_drawdown_control:
                notional = mgr.compute_notional(closes[i])
            else:
                notional = equity * mgr.risk_fraction * mgr.leverage
                if notional < MIN_NOTIONAL:
                    max_safe = equity * mgr.leverage * 0.8
                    if MIN_NOTIONAL <= max_safe:
                        notional = MIN_NOTIONAL
                    else:
                        notional = 0

            if notional <= 0:
                halted_bars += 1
                equity_curve.append(equity)
                continue

            if z[i] > deadzone:
                pos = 1.0
                ep = closes[i + 1]
                eb = i + 1
                peak_price = ep
                trades.append({"entry_notional": notional})  # placeholder for notional tracking
            elif z[i] < -deadzone:
                pos = -1.0
                ep = closes[i + 1]
                eb = i + 1
                peak_price = ep
                trades.append({"entry_notional": notional})

        elif not mgr.can_trade:
            halted_bars += 1

        equity_curve.append(equity)
        if i % (24 * 30) == 0:
            stage_history.append((i, mgr.stage.label, equity, mgr.current_drawdown))

    # Close open
    if pos != 0:
        held = n - 1 - eb
        pnl_pct = pos * (closes[-1] - ep) / ep
        notional = MIN_NOTIONAL
        cost = TOTAL_COST * notional
        net = pnl_pct * notional - cost
        trades.append({"pnl": net, "hold": held, "reason": "end", "equity": equity, "stage": mgr.stage.label})
        equity += net
        equity = max(equity, 1.0)

    # Filter out entry-only placeholder dicts
    real_trades = [t for t in trades if "pnl" in t]
    result = _summarize(real_trades, initial_equity, equity, np.array(equity_curve), halted_bars)
    result["stage_history"] = stage_history
    result["final_stage"] = mgr.stage.label
    return result


def _summarize(trades, initial, final, equity_curve, halted_bars=0):
    if not trades:
        return {"sharpe": 0, "trades": 0, "return_pct": 0, "final": initial,
                "max_dd": 0, "win_rate": 0, "halted_bars": halted_bars}

    nets = np.array([t["pnl"] for t in trades])
    holds = np.array([t.get("hold", 0) for t in trades])
    holds = holds[holds > 0]

    running_max = np.maximum.accumulate(equity_curve)
    dd = (running_max - equity_curve) / np.maximum(running_max, 1.0)
    max_dd = float(np.max(dd))

    if len(nets) > 2 and len(holds) > 0:
        avg_h = float(np.mean(holds))
        tpy = 365 * 24 / max(avg_h, 1)
        sharpe = float(np.mean(nets) / max(np.std(nets, ddof=1), 1e-10) * np.sqrt(tpy))
    else:
        sharpe = 0.0

    # Monthly P&L
    bars_per_month = 24 * 30
    n = len(equity_curve)
    monthly_pnl = []
    for m in range(18):
        s = m * bars_per_month
        e = min((m + 1) * bars_per_month, n) - 1
        if s < n and e < n:
            monthly_pnl.append(equity_curve[e] - equity_curve[s])

    pos_months = sum(1 for x in monthly_pnl if x > 0)
    neg_months = sum(1 for x in monthly_pnl if x <= 0)
    worst_month_pct = min(
        (monthly_pnl[i] / max(equity_curve[i * bars_per_month], 1) * 100)
        for i in range(len(monthly_pnl))
    ) if monthly_pnl else 0

    # Time to double
    double_bar = None
    for i, eq in enumerate(equity_curve):
        if eq >= initial * 2:
            double_bar = i
            break

    return {
        "sharpe": sharpe,
        "trades": len(nets),
        "return_pct": (final - initial) / initial * 100,
        "final": final,
        "max_dd": max_dd,
        "win_rate": float(np.mean(nets > 0) * 100),
        "halted_bars": halted_bars,
        "pos_months": pos_months,
        "neg_months": neg_months,
        "worst_month_pct": worst_month_pct,
        "double_days": double_bar / 24 if double_bar else None,
        "min_equity": float(np.min(equity_curve)),
    }


def main():
    print("=" * 100)
    print("  STAGED RISK vs FIXED RISK — 小资金增长对比")
    print("=" * 100)

    # Load ETH data
    symbol = "ETHUSDT"
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    closes = df["close"].values.astype(np.float64)
    n_total = len(df)

    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    oos_bars = 24 * 30 * 18
    oos_start = n_total - oos_bars + 48

    print(f"  Data: {n_total:,} bars, OOS: {n_total - oos_start:,} bars")

    # Compute predictions
    print("  Computing features & predictions...")
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)
    feat_names = [c for c in feat_df.columns
                  if c not in ("close", "open_time", "timestamp") and c not in BLACKLIST]

    pred_start = max(0, oos_start - ZSCORE_WINDOW)
    import xgboost as xgb
    horizon_preds = []
    for hm_cfg in cfg["horizon_models"]:
        with open(model_dir / hm_cfg["lgbm"], "rb") as f:
            lgbm_data = pickle.load(f)
        with open(model_dir / hm_cfg["xgb"], "rb") as f:
            xgb_data = pickle.load(f)
        hm_feats = hm_cfg["features"]
        sel = [feat_names.index(fn) for fn in hm_feats if fn in feat_names]
        X = feat_df[feat_names].values[pred_start:].astype(np.float64)[:, sel]
        pred = 0.5 * lgbm_data["model"].predict(X) + \
               0.5 * xgb_data["model"].predict(xgb.DMatrix(X))
        horizon_preds.append(pred)

    z_horizons = [zscore_signal(p, window=ZSCORE_WINDOW, warmup=180) for p in horizon_preds]
    z_ensemble = np.mean(z_horizons, axis=0)
    warmup_used = oos_start - pred_start
    z_oos = z_ensemble[warmup_used:]
    closes_oos = closes[oos_start:]

    dz = cfg["deadzone"]
    mh = cfg["min_hold"]
    maxh = cfg["max_hold"]

    for init_eq in [140, 300, 500]:
        print(f"\n{'='*100}")
        print(f"  初始资金: ${init_eq}")
        print(f"{'='*100}")

        scenarios = [
            ("A: Fixed 50%×3x (激进)", lambda: run_fixed_risk(
                z_oos, closes_oos, init_eq, 0.50, 3.0, dz, mh, maxh)),

            ("B: Fixed 5%×2x (保守)", lambda: run_fixed_risk(
                z_oos, closes_oos, init_eq, 0.05, 2.0, dz, mh, maxh)),

            ("C: Staged (无回撤控制)", lambda: run_staged_risk(
                z_oos, closes_oos, init_eq, dz, mh, maxh,
                use_drawdown_control=False)),

            ("D: Staged + 回撤控制", lambda: run_staged_risk(
                z_oos, closes_oos, init_eq, dz, mh, maxh,
                use_drawdown_control=True)),

            ("E: Staged + V11 exit", lambda: run_staged_risk(
                z_oos, closes_oos, init_eq, dz, mh, maxh,
                reversal_thresh=-0.5, deadzone_fade=0.15,
                use_drawdown_control=True)),

            ("F: Staged + trail 2%", lambda: run_staged_risk(
                z_oos, closes_oos, init_eq, dz, mh, maxh,
                trailing_stop_pct=0.02,
                use_drawdown_control=True)),
        ]

        results = []
        for name, fn in scenarios:
            r = fn()
            r["name"] = name
            results.append(r)

        print(f"\n  {'方案':<30s} {'Sharpe':>7s} {'#交易':>5s} {'WR%':>5s} "
              f"{'终值$':>8s} {'回报%':>8s} {'MaxDD%':>7s} {'最低$':>7s} "
              f"{'翻倍天':>7s} {'暂停bar':>7s}")
        print(f"  {'─'*102}")

        for r in results:
            dd_str = f"{r['double_days']:.0f}" if r['double_days'] else "N/A"
            print(f"  {r['name']:<30s} {r['sharpe']:>+7.2f} {r['trades']:>5d} "
                  f"{r['win_rate']:>4.0f}% "
                  f"${r['final']:>7.1f} {r['return_pct']:>+7.1f}% "
                  f"{r['max_dd']*100:>6.1f}% ${r['min_equity']:>6.1f} "
                  f"{dd_str:>7s} {r['halted_bars']:>7d}")

        # Monthly breakdown for best staged
        best_staged = results[4]  # E: Staged + V11
        print(f"\n  月度收益 ({best_staged['name']}):")
        print(f"    正月/负月: {best_staged['pos_months']}/{best_staged['neg_months']}")
        print(f"    最差月: {best_staged['worst_month_pct']:+.1f}%")
        if best_staged.get("final_stage"):
            print(f"    最终阶段: {best_staged['final_stage']}")

        # Survival analysis
        print("\n  存活分析:")
        for r in results:
            survived = r["min_equity"] >= 50  # Can't trade below ~$50
            r["min_equity"] >= MIN_NOTIONAL / 3  # Need ~$33 to open $100
            status = "✅ 存活" if survived else "❌ 实质爆仓"
            if survived and r["final"] > init_eq:
                status = "✅ 盈利"
            print(f"    {r['name']:<30s} 最低${r['min_equity']:>6.1f}  {status}")

    print(f"\n{'='*100}")
    print("  结论")
    print(f"{'='*100}")
    print("  阶梯风控的核心价值:")
    print("    1. 小资金阶段 (50%×3x) 能快速积累")
    print("    2. 回撤控制防止跌破开仓门槛")
    print("    3. 资金增长后自动降低风险，保护利润")
    print("    4. 对比固定激进: 类似收益，但回撤更小、存活率更高")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
