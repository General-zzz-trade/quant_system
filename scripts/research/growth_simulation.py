#!/usr/bin/env python3
"""Growth simulation — how fast can $140 compound trading ETH?

Tests different parameter combinations:
  A. Baseline V10 (current)
  B. Single symbol (5% risk vs 2.5%)
  C. + Trailing stop 2%
  D. + Z-score cap 0.8
  E. + Higher leverage (3x)
  F. Full optimization (all combined)
  G. Kelly fraction sizing
"""
from __future__ import annotations
import sys
import json
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST

# ── Cost model ──
COST_BPS_RT = 8
SLIPPAGE_BPS = 2
FUNDING_BPS_PER_8H = 1.0
TOTAL_ENTRY_EXIT_COST = (COST_BPS_RT + SLIPPAGE_BPS) / 10000

ZSCORE_WINDOW = 720
INITIAL_EQUITY = 140.0


def zscore_signal(pred, window=720, warmup=180):
    """Causal rolling z-score."""
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


@dataclass
class SimConfig:
    name: str
    equity: float = INITIAL_EQUITY
    risk_fraction: float = 0.05  # Single symbol = full 5%
    leverage: float = 2.0
    deadzone: float = 0.3
    min_hold: int = 12
    max_hold: int = 96
    long_only: bool = False
    trailing_stop_pct: float = 0.0  # 0 = disabled
    z_cap: float = 0.0  # 0 = disabled
    dynamic_lev: bool = False
    lev_min: float = 2.0
    lev_max: float = 3.0


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    net_pnl: float
    equity_before: float
    equity_after: float
    leverage: float
    hold_bars: int
    exit_reason: str = ""


def compute_dynamic_leverage(z_val, closes_recent, deadzone, lev_min, lev_max):
    if lev_max <= lev_min:
        return lev_max
    z_excess = abs(z_val) - deadzone
    signal_ramp = min(max(z_excess / deadzone, 0.0), 1.0)
    vol_discount = 1.0
    if len(closes_recent) >= 168:
        rets = np.diff(closes_recent) / closes_recent[:-1]
        current_vol = float(np.std(rets[-168:])) if len(rets) >= 168 else float(np.std(rets))
        long_vol = float(np.std(rets))
        if long_vol > 1e-12:
            vol_ratio = current_vol / long_vol
            vol_discount = 1.0 - min(max((vol_ratio - 0.8) / 0.7, 0.0), 1.0)
    return round(lev_min + (lev_max - lev_min) * signal_ramp * vol_discount, 2)


def run_sim(cfg: SimConfig, z_signal: np.ndarray, closes: np.ndarray) -> tuple:
    """Run backtest with compounding equity."""
    n = len(z_signal)
    trades = []
    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = cfg.leverage
    equity = cfg.equity
    peak_price = 0.0  # For trailing stop

    # Apply z-cap
    z = z_signal.copy()
    if cfg.z_cap > 0:
        z = np.clip(z, -cfg.z_cap, cfg.z_cap)

    # Equity curve (bar-by-bar)
    equity_curve = np.full(n, equity)
    min_notional = 5.0  # Binance minimum

    for i in range(n - 1):
        # ── Check exit ──
        if pos != 0:
            held = i - eb
            should_exit = False
            exit_reason = ""

            # Max hold
            if held >= cfg.max_hold:
                should_exit = True
                exit_reason = "max_hold"
            # Signal reversal / fade
            elif held >= cfg.min_hold:
                if pos * z[i] < -0.3 or abs(z[i]) < 0.2:
                    should_exit = True
                    exit_reason = "signal"

            # Trailing stop
            if not should_exit and cfg.trailing_stop_pct > 0 and pos != 0:
                if pos > 0:
                    peak_price = max(peak_price, closes[i])
                    drawdown = (peak_price - closes[i]) / peak_price
                else:
                    peak_price = min(peak_price, closes[i])
                    drawdown = (closes[i] - peak_price) / peak_price
                if drawdown >= cfg.trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"

            if should_exit:
                exit_price = closes[i + 1]
                pnl_pct = pos * (exit_price - ep) / ep
                notional = max(equity * cfg.risk_fraction * entry_lev, min_notional)
                notional = min(notional, equity * entry_lev)  # Can't exceed equity * leverage

                cost_trading = TOTAL_ENTRY_EXIT_COST * notional
                n_funding = max(held // 8, 1)
                cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding
                if pos < 0:
                    cost_funding = -cost_funding

                gross = pnl_pct * notional
                net = gross - cost_trading - cost_funding
                eq_before = equity
                equity += net
                equity = max(equity, 1.0)  # Can't go below $1

                trades.append(Trade(
                    entry_bar=eb, exit_bar=i+1, direction=int(pos),
                    entry_price=ep, exit_price=exit_price,
                    net_pnl=net, equity_before=eq_before, equity_after=equity,
                    leverage=entry_lev, hold_bars=held, exit_reason=exit_reason
                ))
                pos = 0.0

        # ── Check entry ──
        if pos == 0 and i + 1 < n:
            desired = 0
            if z[i] > cfg.deadzone:
                desired = 1
            elif not cfg.long_only and z[i] < -cfg.deadzone:
                desired = -1

            if desired != 0:
                # Dynamic leverage
                if cfg.dynamic_lev:
                    start = max(0, i - 720)
                    entry_lev = compute_dynamic_leverage(
                        z[i], closes[start:i+1], cfg.deadzone,
                        cfg.lev_min, cfg.lev_max)
                else:
                    entry_lev = cfg.leverage

                # Check if position size meets minimum
                notional = equity * cfg.risk_fraction * entry_lev
                if notional < min_notional:
                    # Auto-scale risk fraction
                    adj_risk = min_notional / (equity * entry_lev)
                    if adj_risk <= 1.0:
                        notional = min_notional
                    else:
                        continue  # Can't even open min position

                pos = float(desired)
                ep = closes[i + 1]
                eb = i + 1
                peak_price = ep

        equity_curve[i] = equity

    # Close open position
    if pos != 0:
        exit_price = closes[-1]
        held = n - 1 - eb
        pnl_pct = pos * (exit_price - ep) / ep
        notional = max(equity * cfg.risk_fraction * entry_lev, min_notional)
        notional = min(notional, equity * entry_lev)
        cost_trading = TOTAL_ENTRY_EXIT_COST * notional
        n_funding = max(held // 8, 1)
        cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding
        if pos < 0:
            cost_funding = -cost_funding
        gross = pnl_pct * notional
        net = gross - cost_trading - cost_funding
        eq_before = equity
        equity += net
        equity = max(equity, 1.0)
        trades.append(Trade(
            entry_bar=eb, exit_bar=n-1, direction=int(pos),
            entry_price=ep, exit_price=exit_price,
            net_pnl=net, equity_before=eq_before, equity_after=equity,
            leverage=entry_lev, hold_bars=held, exit_reason="end"
        ))

    equity_curve[-1] = equity
    return trades, equity, equity_curve


def analyze_sim(name, trades, initial, final, equity_curve):
    if not trades:
        print(f"  {name:<35s} NO TRADES")
        return {}

    nets = np.array([t.net_pnl for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    holds = np.array([t.hold_bars for t in trades])

    # Max drawdown on equity curve
    running_max = np.maximum.accumulate(equity_curve)
    dd = (running_max - equity_curve) / np.maximum(running_max, 1.0)
    max_dd = float(np.max(dd))
    max_dd_dollar = float(np.max(running_max - equity_curve))

    # Sharpe
    if len(nets) > 2:
        avg_h = float(np.mean(holds))
        tpy = 365 * 24 / max(avg_h, 1)
        sharpe = float(np.mean(nets) / max(np.std(nets, ddof=1), 1e-10) * np.sqrt(tpy))
    else:
        sharpe = 0.0

    ret_pct = (final - initial) / initial * 100
    len(trades) / max(len(trades), 1) * 18  # approx

    # Time to double
    eq = initial
    double_bar = None
    for t in trades:
        eq += t.net_pnl
        if eq >= initial * 2 and double_bar is None:
            double_bar = t.exit_bar

    # Monthly returns
    month_nets = {}
    for t in trades:
        m_idx = t.exit_bar // (24 * 30)  # Approximate month index
        month_nets.setdefault(m_idx, 0.0)
        month_nets[m_idx] += t.net_pnl
    pos_months = sum(1 for v in month_nets.values() if v > 0)
    neg_months = sum(1 for v in month_nets.values() if v <= 0)

    # Largest single-trade loss
    worst_trade = float(np.min(nets))
    worst_pct = worst_trade / initial * 100

    return {
        "name": name, "trades": len(trades), "wins": wins,
        "win_rate": wins / len(trades) * 100,
        "sharpe": sharpe, "ret_pct": ret_pct,
        "final": final, "max_dd": max_dd, "max_dd_dollar": max_dd_dollar,
        "double_bar": double_bar,
        "pos_months": pos_months, "neg_months": neg_months,
        "worst_trade": worst_trade, "worst_pct": worst_pct,
        "avg_hold": float(np.mean(holds)),
    }


def main():
    print("=" * 80)
    print("  GROWTH SIMULATION: $140 → ? (ETH V10, 18-month OOS)")
    print("=" * 80)

    # ── Load data ──
    symbol = "ETHUSDT"
    df = pd.read_csv(f"data_files/{symbol}_1h.csv")
    closes = df["close"].values.astype(np.float64)
    n_total = len(df)

    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    oos_bars = 24 * 30 * 18
    embargo = 48
    oos_start = n_total - oos_bars + embargo

    print(f"\n  Data: {n_total:,} bars, OOS start: bar {oos_start}")
    print(f"  OOS period: {n_total - oos_start:,} bars ({(n_total-oos_start)/24:.0f} days)")

    # ── Compute features & predictions ──
    print("  Computing features...")
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

    # Multi-horizon prediction
    print("  Running multi-horizon predictions...")
    pred_start = max(0, oos_start - ZSCORE_WINDOW)
    horizon_preds = []
    for hm_cfg in cfg["horizon_models"]:
        lgbm_path = model_dir / hm_cfg["lgbm"]
        xgb_path = model_dir / hm_cfg["xgb"]
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)
        with open(xgb_path, "rb") as f:
            xgb_data = pickle.load(f)

        hm_feats = hm_cfg["features"]
        sel = [feat_names.index(fn) for fn in hm_feats if fn in feat_names]

        X = feat_df[feat_names].values[pred_start:].astype(np.float64)[:, sel]

        import xgboost as xgb
        pred = 0.5 * lgbm_data["model"].predict(X) + \
               0.5 * xgb_data["model"].predict(xgb.DMatrix(X))
        horizon_preds.append(pred)

    # Z-score each horizon independently, then average
    z_horizons = [zscore_signal(p, window=ZSCORE_WINDOW, warmup=180) for p in horizon_preds]
    z_ensemble = np.mean(z_horizons, axis=0)

    # Trim to OOS
    warmup_used = oos_start - pred_start
    z_oos = z_ensemble[warmup_used:]
    closes_oos = closes[oos_start:]
    n_oos = len(z_oos)
    print(f"  OOS bars: {n_oos:,}, Z-score warmup: {warmup_used}")

    # ── Define scenarios ──
    scenarios = [
        # ── 保守区 ──
        SimConfig(name="A: 保守 5%×2x",
                  risk_fraction=0.05, leverage=2.0),

        SimConfig(name="B: 保守+优化 5%×2x+trail+zcap",
                  risk_fraction=0.05, leverage=2.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        # ── 中等区 ──
        SimConfig(name="C: 中等 20%×3x",
                  risk_fraction=0.20, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        SimConfig(name="D: 中等 30%×3x",
                  risk_fraction=0.30, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        # ── 激进区 (小资金翻倍) ──
        SimConfig(name="E: 激进 50%×3x",
                  risk_fraction=0.50, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        SimConfig(name="F: 激进 50%×5x",
                  risk_fraction=0.50, leverage=5.0,
                  trailing_stop_pct=0.015, z_cap=0.8),

        SimConfig(name="G: 全仓 100%×3x",
                  risk_fraction=1.00, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        SimConfig(name="H: 全仓 100%×5x",
                  risk_fraction=1.00, leverage=5.0,
                  trailing_stop_pct=0.015, z_cap=0.8),

        # ── Kelly最优 ──
        # Kelly f* = p/a - q/b, where p=WR, q=1-WR, a=avg_loss, b=avg_win
        # Approx: WR=56%, avg_win/avg_loss≈1.2 → Kelly≈0.23
        # Half-Kelly (safer) ≈ 0.12
        SimConfig(name="I: Half-Kelly 12%×3x+trail+zcap",
                  risk_fraction=0.12, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),

        SimConfig(name="J: Full-Kelly 25%×3x+trail+zcap",
                  risk_fraction=0.25, leverage=3.0,
                  trailing_stop_pct=0.02, z_cap=0.8),
    ]

    # ── Run all scenarios ──
    results = []
    print(f"\n  Running {len(scenarios)} scenarios...\n")

    for sc in scenarios:
        t0 = time.time()
        trades, final_eq, eq_curve = run_sim(sc, z_oos, closes_oos)
        time.time() - t0
        stats = analyze_sim(sc.name, trades, INITIAL_EQUITY, final_eq, eq_curve)
        results.append(stats)

    # ── Results table ──
    print(f"\n{'='*100}")
    print("  GROWTH COMPARISON: $140 初始资金 ETH 单品种 (18个月OOS, 复利)")
    print(f"{'='*100}")
    print(f"\n  {'方案':<38s} {'Sharpe':>7s} {'#交易':>5s} {'WR':>5s} "
          f"{'终值$':>8s} {'回报%':>8s} {'MaxDD%':>7s} {'MaxDD$':>7s} "
          f"{'翻倍(天)':>8s}")
    print(f"  {'─'*97}")

    for r in results:
        if not r:
            continue
        double_days = f"{r['double_bar']/24:.0f}" if r['double_bar'] else "N/A"
        print(f"  {r['name']:<38s} {r['sharpe']:>+7.2f} {r['trades']:>5d} "
              f"{r['win_rate']:>4.0f}% "
              f"${r['final']:>7.1f} {r['ret_pct']:>+7.1f}% "
              f"{r['max_dd']*100:>6.1f}% ${r['max_dd_dollar']:>6.1f} "
              f"{double_days:>8s}")

    # ── Risk analysis ──
    print(f"\n{'='*100}")
    print("  风险分析")
    print(f"{'='*100}")
    print(f"\n  {'方案':<38s} {'最差单笔$':>10s} {'最差%':>7s} {'正月/负月':>10s} "
          f"{'爆仓风险':>8s}")
    print(f"  {'─'*80}")

    for r in results:
        if not r:
            continue
        # Ruin risk: MaxDD > 80% of equity
        ruin = "高" if r['max_dd'] > 0.60 else ("中" if r['max_dd'] > 0.40 else "低")
        print(f"  {r['name']:<38s} ${r['worst_trade']:>+9.2f} "
              f"{r['worst_pct']:>+6.1f}% "
              f"{r['pos_months']:>3d}/{r['neg_months']:<3d}   "
              f"{'⚠️ ' + ruin if ruin != '低' else '✅ ' + ruin}")

    # ── Recommendation ──
    print(f"\n{'='*100}")
    print("  推荐方案")
    print(f"{'='*100}")

    # Find best risk-adjusted
    valid = [r for r in results if r and r['max_dd'] < 0.50]
    if valid:
        best = max(valid, key=lambda x: x['sharpe'])
        fastest = min([r for r in results if r and r.get('double_bar')],
                      key=lambda x: x['double_bar'], default=None)

        print(f"\n  🏆 最佳风险调整: {best['name']}")
        print(f"     Sharpe {best['sharpe']:.2f}, 终值 ${best['final']:.0f}, "
              f"MaxDD {best['max_dd']*100:.1f}%")

        if fastest:
            print(f"\n  ⚡ 最快翻倍: {fastest['name']}")
            print(f"     翻倍用时 {fastest['double_bar']/24:.0f} 天, "
                  f"但 MaxDD {fastest['max_dd']*100:.1f}%")

    # Monthly equity path for top 3
    print(f"\n{'='*100}")
    print("  月度权益曲线 (前3方案)")
    print(f"{'='*100}")

    top_scenarios = [scenarios[4], scenarios[5], scenarios[6]]  # E, F, G
    top_names = ["E: Trail+Zcap", "F: +DynLev", "G: 激进10%"]

    for sc, name in zip(top_scenarios, top_names):
        trades, final_eq, eq_curve = run_sim(sc, z_oos, closes_oos)
        print(f"\n  {name}:")
        print(f"  {'月份':>6s} {'权益$':>8s} {'月收益$':>9s} {'月收益%':>8s} {'累计%':>8s}")
        print(f"  {'─'*44}")

        bars_per_month = 24 * 30
        for m in range(18):
            start_bar = m * bars_per_month
            end_bar = min((m + 1) * bars_per_month, n_oos) - 1
            if start_bar >= n_oos:
                break
            eq_start = eq_curve[start_bar]
            eq_end = eq_curve[end_bar]
            monthly_ret = eq_end - eq_start
            monthly_pct = monthly_ret / max(eq_start, 1) * 100
            cum_pct = (eq_end - INITIAL_EQUITY) / INITIAL_EQUITY * 100
            print(f"  M{m+1:>4d} ${eq_end:>7.1f} ${monthly_ret:>+8.1f} "
                  f"{monthly_pct:>+7.1f}% {cum_pct:>+7.1f}%")

    print(f"\n{'='*100}")
    print("  ⚠️  注意: 回测≠实盘。实际交易中滑点、延迟、情绪会降低收益。")
    print("  建议: 先用TestNet跑2周验证parity，再用$140实盘。")
    print("  资金管理: MaxDD超过40%时暂停交易，检查策略。")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
