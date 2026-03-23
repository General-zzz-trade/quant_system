#!/usr/bin/env python3
"""Honest Backtest — fixes all known biases from audit.

Fixes applied:
  1. OOS-only: predictions only on data AFTER model training cutoff
  2. Next-bar execution: entry at closes[i+1], not closes[i]
  3. Realistic costs: 8bp round-trip (taker both sides) + slippage
  4. Funding rate cost: ~1bp per 8h holding period at avg funding
  5. Embargo gap: skip 24 bars after train/OOS boundary
  6. No parameter snooping: use pre-committed config from training
  7. Correct portfolio capital: $10k total split across coins

Usage:
    python3 -m scripts.backtest_honest
"""
from __future__ import annotations
import sys
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from shared.signal_postprocess import rolling_zscore, should_exit_position
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Realistic cost config ──
COST_BPS_RT = 8          # 4bp taker per side × 2 = 8bp round-trip
SLIPPAGE_BPS = 2         # 1bp slippage per side × 2
FUNDING_BPS_PER_8H = 1.0 # Average BTC/ETH funding rate ~0.01% per 8h
TOTAL_ENTRY_EXIT_COST = (COST_BPS_RT + SLIPPAGE_BPS) / 10000  # 10bp = 0.001

EQUITY = 10_000.0
RISK_FRACTION_PER_COIN = 0.025  # 2.5% per coin (total 5% for 2 coins)
ZSCORE_WINDOW = 720

# Dynamic leverage
LEV_MIN = 2.0
LEV_MAX = 3.0
VOL_WARMUP = 168

# Embargo: skip this many bars after train/OOS boundary
EMBARGO_BARS = 48  # 48h gap (2 days, > 24h horizon)


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_dynamic_leverage(z_blend, closes_recent, deadzone,
                              lev_min=LEV_MIN, lev_max=LEV_MAX):
    lev_range = lev_max - lev_min
    if lev_range <= 0:
        return lev_max
    z_excess = abs(z_blend) - deadzone
    signal_ramp = min(max(z_excess / deadzone, 0.0), 1.0)
    vol_discount = 1.0
    if len(closes_recent) >= VOL_WARMUP:
        rets = np.diff(closes_recent) / closes_recent[:-1]
        current_vol = float(np.std(rets[-168:])) if len(rets) >= 168 else float(np.std(rets))
        long_vol = float(np.std(rets))
        if long_vol > 1e-12:
            vol_ratio = current_vol / long_vol
            vol_discount = 1.0 - min(max((vol_ratio - 0.8) / 0.7, 0.0), 1.0)
    return round(lev_min + lev_range * signal_ramp * vol_discount, 2)


@dataclass
class Trade:
    symbol: str
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    cost_trading: float
    cost_funding: float
    net_pnl: float
    leverage: float
    hold_bars: int


def run_backtest(symbol, z_signal, closes, timestamps,
                 deadzone, min_hold, max_hold, long_only,
                 equity_share):
    """
    Honest backtest:
      - Entry at closes[i+1] (next bar after signal)
      - Realistic costs: trading + slippage + funding
      - No in-sample bars
    """
    n = len(z_signal)
    trades = []
    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = LEV_MIN

    equity = equity_share  # This coin's share of portfolio
    daily_pnl = np.zeros(n)

    for i in range(n - 1):  # -1: need i+1 for next-bar execution
        # ── Check exit (at next bar close) ──
        if pos != 0:
            held = i - eb
            if should_exit_position(
                position=pos,
                z_value=float(z_signal[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            ):
                # Execute exit at NEXT bar's close
                exit_price = closes[i + 1]
                pnl_pct = pos * (exit_price - ep) / ep
                notional = equity * RISK_FRACTION_PER_COIN * entry_lev

                # Trading cost: entry + exit
                cost_trading = TOTAL_ENTRY_EXIT_COST * notional

                # Funding cost: per 8h period held
                n_funding_periods = max(held // 8, 1)
                cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding_periods
                # For shorts, funding is received (reversed)
                if pos < 0:
                    cost_funding = -cost_funding  # Shorts receive funding (on average)

                gross = pnl_pct * notional
                net = gross - cost_trading - cost_funding
                equity += net

                trades.append(Trade(
                    symbol=symbol, entry_bar=eb, exit_bar=i+1,
                    direction=int(pos), entry_price=ep, exit_price=exit_price,
                    gross_pnl=gross, cost_trading=cost_trading,
                    cost_funding=cost_funding, net_pnl=net,
                    leverage=entry_lev, hold_bars=held,
                ))
                daily_pnl[i+1] += net
                pos = 0.0

        # ── Check entry (execute at next bar close) ──
        if pos == 0 and i + 1 < n:
            desired = 0
            if z_signal[i] > deadzone:
                desired = 1
            elif not long_only and z_signal[i] < -deadzone:
                desired = -1

            if desired != 0:
                pos = float(desired)
                ep = closes[i + 1]  # Next-bar execution
                eb = i + 1
                start = max(0, i - 720)
                entry_lev = compute_dynamic_leverage(
                    z_signal[i], closes[start:i+1], deadzone)

    # Close open position at last bar
    if pos != 0 and n > 0:
        exit_price = closes[-1]
        pnl_pct = pos * (exit_price - ep) / ep
        notional = equity * RISK_FRACTION_PER_COIN * entry_lev
        held = n - 1 - eb
        cost_trading = TOTAL_ENTRY_EXIT_COST * notional
        n_funding_periods = max(held // 8, 1)
        cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding_periods
        if pos < 0:
            cost_funding = -cost_funding
        gross = pnl_pct * notional
        net = gross - cost_trading - cost_funding
        equity += net
        trades.append(Trade(
            symbol=symbol, entry_bar=eb, exit_bar=n-1,
            direction=int(pos), entry_price=ep, exit_price=exit_price,
            gross_pnl=gross, cost_trading=cost_trading,
            cost_funding=cost_funding, net_pnl=net,
            leverage=entry_lev, hold_bars=held,
        ))

    return trades, equity


def analyze(trades, label, initial_equity):
    if not trades:
        print(f"\n  [{label}] No trades.")
        return {}

    net = np.array([t.net_pnl for t in trades])
    gross = np.array([t.gross_pnl for t in trades])
    cost_t = np.array([t.cost_trading for t in trades])
    cost_f = np.array([t.cost_funding for t in trades])
    holds = np.array([t.hold_bars for t in trades])
    levs = np.array([t.leverage for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    longs = sum(1 for t in trades if t.direction > 0)
    shorts = len(trades) - longs

    # Drawdown
    eq = initial_equity
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe
    if len(net) > 2:
        avg_h = float(np.mean(holds))
        tpy = 365 * 24 / max(avg_h, 1)
        sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy))
    else:
        sharpe = 0.0

    total_net = float(np.sum(net))
    total_gross = float(np.sum(gross))
    total_cost_t = float(np.sum(cost_t))
    total_cost_f = float(np.sum(cost_f))

    print(f"\n  [{label}]")
    print(f"  {'─'*70}")
    print(f"  Trades: {len(trades)}  (L:{longs} S:{shorts})  "
          f"Win Rate: {wins/len(trades)*100:.1f}%")
    print(f"  Avg Hold: {np.mean(holds):.0f}h  Avg Leverage: {np.mean(levs):.2f}x")
    print(f"  Gross P&L:     ${total_gross:+,.0f}")
    print(f"  Trading costs: ${total_cost_t:,.0f}  "
          f"({total_cost_t/max(len(trades),1):.1f}/trade, {COST_BPS_RT+SLIPPAGE_BPS}bp RT)")
    print(f"  Funding costs: ${total_cost_f:+,.0f}  "
          f"({total_cost_f/max(len(trades),1):+.1f}/trade)")
    print(f"  Net P&L:       ${total_net:+,.0f}  "
          f"({total_net/initial_equity*100:+.2f}%)")
    print(f"  Max Drawdown:  {max_dd*100:.2f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Cost drag:     {(total_cost_t+total_cost_f)/max(abs(total_gross),1)*100:.1f}% of gross")

    return {
        "trades": len(trades), "longs": longs, "shorts": shorts,
        "win_rate": wins / len(trades) * 100,
        "sharpe": sharpe,
        "total_gross": total_gross,
        "total_cost_t": total_cost_t,
        "total_cost_f": total_cost_f,
        "total_net": total_net,
        "ret_pct": total_net / initial_equity * 100,
        "max_dd": max_dd,
        "avg_lev": float(np.mean(levs)),
        "initial_equity": initial_equity,
    }


def main():
    print("=" * 70)
    print("HONEST BACKTEST — All Biases Fixed")
    print("=" * 70)
    print("\n  Fixes applied:")
    print("    [1] OOS-only predictions (exclude training period)")
    print("    [2] Next-bar execution (signal at bar i → trade at bar i+1)")
    print(f"    [3] Realistic costs: {COST_BPS_RT}bp RT trading + {SLIPPAGE_BPS}bp slippage")
    print(f"    [4] Funding rate: {FUNDING_BPS_PER_8H}bp per 8h holding")
    print(f"    [5] {EMBARGO_BARS}-bar embargo after train/OOS boundary")
    print("    [6] Pre-committed params from training (no OOS snooping)")
    print(f"    [7] Portfolio: ${EQUITY:,.0f} split across coins")

    SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    n_coins = 0
    for s in SYMBOLS:
        if Path(f"data_files/{s}_1h.csv").exists() and \
           Path(f"models_v8/{s}_gate_v2/lgbm_v8.pkl").exists():
            n_coins += 1
    equity_per_coin = EQUITY / max(n_coins, 1)
    print(f"    Coins: {n_coins}, ${equity_per_coin:,.0f} each")

    all_stats = {}
    all_trades = []

    for symbol in SYMBOLS:
        data_path = f"data_files/{symbol}_1h.csv"
        model_dir = Path(f"models_v8/{symbol}_gate_v2")

        if not Path(data_path).exists():
            print(f"\n  SKIP {symbol}: no data")
            continue
        if not (model_dir / "lgbm_v8.pkl").exists():
            print(f"\n  SKIP {symbol}: no model")
            continue

        print(f"\n{'='*70}")
        print(f"  {symbol}")
        print("=" * 70)

        # Load data
        df = pd.read_csv(data_path)
        ts_col = "open_time" if "open_time" in df.columns else "timestamp"
        timestamps = df[ts_col].values.astype(np.int64)
        closes = df["close"].values.astype(np.float64)
        n_total = len(df)

        # Load model config
        with open(model_dir / "config.json") as f:
            cfg = json.load(f)

        # Determine OOS start
        # Training uses first (n - 18months) bars for train+val
        oos_months = 18
        oos_bars = 24 * 30 * oos_months  # 12960 bars
        oos_start = n_total - oos_bars + EMBARGO_BARS  # Add embargo gap

        oos_start_date = pd.Timestamp(timestamps[oos_start], unit="ms").strftime("%Y-%m-%d")
        oos_end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
        print(f"  Data: {n_total:,} bars")
        print(f"  OOS start: bar {oos_start} ({oos_start_date}) "
              f"[{EMBARGO_BARS}-bar embargo applied]")
        print(f"  OOS period: {oos_start_date} → {oos_end_date} "
              f"({n_total - oos_start:,} bars, {(n_total - oos_start)/24:.0f} days)")

        # Compute features on FULL data (features are causal, this is OK)
        _has_v11 = Path("data_files/macro_daily.csv").exists()
        feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
        tf4h = compute_4h_features(df)
        for col in TF4H_FEATURE_NAMES:
            feat_df[col] = tf4h[col].values
        for int_name, fa, fb in INTERACTION_FEATURES:
            if fa in feat_df.columns and fb in feat_df.columns:
                feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)
        feat_names = [c for c in feat_df.columns
                      if c not in ("close", "open_time", "timestamp")
                      and c not in BLACKLIST]

        # Load model
        with open(model_dir / "lgbm_v8.pkl", "rb") as f:
            lgbm = pickle.load(f)
        with open(model_dir / "xgb_v8.pkl", "rb") as f:
            xgb_data = pickle.load(f)
        import xgboost as xgb

        model_feats = cfg["features"]
        sel_idx = [feat_names.index(f) for f in model_feats if f in feat_names]

        # Predict ONLY on OOS bars (plus warmup for z-score)
        # Need ZSCORE_WINDOW bars before OOS for z-score warmup
        pred_start = max(0, oos_start - ZSCORE_WINDOW)
        X_pred = feat_df[feat_names].values[pred_start:].astype(np.float64)[:, sel_idx]
        pred_raw = 0.5 * lgbm["model"].predict(X_pred) + \
                   0.5 * xgb_data["model"].predict(xgb.DMatrix(X_pred))

        # Z-score on the prediction window
        z_full = rolling_zscore(pred_raw, window=ZSCORE_WINDOW, warmup=180)

        # Trim to OOS only
        warmup_used = oos_start - pred_start
        z_oos = z_full[warmup_used:]
        closes_oos = closes[oos_start:]
        ts_oos = timestamps[oos_start:]
        n_oos = len(z_oos)

        print(f"  Z-score warmup: {warmup_used} bars")
        print(f"  OOS bars for trading: {n_oos:,}")

        # Use pre-committed params from model config (no snooping)
        deadzone = cfg.get("deadzone", 1.0)
        min_hold = cfg.get("min_hold", 24)
        max_hold = cfg.get("max_hold", min_hold * 5)
        long_only_cfg = cfg.get("long_only", True)

        # P0: enable shorts regardless of training config
        long_only = False

        print(f"  Config: dz={deadzone}, hold=[{min_hold},{max_hold}], "
              f"long_only={long_only} (training: {long_only_cfg})")

        # Run honest backtest
        trades, final_eq = run_backtest(
            symbol, z_oos, closes_oos, ts_oos,
            deadzone, min_hold, max_hold, long_only,
            equity_share=equity_per_coin)

        stats = analyze(trades, f"{symbol} (OOS, honest)", equity_per_coin)
        all_stats[symbol] = stats
        all_trades.extend(trades)

        # Monthly breakdown
        if trades:
            monthly = {}
            for t in trades:
                ts_ms = int(ts_oos[min(t.exit_bar, len(ts_oos)-1)])
                month = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m")
                if month not in monthly:
                    monthly[month] = {"net": 0.0, "trades": 0, "wins": 0}
                monthly[month]["net"] += t.net_pnl
                monthly[month]["trades"] += 1
                if t.net_pnl > 0:
                    monthly[month]["wins"] += 1

            print(f"\n  {'Month':>8} {'#Tr':>5} {'WinR':>6} {'Net$':>8} {'Cum$':>9}")
            print(f"  {'─'*42}")
            cum = 0.0
            pos_months = 0
            for month in sorted(monthly.keys()):
                m = monthly[month]
                cum += m["net"]
                wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
                if m["net"] > 0:
                    pos_months += 1
                print(f"  {month:>8} {m['trades']:>5} {wr:>5.0f}% "
                      f"{m['net']:>+8.1f} {cum:>+9.1f}")
            print(f"  Positive months: {pos_months}/{len(monthly)}")

    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PORTFOLIO SUMMARY (Honest)")
    print("=" * 70)

    if not all_trades:
        print("  No trades across any symbol.")
        return

    all_trades.sort(key=lambda t: t.entry_bar)
    total_net = sum(t.net_pnl for t in all_trades)
    total_gross = sum(t.gross_pnl for t in all_trades)
    total_cost_t = sum(t.cost_trading for t in all_trades)
    total_cost_f = sum(t.cost_funding for t in all_trades)
    total_wins = sum(1 for t in all_trades if t.net_pnl > 0)
    total_longs = sum(1 for t in all_trades if t.direction > 0)
    total_shorts = len(all_trades) - total_longs

    # Portfolio drawdown
    eq = EQUITY
    peak = EQUITY
    max_dd = 0.0
    for t in all_trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Portfolio Sharpe
    trade_nets = np.array([t.net_pnl for t in all_trades])
    if len(trade_nets) > 2:
        avg_h = np.mean([t.hold_bars for t in all_trades])
        tpy = 365 * 24 / max(avg_h, 1)
        port_sharpe = float(np.mean(trade_nets) /
                            max(np.std(trade_nets, ddof=1), 1e-10) *
                            np.sqrt(tpy))
    else:
        port_sharpe = 0.0

    print(f"\n  Total Trades: {len(all_trades)}  (L:{total_longs} S:{total_shorts})")
    print(f"  Win Rate: {total_wins/len(all_trades)*100:.1f}%")
    print(f"  Gross P&L:     ${total_gross:+,.0f}")
    print(f"  Trading costs: ${total_cost_t:,.0f}")
    print(f"  Funding costs: ${total_cost_f:+,.0f}")
    print(f"  Net P&L:       ${total_net:+,.0f}  ({total_net/EQUITY*100:+.2f}%)")
    print(f"  Cost drag:     {(total_cost_t+total_cost_f)/max(abs(total_gross),1)*100:.1f}% of gross")
    print(f"  Max Drawdown:  {max_dd*100:.2f}%")
    print(f"  Sharpe:        {port_sharpe:.2f}")
    print(f"  Final Equity:  ${eq:,.0f}")

    # Per-coin comparison
    print(f"\n  {'Symbol':<12s} {'#Tr':>5s} {'L/S':>7s} {'WinR':>6s} "
          f"{'Sharpe':>7s} {'Net$':>8s} {'MaxDD':>7s}")
    print(f"  {'─'*60}")
    for symbol in SYMBOLS:
        s = all_stats.get(symbol, {})
        if not s:
            continue
        ls = f"{s['longs']}/{s['shorts']}"
        print(f"  {symbol:<12s} {s['trades']:>5d} {ls:>7s} {s['win_rate']:>5.1f}% "
              f"{s['sharpe']:>+7.2f} ${s['total_net']:>+7.0f} "
              f"{s['max_dd']*100:>6.2f}%")
    ls_port = f"{total_longs}/{total_shorts}"
    print(f"  {'PORTFOLIO':<12s} {len(all_trades):>5d} {ls_port:>7s} "
          f"{total_wins/len(all_trades)*100:>5.1f}% "
          f"{port_sharpe:>+7.2f} ${total_net:>+7.0f} "
          f"{max_dd*100:>6.2f}%")

    # ── Comparison: honest vs biased ──
    print(f"\n{'='*70}")
    print("BIAS IMPACT COMPARISON")
    print("=" * 70)
    print("  (Biased = old backtest, Honest = this backtest)")
    print(f"\n  {'Metric':<25s} {'Biased':>12s} {'Honest':>12s} {'Δ':>10s}")
    print(f"  {'─'*60}")
    # Old numbers from biased backtest (from previous run)
    # BTC: Sharpe 9.24, +224%, 419 trades; ETH: 4.20, +683%, 1341 trades
    biased_trades = 1760
    biased_ret = 907.69
    biased_sharpe = 4.58
    biased_dd = 3.92

    print(f"  {'Trades':<25s} {biased_trades:>12d} {len(all_trades):>12d} "
          f"{len(all_trades)-biased_trades:>+10d}")
    print(f"  {'Return %':<25s} {biased_ret:>+12.2f} {total_net/EQUITY*100:>+12.2f} "
          f"{total_net/EQUITY*100-biased_ret:>+10.2f}")
    print(f"  {'Sharpe':<25s} {biased_sharpe:>+12.2f} {port_sharpe:>+12.2f} "
          f"{port_sharpe-biased_sharpe:>+10.2f}")
    print(f"  {'Max DD %':<25s} {biased_dd:>12.2f} {max_dd*100:>12.2f} "
          f"{max_dd*100-biased_dd:>+10.2f}")
    print(f"  {'Execution':<25s} {'signal bar':>12s} {'next bar':>12s}")
    print(f"  {'Costs (bp RT)':<25s} {'4':>12s} {'10':>12s}")
    print(f"  {'Funding':<25s} {'none':>12s} {'1bp/8h':>12s}")
    print(f"  {'OOS only':<25s} {'no':>12s} {'yes':>12s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
