#!/usr/bin/env python3
"""Multi-Coin Portfolio Backtest — BTC + ETH with P0 improvements.

Tests multi-asset portfolio with:
  - Per-coin 1h model signals (from respective gate_v2 models)
  - Adaptive deadzone (lower when signal is strong)
  - Long + short signals
  - Dynamic leverage 2x–3x
  - Portfolio-level metrics (combined Sharpe, drawdown, correlation)

Usage:
    python3 -m scripts.backtest_multi_coin
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
from features.batch_feature_engine import compute_4h_features, TF4H_FEATURE_NAMES
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Config ──
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
COST_BPS = 4
EQUITY = 10_000.0
RISK_FRACTION_PER_COIN = 0.05  # 5% per coin per trade
DEADZONE_BASE = {"BTCUSDT": 2.5, "ETHUSDT": 1.0}  # From respective model configs
MIN_HOLD = {"BTCUSDT": 24, "ETHUSDT": 12}
MAX_HOLD = {"BTCUSDT": 120, "ETHUSDT": 60}
LONG_ONLY = {"BTCUSDT": False, "ETHUSDT": False}
ZSCORE_WINDOW = 720

# Dynamic leverage
LEV_MIN = 2.0
LEV_MAX = 3.0
VOL_WARMUP = 168

# Adaptive deadzone
ADAPTIVE_DZ = True
DZ_DISCOUNT = 0.8
DZ_MIN = {"BTCUSDT": 2.0, "ETHUSDT": 0.8}


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def zscore_signal(pred, window=720, warmup=50):
    n = len(pred)
    z = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        buf = pred[start:i+1]
        buf = buf[~np.isnan(buf)]
        if len(buf) < warmup:
            continue
        std = np.std(buf)
        if std > 1e-12:
            z[i] = (pred[i] - np.mean(buf)) / std
    return z


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
    net_pnl: float
    leverage: float
    hold_bars: int


def run_single_coin_backtest(symbol, z_signal, closes, timestamps,
                              deadzone_base, min_hold, max_hold,
                              long_only, adaptive_dz=True,
                              dz_min=1.0):
    """Backtest single coin, return trade list and daily PnL series."""
    n = len(z_signal)
    cost_frac = COST_BPS / 10000
    trades = []
    daily_pnl = np.zeros(n)

    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = LEV_MIN
    equity = EQUITY

    for i in range(n):
        # Adaptive deadzone
        eff_dz = deadzone_base
        if adaptive_dz:
            # For single-coin, we use signal strength as proxy
            if abs(z_signal[i]) > deadzone_base * 0.5:
                eff_dz = max(deadzone_base * DZ_DISCOUNT, dz_min)

        # Exit
        if pos != 0:
            held = i - eb
            should_exit = False
            if held >= max_hold:
                should_exit = True
            elif held >= min_hold:
                if pos * z_signal[i] < -0.3 or abs(z_signal[i]) < 0.2:
                    should_exit = True
            if should_exit:
                pnl_pct = pos * (closes[i] - ep) / ep
                notional = equity * RISK_FRACTION_PER_COIN * entry_lev
                net = pnl_pct * notional - cost_frac * notional
                equity += net
                trades.append(Trade(
                    symbol=symbol, entry_bar=eb, exit_bar=i,
                    direction=int(pos), entry_price=ep, exit_price=closes[i],
                    net_pnl=net, leverage=entry_lev, hold_bars=held,
                ))
                daily_pnl[i] += net
                pos = 0.0

        # Entry
        if pos == 0:
            desired = 0
            if z_signal[i] > eff_dz:
                desired = 1
            elif not long_only and z_signal[i] < -eff_dz:
                desired = -1
            if desired != 0:
                pos = float(desired)
                ep = closes[i]
                eb = i
                start = max(0, i - 720)
                entry_lev = compute_dynamic_leverage(
                    z_signal[i], closes[start:i+1], eff_dz)

    # Close open
    if pos != 0:
        pnl_pct = pos * (closes[-1] - ep) / ep
        notional = equity * RISK_FRACTION_PER_COIN * entry_lev
        net = pnl_pct * notional - cost_frac * notional
        trades.append(Trade(
            symbol=symbol, entry_bar=eb, exit_bar=n-1,
            direction=int(pos), entry_price=ep, exit_price=closes[-1],
            net_pnl=net, leverage=entry_lev, hold_bars=n-1-eb,
        ))
        daily_pnl[n-1] += net

    return trades, daily_pnl


def analyze_coin(trades, symbol, n_bars):
    if not trades:
        return {}
    net = np.array([t.net_pnl for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    longs = sum(1 for t in trades if t.direction > 0)
    shorts = len(trades) - longs
    holds = np.array([t.hold_bars for t in trades])
    levs = np.array([t.leverage for t in trades])
    n_bars / 24

    eq = EQUITY
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    if len(net) > 2:
        avg_h = float(np.mean(holds))
        tpy = 365 * 24 / max(avg_h, 1)
        sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy))
    else:
        sharpe = 0.0

    total_net = float(np.sum(net))
    return {
        "symbol": symbol, "trades": len(trades),
        "longs": longs, "shorts": shorts,
        "win_rate": wins / len(trades) * 100,
        "sharpe": sharpe, "total_net": total_net,
        "ret_pct": total_net / EQUITY * 100,
        "max_dd": max_dd, "avg_lev": float(np.mean(levs)),
        "avg_hold": float(np.mean(holds)),
    }


def main():
    print("=" * 70)
    print("MULTI-COIN PORTFOLIO BACKTEST — BTC + ETH")
    print("P0: Adaptive deadzone + Short signals + Dynamic leverage")
    print("=" * 70)

    # ── Load data for each symbol ──
    all_data = {}
    for symbol in SYMBOLS:
        path = f"data_files/{symbol}_1h.csv"
        if not Path(path).exists():
            print(f"  SKIP: {path} not found")
            continue

        df = pd.read_csv(path)
        ts_col = "open_time" if "open_time" in df.columns else "timestamp"
        ts = df[ts_col].values.astype(np.int64)
        closes = df["close"].values.astype(np.float64)

        # Features
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
        model_dir = Path(f"models_v8/{symbol}_gate_v2")
        if not (model_dir / "lgbm_v8.pkl").exists():
            print(f"  SKIP: {model_dir}/lgbm_v8.pkl not found")
            continue

        with open(model_dir / "config.json") as f:
            cfg = json.load(f)
        with open(model_dir / "lgbm_v8.pkl", "rb") as f:
            lgbm = pickle.load(f)
        with open(model_dir / "xgb_v8.pkl", "rb") as f:
            xgb_data = pickle.load(f)

        import xgboost as xgb
        model_feats = cfg["features"]
        sel_idx = [feat_names.index(f) for f in model_feats if f in feat_names]
        X = feat_df[feat_names].values.astype(np.float64)[:, sel_idx]

        pred = 0.5 * lgbm["model"].predict(X) + \
               0.5 * xgb_data["model"].predict(xgb.DMatrix(X))

        start_date = pd.Timestamp(ts[0], unit="ms").strftime("%Y-%m-%d")
        end_date = pd.Timestamp(ts[-1], unit="ms").strftime("%Y-%m-%d")
        print(f"\n  {symbol}: {len(df):,} bars ({start_date} → {end_date}), "
              f"{len(model_feats)} features")

        all_data[symbol] = {
            "predictions": pred, "closes": closes,
            "timestamps": ts, "n_bars": len(df),
            "config": cfg,
        }

    if len(all_data) < 2:
        print("\nNeed at least 2 symbols. Exiting.")
        return

    # ── Align to common timeframe ──
    print("\n[1] Aligning timeframes...")
    # Find common time range
    min_ts = max(d["timestamps"][0] for d in all_data.values())
    max_ts = min(d["timestamps"][-1] for d in all_data.values())
    print(f"  Common range: {pd.Timestamp(min_ts, unit='ms').strftime('%Y-%m-%d')} → "
          f"{pd.Timestamp(max_ts, unit='ms').strftime('%Y-%m-%d')}")

    # Trim each symbol to common range
    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        d = all_data[symbol]
        mask = (d["timestamps"] >= min_ts) & (d["timestamps"] <= max_ts)
        d["predictions"] = d["predictions"][mask]
        d["closes"] = d["closes"][mask]
        d["timestamps"] = d["timestamps"][mask]
        d["n_bars"] = len(d["timestamps"])
        print(f"  {symbol}: {d['n_bars']:,} aligned bars")

    n_common = min(d["n_bars"] for d in all_data.values())

    # ── Z-scores ──
    print("\n[2] Computing z-scores...")
    z_scores = {}
    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        z = zscore_signal(all_data[symbol]["predictions"], window=ZSCORE_WINDOW)
        z_scores[symbol] = z
        print(f"  {symbol}: z-score range [{z.min():.2f}, {z.max():.2f}]")

    # Cross-asset correlation
    if len(z_scores) >= 2:
        symbols_list = [s for s in SYMBOLS if s in z_scores]
        corr = fast_ic(z_scores[symbols_list[0]][:n_common],
                       z_scores[symbols_list[1]][:n_common])
        print(f"  Signal correlation ({symbols_list[0]} vs {symbols_list[1]}): {corr:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # PER-COIN BACKTESTS
    # ═══════════════════════════════════════════════════════════════
    all_trades = {}
    all_daily_pnl = {}
    all_stats = {}

    for symbol in SYMBOLS:
        if symbol not in all_data:
            continue
        d = all_data[symbol]
        dz_base = DEADZONE_BASE.get(symbol, 1.0)
        mh = MIN_HOLD.get(symbol, 24)
        maxh = MAX_HOLD.get(symbol, 120)
        lo = LONG_ONLY.get(symbol, False)
        dz_min = DZ_MIN.get(symbol, dz_base * 0.8)

        print(f"\n{'='*70}")
        print(f"{symbol} — dz={dz_base}, hold=[{mh},{maxh}], long_only={lo}")
        print("=" * 70)

        trades, daily_pnl = run_single_coin_backtest(
            symbol, z_scores[symbol], d["closes"], d["timestamps"],
            dz_base, mh, maxh, lo,
            adaptive_dz=ADAPTIVE_DZ, dz_min=dz_min)

        stats = analyze_coin(trades, symbol, d["n_bars"])
        all_trades[symbol] = trades
        all_daily_pnl[symbol] = daily_pnl
        all_stats[symbol] = stats

        if stats:
            print(f"  Trades: {stats['trades']}  (L:{stats['longs']} S:{stats['shorts']})")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Sharpe: {stats['sharpe']:.2f}")
            print(f"  Return: {stats['ret_pct']:+.2f}%  (${stats['total_net']:+,.0f})")
            print(f"  Max DD: {stats['max_dd']*100:.2f}%")
            print(f"  Avg Leverage: {stats['avg_lev']:.2f}x")
            print(f"  Avg Hold: {stats['avg_hold']:.0f}h")

    # ═══════════════════════════════════════════════════════════════
    # PORTFOLIO ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("PORTFOLIO ANALYSIS — Combined BTC + ETH")
    print("=" * 70)

    # Combine all trades
    combined_trades = []
    for symbol in SYMBOLS:
        if symbol in all_trades:
            combined_trades.extend(all_trades[symbol])
    combined_trades.sort(key=lambda t: t.entry_bar)

    total_trades = len(combined_trades)
    total_net = sum(t.net_pnl for t in combined_trades)
    total_wins = sum(1 for t in combined_trades if t.net_pnl > 0)
    total_longs = sum(1 for t in combined_trades if t.direction > 0)
    total_shorts = total_trades - total_longs

    # Portfolio equity curve
    portfolio_equity = EQUITY
    portfolio_peak = EQUITY
    portfolio_max_dd = 0.0
    for t in combined_trades:
        portfolio_equity += t.net_pnl
        portfolio_peak = max(portfolio_peak, portfolio_equity)
        dd = (portfolio_peak - portfolio_equity) / portfolio_peak
        portfolio_max_dd = max(portfolio_max_dd, dd)

    # Portfolio Sharpe
    trade_nets = np.array([t.net_pnl for t in combined_trades])
    if len(trade_nets) > 2:
        avg_hold = np.mean([t.hold_bars for t in combined_trades])
        tpy = 365 * 24 / max(avg_hold, 1)
        portfolio_sharpe = float(np.mean(trade_nets) /
                                 max(np.std(trade_nets, ddof=1), 1e-10) *
                                 np.sqrt(tpy))
    else:
        portfolio_sharpe = 0.0

    days = n_common / 24
    print(f"\n  Total Trades: {total_trades}  ({total_trades/max(days,1)*7:.1f}/week)")
    print(f"  Long: {total_longs}  Short: {total_shorts}")
    print(f"  Win Rate: {total_wins/max(total_trades,1)*100:.1f}%")
    print(f"  Portfolio Sharpe: {portfolio_sharpe:.2f}")
    print(f"  Total Return: ${total_net:+,.0f}  ({total_net/EQUITY*100:+.2f}%)")
    print(f"  Max Drawdown: {portfolio_max_dd*100:.2f}%")
    print(f"  Final Equity: ${portfolio_equity:,.0f}")

    # ── Per-coin comparison ──
    print(f"\n  {'Symbol':<12s} {'Trades':>7s} {'L/S':>7s} {'WinR':>6s} "
          f"{'Sharpe':>7s} {'Return':>8s} {'MaxDD':>7s} {'AvgLev':>7s}")
    print(f"  {'─'*70}")
    for symbol in SYMBOLS:
        if symbol not in all_stats or not all_stats[symbol]:
            continue
        s = all_stats[symbol]
        ls = f"{s['longs']}/{s['shorts']}"
        print(f"  {symbol:<12s} {s['trades']:>7d} {ls:>7s} {s['win_rate']:>5.1f}% "
              f"{s['sharpe']:>+7.2f} {s['ret_pct']:>+7.2f}% "
              f"{s['max_dd']*100:>6.2f}% {s['avg_lev']:>6.2f}x")
    ls_port = f"{total_longs}/{total_shorts}"
    print(f"  {'PORTFOLIO':<12s} {total_trades:>7d} {ls_port:>7s} "
          f"{total_wins/max(total_trades,1)*100:>5.1f}% "
          f"{portfolio_sharpe:>+7.2f} {total_net/EQUITY*100:>+7.2f}% "
          f"{portfolio_max_dd*100:>6.2f}%")

    # ── Trade overlap analysis ──
    print("\n  Trade Overlap Analysis:")
    btc_trades = all_trades.get("BTCUSDT", [])
    eth_trades = all_trades.get("ETHUSDT", [])
    if btc_trades and eth_trades:
        # Count bars where both coins are in a trade
        btc_in_trade = np.zeros(n_common, dtype=bool)
        for t in btc_trades:
            btc_in_trade[t.entry_bar:min(t.exit_bar+1, n_common)] = True
        eth_in_trade = np.zeros(n_common, dtype=bool)
        for t in eth_trades:
            eth_in_trade[t.entry_bar:min(t.exit_bar+1, n_common)] = True
        both_in_trade = btc_in_trade & eth_in_trade
        print(f"    BTC in trade: {btc_in_trade.sum()} bars ({btc_in_trade.sum()/n_common*100:.1f}%)")
        print(f"    ETH in trade: {eth_in_trade.sum()} bars ({eth_in_trade.sum()/n_common*100:.1f}%)")
        print(f"    Both in trade: {both_in_trade.sum()} bars ({both_in_trade.sum()/n_common*100:.1f}%)")
        print(f"    Diversification: {100 - both_in_trade.sum()/max(btc_in_trade.sum()+eth_in_trade.sum()-both_in_trade.sum(),1)*100:.0f}%")  # noqa: E501

    # ── Monthly table ──
    print("\n  Monthly Portfolio Returns:")
    monthly = {}
    ts_ref = all_data[SYMBOLS[0]]["timestamps"]
    for t in combined_trades:
        ts_ms = int(ts_ref[min(t.exit_bar, len(ts_ref)-1)])
        month = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"net": 0.0, "trades": 0, "btc": 0.0, "eth": 0.0}
        monthly[month]["net"] += t.net_pnl
        monthly[month]["trades"] += 1
        if t.symbol == "BTCUSDT":
            monthly[month]["btc"] += t.net_pnl
        else:
            monthly[month]["eth"] += t.net_pnl

    print(f"  {'Month':>8} {'#Tr':>5} {'BTC$':>8} {'ETH$':>8} {'Total$':>8} {'Cum$':>9}")
    print(f"  {'─'*50}")
    cum = 0.0
    pos_months = 0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        if m["net"] > 0:
            pos_months += 1
        print(f"  {month:>8} {m['trades']:>5} {m['btc']:>+8.1f} {m['eth']:>+8.1f} "
              f"{m['net']:>+8.1f} {cum:>+9.1f}")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # ── vs BTC-only baseline ──
    btc_only = all_stats.get("BTCUSDT", {})
    if btc_only:
        print("\n  Improvement vs BTC-only:")
        print(f"    Trades:  {btc_only.get('trades', 0)} → {total_trades} "
              f"({total_trades/max(btc_only.get('trades',1),1):.1f}x)")
        print(f"    Return:  {btc_only.get('ret_pct', 0):+.2f}% → "
              f"{total_net/EQUITY*100:+.2f}%")
        print(f"    Sharpe:  {btc_only.get('sharpe', 0):+.2f} → {portfolio_sharpe:+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
