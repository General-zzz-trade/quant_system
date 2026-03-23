#!/usr/bin/env python3
"""Analyze WHERE and WHY the ETH V10 alpha loses money.

Loads multi-horizon ETH model, runs OOS backtest, then deeply analyzes
every losing trade to find patterns: regime, time-of-day, z-score magnitude,
holding period, volatility, win-streak overconfidence, loss clustering,
and monthly IC decay.
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
from alpha.utils import fast_ic

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.batch_feature_engine import compute_4h_features, TF4H_FEATURE_NAMES
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST

# ── Cost model (same as backtest_honest) ──
COST_BPS_RT = 8
SLIPPAGE_BPS = 2
FUNDING_BPS_PER_8H = 1.0
TOTAL_ENTRY_EXIT_COST = (COST_BPS_RT + SLIPPAGE_BPS) / 10000
EQUITY = 10_000.0
RISK_FRACTION = 0.025
ZSCORE_WINDOW = 720
EMBARGO_BARS = 48
OOS_MONTHS = 18

LEV_MIN = 2.0
LEV_MAX = 3.0
VOL_WARMUP = 168


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


def compute_dynamic_leverage(z_val, closes_recent, deadzone):
    lev_range = LEV_MAX - LEV_MIN
    if lev_range <= 0:
        return LEV_MAX
    z_excess = abs(z_val) - deadzone
    signal_ramp = min(max(z_excess / deadzone, 0.0), 1.0)
    vol_discount = 1.0
    if len(closes_recent) >= VOL_WARMUP:
        rets = np.diff(closes_recent) / closes_recent[:-1]
        current_vol = float(np.std(rets[-168:])) if len(rets) >= 168 else float(np.std(rets))
        long_vol = float(np.std(rets))
        if long_vol > 1e-12:
            vol_ratio = current_vol / long_vol
            vol_discount = 1.0 - min(max((vol_ratio - 0.8) / 0.7, 0.0), 1.0)
    return round(LEV_MIN + lev_range * signal_ramp * vol_discount, 2)


@dataclass
class Trade:
    entry_bar: int
    exit_bar: int
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    gross_pnl: float
    cost_trading: float
    cost_funding: float
    net_pnl: float
    leverage: float
    hold_bars: int
    z_at_entry: float
    # Extra fields for analysis
    entry_ts: int = 0
    exit_ts: int = 0
    atr_at_entry: float = 0.0
    vol_at_entry: float = 0.0
    prior_wins_streak: int = 0


def run_backtest(z_signal, closes, timestamps, deadzone, min_hold, max_hold):
    n = len(z_signal)
    trades = []
    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = LEV_MIN
    entry_z = 0.0
    equity = EQUITY
    win_streak = 0

    # Precompute ATR (14-bar) for the whole series
    atr = np.zeros(n)
    for i in range(1, n):
        tr = abs(closes[i] - closes[i - 1])  # simplified TR using closes only
        if i >= 14:
            atr[i] = np.mean([abs(closes[j] - closes[j - 1]) for j in range(i - 13, i + 1)])
        else:
            atr[i] = tr

    # Precompute realized vol (168-bar rolling std of returns)
    rets = np.zeros(n)
    rets[1:] = np.diff(closes) / closes[:-1]
    rvol = np.zeros(n)
    for i in range(168, n):
        rvol[i] = np.std(rets[i - 167:i + 1])

    for i in range(n - 1):
        if pos != 0:
            held = i - eb
            should_exit = False
            if held >= max_hold:
                should_exit = True
            elif held >= min_hold:
                if pos * z_signal[i] < -0.3 or abs(z_signal[i]) < 0.2:
                    should_exit = True

            if should_exit:
                exit_price = closes[i + 1]
                pnl_pct = pos * (exit_price - ep) / ep
                notional = equity * RISK_FRACTION * entry_lev
                cost_trading = TOTAL_ENTRY_EXIT_COST * notional
                n_funding = max(held // 8, 1)
                cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding
                if pos < 0:
                    cost_funding = -cost_funding
                gross = pnl_pct * notional
                net = gross - cost_trading - cost_funding
                equity += net

                t = Trade(
                    entry_bar=eb, exit_bar=i + 1, direction=int(pos),
                    entry_price=ep, exit_price=exit_price,
                    gross_pnl=gross, cost_trading=cost_trading,
                    cost_funding=cost_funding, net_pnl=net,
                    leverage=entry_lev, hold_bars=held,
                    z_at_entry=entry_z,
                    entry_ts=int(timestamps[eb]),
                    exit_ts=int(timestamps[min(i + 1, n - 1)]),
                    atr_at_entry=atr[eb],
                    vol_at_entry=rvol[eb],
                    prior_wins_streak=win_streak,
                )
                trades.append(t)

                if net > 0:
                    win_streak += 1
                else:
                    win_streak = 0
                pos = 0.0

        if pos == 0 and i + 1 < n:
            desired = 0
            if z_signal[i] > deadzone:
                desired = 1
            elif z_signal[i] < -deadzone:
                desired = -1

            if desired != 0:
                pos = float(desired)
                ep = closes[i + 1]
                eb = i + 1
                entry_z = z_signal[i]
                start = max(0, i - 720)
                entry_lev = compute_dynamic_leverage(z_signal[i], closes[start:i + 1], deadzone)

    # Close open position
    if pos != 0 and n > 0:
        exit_price = closes[-1]
        held = n - 1 - eb
        pnl_pct = pos * (exit_price - ep) / ep
        notional = equity * RISK_FRACTION * entry_lev
        cost_trading = TOTAL_ENTRY_EXIT_COST * notional
        n_funding = max(held // 8, 1)
        cost_funding = (FUNDING_BPS_PER_8H / 10000) * notional * n_funding
        if pos < 0:
            cost_funding = -cost_funding
        gross = pnl_pct * notional
        net = gross - cost_trading - cost_funding
        equity += net
        trades.append(Trade(
            entry_bar=eb, exit_bar=n - 1, direction=int(pos),
            entry_price=ep, exit_price=exit_price,
            gross_pnl=gross, cost_trading=cost_trading,
            cost_funding=cost_funding, net_pnl=net,
            leverage=entry_lev, hold_bars=held,
            z_at_entry=entry_z,
            entry_ts=int(timestamps[eb]),
            exit_ts=int(timestamps[min(n - 1, n - 1)]),
            atr_at_entry=atr[eb],
            vol_at_entry=rvol[eb],
            prior_wins_streak=win_streak,
        ))

    return trades, equity


def main():
    t_start = time.time()
    print("=" * 80)
    print("  ETH V10 ALPHA LOSS ANALYSIS")
    print("=" * 80)

    # ── Load data ──
    symbol = "ETHUSDT"
    df = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    closes = df["close"].values.astype(np.float64)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    n_total = len(df)

    model_dir = Path(f"/quant_system/models_v8/{symbol}_gate_v2")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    oos_bars = 24 * 30 * OOS_MONTHS
    oos_start = n_total - oos_bars + EMBARGO_BARS

    oos_start_date = pd.Timestamp(timestamps[oos_start], unit="ms").strftime("%Y-%m-%d")
    oos_end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n_total:,} bars")
    print(f"  OOS: bar {oos_start} ({oos_start_date}) -> {oos_end_date}")
    print(f"  OOS bars: {n_total - oos_start:,} ({(n_total - oos_start) / 24:.0f} days)")

    # ── Compute features ──
    print("\n  Computing features...")
    _has_v11 = Path("/quant_system/data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)
    feat_names = [c for c in feat_df.columns
                  if c not in ("close", "open_time", "timestamp") and c not in BLACKLIST]

    # ── Multi-horizon predictions ──
    print("  Running multi-horizon predictions (h12, h24, h48)...")
    pred_start = max(0, oos_start - ZSCORE_WINDOW)

    import xgboost as xgb

    horizon_preds = []
    horizon_labels = []
    for hm_cfg in cfg["horizon_models"]:
        h = hm_cfg["horizon"]
        lgbm_path = model_dir / hm_cfg["lgbm"]
        xgb_path = model_dir / hm_cfg["xgb"]
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)
        with open(xgb_path, "rb") as f:
            xgb_data = pickle.load(f)

        hm_feats = hm_cfg["features"]
        sel = [feat_names.index(fn) for fn in hm_feats if fn in feat_names]
        X = feat_df[feat_names].values[pred_start:].astype(np.float64)[:, sel]

        pred = 0.5 * lgbm_data["model"].predict(X) + \
               0.5 * xgb_data["model"].predict(xgb.DMatrix(X))
        horizon_preds.append(pred)
        horizon_labels.append(f"h{h}")
        print(f"    h{h}: {len(pred):,} predictions, IC features: {hm_feats[:5]}...")

    # Z-score each horizon, then ensemble mean
    z_horizons = [zscore_signal(p, window=ZSCORE_WINDOW, warmup=180) for p in horizon_preds]
    z_ensemble = np.mean(z_horizons, axis=0)

    warmup_used = oos_start - pred_start
    z_oos = z_ensemble[warmup_used:]
    closes_oos = closes[oos_start:]
    ts_oos = timestamps[oos_start:]
    n_oos = len(z_oos)

    # Also keep raw predictions for IC analysis
    raw_preds_oos = [p[warmup_used:] for p in horizon_preds]

    print(f"  OOS bars for trading: {n_oos:,}")

    # ── Config from model ──
    deadzone = cfg.get("deadzone", 0.3)
    min_hold = cfg.get("min_hold", 12)
    max_hold = cfg.get("max_hold", 96)
    print(f"  Config: dz={deadzone}, hold=[{min_hold},{max_hold}], long_only=False")

    # ── Run backtest ──
    print("\n  Running backtest...")
    trades, final_equity = run_backtest(z_oos, closes_oos, ts_oos,
                                         deadzone, min_hold, max_hold)

    winners = [t for t in trades if t.net_pnl > 0]
    losers = [t for t in trades if t.net_pnl <= 0]
    total_net = sum(t.net_pnl for t in trades)
    total_gross = sum(t.gross_pnl for t in trades)

    print(f"\n  Total trades: {len(trades)}")
    print(f"  Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)")
    print(f"  Losers:  {len(losers)} ({len(losers)/len(trades)*100:.1f}%)")
    print(f"  Gross P&L: ${total_gross:+,.2f}")
    print(f"  Net P&L:   ${total_net:+,.2f} ({total_net/EQUITY*100:+.2f}%)")
    print(f"  Final equity: ${final_equity:,.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 1: Market Regime at Entry (Trending vs Ranging, Vol)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 1: MARKET REGIME AT ENTRY")
    print(f"{'='*80}")

    # Compute regime indicators on OOS closes
    # Trend: use 50-bar SMA slope (positive = uptrend, negative = downtrend)
    # Ranging: ADX-like proxy using directional movement
    # Vol regime: 168-bar realized vol vs median vol

    sma50 = np.full(n_oos, np.nan)
    for i in range(49, n_oos):
        sma50[i] = np.mean(closes_oos[i - 49:i + 1])

    sma_slope = np.full(n_oos, np.nan)
    for i in range(59, n_oos):
        sma_slope[i] = (sma50[i] - sma50[i - 10]) / closes_oos[i] * 100  # pct

    rets_oos = np.zeros(n_oos)
    rets_oos[1:] = np.diff(closes_oos) / closes_oos[:-1]
    rvol = np.full(n_oos, np.nan)
    for i in range(167, n_oos):
        rvol[i] = np.std(rets_oos[i - 167:i + 1])

    rvol_median = np.nanmedian(rvol[167:])

    def classify_regime(bar_idx):
        slope = sma_slope[bar_idx] if bar_idx < n_oos and not np.isnan(sma_slope[bar_idx]) else 0
        vol = rvol[bar_idx] if bar_idx < n_oos and not np.isnan(rvol[bar_idx]) else rvol_median

        if abs(slope) > 0.3:
            trend = "TRENDING"
        else:
            trend = "RANGING"

        if vol > rvol_median * 1.3:
            vol_regime = "HIGH_VOL"
        elif vol < rvol_median * 0.7:
            vol_regime = "LOW_VOL"
        else:
            vol_regime = "MED_VOL"

        return trend, vol_regime

    regime_stats = {}
    for t in trades:
        trend, vol_r = classify_regime(t.entry_bar)
        key = f"{trend}_{vol_r}"
        if key not in regime_stats:
            regime_stats[key] = {"wins": 0, "losses": 0, "net": 0.0, "gross_loss": 0.0}
        if t.net_pnl > 0:
            regime_stats[key]["wins"] += 1
        else:
            regime_stats[key]["losses"] += 1
            regime_stats[key]["gross_loss"] += t.net_pnl
        regime_stats[key]["net"] += t.net_pnl

    print(f"\n  {'Regime':<25s} {'Wins':>5s} {'Loss':>5s} {'WR%':>6s} {'NetPnL$':>10s} {'GrossLoss$':>11s}")
    print(f"  {'─'*65}")
    for key in sorted(regime_stats.keys()):
        s = regime_stats[key]
        total = s["wins"] + s["losses"]
        wr = s["wins"] / total * 100 if total > 0 else 0
        print(f"  {key:<25s} {s['wins']:>5d} {s['losses']:>5d} {wr:>5.1f}% "
              f"${s['net']:>+9.2f} ${s['gross_loss']:>+10.2f}")

    # Separate analysis: losers only by regime
    print("\n  Losers by regime:")
    loser_regime = {}
    for t in losers:
        trend, vol_r = classify_regime(t.entry_bar)
        key = f"{trend}_{vol_r}"
        loser_regime.setdefault(key, []).append(t.net_pnl)
    for key in sorted(loser_regime.keys()):
        losses = loser_regime[key]
        print(f"    {key:<25s}: {len(losses):>3d} losses, avg=${np.mean(losses):+.2f}, "
              f"total=${sum(losses):+.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 2: Time-of-Day and Day-of-Week Loss Clustering
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 2: TIME-OF-DAY / DAY-OF-WEEK LOSS CLUSTERING")
    print(f"{'='*80}")

    hour_stats = {h: {"wins": 0, "losses": 0, "net": 0.0} for h in range(24)}
    dow_stats = {d: {"wins": 0, "losses": 0, "net": 0.0} for d in range(7)}

    for t in trades:
        dt = pd.Timestamp(t.entry_ts, unit="ms")
        h = dt.hour
        d = dt.dayofweek
        if t.net_pnl > 0:
            hour_stats[h]["wins"] += 1
            dow_stats[d]["wins"] += 1
        else:
            hour_stats[h]["losses"] += 1
            dow_stats[d]["losses"] += 1
        hour_stats[h]["net"] += t.net_pnl
        dow_stats[d]["net"] += t.net_pnl

    print("\n  Hour-of-day (UTC):")
    print(f"  {'Hour':>6s} {'Wins':>5s} {'Loss':>5s} {'WR%':>6s} {'Net$':>10s}")
    print(f"  {'─'*35}")
    for h in range(24):
        s = hour_stats[h]
        total = s["wins"] + s["losses"]
        if total == 0:
            continue
        wr = s["wins"] / total * 100
        marker = " <<< WORST" if s["net"] < -5 and total >= 3 else ""
        print(f"  {h:>6d} {s['wins']:>5d} {s['losses']:>5d} {wr:>5.1f}% ${s['net']:>+9.2f}{marker}")

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\n  Day-of-week:")
    print(f"  {'Day':>6s} {'Wins':>5s} {'Loss':>5s} {'WR%':>6s} {'Net$':>10s}")
    print(f"  {'─'*35}")
    for d in range(7):
        s = dow_stats[d]
        total = s["wins"] + s["losses"]
        if total == 0:
            continue
        wr = s["wins"] / total * 100
        marker = " <<< WORST" if s["net"] < -5 and total >= 3 else ""
        print(f"  {dow_names[d]:>6s} {s['wins']:>5d} {s['losses']:>5d} {wr:>5.1f}% ${s['net']:>+9.2f}{marker}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 3: Z-Score Magnitude at Entry
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 3: Z-SCORE MAGNITUDE AT ENTRY")
    print(f"{'='*80}")

    z_bins = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 5.0)]
    print(f"\n  {'Z-range':>12s} {'#Tr':>5s} {'Wins':>5s} {'WR%':>6s} {'AvgNet$':>9s} {'TotalNet$':>10s} {'AvgGross$':>10s}")  # noqa: E501
    print(f"  {'─'*65}")
    for lo, hi in z_bins:
        bin_trades = [t for t in trades if lo <= abs(t.z_at_entry) < hi]
        if not bin_trades:
            continue
        wins = sum(1 for t in bin_trades if t.net_pnl > 0)
        net_arr = [t.net_pnl for t in bin_trades]
        gross_arr = [t.gross_pnl for t in bin_trades]
        wr = wins / len(bin_trades) * 100
        print(f"  [{lo:.1f}, {hi:.1f}) {len(bin_trades):>5d} {wins:>5d} {wr:>5.1f}% "
              f"${np.mean(net_arr):>+8.2f} ${sum(net_arr):>+9.2f} ${np.mean(gross_arr):>+9.2f}")

    # Specifically for losers
    print("\n  Losers by z-score magnitude:")
    for lo, hi in z_bins:
        bin_losers = [t for t in losers if lo <= abs(t.z_at_entry) < hi]
        if not bin_losers:
            continue
        avg_loss = np.mean([t.net_pnl for t in bin_losers])
        total_loss = sum(t.net_pnl for t in bin_losers)
        print(f"    [{lo:.1f}, {hi:.1f}): {len(bin_losers):>3d} losers, avg=${avg_loss:+.2f}, total=${total_loss:+.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 4: Holding Period vs P&L
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 4: HOLDING PERIOD VS P&L")
    print(f"{'='*80}")

    hold_bins = [(0, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 200)]
    print(f"\n  {'Hold(bars)':>12s} {'#Tr':>5s} {'Wins':>5s} {'WR%':>6s} {'AvgNet$':>9s} {'TotalNet$':>10s} {'AvgCost$':>9s}")  # noqa: E501
    print(f"  {'─'*65}")
    for lo, hi in hold_bins:
        bin_trades = [t for t in trades if lo <= t.hold_bars < hi]
        if not bin_trades:
            continue
        wins = sum(1 for t in bin_trades if t.net_pnl > 0)
        nets = [t.net_pnl for t in bin_trades]
        costs = [t.cost_trading + t.cost_funding for t in bin_trades]
        wr = wins / len(bin_trades) * 100
        print(f"  [{lo:>3d},{hi:>3d}) {len(bin_trades):>5d} {wins:>5d} {wr:>5.1f}% "
              f"${np.mean(nets):>+8.2f} ${sum(nets):>+9.2f} ${np.mean(costs):>+8.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 5: Volatility at Entry
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 5: ATR / VOLATILITY AT ENTRY")
    print(f"{'='*80}")

    # Use ATR percentiles
    atrs = [t.atr_at_entry for t in trades if t.atr_at_entry > 0]
    if atrs:
        atr_q = np.percentile(atrs, [25, 50, 75])
        print(f"\n  ATR distribution: p25={atr_q[0]:.2f}, median={atr_q[1]:.2f}, p75={atr_q[2]:.2f}")

        atr_bins = [
            ("LOW ATR (p0-25)", 0, atr_q[0]),
            ("MED-LOW ATR (p25-50)", atr_q[0], atr_q[1]),
            ("MED-HIGH ATR (p50-75)", atr_q[1], atr_q[2]),
            ("HIGH ATR (p75-100)", atr_q[2], 1e10),
        ]
        print(f"\n  {'ATR bin':>25s} {'#Tr':>5s} {'Wins':>5s} {'WR%':>6s} {'AvgNet$':>9s} {'TotalNet$':>10s}")
        print(f"  {'─'*60}")
        for label, lo, hi in atr_bins:
            bt = [t for t in trades if t.atr_at_entry > 0 and lo <= t.atr_at_entry < hi]
            if not bt:
                continue
            wins = sum(1 for t in bt if t.net_pnl > 0)
            nets = [t.net_pnl for t in bt]
            wr = wins / len(bt) * 100
            print(f"  {label:>25s} {len(bt):>5d} {wins:>5d} {wr:>5.1f}% "
                  f"${np.mean(nets):>+8.2f} ${sum(nets):>+9.2f}")

    # Same for realized vol
    vols = [t.vol_at_entry for t in trades if t.vol_at_entry > 0]
    if vols:
        vol_q = np.percentile(vols, [25, 50, 75])
        print(f"\n  RVol distribution: p25={vol_q[0]:.5f}, median={vol_q[1]:.5f}, p75={vol_q[2]:.5f}")
        vol_bins = [
            ("LOW VOL (p0-25)", 0, vol_q[0]),
            ("MED-LOW VOL (p25-50)", vol_q[0], vol_q[1]),
            ("MED-HIGH VOL (p50-75)", vol_q[1], vol_q[2]),
            ("HIGH VOL (p75-100)", vol_q[2], 1e10),
        ]
        print(f"\n  {'Vol bin':>25s} {'#Tr':>5s} {'Wins':>5s} {'WR%':>6s} {'AvgNet$':>9s} {'TotalNet$':>10s}")
        print(f"  {'─'*60}")
        for label, lo, hi in vol_bins:
            bt = [t for t in trades if t.vol_at_entry > 0 and lo <= t.vol_at_entry < hi]
            if not bt:
                continue
            wins = sum(1 for t in bt if t.net_pnl > 0)
            nets = [t.net_pnl for t in bt]
            wr = wins / len(bt) * 100
            print(f"  {label:>25s} {len(bt):>5d} {wins:>5d} {wr:>5.1f}% "
                  f"${np.mean(nets):>+8.2f} ${sum(nets):>+9.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 6: Win-Streak Before Losses (Overconfidence?)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 6: WIN-STREAK BEFORE LOSSES (Overconfidence Pattern)")
    print(f"{'='*80}")

    streak_bins = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 10), (11, 100)]
    streak_labels = ["0 (cold)", "1 win", "2 wins", "3-4 wins", "5-10 wins", "11+ wins"]
    print(f"\n  {'Prior streak':>15s} {'#Losers':>8s} {'AvgLoss$':>10s} {'TotalLoss$':>11s} {'% of losses':>12s}")
    print(f"  {'─'*60}")
    for (lo, hi), label in zip(streak_bins, streak_labels):
        bin_losers = [t for t in losers if lo <= t.prior_wins_streak <= hi]
        if not bin_losers:
            continue
        avg_loss = np.mean([t.net_pnl for t in bin_losers])
        total_loss = sum(t.net_pnl for t in bin_losers)
        pct = len(bin_losers) / len(losers) * 100
        print(f"  {label:>15s} {len(bin_losers):>8d} ${avg_loss:>+9.2f} ${total_loss:>+10.2f} {pct:>10.1f}%")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 7: Loss Clustering in Time (Regime Shifts?)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 7: LOSS CLUSTERING IN TIME")
    print(f"{'='*80}")

    # Monthly P&L breakdown
    monthly = {}
    for t in trades:
        dt = pd.Timestamp(t.entry_ts, unit="ms")
        month = dt.strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"wins": 0, "losses": 0, "net": 0.0,
                              "loss_total": 0.0, "win_total": 0.0, "trades": 0}
        monthly[month]["trades"] += 1
        if t.net_pnl > 0:
            monthly[month]["wins"] += 1
            monthly[month]["win_total"] += t.net_pnl
        else:
            monthly[month]["losses"] += 1
            monthly[month]["loss_total"] += t.net_pnl
        monthly[month]["net"] += t.net_pnl

    print(f"\n  {'Month':>8s} {'#Tr':>5s} {'W':>4s} {'L':>4s} {'WR%':>6s} "
          f"{'WinPnL$':>9s} {'LossPnL$':>9s} {'Net$':>9s} {'Cum$':>10s}")
    print(f"  {'─'*72}")
    cum = 0.0
    worst_month = None
    worst_month_net = 0.0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        total = m["wins"] + m["losses"]
        wr = m["wins"] / total * 100 if total > 0 else 0
        marker = ""
        if m["net"] < worst_month_net:
            worst_month_net = m["net"]
            worst_month = month
        if m["net"] < -5:
            marker = " <<<LOSS"
        print(f"  {month:>8s} {m['trades']:>5d} {m['wins']:>4d} {m['losses']:>4d} {wr:>5.1f}% "
              f"${m['win_total']:>+8.2f} ${m['loss_total']:>+8.2f} "
              f"${m['net']:>+8.2f} ${cum:>+9.2f}{marker}")

    if worst_month:
        print(f"\n  Worst month: {worst_month} (${worst_month_net:+.2f})")

    # Consecutive losing trades
    print("\n  Consecutive losing streaks:")
    streak = 0
    max_streak = 0
    max_streak_loss = 0.0
    cur_streak_loss = 0.0
    streaks = []
    for t in trades:
        if t.net_pnl <= 0:
            streak += 1
            cur_streak_loss += t.net_pnl
        else:
            if streak > 0:
                streaks.append((streak, cur_streak_loss))
            if streak > max_streak:
                max_streak = streak
                max_streak_loss = cur_streak_loss
            streak = 0
            cur_streak_loss = 0.0
    if streak > 0:
        streaks.append((streak, cur_streak_loss))
        if streak > max_streak:
            max_streak = streak
            max_streak_loss = cur_streak_loss

    if streaks:
        streak_lens = [s[0] for s in streaks]
        [s[1] for s in streaks]
        print(f"    Total losing streaks: {len(streaks)}")
        print(f"    Max consecutive losses: {max_streak} (total ${max_streak_loss:+.2f})")
        print(f"    Avg streak length: {np.mean(streak_lens):.1f}")
        print(f"    Streaks >= 3: {sum(1 for s in streaks if s[0] >= 3)}")
        print(f"    Streaks >= 5: {sum(1 for s in streaks if s[0] >= 5)}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 8: Longs vs Shorts
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 8: LONGS VS SHORTS")
    print(f"{'='*80}")

    for dir_label, dir_val in [("LONG", 1), ("SHORT", -1)]:
        dir_trades = [t for t in trades if t.direction == dir_val]
        if not dir_trades:
            print(f"\n  {dir_label}: no trades")
            continue
        wins = sum(1 for t in dir_trades if t.net_pnl > 0)
        nets = [t.net_pnl for t in dir_trades]
        print(f"\n  {dir_label}: {len(dir_trades)} trades, {wins} wins ({wins/len(dir_trades)*100:.1f}%), "
              f"net=${sum(nets):+.2f}, avg=${np.mean(nets):+.2f}")
        dir_losers = [t for t in dir_trades if t.net_pnl <= 0]
        if dir_losers:
            print(f"    Losers: {len(dir_losers)}, avg loss=${np.mean([t.net_pnl for t in dir_losers]):+.2f}, "
                  f"worst=${min(t.net_pnl for t in dir_losers):+.2f}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 9: Information Coefficient by Month (Alpha Decay)
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 9: IC BY MONTH (Alpha Decay)")
    print(f"{'='*80}")

    # Compute forward returns for IC calculation
    fwd_rets = {}
    for h in [12, 24, 48]:
        fr = np.full(n_oos, np.nan)
        for i in range(n_oos - h):
            fr[i] = (closes_oos[i + h] - closes_oos[i]) / closes_oos[i]
        fwd_rets[h] = fr

    # Monthly IC for each horizon and ensemble
    ts_dates = pd.to_datetime(ts_oos, unit="ms")
    months_unique = sorted(set(d.strftime("%Y-%m") for d in ts_dates))

    print(f"\n  {'Month':>8s}", end="")
    for h_label in horizon_labels:
        print(f" {h_label+'_IC':>8s}", end="")
    print(f" {'ens_IC':>8s} {'n_bars':>6s}")
    print(f"  {'─'*50}")

    monthly_ics = {h: [] for h in horizon_labels + ["ensemble"]}
    for month in months_unique:
        mask = np.array([d.strftime("%Y-%m") == month for d in ts_dates])
        n_bars_month = mask.sum()
        if n_bars_month < 100:
            continue

        print(f"  {month:>8s}", end="")
        for idx, h_label in enumerate(horizon_labels):
            h_val = cfg["horizon_models"][idx]["horizon"]
            pred_vals = raw_preds_oos[idx][mask]
            fwd_vals = fwd_rets[h_val][mask]
            ic = fast_ic(pred_vals, fwd_vals)
            monthly_ics[h_label].append(ic)
            marker = "*" if (not np.isnan(ic) and abs(ic) < 0.02) else " "
            print(f" {ic:>+7.4f}{marker}", end="")

        # Ensemble IC (use h24 forward return)
        ens_pred = z_oos[mask]
        ens_fwd = fwd_rets[24][mask]
        ens_ic = fast_ic(ens_pred, ens_fwd)
        monthly_ics["ensemble"].append(ens_ic)
        marker = "*" if (not np.isnan(ens_ic) and abs(ens_ic) < 0.02) else " "
        print(f" {ens_ic:>+7.4f}{marker} {n_bars_month:>6d}")

    # IC summary
    print("\n  IC summary (* = weak IC < 0.02):")
    for key in horizon_labels + ["ensemble"]:
        vals = [v for v in monthly_ics[key] if not np.isnan(v)]
        if vals:
            neg_months = sum(1 for v in vals if v < 0)
            weak_months = sum(1 for v in vals if abs(v) < 0.02)
            print(f"    {key:>10s}: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
                  f"negative={neg_months}/{len(vals)}, weak={weak_months}/{len(vals)}")

    # ════════════════════════════════════════════════════════════════
    # ANALYSIS 10: Top-10 Worst Trades Deep Dive
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  ANALYSIS 10: TOP-10 WORST TRADES")
    print(f"{'='*80}")

    worst_10 = sorted(trades, key=lambda t: t.net_pnl)[:10]
    print(f"\n  {'#':>3s} {'Date':>12s} {'Dir':>5s} {'Z':>6s} {'Hold':>5s} "
          f"{'Entry$':>9s} {'Exit$':>9s} {'Gross$':>8s} {'Cost$':>7s} {'Net$':>8s} {'Regime':>20s}")
    print(f"  {'─'*100}")
    for i, t in enumerate(worst_10):
        dt = pd.Timestamp(t.entry_ts, unit="ms").strftime("%Y-%m-%d")
        dir_s = "LONG" if t.direction > 0 else "SHORT"
        trend, vol_r = classify_regime(t.entry_bar)
        cost = t.cost_trading + t.cost_funding
        print(f"  {i+1:>3d} {dt:>12s} {dir_s:>5s} {t.z_at_entry:>+5.2f} {t.hold_bars:>5d} "
              f"${t.entry_price:>8.2f} ${t.exit_price:>8.2f} ${t.gross_pnl:>+7.2f} "
              f"${cost:>6.2f} ${t.net_pnl:>+7.2f} {trend+'_'+vol_r:>20s}")

    # ════════════════════════════════════════════════════════════════
    # SUMMARY OF FINDINGS
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  SUMMARY: WHERE AND WHY THE ALPHA FAILS")
    print(f"{'='*80}")

    # 1. Worst regime
    worst_regime = min(regime_stats.items(), key=lambda x: x[1]["net"])
    print(f"\n  1. WORST REGIME: {worst_regime[0]}")
    print(f"     Net P&L: ${worst_regime[1]['net']:+.2f}, "
          f"WR: {worst_regime[1]['wins']/(worst_regime[1]['wins']+worst_regime[1]['losses'])*100:.1f}%")

    # 2. Worst time
    worst_hour = min(hour_stats.items(), key=lambda x: x[1]["net"])
    worst_dow = min(dow_stats.items(), key=lambda x: x[1]["net"])
    print(f"\n  2. WORST TIME: Hour {worst_hour[0]} UTC (${worst_hour[1]['net']:+.2f}), "
          f"{dow_names[worst_dow[0]]} (${worst_dow[1]['net']:+.2f})")

    # 3. Z-score finding
    extreme_z_trades = [t for t in trades if abs(t.z_at_entry) > 1.5]
    moderate_z_trades = [t for t in trades if 0.5 <= abs(t.z_at_entry) <= 1.0]
    if extreme_z_trades and moderate_z_trades:
        ext_wr = sum(1 for t in extreme_z_trades if t.net_pnl > 0) / len(extreme_z_trades) * 100
        mod_wr = sum(1 for t in moderate_z_trades if t.net_pnl > 0) / len(moderate_z_trades) * 100
        ext_avg = np.mean([t.net_pnl for t in extreme_z_trades])
        mod_avg = np.mean([t.net_pnl for t in moderate_z_trades])
        print("\n  3. Z-SCORE EFFECT:")
        print(f"     Extreme (|z|>1.5): {len(extreme_z_trades)} trades, WR={ext_wr:.1f}%, avg=${ext_avg:+.2f}")
        print(f"     Moderate (0.5-1.0): {len(moderate_z_trades)} trades, WR={mod_wr:.1f}%, avg=${mod_avg:+.2f}")

    # 4. Hold period
    short_holds = [t for t in trades if t.hold_bars <= 24]
    long_holds = [t for t in trades if t.hold_bars >= 72]
    if short_holds and long_holds:
        sh_wr = sum(1 for t in short_holds if t.net_pnl > 0) / len(short_holds) * 100
        lh_wr = sum(1 for t in long_holds if t.net_pnl > 0) / len(long_holds) * 100
        sh_avg = np.mean([t.net_pnl for t in short_holds])
        lh_avg = np.mean([t.net_pnl for t in long_holds])
        print("\n  4. HOLDING PERIOD EFFECT:")
        print(f"     Short (<=24h): {len(short_holds)} trades, WR={sh_wr:.1f}%, avg=${sh_avg:+.2f}")
        print(f"     Long (>=72h): {len(long_holds)} trades, WR={lh_wr:.1f}%, avg=${lh_avg:+.2f}")

    # 5. Longs vs Shorts
    longs = [t for t in trades if t.direction > 0]
    shorts = [t for t in trades if t.direction < 0]
    if longs and shorts:
        l_wr = sum(1 for t in longs if t.net_pnl > 0) / len(longs) * 100
        s_wr = sum(1 for t in shorts if t.net_pnl > 0) / len(shorts) * 100
        print("\n  5. DIRECTION BIAS:")
        print(f"     Longs: {len(longs)} trades, WR={l_wr:.1f}%, net=${sum(t.net_pnl for t in longs):+.2f}")
        print(f"     Shorts: {len(shorts)} trades, WR={s_wr:.1f}%, net=${sum(t.net_pnl for t in shorts):+.2f}")

    # 6. Alpha decay
    ens_ics = [v for v in monthly_ics["ensemble"] if not np.isnan(v)]
    if len(ens_ics) >= 4:
        first_half = ens_ics[:len(ens_ics)//2]
        second_half = ens_ics[len(ens_ics)//2:]
        print("\n  6. ALPHA DECAY:")
        print(f"     First half IC: {np.mean(first_half):+.4f}")
        print(f"     Second half IC: {np.mean(second_half):+.4f}")
        decay = np.mean(second_half) - np.mean(first_half)
        print(f"     Decay: {decay:+.4f} ({'DECAYING' if decay < -0.01 else 'STABLE' if abs(decay) < 0.01 else 'IMPROVING'})")  # noqa: E501

    # 7. Win-streak -> loss
    if losers:
        streak_losers = [t for t in losers if t.prior_wins_streak >= 3]
        cold_losers = [t for t in losers if t.prior_wins_streak == 0]
        if streak_losers and cold_losers:
            print("\n  7. OVERCONFIDENCE AFTER WINS:")
            print(f"     After 3+ wins: {len(streak_losers)} losses, avg=${np.mean([t.net_pnl for t in streak_losers]):+.2f}")  # noqa: E501
            print(f"     After 0 wins:  {len(cold_losers)} losses, avg=${np.mean([t.net_pnl for t in cold_losers]):+.2f}")  # noqa: E501

    # 8. Loss clustering
    print("\n  8. LOSS CLUSTERING:")
    print(f"     Max consecutive losses: {max_streak}")
    print(f"     Worst month: {worst_month} (${worst_month_net:+.2f})")
    neg_month_count = sum(1 for m in monthly.values() if m["net"] < 0)
    print(f"     Negative months: {neg_month_count}/{len(monthly)}")

    # 9. Cost impact
    total_cost = sum(t.cost_trading + t.cost_funding for t in trades)
    trades_that_would_win = [t for t in losers if t.gross_pnl > 0]
    print("\n  9. COST IMPACT:")
    print(f"     Total costs: ${total_cost:+.2f}")
    print(f"     Losers whose GROSS was positive (killed by costs): {len(trades_that_would_win)}")
    if trades_that_would_win:
        print(f"     Those trades' net: ${sum(t.net_pnl for t in trades_that_would_win):+.2f}")

    elapsed = time.time() - t_start
    print(f"\n  Analysis completed in {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
