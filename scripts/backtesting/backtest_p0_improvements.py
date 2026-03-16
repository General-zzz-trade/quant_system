#!/usr/bin/env python3
"""P0 Improvements Backtest — Adaptive Deadzone + Short Signals.

Tests three P0 improvements on BTCUSDT full history:
  A. Baseline: current strategy (dz=2.5, long-only, dynamic lev 2-3x)
  B. Adaptive deadzone: lower dz when both 1h AND 4h z-scores agree
  C. Short signals: allow shorts when z_blend < -deadzone
  D. Combined: adaptive dz + shorts

Each variant uses the same blended z-score (50/50) and dynamic leverage.

Usage:
    python3 -m scripts.backtest_p0_improvements
"""
from __future__ import annotations
import sys
import time
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Config ──
SYMBOL = "BTCUSDT"
COST_BPS = 4
EQUITY = 10_000.0
RISK_FRACTION = 0.05

# Signal params
BLEND_W1H = 0.5
BLEND_W4H = 0.5
DEADZONE_BASE = 2.5
MIN_HOLD = 24
MAX_HOLD = 120
ZSCORE_WINDOW = 720

# Leverage
LEV_MIN = 2.0
LEV_MAX = 3.0
VOL_WARMUP = 168


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


def compute_dynamic_leverage(z_blend, closes_recent, deadzone=DEADZONE_BASE,
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
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    net_pnl: float
    leverage: float
    z_blend: float
    z_1h: float
    z_4h: float
    hold_bars: int
    deadzone_used: float


def run_backtest(
    z_blend, z_1h, z_4h, closes, timestamps,
    deadzone_base=DEADZONE_BASE,
    adaptive_dz=False,
    allow_short=False,
    dz_agreement_discount=0.6,  # dz *= 0.6 when both z agree
    dz_min=1.5,                 # Floor for adaptive deadzone
    cost_bps=COST_BPS,
):
    """Backtest with optional adaptive deadzone and shorts."""
    n = len(z_blend)
    cost_frac = cost_bps / 10000

    trades = []
    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = LEV_MIN
    entry_z = entry_z1 = entry_z4 = 0.0
    entry_dz = deadzone_base

    equity = EQUITY
    equity_curve = [equity]

    for i in range(n):
        # ── Compute effective deadzone ──
        if adaptive_dz:
            # Both z-scores agree in direction → lower deadzone
            both_long = (z_1h[i] > 0.5) and (z_4h[i] > 0.5)
            both_short = (z_1h[i] < -0.5) and (z_4h[i] < -0.5)
            if both_long or both_short:
                eff_dz = max(deadzone_base * dz_agreement_discount, dz_min)
            else:
                eff_dz = deadzone_base
        else:
            eff_dz = deadzone_base

        # ── Check exit ──
        if pos != 0:
            held = i - eb
            should_exit = False
            if held >= MAX_HOLD:
                should_exit = True
            elif held >= MIN_HOLD:
                if pos * z_blend[i] < -0.3 or abs(z_blend[i]) < 0.2:
                    should_exit = True
            if should_exit:
                pnl_pct = pos * (closes[i] - ep) / ep
                notional = equity * RISK_FRACTION * entry_lev
                gross = pnl_pct * notional
                cost = cost_frac * notional
                net = gross - cost
                equity += net
                trades.append(Trade(
                    entry_bar=eb, exit_bar=i, direction=int(pos),
                    entry_price=ep, exit_price=closes[i],
                    gross_pnl=gross, net_pnl=net,
                    leverage=entry_lev,
                    z_blend=entry_z, z_1h=entry_z1, z_4h=entry_z4,
                    hold_bars=held, deadzone_used=entry_dz,
                ))
                pos = 0.0

        # ── Check entry ──
        if pos == 0:
            desired = 0
            if z_blend[i] > eff_dz:
                desired = 1
            elif allow_short and z_blend[i] < -eff_dz:
                desired = -1

            if desired != 0:
                pos = float(desired)
                ep = closes[i]
                eb = i
                entry_z = z_blend[i]
                entry_z1 = z_1h[i]
                entry_z4 = z_4h[i]
                entry_dz = eff_dz
                start = max(0, i - 720)
                entry_lev = compute_dynamic_leverage(
                    z_blend[i], closes[start:i+1], deadzone=eff_dz)

        equity_curve.append(equity)

    # Close open
    if pos != 0:
        pnl_pct = pos * (closes[-1] - ep) / ep
        notional = equity * RISK_FRACTION * entry_lev
        gross = pnl_pct * notional
        cost = cost_frac * notional
        net = gross - cost
        equity += net
        trades.append(Trade(
            entry_bar=eb, exit_bar=n-1, direction=int(pos),
            entry_price=ep, exit_price=closes[-1],
            gross_pnl=gross, net_pnl=net,
            leverage=entry_lev,
            z_blend=entry_z, z_1h=entry_z1, z_4h=entry_z4,
            hold_bars=n-1-eb, deadzone_used=entry_dz,
        ))
        equity_curve.append(equity)

    return {"trades": trades, "equity_curve": equity_curve, "final_equity": equity}


def summarize(result, timestamps, label):
    trades = result["trades"]
    if not trades:
        print(f"\n  [{label}] No trades.")
        return {}

    net = np.array([t.net_pnl for t in trades])
    holds = np.array([t.hold_bars for t in trades])
    levs = np.array([t.leverage for t in trades])
    dzs = np.array([t.deadzone_used for t in trades])
    n_bars = len(timestamps)
    days = n_bars / 24

    wins = sum(1 for t in trades if t.net_pnl > 0)
    longs = sum(1 for t in trades if t.direction > 0)
    shorts = len(trades) - longs

    # Drawdown
    eq = EQUITY
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe
    if len(net) > 2:
        avg_hold_h = float(np.mean(holds))
        trades_per_year = 365 * 24 / max(avg_hold_h, 1)
        sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    total_net = float(np.sum(net))
    ret_pct = total_net / EQUITY * 100

    # Monthly
    monthly = {}
    for t in trades:
        ts_ms = int(timestamps[min(t.exit_bar, len(timestamps)-1)])
        month = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"net": 0.0, "trades": 0, "wins": 0}
        monthly[month]["net"] += t.net_pnl
        monthly[month]["trades"] += 1
        if t.net_pnl > 0:
            monthly[month]["wins"] += 1
    pos_months = sum(1 for m in monthly.values() if m["net"] > 0)

    # Long vs short breakdown
    long_trades = [t for t in trades if t.direction > 0]
    short_trades = [t for t in trades if t.direction < 0]

    print(f"\n  [{label}]")
    print(f"  {'─'*70}")
    print(f"  Trades: {len(trades)}  Long: {longs}  Short: {shorts}  "
          f"({len(trades)/max(days,1)*7:.1f}/week)")
    print(f"  Win Rate: {wins/len(trades)*100:.1f}%  "
          f"Avg Hold: {np.mean(holds):.0f}h")
    print(f"  Total Net: ${total_net:+,.0f}  ({ret_pct:+.2f}%)")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Avg Leverage: {np.mean(levs):.2f}x")
    if _adaptive_dz_used := (dzs < DEADZONE_BASE - 0.01).any():
        n_lowdz = (dzs < DEADZONE_BASE - 0.01).sum()
        print(f"  Adaptive DZ: {n_lowdz}/{len(trades)} trades at reduced dz "
              f"(avg dz={np.mean(dzs):.2f})")
    print(f"  Positive Months: {pos_months}/{len(monthly)}")
    print(f"  Final Equity: ${result['final_equity']:,.0f}")

    # Long vs short analysis
    if long_trades:
        l_net = [t.net_pnl for t in long_trades]
        l_wins = sum(1 for t in long_trades if t.net_pnl > 0)
        print(f"\n  Long:  {longs} trades, WR={l_wins/longs*100:.0f}%, "
              f"total=${sum(l_net):+.0f}, avg=${np.mean(l_net):+.1f}")
    if short_trades:
        s_net = [t.net_pnl for t in short_trades]
        s_wins = sum(1 for t in short_trades if t.net_pnl > 0)
        print(f"  Short: {shorts} trades, WR={s_wins/shorts*100:.0f}%, "
              f"total=${sum(s_net):+.0f}, avg=${np.mean(s_net):+.1f}")

    # Monthly table
    print(f"\n  {'Month':>8} {'#Tr':>5} {'WinR':>6} {'Net$':>8} {'CumNet$':>9}")
    print(f"  {'─'*42}")
    cum = 0.0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        print(f"  {month:>8} {m['trades']:>5} {wr:>5.0f}% {m['net']:>+8.1f} {cum:>+9.1f}")

    return {
        "trades": len(trades), "longs": longs, "shorts": shorts,
        "win_rate": wins/len(trades)*100,
        "sharpe": sharpe, "total_net": total_net, "ret_pct": ret_pct,
        "max_dd": max_dd, "avg_lev": float(np.mean(levs)),
        "pos_months": pos_months, "total_months": len(monthly),
        "final_equity": result["final_equity"],
    }


def main():
    print("=" * 70)
    print("P0 IMPROVEMENTS BACKTEST — Adaptive Deadzone + Short Signals")
    print("=" * 70)

    # ── Load data ──
    print("\n[1] Loading data...")
    t0 = time.time()
    df_1h = pd.read_csv("data_files/BTCUSDT_1h.csv")
    n_1h = len(df_1h)
    ts_col = "open_time" if "open_time" in df_1h.columns else "timestamp"
    timestamps_1h = df_1h[ts_col].values.astype(np.int64)
    closes_1h = df_1h["close"].values.astype(np.float64)

    df_1m = pd.read_csv("data_files/BTCUSDT_1m.csv")
    ts_1m = df_1m["open_time" if "open_time" in df_1m.columns else "timestamp"].values.astype(np.int64)
    groups = ts_1m // (4 * 60 * 60_000)
    work = pd.DataFrame({
        "group": groups, "open_time": ts_1m,
        "open": df_1m["open"].values.astype(np.float64),
        "high": df_1m["high"].values.astype(np.float64),
        "low": df_1m["low"].values.astype(np.float64),
        "close": df_1m["close"].values.astype(np.float64),
        "volume": df_1m["volume"].values.astype(np.float64),
    })
    for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
        work[col] = df_1m[col].values.astype(np.float64) if col in df_1m.columns else 0.0
    df_4h = work.groupby("group", sort=True).agg(
        open_time=("open_time", "first"), open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    ).reset_index(drop=True)
    timestamps_4h = df_4h["open_time"].values.astype(np.int64)
    print(f"  1h: {n_1h:,}  4h: {len(df_4h):,}  ({time.time()-t0:.1f}s)")

    # ── Features ──
    print("\n[2] Computing features...")
    t0 = time.time()
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_1h = compute_features_batch(SYMBOL, df_1h, include_v11=_has_v11)
    tf4h = compute_4h_features(df_1h)
    for col in TF4H_FEATURE_NAMES:
        feat_1h[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_1h.columns and fb in feat_1h.columns:
            feat_1h[int_name] = feat_1h[fa].astype(float) * feat_1h[fb].astype(float)
    feat_names_1h = [c for c in feat_1h.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]

    feat_4h = compute_features_batch(SYMBOL, df_4h, include_v11=_has_v11)
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_4h.columns and fb in feat_4h.columns:
            feat_4h[int_name] = feat_4h[fa].astype(float) * feat_4h[fb].astype(float)
    feat_names_4h = [c for c in feat_4h.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    print(f"  Features done ({time.time()-t0:.1f}s)")

    # ── Models ──
    print("\n[3] Loading models & generating predictions...")
    import xgboost as xgb

    with open("models_v8/BTCUSDT_gate_v2/config.json") as f:
        cfg_1h = json.load(f)
    with open("models_v8/BTCUSDT_gate_v2/lgbm_v8.pkl", "rb") as f:
        lgbm_1h = pickle.load(f)
    with open("models_v8/BTCUSDT_gate_v2/xgb_v8.pkl", "rb") as f:
        xgb_1h = pickle.load(f)

    with open("models_v8/BTCUSDT_4h_v1/config.json") as f:
        cfg_4h = json.load(f)
    with open("models_v8/BTCUSDT_4h_v1/lgbm_v8.pkl", "rb") as f:
        lgbm_4h = pickle.load(f)
    with open("models_v8/BTCUSDT_4h_v1/xgb_v8.pkl", "rb") as f:
        xgb_4h = pickle.load(f)

    # 1h predictions
    sel_1h = [feat_names_1h.index(f) for f in cfg_1h["features"] if f in feat_names_1h]
    X_1h = feat_1h[feat_names_1h].values.astype(np.float64)[:, sel_1h]
    pred_1h = 0.5 * lgbm_1h["model"].predict(X_1h) + \
              0.5 * xgb_1h["model"].predict(xgb.DMatrix(X_1h))

    # 4h predictions
    sel_4h = [feat_names_4h.index(f) for f in cfg_4h["features"] if f in feat_names_4h]
    X_4h = feat_4h[feat_names_4h].values.astype(np.float64)[:, sel_4h]
    pred_4h = 0.5 * lgbm_4h["model"].predict(X_4h) + \
              0.5 * xgb_4h["model"].predict(xgb.DMatrix(X_4h))

    # Align 4h to 1h
    pred_4h_at_1h = np.full(n_1h, np.nan)
    j = 0
    n_4h = len(pred_4h)
    for i in range(n_1h):
        while j < n_4h - 1 and timestamps_4h[j + 1] <= timestamps_1h[i]:
            j += 1
        if j < n_4h and timestamps_4h[j] <= timestamps_1h[i]:
            pred_4h_at_1h[i] = pred_4h[j]

    valid = ~np.isnan(pred_4h_at_1h)
    first_valid = int(np.argmax(valid))
    pred_1h_c = pred_1h[first_valid:]
    pred_4h_c = pred_4h_at_1h[first_valid:]
    closes_c = closes_1h[first_valid:]
    ts_c = timestamps_1h[first_valid:]
    n_c = len(pred_1h_c)
    print(f"  Period: {pd.Timestamp(ts_c[0], unit='ms').strftime('%Y-%m-%d')} → "
          f"{pd.Timestamp(ts_c[-1], unit='ms').strftime('%Y-%m-%d')} ({n_c:,} bars)")

    # ── Z-scores ──
    print("\n[4] Computing z-scores...")
    z_1h = zscore_signal(pred_1h_c, window=ZSCORE_WINDOW)
    z_4h = zscore_signal(pred_4h_c, window=ZSCORE_WINDOW)
    z_blend = BLEND_W1H * z_1h + BLEND_W4H * z_4h

    corr = fast_ic(z_1h, z_4h)
    print(f"  Z-score correlation: {corr:.4f}")

    # Signal opportunity analysis
    n_long = np.sum(z_blend > DEADZONE_BASE)
    n_short = np.sum(z_blend < -DEADZONE_BASE)
    both_agree_long = np.sum((z_1h > 0.5) & (z_4h > 0.5) & (z_blend > 1.5))
    both_agree_short = np.sum((z_1h < -0.5) & (z_4h < -0.5) & (z_blend < -1.5))
    print(f"  Long opportunities  (z_blend > {DEADZONE_BASE}): {n_long} bars ({n_long/n_c*100:.1f}%)")
    print(f"  Short opportunities (z_blend < -{DEADZONE_BASE}): {n_short} bars ({n_short/n_c*100:.1f}%)")
    print(f"  Both agree long  (z1h>0.5 & z4h>0.5, blend>1.5): {both_agree_long} bars")
    print(f"  Both agree short (z1h<-0.5 & z4h<-0.5, blend<-1.5): {both_agree_short} bars")

    # ── Return analysis by regime ──
    ret_1h = np.diff(closes_c) / closes_c[:-1]
    # Check if shorts have alpha
    short_mask = z_blend[:-1] < -DEADZONE_BASE
    long_mask = z_blend[:-1] > DEADZONE_BASE
    if short_mask.sum() > 10:
        short_ret = ret_1h[short_mask]
        print(f"\n  Short signal bars: mean ret = {np.mean(short_ret)*10000:+.2f} bp/bar "
              f"(negative = shorts would profit)")
        print(f"  If shorted: avg = {-np.mean(short_ret)*10000:+.2f} bp/bar, "
              f"win rate = {(short_ret < 0).mean()*100:.0f}%")
    if long_mask.sum() > 10:
        long_ret = ret_1h[long_mask]
        print(f"  Long signal bars: mean ret = {np.mean(long_ret)*10000:+.2f} bp/bar")

    # ═══════════════════════════════════════════════════════════════
    # RUN ALL VARIANTS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("VARIANT A: BASELINE (dz=2.5, long-only)")
    print("=" * 70)
    bt_a = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                        deadzone_base=2.5, adaptive_dz=False, allow_short=False)
    stats_a = summarize(bt_a, ts_c, "Baseline: dz=2.5, long-only")

    print("\n" + "=" * 70)
    print("VARIANT B: ADAPTIVE DEADZONE (dz=2.5→1.5 when models agree)")
    print("=" * 70)
    bt_b = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                        deadzone_base=2.5, adaptive_dz=True, allow_short=False,
                        dz_agreement_discount=0.6, dz_min=1.5)
    stats_b = summarize(bt_b, ts_c, "Adaptive DZ: 2.5→1.5, long-only")

    print("\n" + "=" * 70)
    print("VARIANT C: SHORT SIGNALS (dz=2.5, long+short)")
    print("=" * 70)
    bt_c = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                        deadzone_base=2.5, adaptive_dz=False, allow_short=True)
    stats_c = summarize(bt_c, ts_c, "Shorts enabled: dz=2.5")

    print("\n" + "=" * 70)
    print("VARIANT D: COMBINED (adaptive dz + shorts)")
    print("=" * 70)
    bt_d = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                        deadzone_base=2.5, adaptive_dz=True, allow_short=True,
                        dz_agreement_discount=0.6, dz_min=1.5)
    stats_d = summarize(bt_d, ts_c, "Combined: adaptive DZ + shorts")

    # ── Parameter sweep for adaptive deadzone ──
    print("\n" + "=" * 70)
    print("ADAPTIVE DEADZONE PARAMETER SWEEP")
    print("=" * 70)
    print(f"\n  {'Discount':>8s} {'DZ_min':>6s} {'EffDZ':>6s} {'#Tr':>5s} "
          f"{'Sharpe':>7s} {'Return':>8s} {'WinR':>6s} {'MaxDD':>7s}")
    print(f"  {'─'*60}")

    for discount in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for dz_min in [1.0, 1.5, 2.0]:
            eff_dz = max(DEADZONE_BASE * discount, dz_min)
            bt = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                              deadzone_base=2.5, adaptive_dz=True, allow_short=False,
                              dz_agreement_discount=discount, dz_min=dz_min)
            tr = bt["trades"]
            if not tr:
                continue
            n_net = np.array([t.net_pnl for t in tr])
            n_wins = sum(1 for t in tr if t.net_pnl > 0)
            avg_h = np.mean([t.hold_bars for t in tr])
            tpy = 365 * 24 / max(avg_h, 1)
            sh = float(np.mean(n_net) / max(np.std(n_net, ddof=1), 1e-10) * np.sqrt(tpy)) if len(n_net) > 2 else 0
            eq = EQUITY
            pk = eq
            mdd = 0
            for t in tr:
                eq += t.net_pnl
                pk = max(pk, eq)
                mdd = max(mdd, (pk - eq) / pk)
            ret = sum(t.net_pnl for t in tr) / EQUITY * 100
            n_lowdz = sum(1 for t in tr if t.deadzone_used < DEADZONE_BASE - 0.01)
            print(f"  {discount:>8.1f} {dz_min:>6.1f} {eff_dz:>6.1f} {len(tr):>5d} "
                  f"{sh:>+7.2f} {ret:>+7.2f}% {n_wins/len(tr)*100:>5.1f}% "
                  f"{mdd*100:>6.2f}%  (low_dz={n_lowdz})")

    # ── Short signal deep analysis ──
    print("\n" + "=" * 70)
    print("SHORT SIGNAL ANALYSIS")
    print("=" * 70)

    # Check short alpha in different market regimes
    # Define bear: 30-day return < -10%
    for label, lb in [("30d", 720), ("7d", 168)]:
        bear_mask = np.zeros(n_c, dtype=bool)
        for i in range(lb, n_c):
            ret_lb = (closes_c[i] - closes_c[i-lb]) / closes_c[i-lb]
            if ret_lb < -0.10:
                bear_mask[i] = True
        bear_pct = bear_mask.sum() / n_c * 100

        short_in_bear = (z_blend < -DEADZONE_BASE) & bear_mask
        short_not_bear = (z_blend < -DEADZONE_BASE) & ~bear_mask

        print(f"\n  Bear regime ({label} ret < -10%): {bear_mask.sum()} bars ({bear_pct:.1f}%)")
        if short_in_bear[:-1].sum() > 5:
            r = ret_1h[short_in_bear[:-1]]
            print(f"    Short in bear:     {short_in_bear.sum()} bars, "
                  f"if shorted: {-np.mean(r)*10000:+.2f} bp/bar, "
                  f"win_rate={((r<0).sum()/len(r)*100):.0f}%")
        if short_not_bear[:-1].sum() > 5:
            r = ret_1h[short_not_bear[:-1]]
            print(f"    Short NOT in bear: {short_not_bear.sum()} bars, "
                  f"if shorted: {-np.mean(r)*10000:+.2f} bp/bar, "
                  f"win_rate={((r<0).sum()/len(r)*100):.0f}%")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    all_stats = [
        ("A: Baseline", stats_a),
        ("B: Adaptive DZ", stats_b),
        ("C: +Shorts", stats_c),
        ("D: Combined", stats_d),
    ]

    print(f"\n  {'Variant':<20s} {'Sharpe':>7s} {'Return':>8s} {'Net$':>8s} "
          f"{'MaxDD':>7s} {'#Tr':>5s} {'L/S':>7s} {'WinR':>6s} {'Mo+':>5s}")
    print(f"  {'─'*80}")
    for name, s in all_stats:
        if not s:
            continue
        ls = f"{s['longs']}/{s['shorts']}"
        print(f"  {name:<20s} {s['sharpe']:>+7.2f} {s['ret_pct']:>+7.2f}% "
              f"${s['total_net']:>+7.0f} {s['max_dd']*100:>6.2f}% "
              f"{s['trades']:>5d} {ls:>7s} {s['win_rate']:>5.1f}% "
              f"{s['pos_months']:>2d}/{s['total_months']}")

    # Risk-adjusted
    print("\n  Risk-Adjusted (Sharpe / MaxDD%):")
    for name, s in all_stats:
        if not s or s['max_dd'] == 0:
            continue
        ratio = s['sharpe'] / (s['max_dd'] * 100)
        bar = "█" * int(ratio * 3)
        print(f"    {name:<20s}: {ratio:.3f} {bar}")

    # Trade frequency improvement
    if stats_a and stats_d:
        freq_boost = stats_d["trades"] / max(stats_a["trades"], 1)
        print(f"\n  Trade frequency: {stats_a['trades']} → {stats_d['trades']} "
              f"({freq_boost:.1f}x improvement)")
        print(f"  Return: {stats_a['ret_pct']:+.2f}% → {stats_d['ret_pct']:+.2f}%")
        print(f"  Sharpe: {stats_a['sharpe']:+.2f} → {stats_d['sharpe']:+.2f}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
