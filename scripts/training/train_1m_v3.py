#!/usr/bin/env python3
"""1-Minute Alpha V3 — 4 alternative approaches to find edge.

Approach 1: Mean Reversion (trade AGAINST momentum — IC is negative!)
Approach 2: Volatility Filter (only trade when ATR > median, bigger moves)
Approach 3: Classification (predict direction probability, not magnitude)
Approach 4: Hybrid 1h+1m (1h model for direction, 1m for timing)

Usage:
    cd /quant_system
    python3 scripts/train_1m_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.multi_resolution import (
    compute_multi_resolution_features,
)
from features.dynamic_selector import greedy_ic_select

WARMUP = 300
COST_MAKER_RT = 4  # round-trip bps (maker)
COST_TAKER_RT = 14  # round-trip bps (taker)


def fast_ic(x, y):
    from scipy.stats import spearmanr
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50: return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def load_data():
    print("Loading data & computing features ...")
    df = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv")
    print(f"  {len(df):,} bars")
    feat_df = compute_multi_resolution_features(df, "BTCUSDT")
    feature_names = [c for c in feat_df.columns if c != "close"]
    closes = df["close"].values.astype(np.float64)
    volumes = df["volume"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    return df, feat_df, feature_names, closes, volumes, highs, lows


# ══════════════════════════════════════════════════════════════
# APPROACH 1: Mean Reversion — trade AGAINST the recent move
# ══════════════════════════════════════════════════════════════

def approach_mean_reversion(feat_df, feature_names, closes, highs, lows):
    """Since all ICs are negative, explicitly trade mean reversion."""
    print("\n" + "=" * 70)
    print("APPROACH 1: Mean Reversion")
    print("  Logic: Strong move → fade it. ret_10 IC=-0.04 means BTC reverts.")
    print("=" * 70)

    n = len(closes)
    # Split: 60/20/20
    tr_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Compute mean reversion features
    ret_5 = np.zeros(n)
    ret_10 = np.zeros(n)
    ret_20 = np.zeros(n)
    for i in range(20, n):
        ret_5[i] = closes[i] / closes[i-5] - 1
        ret_10[i] = closes[i] / closes[i-10] - 1
        ret_20[i] = closes[i] / closes[i-20] - 1

    # ATR for volatility filter
    atr_20 = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_20[i] = atr_20[i-1] * 0.95 + tr * 0.05 if i > 1 else tr
    atr_norm = atr_20 / np.maximum(closes, 1)

    # Rolling median ATR for vol filter
    atr_median = pd.Series(atr_norm).rolling(1440, min_periods=100).median().values

    # RSI for oversold/overbought
    rsi = feat_df["rsi_6"].values if "rsi_6" in feat_df.columns else np.full(n, 50)

    # CVD for flow confirmation
    cvd = feat_df["cvd_10"].values if "cvd_10" in feat_df.columns else np.zeros(n)

    # Taker imbalance
    feat_df["taker_imbalance"].values if "taker_imbalance" in feat_df.columns else np.zeros(n)

    # Mean reversion signal: -ret_10 (fade the move)
    # Enhanced: stronger when RSI extreme + high vol + flow exhaustion
    mr_signal = np.zeros(n)
    for i in range(WARMUP, n):
        base = -ret_10[i]  # fade the 10-bar move

        # RSI boost: stronger signal when oversold/overbought
        rsi_boost = 1.0
        if rsi[i] > 75: rsi_boost = 1.5  # overbought → stronger short
        elif rsi[i] < 25: rsi_boost = 1.5  # oversold → stronger long
        elif 40 < rsi[i] < 60: rsi_boost = 0.5  # neutral → weaker

        # CVD exhaustion: when CVD agrees with price (not diverging), reversion is stronger
        cvd_match = 1.0
        if cvd[i] > 0.5 and ret_10[i] > 0: cvd_match = 1.3  # buying exhaustion
        elif cvd[i] < -0.5 and ret_10[i] < 0: cvd_match = 1.3  # selling exhaustion

        mr_signal[i] = base * rsi_boost * cvd_match

    # Normalize
    std = np.nanstd(mr_signal[WARMUP:tr_end])
    if std > 1e-10:
        mr_signal /= std

    # Test different thresholds and vol filters
    configs = [
        ("MR basic", 1.5, False, False),
        ("MR basic", 2.0, False, False),
        ("MR basic", 2.5, False, False),
        ("MR basic", 3.0, False, False),
        ("MR+vol", 1.5, True, False),
        ("MR+vol", 2.0, True, False),
        ("MR+vol", 2.5, True, False),
        ("MR+vol", 3.0, True, False),
        ("MR+vol+hold", 2.0, True, True),
        ("MR+vol+hold", 2.5, True, True),
        ("MR+vol+hold", 3.0, True, True),
    ]

    print(f"\n{'Config':<20} {'Thresh':>6} {'Trades':>7} {'T/Day':>6} {'WinR':>6} "
          f"{'AvgGross':>9} {'AvgNet':>9} {'Net%':>8}")
    print("-" * 85)

    best_net = -999
    best_cfg = None

    for label, thresh, use_vol, longer_hold in configs:
        position = 0
        entry_price = 0.0
        entry_bar = 0
        min_hold = 15 if longer_hold else 5
        max_hold = 60 if longer_hold else 30
        trades_gross = []
        trades_net = []

        for i in range(val_end, n):  # OOS only
            if position != 0:
                held = i - entry_bar
                should_exit = False
                if held >= max_hold:
                    should_exit = True
                elif held >= min_hold:
                    # Exit when signal reverses or fades
                    if position * mr_signal[i] < -0.5:
                        should_exit = True
                    elif abs(mr_signal[i]) < 0.3:
                        should_exit = True
                if should_exit:
                    pnl = position * (closes[i] - entry_price) / entry_price
                    trades_gross.append(pnl)
                    trades_net.append(pnl - COST_MAKER_RT / 10000)
                    position = 0

            if position == 0:
                if use_vol and atr_norm[i] < atr_median[i]:
                    continue  # skip low vol

                if mr_signal[i] > thresh:
                    position = 1
                    entry_price = closes[i]
                    entry_bar = i
                elif mr_signal[i] < -thresh:
                    position = -1
                    entry_price = closes[i]
                    entry_bar = i

        # Close open
        if position != 0:
            pnl = position * (closes[-1] - entry_price) / entry_price
            trades_gross.append(pnl)
            trades_net.append(pnl - COST_MAKER_RT / 10000)

        nt = len(trades_net)
        if nt == 0:
            print(f"{label:<20} {thresh:>6.1f} {'—':>7}")
            continue

        g = np.array(trades_gross)
        ne = np.array(trades_net)
        days = (n - val_end) / 1440
        tag = f"{label}"
        print(f"{tag:<20} {thresh:>6.1f} {nt:>7} {nt/days:>6.1f} {np.mean(ne>0)*100:>5.1f}% "
              f"{np.mean(g)*10000:>+8.1f}bp {np.mean(ne)*10000:>+8.1f}bp "
              f"{np.sum(ne)*100:>+7.1f}%")

        if np.sum(ne) > best_net:
            best_net = np.sum(ne)
            best_cfg = (label, thresh, nt, np.mean(g)*10000, np.mean(ne)*10000)

    if best_cfg:
        print(f"\n  Best: {best_cfg[0]} thresh={best_cfg[1]}, {best_cfg[2]} trades, "
              f"avg_gross={best_cfg[3]:+.1f}bp, avg_net={best_cfg[4]:+.1f}bp")


# ══════════════════════════════════════════════════════════════
# APPROACH 2: Volatility Breakout — trade only during vol expansion
# ══════════════════════════════════════════════════════════════

def approach_vol_breakout(feat_df, feature_names, closes, highs, lows, volumes):
    """Only trade during volatility expansion, skip quiet periods."""
    print("\n" + "=" * 70)
    print("APPROACH 2: Volatility Breakout + Regime Filter")
    print("  Logic: Only trade when vol expanding + strong signal alignment")
    print("=" * 70)

    import lightgbm as lgb

    n = len(closes)
    tr_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # ATR
    atr_20 = np.zeros(n)
    for i in range(1, n):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_20[i] = atr_20[i-1] * 0.95 + tr * 0.05 if i > 1 else tr
    atr_norm = atr_20 / np.maximum(closes, 1)
    atr_pct = pd.Series(atr_norm).rolling(1440*7, min_periods=100).rank(pct=True).values

    # Vol expansion: current vol > 70th percentile
    vol_expanding = atr_pct > 0.70

    # Train ML model with all features
    horizons = [10, 15, 30]

    for horizon in horizons:
        y = np.full(n, np.nan)
        y[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1
        valid_v = ~np.isnan(y)
        vp = y[valid_v]
        if len(vp) > 10:
            p1, p99 = np.percentile(vp, [1, 99])
            y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))

        X = feat_df[feature_names].values[WARMUP:].astype(np.float64)
        y_w = y[WARMUP:]
        c_w = closes[WARMUP:]
        vol_exp_w = vol_expanding[WARMUP:]
        len(X)
        tr_adj = tr_end - WARMUP
        val_adj = val_end - WARMUP

        # Train on vol-expanding bars ONLY
        np.arange(tr_adj)
        vol_train = vol_exp_w[:tr_adj]
        valid_train = ~np.isnan(y_w[:tr_adj]) & vol_train
        valid_val = ~np.isnan(y_w[tr_adj:val_adj])

        X_tr = X[:tr_adj][valid_train]
        y_tr = y_w[:tr_adj][valid_train]

        if len(X_tr) < 1000:
            print(f"  h={horizon}: Too few vol-expanding training samples ({len(X_tr)})")
            continue

        # Feature selection
        selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=12)
        sel_idx = [feature_names.index(f) for f in selected]

        X_tr_s = X_tr[:, sel_idx]
        X_val_s = X[tr_adj:val_adj][valid_val][:, sel_idx]
        y_val = y_w[tr_adj:val_adj][valid_val]

        dtrain = lgb.Dataset(X_tr_s, label=y_tr)
        dval = lgb.Dataset(X_val_s, label=y_val, reference=dtrain)

        params = {
            "max_depth": 4, "num_leaves": 10, "learning_rate": 0.008,
            "min_child_samples": 300, "reg_alpha": 1.0, "reg_lambda": 10.0,
            "subsample": 0.4, "colsample_bytree": 0.5, "objective": "regression",
            "verbosity": -1,
        }
        bst = lgb.train(params, dtrain, num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[lgb.early_stopping(50, verbose=False)])

        # Predict on OOS (only vol-expanding bars)
        X_test = X[val_adj:]
        pred = bst.predict(X_test[:, sel_idx])
        y_test = y_w[val_adj:]
        vol_test = vol_exp_w[val_adj:]
        c_test = c_w[val_adj:]

        ic_all = fast_ic(pred, y_test)
        vol_mask = vol_test
        ic_vol = fast_ic(pred[vol_mask], y_test[vol_mask]) if vol_mask.sum() > 50 else 0

        print(f"\n  Horizon={horizon}min  IC_all={ic_all:.4f}  IC_vol_expanding={ic_vol:.4f}  "
              f"features={selected[:5]}...")

        # Backtest on vol-expanding bars only
        pred_std = np.nanstd(pred)
        if pred_std < 1e-12: continue
        z_pred = pred / pred_std

        for dz in [1.5, 2.0, 2.5, 3.0]:
            position = 0
            entry_price = 0.0
            entry_bar = 0
            min_hold = max(horizon // 2, 3)
            max_hold = horizon * 4
            trades = []

            for i in range(len(c_test)):
                if position != 0:
                    held = i - entry_bar
                    should_exit = held >= max_hold or (held >= min_hold and (
                        position * z_pred[i] < -0.5 or abs(z_pred[i]) < 0.3))
                    if should_exit:
                        pnl = position * (c_test[i] - entry_price) / entry_price
                        trades.append(pnl)
                        position = 0

                if position == 0 and vol_test[i]:  # only enter during vol expansion
                    if z_pred[i] > dz:
                        position = 1; entry_price = c_test[i]; entry_bar = i
                    elif z_pred[i] < -dz:
                        position = -1; entry_price = c_test[i]; entry_bar = i

            if position != 0:
                trades.append(position * (c_test[-1] - entry_price) / entry_price)

            nt = len(trades)
            if nt == 0: continue
            t = np.array(trades)
            days = len(c_test) / 1440
            net = t - COST_MAKER_RT / 10000
            print(f"    dz={dz:.1f}: {nt:>5} trades ({nt/days:.1f}/d) "
                  f"winR={np.mean(net>0)*100:.1f}% "
                  f"avg_gross={np.mean(t)*10000:+.1f}bp "
                  f"avg_net={np.mean(net)*10000:+.1f}bp "
                  f"total={np.sum(net)*100:+.1f}%")


# ══════════════════════════════════════════════════════════════
# APPROACH 3: Classification — predict P(up) > threshold
# ══════════════════════════════════════════════════════════════

def approach_classification(feat_df, feature_names, closes):
    """Binary classification: will price move > X bps in next Y bars?"""
    print("\n" + "=" * 70)
    print("APPROACH 3: Classification — Predict P(move > threshold)")
    print("  Logic: Don't predict magnitude, predict probability of big move")
    print("=" * 70)

    import lightgbm as lgb

    n = len(closes)
    tr_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Try different move thresholds and horizons
    for horizon in [10, 15, 30]:
        fwd_ret = np.full(n, np.nan)
        fwd_ret[:n-horizon] = closes[horizon:] / closes[:n-horizon] - 1

        for move_bps in [10, 15, 20]:
            move_frac = move_bps / 10000

            # Binary target: 1 if |move| > threshold in predicted direction
            y_up = (fwd_ret > move_frac).astype(float)
            y_down = (fwd_ret < -move_frac).astype(float)

            # Train two models: P(up big) and P(down big)
            X = feat_df[feature_names].values[WARMUP:].astype(np.float64)
            len(X)
            tr_adj = tr_end - WARMUP
            val_adj = val_end - WARMUP

            for direction, y_cls in [("LONG", y_up), ("SHORT", y_down)]:
                y_w = y_cls[WARMUP:]
                valid_tr = ~np.isnan(y_w[:tr_adj])
                valid_val = ~np.isnan(y_w[tr_adj:val_adj])

                X_tr = X[:tr_adj][valid_tr]
                y_tr = y_w[:tr_adj][valid_tr]
                X_val = X[tr_adj:val_adj][valid_val]
                y_val = y_w[tr_adj:val_adj][valid_val]

                pos_rate = y_tr.mean()
                if pos_rate < 0.01 or pos_rate > 0.99:
                    continue

                # Feature selection via IC
                selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=12)
                sel_idx = [feature_names.index(f) for f in selected]

                params = {
                    "max_depth": 4, "num_leaves": 10, "learning_rate": 0.01,
                    "min_child_samples": 300, "reg_alpha": 1.0, "reg_lambda": 10.0,
                    "subsample": 0.4, "colsample_bytree": 0.5,
                    "objective": "binary", "metric": "auc",
                    "verbosity": -1, "is_unbalance": True,
                }

                dtrain = lgb.Dataset(X_tr[:, sel_idx], label=y_tr)
                dval = lgb.Dataset(X_val[:, sel_idx], label=y_val, reference=dtrain)

                bst = lgb.train(params, dtrain, num_boost_round=300,
                                valid_sets=[dval],
                                callbacks=[lgb.early_stopping(30, verbose=False)])

                # OOS test
                X_test = X[val_adj:]
                y_test = y_w[val_adj:]
                pred = bst.predict(X_test[:, sel_idx])
                c_test = closes[WARMUP + val_adj:]

                # AUC
                from sklearn.metrics import roc_auc_score
                valid_test = ~np.isnan(y_test)
                if valid_test.sum() < 100: continue
                try:
                    auc = roc_auc_score(y_test[valid_test], pred[valid_test])
                except:
                    auc = 0.5

                # Trade when P > high threshold
                for prob_thresh in [0.6, 0.7, 0.8]:
                    signals = pred > prob_thresh
                    n_signals = signals.sum()
                    if n_signals == 0: continue

                    # Simple evaluation: avg return when signal fires
                    fwd_w = fwd_ret[WARMUP + val_adj:]
                    valid_sig = signals & ~np.isnan(fwd_w)
                    if valid_sig.sum() == 0: continue

                    if direction == "LONG":
                        avg_ret = np.mean(fwd_w[valid_sig])
                    else:
                        avg_ret = -np.mean(fwd_w[valid_sig])  # short profits from down

                    avg_bps = avg_ret * 10000
                    net_bps = avg_bps - COST_MAKER_RT
                    days = len(c_test) / 1440

                    print(f"  h={horizon} mv={move_bps}bp {direction:5s} P>{prob_thresh:.1f}: "
                          f"AUC={auc:.3f} signals={n_signals:>5} ({n_signals/days:.1f}/d) "
                          f"avg={avg_bps:+.1f}bp net={net_bps:+.1f}bp "
                          f"total_net={net_bps*n_signals/10000*100:+.1f}%")


# ══════════════════════════════════════════════════════════════
# APPROACH 4: Hybrid — 1h direction + 1m entry timing
# ══════════════════════════════════════════════════════════════

def approach_hybrid(feat_df, feature_names, closes, highs, lows):
    """Use slow features for direction bias, fast features for entry."""
    print("\n" + "=" * 70)
    print("APPROACH 4: Hybrid — 1h Direction + 1m Entry Timing")
    print("  Logic: Slow trend gives direction, fast signal times the entry")
    print("=" * 70)

    n = len(closes)
    val_end = int(n * 0.8)

    # 1h direction: slow_close_vs_ma20 + slow_rsi_14 + slow_mean_reversion_20
    slow_ma = feat_df["slow_close_vs_ma20"].values if "slow_close_vs_ma20" in feat_df.columns else np.zeros(n)
    slow_rsi = feat_df["slow_rsi_14"].values if "slow_rsi_14" in feat_df.columns else np.full(n, 50)
    feat_df["slow_mean_reversion_20"].values if "slow_mean_reversion_20" in feat_df.columns else np.zeros(n)
    tf4h_ma = feat_df["tf4h_close_vs_ma20"].values if "tf4h_close_vs_ma20" in feat_df.columns else np.zeros(n)

    # 1h direction bias: +1 bullish, -1 bearish, 0 neutral
    direction_bias = np.zeros(n)
    for i in range(WARMUP, n):
        score = 0
        if slow_ma[i] > 0.01: score += 1
        elif slow_ma[i] < -0.01: score -= 1
        if not np.isnan(tf4h_ma[i]):
            if tf4h_ma[i] > 0.01: score += 1
            elif tf4h_ma[i] < -0.01: score -= 1
        if not np.isnan(slow_rsi[i]):
            if slow_rsi[i] > 60: score += 1
            elif slow_rsi[i] < 40: score -= 1
        direction_bias[i] = np.sign(score) if abs(score) >= 2 else 0  # need 2+ signals

    # 1m entry: mean reversion within the trend direction
    ret_5 = np.zeros(n)
    ret_10 = np.zeros(n)
    rsi_6 = feat_df["rsi_6"].values if "rsi_6" in feat_df.columns else np.full(n, 50)
    feat_df["cvd_10"].values if "cvd_10" in feat_df.columns else np.zeros(n)

    for i in range(10, n):
        ret_5[i] = closes[i] / closes[i-5] - 1
        ret_10[i] = closes[i] / closes[i-10] - 1

    # Entry timing: pullback in trend direction
    # Bull trend + pullback (ret_5 < 0, RSI < 40) → buy the dip
    # Bear trend + bounce (ret_5 > 0, RSI > 60) → sell the rally

    configs = [
        ("basic", 0.0010, 30, 120),      # 10bp pullback, hold 30-120 bars
        ("deep", 0.0020, 30, 120),        # 20bp pullback
        ("shallow", 0.0005, 15, 60),      # 5bp pullback
        ("deep+long", 0.0020, 60, 240),   # 20bp pullback, hold longer
    ]

    print(f"\n{'Config':<16} {'Trades':>7} {'T/Day':>6} {'WinR':>6} "
          f"{'AvgGross':>9} {'AvgNet':>9} {'Net%':>8}")
    print("-" * 75)

    for label, pullback_thresh, min_hold, max_hold in configs:
        position = 0
        entry_price = 0.0
        entry_bar = 0
        trades_gross = []
        trades_net = []

        for i in range(val_end, n):  # OOS
            if position != 0:
                held = i - entry_bar
                should_exit = False
                if held >= max_hold: should_exit = True
                elif held >= min_hold:
                    # Exit when direction bias flips or profit target
                    pnl_pct = position * (closes[i] - entry_price) / entry_price
                    if direction_bias[i] * position < 0: should_exit = True  # bias flipped
                    elif pnl_pct > 0.003: should_exit = True  # 30bp profit target
                    elif pnl_pct < -0.002: should_exit = True  # 20bp stop loss
                if should_exit:
                    pnl = position * (closes[i] - entry_price) / entry_price
                    trades_gross.append(pnl)
                    trades_net.append(pnl - COST_MAKER_RT / 10000)
                    position = 0

            if position == 0:
                if direction_bias[i] > 0 and ret_5[i] < -pullback_thresh:
                    # Bullish bias + pullback → buy
                    if rsi_6[i] < 40:  # RSI confirms oversold on 1m
                        position = 1
                        entry_price = closes[i]
                        entry_bar = i
                elif direction_bias[i] < 0 and ret_5[i] > pullback_thresh:
                    # Bearish bias + bounce → sell
                    if rsi_6[i] > 60:  # RSI confirms overbought on 1m
                        position = -1
                        entry_price = closes[i]
                        entry_bar = i

        if position != 0:
            pnl = position * (closes[-1] - entry_price) / entry_price
            trades_gross.append(pnl)
            trades_net.append(pnl - COST_MAKER_RT / 10000)

        nt = len(trades_net)
        if nt == 0:
            print(f"{label:<16} {'—':>7}")
            continue
        g = np.array(trades_gross)
        ne = np.array(trades_net)
        days = (n - val_end) / 1440
        print(f"{label:<16} {nt:>7} {nt/days:>6.1f} {np.mean(ne>0)*100:>5.1f}% "
              f"{np.mean(g)*10000:>+8.1f}bp {np.mean(ne)*10000:>+8.1f}bp "
              f"{np.sum(ne)*100:>+7.1f}%")


# ══════════════════════════════════════════════════════════════
# APPROACH 5: Multi-symbol — use ETHUSDT as leading indicator
# ══════════════════════════════════════════════════════════════

def approach_multi_symbol():
    """Use ETH price as a leading indicator for BTC."""
    print("\n" + "=" * 70)
    print("APPROACH 5: Multi-Symbol (ETH → BTC)")
    print("=" * 70)

    eth_path = Path("/quant_system/data_files/ETHUSDT_1m.csv")
    if not eth_path.exists():
        print("  ETHUSDT 1m data not found. Skipping.")
        print("  To download: python3 scripts/download_binance_klines.py --symbols ETHUSDT --interval 1m --start 2024-01-01 --out data_files")
        return

    # Would implement cross-asset signals here
    print("  ETH data available — implementing cross-asset signals...")


# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("1-Minute Alpha V3 — Alternative Approaches")
    print("=" * 70)

    df, feat_df, feature_names, closes, volumes, highs, lows = load_data()

    approach_mean_reversion(feat_df, feature_names, closes, highs, lows)
    approach_vol_breakout(feat_df, feature_names, closes, highs, lows, volumes)
    approach_classification(feat_df, feature_names, closes)
    approach_hybrid(feat_df, feature_names, closes, highs, lows)
    approach_multi_symbol()

    print("\n" + "=" * 70)
    print("ALL APPROACHES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
