#!/usr/bin/env python3
"""Multi-Timeframe Ensemble Backtest — 1h + 4h signal fusion.

Tests fusion methods on overlapping OOS period (2024-09 → 2026-03):
  1. Baseline: 1h only, 4h only (independent)
  2. Cascade: 4h regime filter → only trade 1h when 4h agrees
  3. Weighted vote: blend z-scores (w1h * z1h + w4h * z4h)
  4. Agreement: only trade when both signals same direction
  5. Confidence-scaled: agreement → 1.5x size, disagreement → 0.5x

Usage:
    python3 -m scripts.backtest_multi_tf
"""
from __future__ import annotations
import sys, time, json, pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from features.dynamic_selector import _rankdata, _spearman_ic
from scripts.signal_postprocess import should_exit_position
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scipy.stats import spearmanr

# ── Config ──
SYMBOL = "BTCUSDT"
COST_BPS = 6e-4  # 6 bps (maker + slippage)


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


# ── Signal Generation ──

def zscore_signal(pred: np.ndarray, window: int = 720) -> np.ndarray:
    """Rolling z-score normalization of raw predictions."""
    n = len(pred)
    z = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        buf = pred[start:i+1]
        buf = buf[~np.isnan(buf)]
        if len(buf) < 50:
            z[i] = 0.0
        else:
            std = np.std(buf)
            if std > 1e-12:
                z[i] = (pred[i] - np.mean(buf)) / std
            else:
                z[i] = 0.0
    return z


def apply_signal(z: np.ndarray, deadzone: float, min_hold: int,
                 max_hold: int, long_only: bool = True) -> np.ndarray:
    """Z-score → position signal with hold constraints."""
    n = len(z)
    pos = np.zeros(n)
    current = 0.0
    entry_bar = 0

    for i in range(n):
        if current != 0:
            held = i - entry_bar
            if should_exit_position(
                position=current,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            ):
                current = 0.0
        if current == 0:
            if z[i] > deadzone:
                current = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                current = -1.0
                entry_bar = i
        pos[i] = current
    return pos


def backtest_pnl(signal: np.ndarray, closes: np.ndarray,
                 cost_per_trade: float = COST_BPS) -> Dict[str, Any]:
    """Compute PnL from signal array."""
    ret_1bar = np.diff(closes) / closes[:-1]
    sig = signal[:len(ret_1bar)]
    turnover = np.abs(np.diff(sig, prepend=0))
    gross = sig * ret_1bar
    cost = turnover * cost_per_trade
    net = gross - cost

    active = sig != 0
    n_active = int(active.sum())

    # Count trades
    entries = (sig != 0) & (np.concatenate([[0], sig[:-1]]) == 0)
    n_trades = int(entries.sum())

    # Per-trade PnL
    trade_pnls = []
    in_trade = False
    tstart = 0
    for i in range(len(sig)):
        if sig[i] != 0 and not in_trade:
            in_trade = True; tstart = i
        elif sig[i] == 0 and in_trade:
            trade_pnls.append(float(np.sum(net[tstart:i])))
            in_trade = False
    if in_trade:
        trade_pnls.append(float(np.sum(net[tstart:])))

    win_rate = float(np.mean([p > 0 for p in trade_pnls]) * 100) if trade_pnls else 0

    # Sharpe
    sharpe = 0.0
    if n_active > 10:
        active_pnl = net[active]
        std_a = float(np.std(active_pnl, ddof=1))
        if std_a > 0:
            sharpe = float(np.mean(active_pnl)) / std_a * np.sqrt(8760)

    total_return = float(np.sum(net))
    total_gross = float(np.sum(gross))

    # Max drawdown
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    # Monthly returns
    bars_per_month = 24 * 30
    n_months = len(net) // bars_per_month
    monthly = []
    for m in range(n_months):
        monthly.append(float(np.sum(net[m*bars_per_month:(m+1)*bars_per_month])))
    pos_months = sum(1 for r in monthly if r > 0)

    return {
        "sharpe": sharpe,
        "total_return": total_return,
        "total_gross": total_gross,
        "trades": n_trades,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "n_active": n_active,
        "n_total": len(ret_1bar),
        "pos_months": pos_months,
        "n_months": n_months,
        "monthly": monthly,
        "avg_net_bps": total_return / max(n_trades, 1) * 10000,
    }


# ── Main ──

def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME ENSEMBLE BACKTEST — 1h + 4h")
    print("=" * 70)

    # ── Load 1h data & model ──
    print("\n  Loading 1h data...")
    df_1h = pd.read_csv("data_files/BTCUSDT_1h.csv")
    n_1h = len(df_1h)
    print(f"  1h bars: {n_1h:,}")

    # 1h features
    print("  Computing 1h features...")
    t0 = time.time()
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_1h = compute_features_batch(SYMBOL, df_1h, include_v11=_has_v11)

    # Add 4h multi-timeframe features (used by 1h model)
    tf4h = compute_4h_features(df_1h)
    for col in TF4H_FEATURE_NAMES:
        feat_1h[col] = tf4h[col].values

    # Add interaction features
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_1h.columns and fb in feat_1h.columns:
            feat_1h[int_name] = feat_1h[fa].astype(float) * feat_1h[fb].astype(float)

    feat_names_1h = [c for c in feat_1h.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    closes_1h = df_1h["close"].values.astype(np.float64)
    timestamps_1h = df_1h["open_time" if "open_time" in df_1h.columns else "timestamp"].values.astype(np.int64)
    print(f"  1h features: {len(feat_names_1h)} in {time.time()-t0:.1f}s")

    # Load 1h models
    with open("models_v8/BTCUSDT_gate_v2/config.json") as f:
        cfg_1h = json.load(f)
    with open("models_v8/BTCUSDT_gate_v2/lgbm_v8.pkl", "rb") as f:
        lgbm_1h = pickle.load(f)
    with open("models_v8/BTCUSDT_gate_v2/xgb_v8.pkl", "rb") as f:
        xgb_1h = pickle.load(f)

    model_feats_1h = cfg_1h["features"]
    sel_idx_1h = [feat_names_1h.index(f) for f in model_feats_1h if f in feat_names_1h]
    actual_feats_1h = [feat_names_1h[i] for i in sel_idx_1h]
    print(f"  1h model features: {len(actual_feats_1h)}/{len(model_feats_1h)} available")

    # ── Load 4h data & model ──
    print("\n  Loading 1m data for 4h resampling...")
    df_1m = pd.read_csv("data_files/BTCUSDT_1m.csv")
    print(f"  1m bars: {len(df_1m):,}")

    # Resample to 4h
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts_1m = df_1m[ts_col].values.astype(np.int64)
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

    n_4h = len(df_4h)
    print(f"  4h bars: {n_4h:,}")

    # 4h features
    print("  Computing 4h features...")
    t0 = time.time()
    feat_4h = compute_features_batch(SYMBOL, df_4h, include_v11=_has_v11)
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_4h.columns and fb in feat_4h.columns:
            feat_4h[int_name] = feat_4h[fa].astype(float) * feat_4h[fb].astype(float)
    feat_names_4h = [c for c in feat_4h.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    closes_4h = df_4h["close"].values.astype(np.float64)
    timestamps_4h = df_4h["open_time"].values.astype(np.int64)
    print(f"  4h features: {len(feat_names_4h)} in {time.time()-t0:.1f}s")

    # Load 4h models
    with open("models_v8/BTCUSDT_4h_v1/config.json") as f:
        cfg_4h = json.load(f)
    with open("models_v8/BTCUSDT_4h_v1/lgbm_v8.pkl", "rb") as f:
        lgbm_4h = pickle.load(f)
    with open("models_v8/BTCUSDT_4h_v1/xgb_v8.pkl", "rb") as f:
        xgb_4h = pickle.load(f)

    model_feats_4h = cfg_4h["features"]
    sel_idx_4h = [feat_names_4h.index(f) for f in model_feats_4h if f in feat_names_4h]
    actual_feats_4h = [feat_names_4h[i] for i in sel_idx_4h]
    print(f"  4h model features: {len(actual_feats_4h)}/{len(model_feats_4h)} available")

    # ── Generate predictions ──
    print("\n  Generating predictions...")
    import lightgbm as lgb
    import xgboost as xgb

    # 1h predictions (OOS = last 13140 bars)
    oos_1h = 13140
    X_1h_oos = feat_1h[feat_names_1h].values[-oos_1h:].astype(np.float64)[:, sel_idx_1h]
    lgbm_pred_1h = lgbm_1h["model"].predict(X_1h_oos)
    xgb_pred_1h = xgb_1h["model"].predict(xgb.DMatrix(X_1h_oos))
    pred_1h = 0.5 * lgbm_pred_1h + 0.5 * xgb_pred_1h
    close_1h_oos = closes_1h[-oos_1h:]
    ts_1h_oos = timestamps_1h[-oos_1h:]

    # 4h predictions (OOS = last 3240 bars)
    oos_4h = 3240
    X_4h_oos = feat_4h[feat_names_4h].values[-oos_4h:].astype(np.float64)[:, sel_idx_4h]
    lgbm_pred_4h = lgbm_4h["model"].predict(X_4h_oos)
    xgb_pred_4h = xgb_4h["model"].predict(xgb.DMatrix(X_4h_oos))
    pred_4h = 0.5 * lgbm_pred_4h + 0.5 * xgb_pred_4h
    ts_4h_oos = timestamps_4h[-oos_4h:]

    print(f"  1h OOS: {oos_1h} bars, start={pd.Timestamp(ts_1h_oos[0], unit='ms').strftime('%Y-%m-%d')}")
    print(f"  4h OOS: {oos_4h} bars, start={pd.Timestamp(ts_4h_oos[0], unit='ms').strftime('%Y-%m-%d')}")

    # ── Align 4h predictions to 1h frequency ──
    # Forward-fill: each 4h prediction applies to the next 4 1h bars
    pred_4h_at_1h = np.full(oos_1h, np.nan)
    j = 0
    for i in range(oos_1h):
        # Find the latest 4h bar that started at or before this 1h bar
        while j < oos_4h - 1 and ts_4h_oos[j + 1] <= ts_1h_oos[i]:
            j += 1
        if ts_4h_oos[j] <= ts_1h_oos[i]:
            pred_4h_at_1h[i] = pred_4h[j]

    valid_both = ~np.isnan(pred_4h_at_1h)
    n_aligned = valid_both.sum()
    print(f"  Aligned bars (both models have predictions): {n_aligned:,}")

    # Trim to common range
    first_valid = np.argmax(valid_both)
    pred_1h = pred_1h[first_valid:]
    pred_4h_aligned = pred_4h_at_1h[first_valid:]
    close_common = close_1h_oos[first_valid:]
    ts_common = ts_1h_oos[first_valid:]
    n_common = len(pred_1h)
    print(f"  Common period: {n_common} bars "
          f"({pd.Timestamp(ts_common[0], unit='ms').strftime('%Y-%m-%d')} → "
          f"{pd.Timestamp(ts_common[-1], unit='ms').strftime('%Y-%m-%d')})")

    # ── Z-score normalization ──
    print("\n  Computing z-scores...")
    z_1h = zscore_signal(pred_1h, window=720)    # 30 days for 1h
    z_4h = zscore_signal(pred_4h_aligned, window=720)  # same window (in 1h bars)

    # Check signal correlation
    corr_pred = fast_ic(pred_1h, pred_4h_aligned)
    corr_z = fast_ic(z_1h, z_4h)
    print(f"  Raw prediction correlation: {corr_pred:.4f}")
    print(f"  Z-score correlation: {corr_z:.4f}")

    # ── 1h target for IC measurement ──
    y_1h = np.full(n_common, np.nan)
    h = 24  # 1h model horizon
    y_1h[:n_common-h] = close_common[h:] / close_common[:n_common-h] - 1
    ic_1h = fast_ic(pred_1h, y_1h)
    ic_4h = fast_ic(pred_4h_aligned, y_1h)
    print(f"  IC (1h pred vs 24h return): {ic_1h:.4f}")
    print(f"  IC (4h pred vs 24h return): {ic_4h:.4f}")

    # ═══════════════════════════════════════════════════════════
    # TEST ALL FUSION METHODS
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FUSION METHOD COMPARISON")
    print("=" * 70)

    # 1h model config
    dz_1h = cfg_1h["deadzone"]        # 0.5
    mh_1h = cfg_1h["min_hold"]        # 24
    maxh_1h = mh_1h * 5               # 120

    # 4h model config (in 1h bars: multiply by 4)
    dz_4h = cfg_4h["deadzone"]        # 2.0
    mh_4h_1h = cfg_4h["min_hold"] * 4  # 3*4=12 (in 1h bars)
    maxh_4h_1h = cfg_4h["max_hold"] * 4  # 36*4=144

    results = {}

    # ── Method 0a: 1h Only (baseline) ──
    sig_1h = apply_signal(z_1h, dz_1h, mh_1h, maxh_1h, long_only=True)
    r = backtest_pnl(sig_1h, close_common)
    results["1h Only"] = r

    # ── Method 0b: 4h Only (baseline, at 1h frequency) ──
    sig_4h = apply_signal(z_4h, dz_4h, mh_4h_1h, maxh_4h_1h, long_only=True)
    r = backtest_pnl(sig_4h, close_common)
    results["4h Only"] = r

    # ── Method 1: Cascade — 4h filters 1h ──
    # Only take 1h long signals when 4h is also long
    sig_cascade = sig_1h.copy()
    sig_cascade[sig_4h <= 0] = 0  # Kill 1h signal when 4h not bullish
    r = backtest_pnl(sig_cascade, close_common)
    results["Cascade (4h filter)"] = r

    # ── Method 2: Weighted Z-score blend ──
    for w1h, w4h in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        z_blend = w1h * z_1h + w4h * z_4h
        sig_blend = apply_signal(z_blend, dz_1h, mh_1h, maxh_1h, long_only=True)
        r = backtest_pnl(sig_blend, close_common)
        results[f"Weighted ({w1h:.0%}/{w4h:.0%})"] = r

    # ── Method 3: Agreement Only ──
    # Both must be positive z-score to enter
    sig_agree = sig_1h.copy()
    sig_agree[z_4h <= 0] = 0  # Need 4h z > 0 (not just position, but raw signal positive)
    r = backtest_pnl(sig_agree, close_common)
    results["Agreement (both z>0)"] = r

    # ── Method 4: Confidence-Scaled ──
    # Agreement → full size, disagreement → half size
    sig_conf = sig_1h.copy()
    agreement = (z_1h > 0) & (z_4h > 0)
    sig_conf[~agreement & (sig_1h > 0)] *= 0.5  # Half size on disagreement
    r = backtest_pnl(sig_conf, close_common)
    results["Confidence-Scaled"] = r

    # ── Method 5: OR logic (either signal triggers) ──
    sig_or = np.maximum(sig_1h, sig_4h)
    r = backtest_pnl(sig_or, close_common)
    results["OR (either triggers)"] = r

    # ── Method 6: Adaptive blend (split into halves, optimize on first, test on second) ──
    half = n_common // 2

    # Find best weight on first half
    best_w = 0.5
    best_sharpe = -999
    for w in np.arange(0.0, 1.05, 0.1):
        z_test = w * z_1h[:half] + (1-w) * z_4h[:half]
        sig_test = apply_signal(z_test, dz_1h, mh_1h, maxh_1h, long_only=True)
        r_test = backtest_pnl(sig_test, close_common[:half])
        if r_test["sharpe"] > best_sharpe:
            best_sharpe = r_test["sharpe"]
            best_w = w

    # Apply to second half
    z_adaptive = best_w * z_1h[half:] + (1-best_w) * z_4h[half:]
    sig_adaptive = apply_signal(z_adaptive, dz_1h, mh_1h, maxh_1h, long_only=True)
    r = backtest_pnl(sig_adaptive, close_common[half:])
    results[f"Adaptive (w1h={best_w:.1f})"] = r
    print(f"\n  Adaptive weight (optimized on H1): w_1h={best_w:.1f}, w_4h={1-best_w:.1f}")

    # ── Method 7: Deadzone sweep on blended z ──
    best_dz_blend = 0
    best_sharpe_dz = -999
    for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5]:
        z_b = 0.5 * z_1h + 0.5 * z_4h
        sig_b = apply_signal(z_b, dz, mh_1h, maxh_1h, long_only=True)
        r_b = backtest_pnl(sig_b, close_common)
        if r_b["sharpe"] > best_sharpe_dz and r_b["trades"] >= 10:
            best_sharpe_dz = r_b["sharpe"]
            best_dz_blend = dz

    z_best_dz = 0.5 * z_1h + 0.5 * z_4h
    sig_best_dz = apply_signal(z_best_dz, best_dz_blend, mh_1h, maxh_1h, long_only=True)
    r = backtest_pnl(sig_best_dz, close_common)
    results[f"Blend dz={best_dz_blend}"] = r

    # ═══════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  {'Method':<28s} {'Sharpe':>7s} {'Return':>8s} {'Trades':>7s} "
          f"{'WinR':>6s} {'bp/tr':>7s} {'MaxDD':>7s} {'Mo+':>5s}")
    print(f"  {'─'*75}")

    # Sort by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["sharpe"])
    for name, r in sorted_results:
        print(f"  {name:<28s} {r['sharpe']:>+7.2f} {r['total_return']*100:>+7.2f}% "
              f"{r['trades']:>7d} {r['win_rate']:>5.1f}% "
              f"{r['avg_net_bps']:>+7.1f} {r['max_dd']*100:>6.2f}% "
              f"{r['pos_months']:>2d}/{r['n_months']}")

    # ── Best method detail ──
    best_name, best_r = sorted_results[0]
    print(f"\n  BEST: {best_name}")
    print(f"  Sharpe: {best_r['sharpe']:+.2f}")
    print(f"  Total return: {best_r['total_return']*100:+.2f}%")

    if best_r.get("monthly"):
        print(f"\n  Monthly returns ({best_name}):")
        for i, m in enumerate(best_r["monthly"]):
            marker = "+" if m > 0 else " "
            print(f"    Month {i+1:2d}: {marker}{m*100:+.2f}%")

    # ── Signal agreement analysis ──
    print(f"\n{'='*70}")
    print(f"  SIGNAL AGREEMENT ANALYSIS")
    print(f"{'='*70}")

    both_long = (sig_1h > 0) & (sig_4h > 0)
    only_1h = (sig_1h > 0) & (sig_4h <= 0)
    only_4h = (sig_1h <= 0) & (sig_4h > 0)
    neither = (sig_1h <= 0) & (sig_4h <= 0)

    ret_1bar = np.diff(close_common) / close_common[:-1]

    for label, mask in [("Both long", both_long[:-1]),
                        ("Only 1h long", only_1h[:-1]),
                        ("Only 4h long", only_4h[:-1]),
                        ("Neither", neither[:-1])]:
        if mask.sum() > 0:
            rets = ret_1bar[mask]
            avg_ret = np.mean(rets) * 10000
            sr = np.mean(rets) / max(np.std(rets), 1e-12) * np.sqrt(8760)
            print(f"  {label:<20s}: {mask.sum():>5d} bars ({mask.sum()/len(mask)*100:>5.1f}%), "
                  f"avg={avg_ret:+.1f}bp/bar, Sharpe={sr:+.2f}")

    # ── Improvement over baseline ──
    baseline_sharpe = results["1h Only"]["sharpe"]
    best_sharpe = best_r["sharpe"]
    improvement = (best_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.01)
    print(f"\n  Improvement over 1h baseline: {improvement*100:+.1f}% "
          f"(Sharpe {baseline_sharpe:.2f} → {best_sharpe:.2f})")


if __name__ == "__main__":
    main()
