#!/usr/bin/env python3
"""Full-History Multi-Timeframe Backtest — 1h + 4h blend with dynamic leverage.

Validates the production MultiTimeframeSignal on all available data:
  - Blended z-score (50/50), deadzone=2.5, hold=[24,120] 1h bars
  - Dynamic leverage: 2x–3x based on signal strength × volatility discount
  - Trade-by-trade PnL with leverage-adjusted sizing
  - Monthly breakdown, drawdown analysis, leverage distribution

Usage:
    python3 -m scripts.backtest_multi_tf_full
"""
from __future__ import annotations
import sys
import time
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from alpha.utils import fast_ic

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.batch_feature_engine import compute_4h_features, TF4H_FEATURE_NAMES
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST

# ── Config ──
SYMBOL = "BTCUSDT"
COST_BPS = 4          # Maker round-trip bps
EQUITY = 10_000.0
RISK_FRACTION = 0.05  # 5% per trade (base, before leverage)

# Multi-TF signal config (matches production)
BLEND_W1H = 0.5
BLEND_W4H = 0.5
DEADZONE = 2.5
MIN_HOLD = 24         # 1h bars
MAX_HOLD = 120        # 1h bars
LONG_ONLY = True
ZSCORE_WINDOW_1H = 720
ZSCORE_WINDOW_4H = 720   # In 1h bars (aligned)

# Dynamic leverage config
LEV_MIN = 2.0
LEV_MAX = 3.0
VOL_WARMUP = 168      # 7 days of 1h bars


# ── Z-score ──

def zscore_signal(pred: np.ndarray, window: int = 720,
                  warmup: int = 50) -> np.ndarray:
    n = len(pred)
    z = np.zeros(n)
    for i in range(n):
        start = max(0, i - window + 1)
        buf = pred[start:i+1]
        buf = buf[~np.isnan(buf)]
        if len(buf) < warmup:
            z[i] = 0.0
        else:
            std = np.std(buf)
            z[i] = (pred[i] - np.mean(buf)) / std if std > 1e-12 else 0.0
    return z


# ── Dynamic leverage ──

def compute_dynamic_leverage(
    z_blend: float,
    closes_recent: np.ndarray,
    deadzone: float = DEADZONE,
    lev_min: float = LEV_MIN,
    lev_max: float = LEV_MAX,
    vol_warmup: int = VOL_WARMUP,
) -> float:
    """Mirror of MultiTimeframeSignal._compute_dynamic_leverage."""
    lev_range = lev_max - lev_min
    if lev_range <= 0:
        return lev_max

    # Factor 1: Signal strength ramp
    z_excess = abs(z_blend) - deadzone
    signal_ramp = min(max(z_excess / deadzone, 0.0), 1.0)

    # Factor 2: Volatility discount
    vol_discount = 1.0
    if len(closes_recent) >= vol_warmup:
        rets = np.diff(closes_recent) / closes_recent[:-1]
        current_vol = float(np.std(rets[-168:])) if len(rets) >= 168 else float(np.std(rets))
        long_vol = float(np.std(rets))
        if long_vol > 1e-12:
            vol_ratio = current_vol / long_vol
            vol_discount = 1.0 - min(max((vol_ratio - 0.8) / 0.7, 0.0), 1.0)

    return round(lev_min + lev_range * signal_ramp * vol_discount, 2)


# ── Trade tracking ──

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


def run_backtest(
    z_blend: np.ndarray,
    z_1h: np.ndarray,
    z_4h: np.ndarray,
    closes: np.ndarray,
    timestamps: np.ndarray,
    cost_bps: float = COST_BPS,
    dynamic_lev: bool = True,
    lev_min: float = LEV_MIN,
    lev_max: float = LEV_MAX,
) -> Dict[str, Any]:
    """Full backtest with dynamic leverage sizing."""
    n = len(z_blend)
    cost_frac = cost_bps / 10000

    trades: List[Trade] = []
    pos = 0.0
    ep = 0.0
    eb = 0
    entry_lev = lev_min
    entry_z = 0.0
    entry_z1 = 0.0
    entry_z4 = 0.0

    equity = EQUITY
    equity_curve = [equity]
    leverage_history = []

    for i in range(n):
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
                    hold_bars=held,
                ))
                pos = 0.0

        # ── Check entry ──
        if pos == 0:
            if z_blend[i] > DEADZONE:
                pos = 1.0
                ep = closes[i]
                eb = i
                entry_z = z_blend[i]
                entry_z1 = z_1h[i]
                entry_z4 = z_4h[i]

                if dynamic_lev:
                    start = max(0, i - 720)
                    entry_lev = compute_dynamic_leverage(
                        z_blend[i], closes[start:i+1],
                        lev_min=lev_min, lev_max=lev_max)
                else:
                    entry_lev = lev_max

                leverage_history.append(entry_lev)

            elif not LONG_ONLY and z_blend[i] < -DEADZONE:
                pos = -1.0
                ep = closes[i]
                eb = i
                entry_z = z_blend[i]
                entry_z1 = z_1h[i]
                entry_z4 = z_4h[i]
                if dynamic_lev:
                    start = max(0, i - 720)
                    entry_lev = compute_dynamic_leverage(
                        z_blend[i], closes[start:i+1],
                        lev_min=lev_min, lev_max=lev_max)
                else:
                    entry_lev = lev_max
                leverage_history.append(entry_lev)

        equity_curve.append(equity)

    # Close open position at end
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
            hold_bars=n-1-eb,
        ))
        equity_curve.append(equity)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "leverage_history": leverage_history,
    }


def analyze(result: Dict[str, Any], timestamps: np.ndarray, label: str) -> Dict[str, Any]:
    """Print detailed analysis."""
    trades = result["trades"]
    result["leverage_history"]

    if not trades:
        print(f"\n  [{label}] No trades.")
        return {}

    np.array([t.gross_pnl for t in trades])
    net = np.array([t.net_pnl for t in trades])
    holds = np.array([t.hold_bars for t in trades])
    levs = np.array([t.leverage for t in trades])
    n_bars = len(timestamps)
    days = n_bars / 24

    wins = sum(1 for t in trades if t.net_pnl > 0)

    # Monthly breakdown
    monthly: Dict[str, Dict] = {}
    for t in trades:
        ts_ms = int(timestamps[min(t.exit_bar, len(timestamps)-1)])
        month = pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m")
        if month not in monthly:
            monthly[month] = {"gross": 0.0, "net": 0.0, "trades": 0,
                              "wins": 0, "levs": []}
        monthly[month]["gross"] += t.gross_pnl
        monthly[month]["net"] += t.net_pnl
        monthly[month]["trades"] += 1
        monthly[month]["levs"].append(t.leverage)
        if t.net_pnl > 0:
            monthly[month]["wins"] += 1

    # Drawdown
    eq = EQUITY
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t.net_pnl
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (annualized)
    if len(net) > 2:
        avg_hold_h = float(np.mean(holds))
        trades_per_year = 365 * 24 / max(avg_hold_h, 1)
        sharpe = float(np.mean(net) / max(np.std(net, ddof=1), 1e-10)
                       * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    total_net = float(np.sum(net))
    ret_pct = total_net / EQUITY * 100

    print(f"\n  [{label}]")
    print(f"  {'─'*70}")
    print(f"  Trades: {len(trades)}  ({len(trades)/max(days,1)*7:.1f}/week)  "
          f"Win Rate: {wins/len(trades)*100:.1f}%")
    print(f"  Avg Hold: {np.mean(holds):.0f} bars ({np.mean(holds):.0f}h)  "
          f"Median Hold: {np.median(holds):.0f}h")
    print(f"  Total Net: ${total_net:+,.0f}  ({ret_pct:+.2f}% on ${EQUITY:,.0f})")
    print(f"  Avg Net/trade: ${np.mean(net):+.1f}  "
          f"({np.mean(net)/EQUITY*10000:+.1f} bp of equity)")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Sharpe (annual): {sharpe:.2f}")
    print(f"  Final Equity: ${result['final_equity']:,.0f}")

    # Leverage stats
    print("\n  Leverage Distribution:")
    print(f"    Mean: {np.mean(levs):.2f}x   Median: {np.median(levs):.2f}x   "
          f"Min: {np.min(levs):.2f}x   Max: {np.max(levs):.2f}x")
    for lo, hi in [(2.0, 2.25), (2.25, 2.5), (2.5, 2.75), (2.75, 3.01)]:
        count = sum(1 for ll in levs if lo <= ll < hi)
        pct = count / len(levs) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lo:.2f}–{hi:.2f}x: {count:>3d} ({pct:>5.1f}%) {bar}")

    # Leverage vs outcome
    print("\n  Leverage vs Outcome:")
    lev_mid = np.median(levs)
    low_lev = [t for t in trades if t.leverage < lev_mid]
    high_lev = [t for t in trades if t.leverage >= lev_mid]
    if low_lev:
        low_wr = sum(1 for t in low_lev if t.net_pnl > 0) / len(low_lev) * 100
        low_avg = np.mean([t.net_pnl for t in low_lev])
        print(f"    Low lev  (<{lev_mid:.2f}x): {len(low_lev):>3d} trades, "
              f"WR={low_wr:.0f}%, avg=${low_avg:+.1f}")
    if high_lev:
        high_wr = sum(1 for t in high_lev if t.net_pnl > 0) / len(high_lev) * 100
        high_avg = np.mean([t.net_pnl for t in high_lev])
        print(f"    High lev (≥{lev_mid:.2f}x): {len(high_lev):>3d} trades, "
              f"WR={high_wr:.0f}%, avg=${high_avg:+.1f}")

    # Monthly table
    print(f"\n  {'Month':>8} {'#Tr':>5} {'WinR':>6} {'Net$':>8} "
          f"{'CumNet$':>9} {'AvgLev':>7}")
    print(f"  {'─'*50}")
    cum = 0.0
    pos_months = 0
    for month in sorted(monthly.keys()):
        m = monthly[month]
        cum += m["net"]
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        avg_lev = np.mean(m["levs"]) if m["levs"] else 0
        if m["net"] > 0:
            pos_months += 1
        print(f"  {month:>8} {m['trades']:>5} {wr:>5.0f}% {m['net']:>+8.1f} "
              f"{cum:>+9.1f} {avg_lev:>6.2f}x")
    print(f"  Positive months: {pos_months}/{len(monthly)}")

    # Top / bottom trades
    sorted_trades = sorted(trades, key=lambda t: t.net_pnl)
    print("\n  Bottom 5 trades:")
    for t in sorted_trades[:5]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%Y-%m-%d %H:%M")
        print(f"    {ts}  L @ ${t.entry_price:,.0f}  hold={t.hold_bars}h  "
              f"lev={t.leverage:.2f}x  z={t.z_blend:+.2f}  net=${t.net_pnl:+.1f}")
    print("  Top 5 trades:")
    for t in sorted_trades[-5:]:
        ts = pd.Timestamp(int(timestamps[t.entry_bar]), unit="ms").strftime("%Y-%m-%d %H:%M")
        print(f"    {ts}  L @ ${t.entry_price:,.0f}  hold={t.hold_bars}h  "
              f"lev={t.leverage:.2f}x  z={t.z_blend:+.2f}  net=${t.net_pnl:+.1f}")

    return {
        "trades": len(trades), "win_rate": wins/len(trades)*100,
        "sharpe": sharpe, "total_net": total_net, "ret_pct": ret_pct,
        "max_dd": max_dd, "avg_lev": float(np.mean(levs)),
        "pos_months": pos_months, "total_months": len(monthly),
        "final_equity": result["final_equity"],
    }


# ── Main ──

def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME FULL HISTORY BACKTEST")
    print(f"Config: blend={BLEND_W1H}/{BLEND_W4H}, dz={DEADZONE}, "
          f"hold=[{MIN_HOLD},{MAX_HOLD}], lev=[{LEV_MIN},{LEV_MAX}]")
    print("=" * 70)

    # ── Load 1h data ──
    print("\n[1] Loading data...")
    t0 = time.time()
    df_1h = pd.read_csv("data_files/BTCUSDT_1h.csv")
    n_1h = len(df_1h)
    ts_col_1h = "open_time" if "open_time" in df_1h.columns else "timestamp"
    timestamps_1h = df_1h[ts_col_1h].values.astype(np.int64)
    closes_1h = df_1h["close"].values.astype(np.float64)
    start_date = pd.Timestamp(timestamps_1h[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps_1h[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"  1h bars: {n_1h:,} ({start_date} → {end_date})")

    # ── Load 1m for 4h resampling ──
    df_1m = pd.read_csv("data_files/BTCUSDT_1m.csv")
    print(f"  1m bars: {len(df_1m):,}")

    # Resample to 4h
    ts_col_1m = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts_1m = df_1m[ts_col_1m].values.astype(np.int64)
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
    print(f"  4h bars: {len(df_4h):,}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ── Compute features ──
    print("\n[2] Computing features...")
    t0 = time.time()
    _has_v11 = Path("data_files/macro_daily.csv").exists()

    # 1h features
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
    print(f"  1h features: {len(feat_names_1h)}")

    # 4h features
    feat_4h = compute_features_batch(SYMBOL, df_4h, include_v11=_has_v11)
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_4h.columns and fb in feat_4h.columns:
            feat_4h[int_name] = feat_4h[fa].astype(float) * feat_4h[fb].astype(float)
    feat_names_4h = [c for c in feat_4h.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    print(f"  4h features: {len(feat_names_4h)}")
    print(f"  Feature time: {time.time()-t0:.1f}s")

    # ── Load models ──
    print("\n[3] Loading models...")
    import xgboost as xgb

    # 1h model
    with open("models_v8/BTCUSDT_gate_v2/config.json") as f:
        cfg_1h = json.load(f)
    with open("models_v8/BTCUSDT_gate_v2/lgbm_v8.pkl", "rb") as f:
        lgbm_1h = pickle.load(f)
    with open("models_v8/BTCUSDT_gate_v2/xgb_v8.pkl", "rb") as f:
        xgb_1h = pickle.load(f)
    model_feats_1h = cfg_1h["features"]
    sel_idx_1h = [feat_names_1h.index(f) for f in model_feats_1h if f in feat_names_1h]
    print(f"  1h model: {len(model_feats_1h)} features")

    # 4h model
    with open("models_v8/BTCUSDT_4h_v1/config.json") as f:
        cfg_4h = json.load(f)
    with open("models_v8/BTCUSDT_4h_v1/lgbm_v8.pkl", "rb") as f:
        lgbm_4h = pickle.load(f)
    with open("models_v8/BTCUSDT_4h_v1/xgb_v8.pkl", "rb") as f:
        xgb_4h = pickle.load(f)
    model_feats_4h = cfg_4h["features"]
    sel_idx_4h = [feat_names_4h.index(f) for f in model_feats_4h if f in feat_names_4h]
    print(f"  4h model: {len(model_feats_4h)} features")

    # ── Generate predictions (full history) ──
    print("\n[4] Generating predictions (full history)...")
    t0 = time.time()

    # 1h predictions — full history
    X_1h = feat_1h[feat_names_1h].values.astype(np.float64)[:, sel_idx_1h]
    lgbm_pred_1h = lgbm_1h["model"].predict(X_1h)
    xgb_pred_1h = xgb_1h["model"].predict(xgb.DMatrix(X_1h))
    pred_1h = 0.5 * lgbm_pred_1h + 0.5 * xgb_pred_1h

    # 4h predictions — full history
    X_4h = feat_4h[feat_names_4h].values.astype(np.float64)[:, sel_idx_4h]
    lgbm_pred_4h = lgbm_4h["model"].predict(X_4h)
    xgb_pred_4h = xgb_4h["model"].predict(xgb.DMatrix(X_4h))
    pred_4h = 0.5 * lgbm_pred_4h + 0.5 * xgb_pred_4h

    print(f"  1h predictions: {len(pred_1h):,}")
    print(f"  4h predictions: {len(pred_4h):,}")
    print(f"  Predict time: {time.time()-t0:.1f}s")

    # ── Align 4h to 1h frequency ──
    print("\n[5] Aligning timeframes...")
    pred_4h_at_1h = np.full(n_1h, np.nan)
    j = 0
    n_4h = len(pred_4h)
    for i in range(n_1h):
        while j < n_4h - 1 and timestamps_4h[j + 1] <= timestamps_1h[i]:
            j += 1
        if j < n_4h and timestamps_4h[j] <= timestamps_1h[i]:
            pred_4h_at_1h[i] = pred_4h[j]

    valid_both = ~np.isnan(pred_4h_at_1h)
    first_valid = int(np.argmax(valid_both))
    n_common = int(valid_both.sum())
    print(f"  Aligned bars: {n_common:,} (from bar {first_valid})")

    # Trim to common range
    pred_1h_c = pred_1h[first_valid:]
    pred_4h_c = pred_4h_at_1h[first_valid:]
    closes_c = closes_1h[first_valid:]
    ts_c = timestamps_1h[first_valid:]
    n_c = len(pred_1h_c)

    date_start = pd.Timestamp(ts_c[0], unit="ms").strftime("%Y-%m-%d")
    date_end = pd.Timestamp(ts_c[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"  Period: {date_start} → {date_end} ({n_c:,} bars, {n_c/24:.0f} days)")

    # ── Z-score normalization ──
    print("\n[6] Computing z-scores...")
    t0 = time.time()
    warmup_1h = max(ZSCORE_WINDOW_1H // 4, 20)
    warmup_4h = max(ZSCORE_WINDOW_4H // 4, 20)
    z_1h = zscore_signal(pred_1h_c, window=ZSCORE_WINDOW_1H, warmup=warmup_1h)
    z_4h = zscore_signal(pred_4h_c, window=ZSCORE_WINDOW_4H, warmup=warmup_4h)

    z_blend = BLEND_W1H * z_1h + BLEND_W4H * z_4h
    print(f"  Z-score time: {time.time()-t0:.1f}s")

    # Correlation
    corr_raw = fast_ic(pred_1h_c, pred_4h_c)
    corr_z = fast_ic(z_1h, z_4h)
    print(f"  Raw prediction correlation: {corr_raw:.4f}")
    print(f"  Z-score correlation: {corr_z:.4f}")

    # ── Signal stats ──
    n_long_sig = np.sum(z_blend > DEADZONE)
    print(f"  Bars with z_blend > {DEADZONE}: {n_long_sig} ({n_long_sig/n_c*100:.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # BACKTEST 1: Dynamic leverage (2x–3x)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BACKTEST A: DYNAMIC LEVERAGE (2x–3x)")
    print("=" * 70)

    bt_dyn = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                          cost_bps=COST_BPS, dynamic_lev=True)
    stats_dyn = analyze(bt_dyn, ts_c, "Dynamic Lev 2x–3x, Maker 4bp")

    # ═══════════════════════════════════════════════════════════════
    # BACKTEST 2: Fixed leverage (3x) — comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BACKTEST B: FIXED LEVERAGE (3x) — comparison")
    print("=" * 70)

    bt_fix3 = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                           cost_bps=COST_BPS, dynamic_lev=False,
                           lev_min=3.0, lev_max=3.0)
    stats_fix3 = analyze(bt_fix3, ts_c, "Fixed 3x, Maker 4bp")

    # ═══════════════════════════════════════════════════════════════
    # BACKTEST 3: Fixed leverage (2x) — comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BACKTEST C: FIXED LEVERAGE (2x) — comparison")
    print("=" * 70)

    bt_fix2 = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                           cost_bps=COST_BPS, dynamic_lev=False,
                           lev_min=2.0, lev_max=2.0)
    stats_fix2 = analyze(bt_fix2, ts_c, "Fixed 2x, Maker 4bp")

    # ═══════════════════════════════════════════════════════════════
    # BACKTEST 4: No leverage (1x) — baseline
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BACKTEST D: NO LEVERAGE (1x) — baseline")
    print("=" * 70)

    bt_1x = run_backtest(z_blend, z_1h, z_4h, closes_c, ts_c,
                         cost_bps=COST_BPS, dynamic_lev=False,
                         lev_min=1.0, lev_max=1.0)
    stats_1x = analyze(bt_1x, ts_c, "No leverage 1x, Maker 4bp")

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    all_stats = [
        ("No leverage (1x)", stats_1x),
        ("Fixed 2x", stats_fix2),
        ("Dynamic 2x–3x", stats_dyn),
        ("Fixed 3x", stats_fix3),
    ]

    print(f"\n  {'Method':<20s} {'Sharpe':>7s} {'Return':>8s} {'Net$':>8s} "
          f"{'MaxDD':>7s} {'#Tr':>5s} {'WinR':>6s} {'AvgLev':>7s} "
          f"{'Mo+':>5s}")
    print(f"  {'─'*80}")
    for name, s in all_stats:
        if not s:
            continue
        print(f"  {name:<20s} {s['sharpe']:>+7.2f} {s['ret_pct']:>+7.2f}% "
              f"${s['total_net']:>+7.0f} {s['max_dd']*100:>6.2f}% "
              f"{s['trades']:>5d} {s['win_rate']:>5.1f}% "
              f"{s.get('avg_lev', 0):>6.2f}x "
              f"{s['pos_months']:>2d}/{s['total_months']}")

    # Risk-adjusted comparison
    print("\n  Risk-Adjusted (Sharpe / MaxDD):")
    for name, s in all_stats:
        if not s or s['max_dd'] == 0:
            continue
        ratio = s['sharpe'] / (s['max_dd'] * 100)
        bar = "█" * int(ratio * 5)
        print(f"    {name:<20s}: {ratio:.3f} {bar}")

    print("\n  Key insight: Dynamic leverage targets higher leverage only when")
    print("  signal is strong AND volatility is low — protecting capital during")
    print("  turbulent markets while capturing more from high-conviction trades.")
    print("\n  Done.")


if __name__ == "__main__":
    main()
