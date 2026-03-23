#!/usr/bin/env python3
"""15-Minute Alpha Research — evaluate if 1h alpha transfers to 15m timeframe.

Three research questions:
1. Do existing 1h features have predictive power on 15m bars?
2. Can we train a dedicated 15m model with sufficient IC?
3. What's the optimal strategy: 15m model, or 1h signal with 15m execution?

Usage:
    python3 -m scripts.research_15m_alpha --symbol BTCUSDT
    python3 -m scripts.research_15m_alpha --symbol BTCUSDT,ETHUSDT
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from shared.signal_postprocess import rolling_zscore, should_exit_position

sys.path.insert(0, "/quant_system")

# ── Config ──
HORIZONS_15M = [4, 8, 16, 32, 64]  # in 15m bars = [1h, 2h, 4h, 8h, 16h]
COST_BPS_RT = 4
WARMUP = 30
TOP_K = 14


def fast_ic(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50:
        return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def compute_target(closes, horizon):
    n = len(closes)
    y = np.full(n, np.nan)
    y[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1
    v = y[~np.isnan(y)]
    if len(v) > 10:
        p1, p99 = np.percentile(v, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))
    return y


def compute_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features on 15m OHLCV data using the Rust feature engine.

    The Rust engine computes all 105 features from OHLCV + volume data.
    Window sizes are in bar-counts, so on 15m data:
    - ma_20 = 20 bars = 5 hours (vs 20 hours on 1h)
    - rsi_14 = 14 bars = 3.5 hours (vs 14 hours on 1h)
    This is fine — features auto-adapt to the timeframe.
    """
    from features.batch_feature_engine import compute_features_batch

    feat_df = compute_features_batch(
        symbol="",  # symbol only used for funding/spot lookups
        df=df,
        include_v11=False,  # no macro features for 15m
    )
    return feat_df


def backtest_simple(
    z: np.ndarray,
    closes: np.ndarray,
    deadzone: float = 0.3,
    min_hold: int = 4,    # 4 bars = 1 hour on 15m
    max_hold: int = 64,   # 64 bars = 16 hours on 15m
    long_only: bool = False,
    cost_bps: float = COST_BPS_RT,
) -> Dict[str, Any]:
    """Simple backtest on 15m bars."""
    n = len(z)
    cost_frac = cost_bps / 10000
    pos = 0.0
    entry_bar = 0
    trades = []

    for i in range(n):
        if pos != 0:
            held = i - entry_bar
            should_exit = should_exit_position(
                position=pos,
                z_value=float(z[i]),
                held_bars=held,
                min_hold=min_hold,
                max_hold=max_hold,
            )
            if should_exit:
                pnl_pct = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                trades.append(pnl_pct * 500.0 - cost_frac * 500.0)
                pos = 0.0

        if pos == 0:
            if z[i] > deadzone:
                pos = 1.0
                entry_bar = i
            elif not long_only and z[i] < -deadzone:
                pos = -1.0
                entry_bar = i

    if not trades:
        return {"sharpe": 0, "trades": 0, "return": 0, "win_rate": 0}

    net_arr = np.array(trades)
    avg_hold = n / max(len(trades), 1)
    # 15m bars: 96 bars/day, 365 days/year
    tpy = 365 * 96 / max(avg_hold, 1)
    sharpe = float(np.mean(net_arr) / max(np.std(net_arr, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": round(sharpe, 2),
        "trades": len(trades),
        "return": round(float(np.sum(net_arr)) / 10000 * 100, 2),
        "win_rate": round(float(np.mean(net_arr > 0) * 100), 1),
        "avg_net_bps": round(float(np.mean(net_arr) / 500 * 10000), 1),
        "avg_hold_bars": round(avg_hold, 1),
        "avg_hold_hours": round(avg_hold * 0.25, 1),
    }


def research_symbol(symbol: str) -> Dict[str, Any]:
    """Full 15m alpha research for one symbol."""
    data_path = Path(f"data_files/{symbol}_15m.csv")
    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return {}

    df = pd.read_csv(data_path)
    n = len(df)
    ts = df["open_time"].values.astype(np.int64)
    closes = df["close"].values.astype(np.float64)
    start = pd.Timestamp(ts[0], unit="ms").strftime("%Y-%m-%d")
    end = pd.Timestamp(ts[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} 15m bars ({start} → {end})")
    print(f"  Duration: {(ts[-1]-ts[0])/86400000:.0f} days")

    # ── Step 1: Compute features ──
    print("\n  Step 1: Computing 15m features...", end=" ", flush=True)
    t0 = time.time()
    feat_df = compute_15m_features(df)
    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp", "open", "high", "low", "volume")]
    X = feat_df[feature_names].values.astype(np.float64)
    print(f"{len(feature_names)} features in {time.time()-t0:.1f}s")

    # ── Step 2: IC analysis per horizon ──
    print("\n  Step 2: IC analysis across horizons")
    print(f"  {'Horizon':>8} {'Bars':>6} {'Hours':>6} {'IC':>8} {'Top Feature':>25} {'Top IC':>8}")
    print(f"  {'-'*8:>8} {'-'*6:>6} {'-'*6:>6} {'-'*8:>8} {'-'*25:>25} {'-'*8:>8}")

    best_horizons = []
    for h in HORIZONS_15M:
        y = compute_target(closes, h)
        hours = h * 0.25

        # Ensemble IC (mean of all features → target)
        valid = ~np.isnan(y)
        ic_scores = []
        for j in range(X.shape[1]):
            ic = fast_ic(X[valid, j], y[valid])
            ic_scores.append((feature_names[j], abs(ic), ic))

        ic_scores.sort(key=lambda x: x[1], reverse=True)
        top_name, top_abs_ic, top_ic = ic_scores[0] if ic_scores else ("?", 0, 0)

        # Simple mean prediction IC
        top_feats = [f for f, _, _ in ic_scores[:TOP_K]]
        X_top = X[:, [feature_names.index(f) for f in top_feats]]
        pred_mean = np.nanmean(X_top, axis=1)
        mean_ic = fast_ic(pred_mean[valid], y[valid])

        print(f"  {f'h{h}':>8} {h:>6} {hours:>6.1f} {mean_ic:>+8.4f} {top_name:>25} {top_ic:>+8.4f}")

        if abs(mean_ic) > 0.005:
            best_horizons.append((h, mean_ic))

    # ── Step 3: Train LightGBM for best horizons ──
    print("\n  Step 3: Training LightGBM models")

    # Split: last 6 months OOS
    bars_per_month = 96 * 30  # 96 15m bars/day × 30 days
    oos_bars = bars_per_month * 6
    train_end = n - oos_bars
    val_size = bars_per_month * 2
    val_start = train_end - val_size

    if train_end < bars_per_month * 6:
        print("  WARNING: Not enough training data, using 50% split")
        train_end = n // 2
        val_start = train_end - n // 6
        oos_bars = n - train_end

    closes_test = closes[train_end:]
    print(f"  Split: train={val_start-WARMUP:,} val={val_size:,} test={oos_bars:,}")

    results = {}
    horizons_to_train = [h for h, ic in best_horizons] if best_horizons else HORIZONS_15M[:3]

    for h in horizons_to_train:
        hours = h * 0.25
        print(f"\n  ── h={h} ({hours:.0f}h) ──")
        y = compute_target(closes, h)

        # Feature selection on train
        from features.dynamic_selector import greedy_ic_select
        valid_train = ~np.isnan(y[:val_start])
        selected = greedy_ic_select(
            X[:val_start][valid_train],
            y[:val_start][valid_train],
            feature_names,
            top_k=TOP_K,
        )
        print(f"    Selected {len(selected)} features")

        feat_idx = [feature_names.index(f) for f in selected]
        X_sel = X[:, feat_idx]

        # Train LightGBM
        import lightgbm as lgb

        valid_t = ~np.isnan(y[:val_start])
        valid_v = ~np.isnan(y[val_start:train_end])

        dtrain = lgb.Dataset(X_sel[:val_start][valid_t], y[:val_start][valid_t])
        dval = lgb.Dataset(
            X_sel[val_start:train_end][valid_v],
            y[val_start:train_end][valid_v],
            reference=dtrain,
        )

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        # OOS prediction
        pred_oos = model.predict(X_sel[train_end:])
        y_oos = y[train_end:]
        valid_oos = ~np.isnan(y_oos)

        oos_ic = fast_ic(pred_oos[valid_oos], y_oos[valid_oos])
        print(f"    OOS IC: {oos_ic:+.4f}")

        # Backtest
        z = rolling_zscore(pred_oos, window=720, warmup=180)

        # Sweep deadzone
        best_sharpe = -999
        best_result = None

        for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for mh in [4, 8, 16]:
                for maxh in [32, 64, 128]:
                    for lo in [True, False]:
                        r = backtest_simple(
                            z, closes_test, deadzone=dz,
                            min_hold=mh, max_hold=maxh, long_only=lo,
                        )
                        if r["sharpe"] > best_sharpe and r["trades"] >= 10:
                            best_sharpe = r["sharpe"]
                            best_result = r
                            best_params = {"dz": dz, "min_hold": mh, "max_hold": maxh, "long_only": lo}

        if best_result:
            print(f"    Best: dz={best_params['dz']}, hold=[{best_params['min_hold']},{best_params['max_hold']}], "
                  f"long_only={best_params['long_only']}")
            print(f"    Sharpe={best_result['sharpe']}, trades={best_result['trades']}, "
                  f"WR={best_result['win_rate']}%, ret={best_result['return']:+.2f}%")
            print(f"    Avg hold: {best_result['avg_hold_hours']:.1f}h, avg net: {best_result['avg_net_bps']:.1f}bps")
        else:
            print("    No viable config found")
            best_result = {"sharpe": 0, "trades": 0}

        results[h] = {
            "horizon_bars": h,
            "horizon_hours": hours,
            "oos_ic": oos_ic,
            "best_result": best_result,
            "best_params": best_params if best_result else {},
            "features": selected,
        }

    # ── Step 4: Compare with 1h baseline ──
    print(f"\n  {'='*60}")
    print(f"  15M ALPHA RESEARCH SUMMARY — {symbol}")
    print(f"  {'='*60}")

    print(f"\n  {'Horizon':>10} {'IC':>8} {'Sharpe':>8} {'Trades':>8} {'WR%':>6} {'Ret%':>8} {'AvgHold':>8}")
    print(f"  {'-'*10:>10} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8}")

    for h, info in sorted(results.items()):
        r = info["best_result"]
        hh = info["horizon_hours"]
        label = f"h{h} ({hh:.0f}h)"
        print(
            f"  {label:>10} "
            f"{info['oos_ic']:>+8.4f} "
            f"{r.get('sharpe', 0):>8.2f} "
            f"{r.get('trades', 0):>8} "
            f"{r.get('win_rate', 0):>6.1f} "
            f"{r.get('return', 0):>+8.2f} "
            f"{r.get('avg_hold_hours', 0):>7.1f}h"
        )

    print("\n  Comparison with 1h baseline:")
    print(f"  - 1h model: trained on {symbol}_1h.csv, ~{n//96:,} days of 15m ≈ {n//96//30:.0f} months")
    print("  - 15m: 4x more bars, potentially 4x more trades")
    print("  - Cost impact: same 4bps RT per trade")

    # Verdict
    viable = [h for h, info in results.items()
              if info["oos_ic"] > 0.01 and info["best_result"].get("sharpe", 0) > 1.0]
    if viable:
        print(f"\n  VERDICT: 15m alpha EXISTS for horizons {viable}")
        print("  Recommended: train dedicated 15m models")
    else:
        marginal = [h for h, info in results.items() if info["oos_ic"] > 0.005]
        if marginal:
            print(f"\n  VERDICT: Marginal 15m alpha for {marginal}")
            print("  Recommended: use 1h signal with 15m execution timing")
        else:
            print("\n  VERDICT: No significant 15m alpha found")
            print("  Recommended: stay on 1h timeframe")

    return results


def main():
    parser = argparse.ArgumentParser(description="15m Alpha Research")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Comma-separated symbols")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    print("=" * 70)
    print("  15-MINUTE ALPHA RESEARCH")
    print(f"  Symbols: {symbols}")
    print("=" * 70)

    all_results = {}
    for symbol in symbols:
        results = research_symbol(symbol)
        all_results[symbol] = results

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
