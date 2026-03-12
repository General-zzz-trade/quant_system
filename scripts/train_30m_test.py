#!/usr/bin/env python3
"""Test 30m/15m/5m models — resample 1m data to different timeframes and evaluate.

Answers: how does alpha decay as we go from 1h → 30m → 15m → 5m?
"""
from __future__ import annotations
import sys, time, json, pickle, logging
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")
from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from features.dynamic_selector import greedy_ic_select, _rankdata, _spearman_ic

WARMUP = 100
COST_MAKER_RT_BPS = 4  # maker round-trip

def fast_ic(x, y):
    from scipy.stats import spearmanr
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 50: return 0.0
    r, _ = spearmanr(x[m], y[m])
    return float(r) if not np.isnan(r) else 0.0


def resample_1m_to(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m bars to N-minute bars."""
    ts_col = "open_time" if "open_time" in df_1m.columns else "timestamp"
    ts = df_1m[ts_col].values.astype(np.int64)
    group_ms = minutes * 60_000
    groups = ts // group_ms

    work = pd.DataFrame({
        "group": groups,
        "open_time": ts,
        "open": df_1m["open"].values.astype(np.float64),
        "high": df_1m["high"].values.astype(np.float64),
        "low": df_1m["low"].values.astype(np.float64),
        "close": df_1m["close"].values.astype(np.float64),
        "volume": df_1m["volume"].values.astype(np.float64),
    })
    for col in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"):
        if col in df_1m.columns:
            work[col] = df_1m[col].values.astype(np.float64)
        else:
            work[col] = 0.0

    agg = work.groupby("group", sort=True).agg(
        open_time=("open_time", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
        trades=("trades", "sum"),
        taker_buy_volume=("taker_buy_volume", "sum"),
        taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
    ).reset_index(drop=True)
    return agg


def compute_features_for_tf(df_bars, symbol, tf_label):
    """Compute 105 features using the Rust batch engine."""
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df_bars, include_v11=_has_v11)
    return feat_df


def evaluate_timeframe(df_bars, symbol, tf_label, tf_minutes, horizons_bars):
    """Full evaluation: IC analysis + walk-forward backtest."""
    print(f"\n{'='*70}")
    print(f"TIMEFRAME: {tf_label} ({len(df_bars):,} bars, {len(df_bars)*tf_minutes/1440:.0f} days)")
    print(f"{'='*70}")

    t0 = time.time()
    feat_df = compute_features_for_tf(df_bars, symbol, tf_label)
    elapsed = time.time() - t0

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")]
    closes = df_bars["close"].values.astype(np.float64)
    n = len(closes)

    print(f"  {len(feature_names)} features computed in {elapsed:.1f}s")

    # IC analysis
    print(f"\n  --- IC Analysis ---")
    best_horizon = None
    best_ic_sum = 0
    best_ics = {}

    for h_bars in horizons_bars:
        h_minutes = h_bars * tf_minutes
        y = np.full(n, np.nan)
        y[:n-h_bars] = closes[h_bars:] / closes[:n-h_bars] - 1
        valid = ~np.isnan(y)
        vv = y[valid]
        if len(vv) > 10:
            p1, p99 = np.percentile(vv, [1, 99])
            y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))

        ics = []
        for fname in feature_names:
            vals = feat_df[fname].values if fname in feat_df.columns else np.zeros(n)
            ic = fast_ic(vals[WARMUP:], y[WARMUP:])
            ics.append((fname, ic))
        ics.sort(key=lambda x: abs(x[1]), reverse=True)

        n_pass = sum(1 for _, ic in ics if abs(ic) >= 0.01)
        ic_sum = sum(abs(ic) for _, ic in ics if abs(ic) >= 0.01)

        print(f"\n  Horizon = {h_bars} bars ({h_minutes}min): {n_pass} features pass IC≥0.01")
        for fname, ic in ics[:8]:
            tag = "✓" if abs(ic) >= 0.01 else " "
            print(f"    {tag} {fname:30s}  IC={ic:+.5f}")

        if ic_sum > best_ic_sum:
            best_ic_sum = ic_sum
            best_horizon = h_bars
            best_ics = {f: ic for f, ic in ics}

    if best_horizon is None:
        print("  No viable horizon found.")
        return None

    h_minutes = best_horizon * tf_minutes
    print(f"\n  Best horizon: {best_horizon} bars ({h_minutes} min), total |IC|={best_ic_sum:.4f}")

    # Walk-forward backtest
    print(f"\n  --- Walk-Forward Backtest ---")
    import lightgbm as lgb

    y = np.full(n, np.nan)
    y[:n-best_horizon] = closes[best_horizon:] / closes[:n-best_horizon] - 1
    valid = ~np.isnan(y)
    vv = y[valid]
    if len(vv) > 10:
        p1, p99 = np.percentile(vv, [1, 99])
        y = np.where(np.isnan(y), np.nan, np.clip(y, p1, p99))

    X = feat_df[feature_names].values[WARMUP:].astype(np.float64)
    y_w = y[WARMUP:]
    c_w = closes[WARMUP:]
    n_w = len(X)

    tr_end = int(n_w * 0.6)
    val_end = int(n_w * 0.8)

    valid_tr = ~np.isnan(y_w[:tr_end])
    valid_val = ~np.isnan(y_w[tr_end:val_end])

    X_tr = X[:tr_end][valid_tr]
    y_tr = y_w[:tr_end][valid_tr]
    X_val = X[tr_end:val_end][valid_val]
    y_val = y_w[tr_end:val_end][valid_val]

    print(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {n_w - val_end:,}")

    # Feature selection
    selected = greedy_ic_select(X_tr, y_tr, feature_names, top_k=15)
    sel_idx = [feature_names.index(f) for f in selected]
    print(f"  Selected: {selected}")

    params = {
        "max_depth": 4, "num_leaves": 12, "learning_rate": 0.01,
        "min_child_samples": 200, "reg_alpha": 0.5, "reg_lambda": 5.0,
        "subsample": 0.5, "colsample_bytree": 0.6,
        "objective": "regression", "verbosity": -1,
    }

    dtrain = lgb.Dataset(X_tr[:, sel_idx], label=y_tr)
    dval = lgb.Dataset(X_val[:, sel_idx], label=y_val, reference=dtrain)
    bst = lgb.train(params, dtrain, num_boost_round=500,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])

    # OOS test
    X_test = X[val_end:]
    pred = bst.predict(X_test[:, sel_idx])
    y_test = y_w[val_end:]
    c_test = c_w[val_end:]

    oos_ic = fast_ic(pred, y_test)
    print(f"  OOS IC: {oos_ic:.4f}")

    # Backtest with multiple deadzones
    pred_std = np.nanstd(pred)
    if pred_std < 1e-12:
        print("  Prediction variance too low.")
        return None
    z_pred = pred / pred_std

    print(f"\n  {'DZ':>4} {'Trades':>7} {'T/Day':>6} {'WinR':>6} {'AvgGross':>9} {'AvgNet':>9} {'TotalNet':>9} {'Sharpe':>7}")
    print(f"  {'-'*65}")

    best_result = None
    best_net_total = -999

    for dz in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        position = 0
        entry_price = 0.0
        entry_bar = 0
        min_hold = max(best_horizon // 2, 2)
        max_hold = best_horizon * 6
        trades_g = []
        trades_n = []

        for i in range(len(c_test)):
            if position != 0:
                held = i - entry_bar
                should_exit = False
                if held >= max_hold: should_exit = True
                elif held >= min_hold:
                    if position * z_pred[i] < -0.3: should_exit = True
                    elif abs(z_pred[i]) < 0.2: should_exit = True
                if should_exit:
                    pnl = position * (c_test[i] - entry_price) / entry_price
                    trades_g.append(pnl)
                    trades_n.append(pnl - COST_MAKER_RT_BPS / 10000)
                    position = 0

            if position == 0:
                if z_pred[i] > dz:
                    position = 1; entry_price = c_test[i]; entry_bar = i
                elif z_pred[i] < -dz:
                    position = -1; entry_price = c_test[i]; entry_bar = i

        if position != 0:
            pnl = position * (c_test[-1] - entry_price) / entry_price
            trades_g.append(pnl)
            trades_n.append(pnl - COST_MAKER_RT_BPS / 10000)

        nt = len(trades_n)
        if nt == 0:
            print(f"  {dz:>4.1f} {'—':>7}")
            continue

        g = np.array(trades_g)
        ne = np.array(trades_n)
        days = len(c_test) * tf_minutes / 1440

        sharpe = float(np.mean(ne) / max(np.std(ne), 1e-10) * np.sqrt(252 * (1440/tf_minutes) / max(len(c_test)/nt, 1)))

        print(f"  {dz:>4.1f} {nt:>7} {nt/days:>6.1f} {np.mean(ne>0)*100:>5.1f}% "
              f"{np.mean(g)*10000:>+8.1f}bp {np.mean(ne)*10000:>+8.1f}bp "
              f"{np.sum(ne)*100:>+8.1f}% {sharpe:>7.2f}")

        if np.sum(ne) > best_net_total:
            best_net_total = np.sum(ne)
            best_result = {
                "tf": tf_label, "tf_minutes": tf_minutes,
                "horizon_bars": best_horizon, "horizon_min": h_minutes,
                "deadzone": dz, "trades": nt, "trades_per_day": nt/days,
                "oos_ic": oos_ic, "win_rate": float(np.mean(ne>0)*100),
                "avg_gross_bps": float(np.mean(g)*10000),
                "avg_net_bps": float(np.mean(ne)*10000),
                "total_net_pct": float(np.sum(ne)*100),
                "sharpe": sharpe,
                "features": selected,
                "model": bst,
                "sel_idx": sel_idx,
            }

    return best_result


def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME COMPARISON: 1h vs 30m vs 15m vs 5m")
    print("=" * 70)

    symbol = "BTCUSDT"
    df_1m = pd.read_csv(f"/quant_system/data_files/{symbol}_1m.csv")
    print(f"Loaded {len(df_1m):,} 1m bars")

    results = {}

    # 1h bars
    df_1h = resample_1m_to(df_1m, 60)
    r = evaluate_timeframe(df_1h, symbol, "1h", 60,
                           horizons_bars=[1, 2, 3, 6, 12, 24])
    if r: results["1h"] = r

    # 30m bars
    df_30m = resample_1m_to(df_1m, 30)
    r = evaluate_timeframe(df_30m, symbol, "30m", 30,
                           horizons_bars=[1, 2, 4, 8, 16])
    if r: results["30m"] = r

    # 15m bars
    df_15m = resample_1m_to(df_1m, 15)
    r = evaluate_timeframe(df_15m, symbol, "15m", 15,
                           horizons_bars=[1, 2, 4, 8, 16])
    if r: results["15m"] = r

    # 5m bars
    df_5m = resample_1m_to(df_1m, 5)
    r = evaluate_timeframe(df_5m, symbol, "5m", 5,
                           horizons_bars=[2, 4, 6, 12, 24])
    if r: results["5m"] = r

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"\n{'TF':>4} {'Horizon':>8} {'IC':>6} {'DZ':>4} {'T/Day':>6} {'WinR':>6} "
          f"{'AvgGross':>9} {'AvgNet':>9} {'TotalNet':>9} {'Sharpe':>7}")
    print("-" * 80)

    for tf in ["1h", "30m", "15m", "5m"]:
        if tf not in results:
            print(f"{tf:>4} — no profitable config")
            continue
        r = results[tf]
        print(f"{tf:>4} {r['horizon_min']:>5}min {r['oos_ic']:>6.4f} {r['deadzone']:>4.1f} "
              f"{r['trades_per_day']:>6.1f} {r['win_rate']:>5.1f}% "
              f"{r['avg_gross_bps']:>+8.1f}bp {r['avg_net_bps']:>+8.1f}bp "
              f"{r['total_net_pct']:>+8.1f}% {r['sharpe']:>7.2f}")

    # Save best model if profitable
    best_tf = max(results.keys(), key=lambda k: results[k]["total_net_pct"]) if results else None
    if best_tf and results[best_tf]["total_net_pct"] > 0:
        r = results[best_tf]
        print(f"\n  BEST: {best_tf} — Net {r['total_net_pct']:+.1f}%, Sharpe {r['sharpe']:.2f}")

        out_dir = Path(f"models_v8/{symbol}_{best_tf}_v1")
        out_dir.mkdir(parents=True, exist_ok=True)
        r["model"].save_model(str(out_dir / f"lgbm_{best_tf}.txt"))
        with open(out_dir / f"lgbm_{best_tf}.pkl", "wb") as f:
            pickle.dump(r["model"], f)
        config = {
            "version": f"v8_{best_tf}",
            "symbol": symbol,
            "features": r["features"],
            "deadzone": r["deadzone"],
            "min_hold": max(r["horizon_bars"] // 2, 2),
            "horizon": r["horizon_bars"],
            "horizon_minutes": r["horizon_min"],
            "timeframe": best_tf,
            "oos_ic": r["oos_ic"],
            "sharpe": r["sharpe"],
            "avg_net_bps": r["avg_net_bps"],
        }
        with open(out_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved to {out_dir}")
    else:
        print(f"\n  No timeframe produced positive net P&L.")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
