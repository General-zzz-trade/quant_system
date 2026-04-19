"""Walk-forward backtest validation for 1D Phase A + B model changes.

Compares 4 training methodologies on the SAME WF setup as daily_poc.py
(6mo train / 3mo test, full 5-year OOS):

  V0  ORIGINAL          h=1 only, no sample weights        (Phase 1 baseline)
  V1  RECENCY           h=1 only, recency weights          (Phase A only)
  V2  MULTI_HORIZON     h=1+3+7 IC-weighted, no weights    (Phase B only)
  V3  AB_COMBINED       h=1+3+7 IC-weighted, recency       (Phase A + B, what we shipped)

Per-symbol opt-in for recency (BTC = no, ETH = yes) is the production
config (PROD_CONFIG.use_recency_weights). For backtesting we test both
modes per symbol so we can see the actual marginal impact.

Run:
  python3 -m scripts.daily_phase_ab_backtest
  python3 -m scripts.daily_phase_ab_backtest --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from scripts.daily_poc import (  # noqa: E402
    BARS_PER_MONTH, COST_TAKER_RT, MAX_HOLD, MIN_HOLD,
    compute_1d_features, resample_1h_to_1d, run_backtest,
)
from scripts.train_1d_production import _recency_weights  # noqa: E402

DZ_BEST = {"BTCUSDT": 1.75, "ETHUSDT": 1.75}
HORIZONS = [1, 3, 7]
TRAIN_BARS = BARS_PER_MONTH * 6   # 6 months
TEST_BARS = BARS_PER_MONTH * 3    # 3 months


def _make_target(closes, horizon):
    log_rets = np.log(closes[horizon:] / closes[:-horizon])
    target = np.full(len(closes), np.nan)
    target[:-horizon] = log_rets
    return target


def _train_one(X_tr, y_tr, sample_weights=None):
    import lightgbm as lgb
    import xgboost as xgb

    valid = ~np.isnan(y_tr)
    X_tr = X_tr[valid]
    y_tr = y_tr[valid]
    sw = sample_weights[valid] if sample_weights is not None else None
    if len(X_tr) < 60:
        return None, None

    lgb_params = {"objective": "regression", "metric": "rmse",
                  "num_leaves": 31, "learning_rate": 0.03,
                  "feature_fraction": 0.85, "bagging_fraction": 0.85,
                  "bagging_freq": 3, "min_data_in_leaf": 20,
                  "verbosity": -1}
    lgb_dataset = lgb.Dataset(X_tr, label=y_tr, weight=sw)
    lgb_model = lgb.train(lgb_params, lgb_dataset, num_boost_round=200)

    xgb_params = {"objective": "reg:squarederror", "eta": 0.03,
                  "max_depth": 5, "subsample": 0.85, "colsample_bytree": 0.85,
                  "min_child_weight": 10, "verbosity": 0}
    xgb_dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw)
    xgb_model = xgb.train(xgb_params, xgb_dtrain, num_boost_round=200)
    return lgb_model, xgb_model


def walk_forward_v0_v1(X, closes, dates, recency: bool):
    """Single-horizon (h=1) walk-forward, optional recency weights."""
    import xgboost as xgb
    n = len(X)
    preds = np.full(n, np.nan)
    target = _make_target(closes, 1)

    start = TRAIN_BARS
    while start + TEST_BARS <= n:
        tr_start = max(0, start - TRAIN_BARS)
        X_tr = X[tr_start:start]
        y_tr = target[tr_start:start]
        d_tr = dates[tr_start:start]

        sw = _recency_weights(d_tr) if recency else None
        lgb_m, xgb_m = _train_one(X_tr, y_tr, sw)
        if lgb_m is None:
            start += TEST_BARS
            continue

        end = min(start + TEST_BARS, n)
        X_te = X[start:end]
        p_lgb = lgb_m.predict(X_te)
        p_xgb = xgb_m.predict(xgb.DMatrix(X_te))
        preds[start:end] = 0.5 * p_lgb + 0.5 * p_xgb
        start += TEST_BARS
    return preds


def walk_forward_v2_v3(X, closes, dates, recency: bool):
    """Multi-horizon (h=1/3/7) IC-weighted walk-forward."""
    import xgboost as xgb
    n = len(X)
    preds = np.full(n, np.nan)

    start = TRAIN_BARS
    while start + TEST_BARS <= n:
        tr_start = max(0, start - TRAIN_BARS)
        end = min(start + TEST_BARS, n)
        X_tr_raw = X[tr_start:start]
        d_tr = dates[tr_start:start]
        sw_full = _recency_weights(d_tr) if recency else None

        # Train + measure IC per horizon (use within-train holdout = last 25%)
        h_models = {}
        h_ic = {}
        train_split = int(0.75 * (start - tr_start))

        for horizon in HORIZONS:
            target_h = _make_target(closes, horizon)
            y_tr = target_h[tr_start:start]
            sw = sw_full[:train_split] if sw_full is not None else None
            lgb_m, xgb_m = _train_one(
                X_tr_raw[:train_split], y_tr[:train_split], sw,
            )
            if lgb_m is None:
                continue
            X_val = X_tr_raw[train_split:]
            y_val = y_tr[train_split:]
            valid = ~np.isnan(y_val)
            if valid.sum() < 10:
                continue
            p_lgb = lgb_m.predict(X_val)
            p_xgb = xgb_m.predict(xgb.DMatrix(X_val))
            p_ens = 0.5 * p_lgb + 0.5 * p_xgb
            ic = float(np.corrcoef(p_ens[valid], y_val[valid])[0, 1])
            h_models[horizon] = (lgb_m, xgb_m)
            h_ic[horizon] = ic

        if not h_models:
            start += TEST_BARS
            continue

        # IC-weighted
        weights = np.array([max(h_ic.get(h, 0.0), 0.0) for h in HORIZONS])
        if weights.sum() <= 0:
            weights = np.ones(len(HORIZONS)) / len(HORIZONS)
        else:
            weights = weights / weights.sum()

        # Predict on test
        X_te = X[start:end]
        combined = np.zeros(end - start)
        for i, h in enumerate(HORIZONS):
            if h in h_models:
                lgb_m, xgb_m = h_models[h]
                p_lgb = lgb_m.predict(X_te)
                p_xgb = xgb_m.predict(xgb.DMatrix(X_te))
                combined += weights[i] * (0.5 * p_lgb + 0.5 * p_xgb)
        preds[start:end] = combined
        start += TEST_BARS
    return preds


def _summarize(trades, label):
    if not trades:
        print(f"  [{label:18s}] no trades"); return None
    nets = np.array([t.net_pnl for t in trades])
    wins = sum(1 for t in trades if t.net_pnl > 0)
    wr = wins / len(trades) * 100
    avg_bp = float(np.mean(nets) / 500 * 10000)
    total = float(np.sum(nets) / 10000 * 100)
    eq, peak, mdd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t.net_pnl; peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    holds = np.array([t.hold_days for t in trades])
    tpy = 365 / max(np.mean(holds), 0.5)
    sharpe = float(np.mean(nets) / max(np.std(nets, ddof=1), 1e-10)
                   * np.sqrt(tpy)) if len(nets) > 1 else 0.0
    print(f"  [{label:18s}] n={len(trades):>3d}  WR={wr:>4.0f}%  "
          f"hold={np.mean(holds):>4.1f}d  avg={avg_bp:>+6.1f}bp  "
          f"total={total:>+6.2f}%  DD={mdd*100:>5.2f}%  Sh={sharpe:>+5.2f}")
    return {"sharpe": sharpe, "total": total, "mdd": mdd*100,
            "wr": wr, "n": len(trades)}


def run_symbol(symbol):
    print(f"\n{symbol} 1D walk-forward backtest (Phase A+B validation)")
    df = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df)
    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    df_1d = df_1d[df_1d["date"] >= macro["date"].min()].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    closes = df_1d["close"].values.astype(np.float64)
    dates = df_1d["date"].values
    X = feat_df[feature_names].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Total: {len(X):,} days, features: {len(feature_names)}")
    print()

    dz = DZ_BEST[symbol]
    results = {}

    print("Variants:")
    for label, recency, multi in [
        ("V0_ORIGINAL", False, False),
        ("V1_RECENCY", True, False),
        ("V2_MULTI_HORIZON", False, True),
        ("V3_AB_COMBINED", True, True),
    ]:
        if multi:
            preds = walk_forward_v2_v3(X, closes, dates, recency)
        else:
            preds = walk_forward_v0_v1(X, closes, dates, recency)
        valid = ~np.isnan(preds)
        if valid.sum() < 30:
            print(f"  [{label:18s}] insufficient predictions"); continue
        oos_p = preds[valid]
        oos_c = closes[valid]
        trades = run_backtest(oos_p, oos_c, dz, COST_TAKER_RT,
                              long_only=False, min_hold=MIN_HOLD, max_hold=MAX_HOLD)
        r = _summarize(trades, label)
        if r:
            results[label] = r
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", choices=["BTCUSDT", "ETHUSDT", "BOTH"], default="BOTH")
    args = p.parse_args()

    print("=" * 90)
    print("1D Phase A + B walk-forward backtest validation")
    print(f"  WF: train={TRAIN_BARS//BARS_PER_MONTH}mo, test={TEST_BARS//BARS_PER_MONTH}mo, "
          f"taker cost={COST_TAKER_RT}bp")
    print("  V0=original h=1 unweighted (Phase 1 baseline)")
    print("  V1=h=1 + recency weights (Phase A only)")
    print("  V2=multi-horizon h=1/3/7 IC-weighted (Phase B only)")
    print("  V3=multi-horizon + recency (Phase A+B = what we shipped)")
    print("=" * 90)

    syms = ["BTCUSDT", "ETHUSDT"] if args.symbol == "BOTH" else [args.symbol]
    all_results = {}
    for s in syms:
        all_results[s] = run_symbol(s)

    print("\n" + "=" * 90)
    print("RANKING by Sharpe")
    print("=" * 90)
    for sym, results in all_results.items():
        if not results:
            continue
        baseline = results.get("V0_ORIGINAL", {}).get("sharpe", 0)
        ranked = sorted(results.items(), key=lambda kv: -kv[1]["sharpe"])
        print(f"\n{sym}:")
        for label, r in ranked:
            marker = "🏆" if label == ranked[0][0] else "  "
            delta = r["sharpe"] - baseline
            print(f"  {marker} {label:18s} Sharpe={r['sharpe']:+.2f} "
                  f"(Δ={delta:+.2f})  return={r['total']:+.2f}% "
                  f"DD={r['mdd']:.2f}% trades={r['n']}")


if __name__ == "__main__":
    main()
