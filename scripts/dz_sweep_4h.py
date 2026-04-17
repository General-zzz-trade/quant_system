#!/usr/bin/env python3
"""Quick deadzone sweep for BTC 4h model on 18-month OOS.

Loads the production ensemble once, computes predictions once, then
reruns run_backtest() at different deadzone values. Much faster than
the full walk-forward since we skip retraining.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from research.backtesting.backtest_4h_full import (  # noqa: E402
    BARS_PER_DAY,
    BARS_PER_MONTH,
    COST_MAKER_RT,
    COST_TAKER_RT,
    compute_features,
    resample_1m_to_4h,
    run_backtest,
)
from alpha.utils import compute_target, fast_ic  # noqa: E402
import json  # noqa: E402


def _load_pkl(path: str):
    # The pickle files here are our own model artefacts, produced by the
    # retrain pipeline, not untrusted input. Using joblib-free pickle keeps
    # parity with the live loader in decision/signals/alpha_signal.py.
    import pickle  # noqa: S403 — trusted local artefact

    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def main():
    print("=" * 72)
    print("BTC 4h deadzone sweep — 18-month OOS, production ensemble")
    print("=" * 72)

    df_1m = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv")
    df_4h = resample_1m_to_4h(df_1m)
    n = len(df_4h)
    print(f"Loaded {n:,} 4h bars")

    feat_df, feature_names = compute_features(df_4h)
    closes = df_4h["close"].values.astype(np.float64)
    timestamps = df_4h["open_time"].values.astype(np.int64)

    config_path = Path("models_v8/BTCUSDT_4h/config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    horizon = cfg["horizon"]
    base_dz = cfg["deadzone"]
    min_hold = cfg["min_hold"]
    max_hold = cfg["max_hold"]
    long_only = cfg["long_only"]

    print(
        f"Production: horizon={horizon} ({horizon*4}h), "
        f"dz={base_dz}, min_hold={min_hold}, max_hold={max_hold}, "
        f"long_only={long_only}"
    )

    lgbm_data = _load_pkl("models_v8/BTCUSDT_4h/lgbm_v8.pkl")
    xgb_data = _load_pkl("models_v8/BTCUSDT_4h/xgb_v8.pkl")

    import xgboost as xgb

    lgbm_model = lgbm_data["model"]
    xgb_model = xgb_data["model"]
    model_features = lgbm_data["features"]

    oos_start = n - BARS_PER_MONTH * 18
    oos_feat = feat_df[feature_names].values[oos_start:].astype(np.float64)
    oos_closes = closes[oos_start:]
    oos_ts = timestamps[oos_start:]

    X_oos = np.zeros((len(oos_feat), len(model_features)))
    for j, mf in enumerate(model_features):
        if mf in feature_names:
            X_oos[:, j] = oos_feat[:, feature_names.index(mf)]

    lgbm_pred = lgbm_model.predict(X_oos)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_oos))
    oos_pred = 0.5 * lgbm_pred + 0.5 * xgb_pred

    oos_ic = fast_ic(oos_pred, compute_target(oos_closes, horizon))
    print(
        f"OOS: {len(oos_closes)} bars ({len(oos_closes)/BARS_PER_DAY:.0f} days), "
        f"IC={oos_ic:.4f}"
    )

    pred_std = float(np.nanstd(oos_pred))
    z_scores = oos_pred / pred_std
    print("\nZ-score crossing frequency over 18m OOS:")
    for pct in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2):
        n_cross = int(np.sum(np.abs(z_scores) > pct))
        print(
            f"  |z| > {pct:.1f}: {n_cross} bars "
            f"({n_cross/len(z_scores)*100:.1f}%)"
        )

    print()
    print(
        f"{'dz':>6} {'cost':>8} {'trades':>7} {'winR':>6} "
        f"{'avg_net_bp':>11} {'total_ret':>10} {'maxDD':>7} {'sharpe':>7}"
    )
    print("-" * 72)

    for dz in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1):
        for cost_label, cost_bps in [("maker", COST_MAKER_RT), ("taker", COST_TAKER_RT)]:
            bt = run_backtest(
                oos_pred,
                oos_closes,
                oos_ts,
                horizon,
                dz,
                cost_bps,
                long_only,
                min_hold,
                max_hold,
            )
            trades = bt["trades"]
            if not trades:
                print(f"{dz:>6.2f} {cost_label:>8} {'0':>7}")
                continue
            net = np.array([t.net_pnl for t in trades])
            wins = sum(1 for t in trades if t.net_pnl > 0)
            wr = wins / len(trades) * 100
            avg_net_bp = float(np.mean(net) / 500 * 10000)
            total_ret = float(np.sum(net) / 10000 * 100)

            eq = 10000.0
            peak = eq
            max_dd = 0.0
            for t in trades:
                eq += t.net_pnl
                peak = max(peak, eq)
                max_dd = max(max_dd, (peak - eq) / peak)

            holds = np.array([t.hold_bars for t in trades])
            avg_hold_days = np.mean(holds) * 4 / 24
            tpy = 365 / max(avg_hold_days, 0.5)
            sharpe = float(
                np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy)
            ) if len(net) > 1 else 0.0

            print(
                f"{dz:>6.2f} {cost_label:>8} {len(trades):>7} "
                f"{wr:>5.1f}% {avg_net_bp:>+10.1f} "
                f"{total_ret:>+9.2f}% {max_dd*100:>6.2f}% {sharpe:>+7.2f}"
            )
        print()


if __name__ == "__main__":
    main()
