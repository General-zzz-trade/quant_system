#!/usr/bin/env python3
"""Trend-bypass signal backtest for BTC 4h.

Hypothesis: when the model's z-score is in the "noisy zone" (1.5 < |z| < 2.0)
but the market has a strong trend (24h price change >= +3%), a partial
position (25% size) might capture the rally that the main z>2.0 rule
misses.

Compares A) main only, B) trend only, C) main + trend-bypass combined.
Uses the same 18-month OOS as dz_sweep_4h.py.
"""
from __future__ import annotations
import sys
import json

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
)
from alpha.utils import compute_target, fast_ic  # noqa: E402

BASELINE_DZ = 2.0
TREND_Z_MIN = 1.5
TREND_RET_PCT = 0.03
TREND_SIZE = 0.25
MIN_HOLD = 3
MAX_HOLD = 36
NOTIONAL = 500.0


def _load_model(path: str):
    # Loaded from our own retrain pipeline output; parity with live loader.
    import pickle as _pkl  # noqa: S403

    with open(path, "rb") as f:
        return _pkl.load(f)  # noqa: S301


def _run_strategy(
    z, closes, timestamps, cost_bps,
    *,
    main_dz, main_size=1.0,
    trend_z_min=None, trend_ret_pct=None, trend_size=0.25,
    ret_24h=None, long_only=True,
):
    n = len(z)
    cost_frac = cost_bps / 10000
    trades = []
    pos = 0
    pos_size = 0.0
    ep = 0.0
    eb = 0
    score = 0.0
    equity = 10000.0

    trend_enabled = trend_z_min is not None and trend_ret_pct is not None

    for i in range(n):
        if pos != 0:
            held = i - eb
            should_exit = False
            if held >= MAX_HOLD:
                should_exit = True
            elif held >= MIN_HOLD:
                if pos * z[i] < -0.3 or abs(z[i]) < 0.2:
                    should_exit = True
            if should_exit:
                pnl_pct = pos * (closes[i] - ep) / ep
                gross = pnl_pct * NOTIONAL * pos_size
                cost = cost_frac * NOTIONAL * pos_size
                net = gross - cost
                equity += net
                trades.append({
                    "direction": pos, "size": pos_size,
                    "entry_price": ep, "exit_price": closes[i],
                    "gross_pnl": gross, "net_pnl": net,
                    "hold_bars": held,
                    "source": "main" if pos_size == main_size else "trend",
                })
                pos = 0

        if pos == 0:
            if z[i] > main_dz:
                pos, pos_size = 1, main_size
                ep, eb, score = closes[i], i, z[i]
            elif not long_only and z[i] < -main_dz:
                pos, pos_size = -1, main_size
                ep, eb, score = closes[i], i, z[i]
            elif trend_enabled:
                ret = ret_24h[i] if ret_24h is not None else 0.0
                if z[i] > trend_z_min and ret >= trend_ret_pct:
                    pos, pos_size = 1, trend_size
                    ep, eb, score = closes[i], i, z[i]
                elif (not long_only and z[i] < -trend_z_min
                      and ret <= -trend_ret_pct):
                    pos, pos_size = -1, trend_size
                    ep, eb = closes[i], i
                    score = z[i]  # noqa: F841 — kept for potential logging

    if pos != 0:
        pnl_pct = pos * (closes[-1] - ep) / ep
        gross = pnl_pct * NOTIONAL * pos_size
        cost = cost_frac * NOTIONAL * pos_size
        net = gross - cost
        trades.append({
            "direction": pos, "size": pos_size,
            "entry_price": ep, "exit_price": closes[-1],
            "gross_pnl": gross, "net_pnl": net,
            "hold_bars": n - 1 - eb,
            "source": "main" if pos_size == main_size else "trend",
        })

    return trades


def _summarize(trades, label):
    if not trades:
        print(f"  [{label}] no trades")
        return
    net = np.array([t["net_pnl"] for t in trades])
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    wr = wins / len(trades) * 100
    avg_net_bp = float(np.mean(net) / NOTIONAL * 10000)
    total_ret = float(np.sum(net) / 10000 * 100)

    eq, peak, max_dd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t["net_pnl"]
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    holds = np.array([t["hold_bars"] for t in trades])
    avg_hold_days = np.mean(holds) * 4 / 24
    tpy = 365 / max(avg_hold_days, 0.5)
    sharpe = float(
        np.mean(net) / max(np.std(net, ddof=1), 1e-10) * np.sqrt(tpy)
    ) if len(net) > 1 else 0.0

    main_n = sum(1 for t in trades if t["source"] == "main")
    trend_n = len(trades) - main_n
    source_str = f"(main={main_n}, trend={trend_n})" if trend_n else ""

    print(
        f"  [{label}] trades={len(trades)} {source_str}  "
        f"WR={wr:.1f}%  avg_net={avg_net_bp:+.1f}bp  "
        f"total={total_ret:+.2f}%  maxDD={max_dd*100:.2f}%  "
        f"Sharpe={sharpe:+.2f}"
    )


def main():
    print("=" * 72)
    print("BTC 4h trend-bypass backtest — 18 month OOS")
    print(f"  A) main: z > {BASELINE_DZ} → 100% long")
    print(f"  B) trend: {TREND_Z_MIN} < z < {BASELINE_DZ} AND "
          f"24h ret >= {TREND_RET_PCT*100:.0f}% → {TREND_SIZE*100:.0f}% long")
    print("  C) main + trend combined")
    print("=" * 72)

    df_1m = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv")
    df_4h = resample_1m_to_4h(df_1m)
    n = len(df_4h)

    feat_df, feature_names = compute_features(df_4h)
    closes = df_4h["close"].values.astype(np.float64)
    timestamps = df_4h["open_time"].values.astype(np.int64)

    with open("models_v8/BTCUSDT_4h/config.json") as f:
        cfg = json.load(f)
    horizon = cfg["horizon"]
    long_only = cfg["long_only"]

    lgbm_data = _load_model("models_v8/BTCUSDT_4h/lgbm_v8.pkl")
    xgb_data = _load_model("models_v8/BTCUSDT_4h/xgb_v8.pkl")
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
    print(f"OOS IC: {oos_ic:.4f}  bars: {len(oos_closes)} "
          f"({len(oos_closes)/BARS_PER_DAY:.0f} days)")

    pred_std = float(np.nanstd(oos_pred))
    z = oos_pred / pred_std

    ret_24h = np.zeros(len(oos_closes))
    ret_24h[6:] = (oos_closes[6:] - oos_closes[:-6]) / oos_closes[:-6]

    bypass_cand = int(np.sum(
        (z > TREND_Z_MIN) & (z <= BASELINE_DZ) & (ret_24h >= TREND_RET_PCT)
    ))
    print(f"Bypass-candidate bars (long only): {bypass_cand} "
          f"({bypass_cand/len(z)*100:.1f}%)")

    for cost_label, cost_bps in [("maker", COST_MAKER_RT), ("taker", COST_TAKER_RT)]:
        print(f"\n--- {cost_label} ({cost_bps}bp round-trip) ---")

        trades_main = _run_strategy(
            z, oos_closes, oos_ts, cost_bps,
            main_dz=BASELINE_DZ, main_size=1.0, long_only=long_only,
        )
        _summarize(trades_main, "A) main only")

        trades_trend = _run_strategy(
            z, oos_closes, oos_ts, cost_bps,
            main_dz=99, main_size=1.0,
            trend_z_min=TREND_Z_MIN, trend_ret_pct=TREND_RET_PCT,
            trend_size=TREND_SIZE, ret_24h=ret_24h, long_only=long_only,
        )
        _summarize(trades_trend, "B) trend only")

        trades_combined = _run_strategy(
            z, oos_closes, oos_ts, cost_bps,
            main_dz=BASELINE_DZ, main_size=1.0,
            trend_z_min=TREND_Z_MIN, trend_ret_pct=TREND_RET_PCT,
            trend_size=TREND_SIZE, ret_24h=ret_24h, long_only=long_only,
        )
        _summarize(trades_combined, "C) main + trend")


if __name__ == "__main__":
    main()
