"""Replay the 1D paper runner over the last 90 days.

For each day in the holdout window we feed the SAME code path the live
paper runner uses (runner.daily_paper_runner.run_symbol-equivalent
logic) and aggregate the resulting trades. The output should match the
walk-forward holdout numbers reported in models_v8/{sym}_1d/config.json
within statistical noise.

Why: validates that the live runner's signal/position state machine is
implemented correctly, without waiting a real week. Catches off-by-one
bugs, wrong dz/min_hold lookup, sign errors, etc. Does NOT validate
data freshness or systemd plumbing — those still need real time.

Run:
  python3 -m scripts.daily_replay_validate
  python3 -m scripts.daily_replay_validate --days 180   # longer window
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from runner.daily_paper_runner import (  # noqa: E402
    COST_BPS,
    NOTIONAL,
    _decide_action,
    _load_model_bundle,
    _predict,
)
from scripts.daily_poc import (  # noqa: E402
    compute_1d_features,
    resample_1h_to_1d,
)


def replay(symbol: str, model_dir: str, macro: pd.DataFrame, days: int):
    lgbm, xgb_m, cfg = _load_model_bundle(model_dir)
    feature_names_cfg = cfg["features"]
    dz = cfg["deadzone"]
    min_hold = cfg["min_hold"]
    max_hold = cfg["max_hold"]
    pred_mean = cfg["zscore_pred_mean"]
    pred_std = max(cfg["zscore_pred_std"], 1e-10)

    df_1h = pd.read_csv(f"/quant_system/data_files/{symbol}_1h.csv")
    df_1d = resample_1h_to_1d(df_1h)
    df_1d = df_1d[df_1d["date"] >= macro["date"].min()].reset_index(drop=True)

    feat_df, feature_names = compute_1d_features(df_1d, macro)
    X_all = feat_df[feature_names].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    closes = df_1d["close"].values.astype(np.float64)
    dates = df_1d["date"].values

    # Map config feature order → live X column index
    col_idx = []
    for name in feature_names_cfg:
        if name in feature_names:
            col_idx.append(feature_names.index(name))
        else:
            col_idx.append(-1)

    n = len(df_1d)
    start_i = max(0, n - days)

    state = {
        "signal": 0, "entry_price": 0.0, "entry_idx": -1,
        "bars_held": 0, "cum_pnl": 0.0, "trades": 0, "wins": 0,
    }
    trades = []

    for i in range(start_i, n):
        # Build aligned X_row
        X_row = np.zeros(len(feature_names_cfg), dtype=np.float64)
        for j, idx in enumerate(col_idx):
            if idx >= 0:
                X_row[j] = X_all[i, idx]
        pred = _predict(lgbm, xgb_m, X_row)
        z = (pred - pred_mean) / pred_std
        new_signal, reason = _decide_action(z, dz, state["signal"])

        if state["signal"] == 0 and new_signal != 0:
            state["signal"] = new_signal
            state["entry_price"] = closes[i]
            state["entry_idx"] = i
            state["bars_held"] = 0
        elif state["signal"] != 0:
            state["bars_held"] += 1
            force_close = False
            close_reason = ""
            if state["bars_held"] >= max_hold:
                force_close = True
                close_reason = "max_hold"
            elif state["bars_held"] >= min_hold:
                if reason in ("z_reversal", "z_decay"):
                    force_close = True
                    close_reason = reason
            if force_close:
                ep = state["entry_price"]
                pnl_pct = state["signal"] * (closes[i] - ep) / ep
                gross = pnl_pct * NOTIONAL
                cost = COST_BPS / 10000 * NOTIONAL
                net = gross - cost
                state["cum_pnl"] += net
                state["trades"] += 1
                if net > 0:
                    state["wins"] += 1
                trades.append({
                    "entry_date": str(dates[state["entry_idx"]]),
                    "exit_date": str(dates[i]),
                    "direction": state["signal"],
                    "entry_price": float(ep),
                    "exit_price": float(closes[i]),
                    "bars_held": state["bars_held"],
                    "gross_pnl": float(gross),
                    "net_pnl": float(net),
                    "close_reason": close_reason,
                })
                state["signal"] = 0
                state["entry_price"] = 0.0
                state["entry_idx"] = -1
                state["bars_held"] = 0

    return trades, state, dates[start_i], dates[n - 1], cfg


def summarize(trades, label):
    if not trades:
        print(f"  [{label}] no trades in window")
        return
    nets = np.array([t["net_pnl"] for t in trades])
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    wr = wins / len(trades) * 100
    avg_bp = float(np.mean(nets) / NOTIONAL * 10000)
    total = float(np.sum(nets) / 10000 * 100)

    eq, peak, max_dd = 10000.0, 10000.0, 0.0
    for t in trades:
        eq += t["net_pnl"]
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)

    holds = np.array([t["bars_held"] for t in trades])
    tpy = 365 / max(np.mean(holds), 0.5)
    sharpe = float(
        np.mean(nets) / max(np.std(nets, ddof=1), 1e-10) * np.sqrt(tpy)
    ) if len(nets) > 1 else 0.0

    longs = sum(1 for t in trades if t["direction"] > 0)
    shorts = len(trades) - longs

    print(f"  [{label}] trades={len(trades)} (L={longs}/S={shorts})  "
          f"WR={wr:.0f}%  hold={np.mean(holds):.1f}d  "
          f"avg={avg_bp:+.1f}bp  total={total:+.2f}%  "
          f"maxDD={max_dd*100:.2f}%  Sharpe={sharpe:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Replay 1D paper runner over recent days")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    args = parser.parse_args()

    macro = pd.read_csv("/quant_system/data_files/cross_market_daily.csv")
    macro["date"] = pd.to_datetime(macro["date"]).dt.date

    print("=" * 72)
    print(f"1D paper replay validation — last {args.days} days")
    print("=" * 72)

    for sym in args.symbols:
        model_dir = f"/quant_system/models_v8/{sym}_1d"
        if not Path(model_dir).exists():
            print(f"  {sym}: no model — skipping")
            continue
        trades, state, d0, d1, cfg = replay(sym, model_dir, macro, args.days)
        wf = cfg.get("walk_forward", {})
        print(f"\n{sym} ({d0} → {d1}, dz={cfg['deadzone']}):")
        summarize(trades, "live-replay")
        print(f"  walk-forward expected: Sharpe={wf.get('sharpe', 0):.2f}, "
              f"WR={wf.get('win_rate', 0):.1f}%, "
              f"trades={wf.get('trades', 0)} (over 5y), "
              f"holdout_IC={cfg.get('holdout_ic_90d', 0):+.4f}")

        # Pass/fail gate
        if trades:
            nets = [t["net_pnl"] for t in trades]
            sharpe = float(np.mean(nets) / max(np.std(nets, ddof=1), 1e-10)
                           * np.sqrt(365 / max(np.mean([t["bars_held"] for t in trades]), 0.5))
                           ) if len(nets) > 1 else 0.0
            wf_sharpe = wf.get("sharpe", 0)
            ratio = sharpe / wf_sharpe if wf_sharpe != 0 else 0
            verdict = "PASS" if ratio > 0.5 else (
                "DEGRADED" if ratio > 0 else "FAIL"
            )
            print(f"  → ratio (replay/WF Sharpe) = {ratio:+.2f}  → {verdict}")
        else:
            n_bars_in = args.days
            print(f"  → no trades in {n_bars_in} days "
                  f"(expected ~{wf.get('trades', 0)/(5*365)*n_bars_in:.1f} from WF rate)")


if __name__ == "__main__":
    main()
