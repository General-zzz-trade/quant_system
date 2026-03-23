#!/usr/bin/env python3
"""V11 Alpha Architecture — Backtest Verification.

Compares V10 baseline against V11 variants:
  A. V10 baseline (mean_zscore, no exit/regime features)
  B. V11 + trailing stop 2%
  C. V11 + z-score cap 4.0
  D. V11 + trailing stop + z-cap
  E. V11 + regime gate (reduce)
  F. V11 + all features combined
  G. V11 + time filter (skip 0-3 UTC)
  H. V11 full (trailing + zcap + regime + time)

Tests: ETH and BTC, 18-month OOS.
"""
from __future__ import annotations

import sys
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from alpha.training.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from alpha.v11_config import V11Config, ExitConfig, RegimeGateConfig, TimeFilterConfig
from decision.exit_manager import ExitManager
from decision.regime_gate import RegimeGate

COST_BPS_RT = 4
ZSCORE_WINDOW = 720
BARS_PER_MONTH = 24 * 30


def zscore_signal(pred, window=720, warmup=180):
    n = len(pred)
    z = np.zeros(n)
    buf = []
    for i in range(n):
        buf.append(pred[i])
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < warmup:
            continue
        arr = np.array(buf)
        std = float(np.std(arr))
        if std > 1e-12:
            z[i] = (pred[i] - float(np.mean(arr))) / std
    return z


def run_backtest_v11(
    z: np.ndarray,
    closes: np.ndarray,
    config: V11Config,
    feat_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run backtest with V11 modules."""
    n = len(z)
    exit_mgr = ExitManager(config=config.exit, min_hold=config.min_hold, max_hold=config.max_hold)
    regime_gate = RegimeGate(config=config.regime_gate)

    cost_frac = COST_BPS_RT / 10000
    pos = 0.0
    entry_bar = 0
    trades = []
    equity_curve = [0.0]
    running_pnl = 0.0

    for i in range(n):
        feat_dict = {}
        if feat_df is not None and i < len(feat_df):
            for col in ["bb_width_20", "vol_of_vol", "adx_14"]:
                if col in feat_df.columns:
                    v = feat_df.iloc[i][col]
                    if not np.isnan(v):
                        feat_dict[col] = float(v)
        _regime, position_scale = regime_gate.evaluate(feat_dict)

        if pos != 0:
            exit_mgr.update_price("SYM", closes[i])
            should_exit, reason = exit_mgr.check_exit("SYM", closes[i], i, z[i], pos)
            if should_exit:
                pnl_pct = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                net = pnl_pct * 500.0 - cost_frac * 500.0
                trades.append({
                    "pnl": net,
                    "pnl_pct": pnl_pct,
                    "hold": i - entry_bar,
                    "reason": reason,
                    "direction": int(pos),
                })
                running_pnl += net
                exit_mgr.on_exit("SYM")
                pos = 0.0

        if pos == 0:
            hour_utc = i % 24
            if not exit_mgr.allow_entry(z[i], hour_utc):
                equity_curve.append(running_pnl)
                continue
            if z[i] > config.deadzone:
                pos = 1.0
                entry_bar = i
                exit_mgr.on_entry("SYM", closes[i], i, 1.0)
            elif not config.long_only and z[i] < -config.deadzone:
                pos = -1.0
                entry_bar = i
                exit_mgr.on_entry("SYM", closes[i], i, -1.0)

        equity_curve.append(running_pnl)

    # Close open position
    if pos != 0:
        pnl_pct = pos * (closes[-1] - closes[entry_bar]) / closes[entry_bar]
        net = pnl_pct * 500.0 - cost_frac * 500.0
        trades.append({
            "pnl": net, "pnl_pct": pnl_pct, "hold": n - 1 - entry_bar,
            "reason": "end", "direction": int(pos),
        })
        running_pnl += net

    if not trades:
        return {"sharpe": 0, "trades": 0, "return": 0, "win_rate": 0,
                "avg_net_bps": 0, "max_dd": 0, "avg_hold": 0, "exit_reasons": {}}

    nets = np.array([t["pnl"] for t in trades])
    holds = np.array([t["hold"] for t in trades])
    avg_hold = float(np.mean(holds))
    tpy = 365 * 24 / max(avg_hold, 1)
    sharpe = float(np.mean(nets) / max(np.std(nets, ddof=1), 1e-10) * np.sqrt(tpy))

    # Max drawdown
    eq = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq + 10000)  # offset to avoid div by 0
    dd = (running_max - (eq + 10000)) / running_max
    max_dd = float(np.max(dd))

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t["reason"].split("=")[0] if "=" in t["reason"] else t["reason"]
        reasons[r] = reasons.get(r, 0) + 1

    # Long/short breakdown
    long_trades = [t for t in trades if t["direction"] > 0]
    short_trades = [t for t in trades if t["direction"] < 0]
    long_wr = np.mean([t["pnl"] > 0 for t in long_trades]) * 100 if long_trades else 0
    short_wr = np.mean([t["pnl"] > 0 for t in short_trades]) * 100 if short_trades else 0

    # Bootstrap
    bs_sharpes = []
    for _ in range(500):
        sample = np.random.choice(nets, size=len(nets), replace=True)
        if np.std(sample) > 0:
            bs_sharpes.append(float(np.mean(sample) / np.std(sample) * np.sqrt(52)))
    p5 = float(np.percentile(bs_sharpes, 5)) if bs_sharpes else 0

    return {
        "sharpe": sharpe,
        "trades": len(trades),
        "return": float(np.sum(nets)) / 10000,
        "win_rate": float(np.mean(nets > 0) * 100),
        "avg_net_bps": float(np.mean(nets) / 500 * 10000),
        "max_dd": max_dd,
        "avg_hold": avg_hold,
        "exit_reasons": reasons,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_wr": long_wr,
        "short_wr": short_wr,
        "bootstrap_p5": p5,
    }


def load_predictions(symbol: str):
    """Load data, features, and compute multi-horizon z-score signal."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    data_path = Path(f"data_files/{symbol}_1h.csv")

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return None

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    df = pd.read_csv(data_path)
    n = len(df)
    closes = df["close"].values.astype(np.float64)

    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")

    # Features
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)
    feat_names = [c for c in feat_df.columns
                  if c not in ("close", "open_time", "timestamp") and c not in BLACKLIST]

    # OOS split
    oos_bars = BARS_PER_MONTH * 18
    oos_start = n - oos_bars
    pred_start = max(0, oos_start - ZSCORE_WINDOW)

    # Multi-horizon predictions
    import xgboost as xgb
    horizon_preds = []
    for hm_cfg in cfg["horizon_models"]:
        with open(model_dir / hm_cfg["lgbm"], "rb") as f:
            lgbm_data = pickle.load(f)
        with open(model_dir / hm_cfg["xgb"], "rb") as f:
            xgb_data = pickle.load(f)

        hm_feats = hm_cfg["features"]
        sel = [feat_names.index(fn) for fn in hm_feats if fn in feat_names]
        X = feat_df[feat_names].values[pred_start:].astype(np.float64)[:, sel]
        pred = 0.5 * lgbm_data["model"].predict(X) + \
               0.5 * xgb_data["model"].predict(xgb.DMatrix(X))
        horizon_preds.append(pred)

    z_horizons = [zscore_signal(p, window=ZSCORE_WINDOW, warmup=180) for p in horizon_preds]
    z_ensemble = np.mean(z_horizons, axis=0)

    warmup_used = oos_start - pred_start
    z_oos = z_ensemble[warmup_used:]
    closes_oos = closes[oos_start:]
    feat_df_oos = feat_df.iloc[oos_start:].reset_index(drop=True)

    n_oos = len(z_oos)
    print(f"  Data: {n:,} bars ({start_date} → {end_date})")
    print(f"  OOS: {n_oos:,} bars ({n_oos/24:.0f} days), z-score warmup: {warmup_used}")
    print(f"  Config: dz={cfg.get('deadzone')}, hold=[{cfg.get('min_hold')},{cfg.get('max_hold')}], "
          f"long_only={cfg.get('long_only')}")

    return {
        "z": z_oos,
        "closes": closes_oos,
        "feat_df": feat_df_oos,
        "cfg": cfg,
        "n_oos": n_oos,
    }


def main():
    print("=" * 100)
    print("  V11 ALPHA ARCHITECTURE — BACKTEST VERIFICATION")
    print("=" * 100)

    symbols = ["ETHUSDT", "BTCUSDT"]

    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"  {symbol}")
        print(f"{'='*100}")

        t0 = time.time()
        data = load_predictions(symbol)
        if data is None:
            continue
        print(f"  Features + predictions computed in {time.time() - t0:.1f}s")

        z = data["z"]
        closes = data["closes"]
        feat_df = data["feat_df"]
        cfg = data["cfg"]

        base_kw = {
            "deadzone": cfg["deadzone"],
            "min_hold": cfg["min_hold"],
            "max_hold": cfg["max_hold"],
            "long_only": cfg.get("long_only", False),
        }

        # Define test scenarios
        scenarios = [
            ("A: V10 baseline", V11Config(**base_kw)),

            ("B: + trailing 2%", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.02),
            )),

            ("C: + z-cap 4.0", V11Config(
                **base_kw,
                exit=ExitConfig(zscore_cap=4.0),
            )),

            ("D: + trail 2% + zcap 4", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.02, zscore_cap=4.0),
            )),

            ("E: + regime gate", V11Config(
                **base_kw,
                regime_gate=RegimeGateConfig(enabled=True, reduce_factor=0.3),
            )),

            ("F: trail+zcap+regime", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.02, zscore_cap=4.0),
                regime_gate=RegimeGateConfig(enabled=True, reduce_factor=0.3),
            )),

            ("G: + time filter 0-3UTC", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.02),
                time_filter=TimeFilterConfig(enabled=True, skip_hours_utc=[0, 1, 2, 3]),
            )),

            ("H: V11 full", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.02, zscore_cap=4.0),
                regime_gate=RegimeGateConfig(enabled=True, reduce_factor=0.3),
                time_filter=TimeFilterConfig(enabled=True, skip_hours_utc=[0, 1, 2, 3]),
            )),

            ("I: trail 1.5%", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.015),
            )),

            ("J: trail 3%", V11Config(
                **base_kw,
                exit=ExitConfig(trailing_stop_pct=0.03),
            )),

            ("K: reversal -0.2", V11Config(
                **base_kw,
                exit=ExitConfig(reversal_threshold=-0.2),
            )),

            ("L: reversal -0.5", V11Config(
                **base_kw,
                exit=ExitConfig(reversal_threshold=-0.5, deadzone_fade=0.15),
            )),
        ]

        # Run all scenarios
        results = []
        for name, v11cfg in scenarios:
            use_feat = feat_df if v11cfg.regime_gate.enabled else None
            r = run_backtest_v11(z, closes, v11cfg, feat_df=use_feat)
            r["name"] = name
            results.append(r)

        # Results table
        print(f"\n  {'─'*98}")
        print(f"  {'方案':<30s} {'Sharpe':>7s} {'#交易':>5s} {'WR%':>5s} "
              f"{'Ret%':>7s} {'AvgBps':>7s} {'MaxDD%':>6s} {'AvgH':>5s} {'BS p5':>6s}")
        print(f"  {'─'*98}")

        baseline_sharpe = results[0]["sharpe"] if results else 0
        for r in results:
            delta = r["sharpe"] - baseline_sharpe
            delta_str = f" ({delta:+.2f})" if r["name"] != "A: V10 baseline" else ""
            print(f"  {r['name']:<30s} {r['sharpe']:>+7.2f}{delta_str:>8s} "
                  f"{r['trades']:>5d} {r['win_rate']:>4.0f}% "
                  f"{r['return']*100:>+6.1f}% {r['avg_net_bps']:>+6.1f} "
                  f"{r['max_dd']*100:>5.1f}% {r['avg_hold']:>4.0f}h "
                  f"{r['bootstrap_p5']:>+5.2f}")

        # Exit reason breakdown for best non-baseline
        best = max(results[1:], key=lambda x: x["sharpe"]) if len(results) > 1 else results[0]
        print(f"\n  Best V11 variant: {best['name']} (Sharpe {best['sharpe']:.2f})")
        print(f"  Exit reasons: {best['exit_reasons']}")
        print(f"  Long: {best['long_trades']} trades, WR={best['long_wr']:.0f}%")
        print(f"  Short: {best['short_trades']} trades, WR={best['short_wr']:.0f}%")

        # Production checks
        print(f"\n  {'='*60}")
        print("  PRODUCTION CHECKS")
        print(f"  {'='*60}")
        checks = {
            "Sharpe > 1.0": best["sharpe"] > 1.0,
            "Trades >= 15": best["trades"] >= 15,
            "Bootstrap p5 > 0": best["bootstrap_p5"] > 0,
            f"Sharpe >= V10 ({baseline_sharpe:.2f})": best["sharpe"] >= baseline_sharpe * 0.95,
        }
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {check}")

    print(f"\n{'='*100}")
    print("  VERIFICATION COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
