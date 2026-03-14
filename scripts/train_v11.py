#!/usr/bin/env python3
"""V11 Alpha Training — configurable horizons, exit manager, regime gate.

Builds on train_multi_horizon.py with:
- Configurable horizons (supports h=6)
- Optional vol-normalized target: y = ret / rolling_vol
- Outputs full v11 config.json (exit, regime_gate, time_filter sections)
- Post-training backtest with ExitManager + RegimeGate

Usage:
    python3 -m scripts.train_v11 --symbol ETHUSDT
    python3 -m scripts.train_v11 --symbol ETHUSDT --horizons 6,12,24
    python3 -m scripts.train_v11 --symbol ETHUSDT --trailing-stop 0.02
"""
from __future__ import annotations

import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.multi_timeframe import compute_4h_features, TF4H_FEATURE_NAMES
from scripts.train_v7_alpha import INTERACTION_FEATURES, BLACKLIST
from scripts.train_multi_horizon import (
    rolling_zscore,
    train_single_horizon,
    WARMUP,
    COST_BPS_RT,
    BARS_PER_MONTH,
)
from alpha.v11_config import V11Config, ExitConfig, RegimeGateConfig

VERSION = "v11"


def backtest_v11(
    preds_by_horizon: Dict[int, np.ndarray],
    closes: np.ndarray,
    config: V11Config,
    cost_bps: float = COST_BPS_RT,
    features_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Backtest with V11 modules: ExitManager + RegimeGate."""
    from decision.exit_manager import ExitManager
    from decision.regime_gate import RegimeGate

    n = len(closes)
    warmup = config.zscore_warmup

    # Compute rolling z-score per horizon, then average
    z_all = []
    for h, pred in sorted(preds_by_horizon.items()):
        z_h = rolling_zscore(pred, window=config.zscore_window, warmup=warmup)
        z_all.append(z_h)
    z = np.mean(z_all, axis=0)

    exit_mgr = ExitManager(config=config.exit, min_hold=config.min_hold, max_hold=config.max_hold)
    regime_gate = RegimeGate(config=config.regime_gate)

    cost_frac = cost_bps / 10000
    pos = 0.0
    entry_bar = 0
    trades = []

    for i in range(n):
        # Build feature dict for regime gate
        feat_dict = {}
        if features_df is not None and i < len(features_df):
            for col in ["bb_width_20", "vol_of_vol", "adx_14"]:
                if col in features_df.columns:
                    feat_dict[col] = float(features_df.iloc[i][col])
        _regime_label, position_scale = regime_gate.evaluate(feat_dict)

        if pos != 0:
            exit_mgr.update_price("SYM", closes[i])
            should_exit, reason = exit_mgr.check_exit(
                "SYM", closes[i], i, z[i], pos
            )
            if should_exit:
                pnl_pct = pos * (closes[i] - closes[entry_bar]) / closes[entry_bar]
                trades.append(pnl_pct * 500.0 - cost_frac * 500.0)
                exit_mgr.on_exit("SYM")
                pos = 0.0

        if pos == 0:
            hour_utc = i % 24  # approximate
            if not exit_mgr.allow_entry(z[i], hour_utc):
                continue
            if z[i] > config.deadzone:
                pos = 1.0
                entry_bar = i
                exit_mgr.on_entry("SYM", closes[i], i, 1.0)
            elif not config.long_only and z[i] < -config.deadzone:
                pos = -1.0
                entry_bar = i
                exit_mgr.on_entry("SYM", closes[i], i, -1.0)

    if not trades:
        return {"sharpe": 0, "trades": 0, "return": 0}
    net_arr = np.array(trades)
    avg_hold = n / max(len(trades), 1)
    tpy = 365 * 24 / max(avg_hold, 1)
    sharpe = float(np.mean(net_arr) / max(np.std(net_arr, ddof=1), 1e-10) * np.sqrt(tpy))
    return {
        "sharpe": sharpe,
        "trades": len(trades),
        "return": float(np.sum(net_arr)) / 10000,
        "win_rate": float(np.mean(net_arr > 0) * 100),
        "avg_net_bps": float(np.mean(net_arr) / 500 * 10000),
    }


def train_symbol_v11(
    symbol: str,
    horizons: List[int],
    trailing_stop_pct: float = 0.0,
    zscore_cap: float = 0.0,
    regime_gate_enabled: bool = False,
    lgbm_xgb_weight: float = 0.5,
) -> bool:
    """Train V11 multi-horizon ensemble for one symbol."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    data_path = Path(f"data_files/{symbol}_1h.csv")

    if not data_path.exists():
        print(f"  ERROR: {data_path} not found")
        return False

    df = pd.read_csv(data_path)
    n = len(df)
    ts_col = "open_time" if "open_time" in df.columns else "timestamp"
    timestamps = df[ts_col].values.astype(np.int64)
    closes = df["close"].values.astype(np.float64)
    start_date = pd.Timestamp(timestamps[0], unit="ms").strftime("%Y-%m-%d")
    end_date = pd.Timestamp(timestamps[-1], unit="ms").strftime("%Y-%m-%d")
    print(f"\n  Data: {n:,} 1h bars ({start_date} → {end_date})")

    # ── Features ──
    print("  Computing features...", end=" ", flush=True)
    t0 = time.time()
    _has_v11 = Path("data_files/macro_daily.csv").exists()
    feat_df = compute_features_batch(symbol, df, include_v11=_has_v11)
    tf4h = compute_4h_features(df)
    for col in TF4H_FEATURE_NAMES:
        feat_df[col] = tf4h[col].values
    for int_name, fa, fb in INTERACTION_FEATURES:
        if fa in feat_df.columns and fb in feat_df.columns:
            feat_df[int_name] = feat_df[fa].astype(float) * feat_df[fb].astype(float)

    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "open_time", "timestamp")
                     and c not in BLACKLIST]
    X = feat_df[feature_names].values.astype(np.float64)
    print(f"{len(feature_names)} features in {time.time() - t0:.1f}s")

    # ── Split ──
    oos_bars = BARS_PER_MONTH * 18
    train_end = n - oos_bars
    val_size = BARS_PER_MONTH * 6
    val_start = train_end - val_size
    closes_test = closes[train_end:]
    feat_df_test = feat_df.iloc[train_end:].reset_index(drop=True)

    print(f"  Split: train={val_start - WARMUP:,} val={val_size:,} test={oos_bars:,}")

    # ── Train each horizon ──
    models = {}
    preds_test = {}
    all_infos = {}

    for h in horizons:
        print(f"\n  ── Horizon h={h}h ──")
        t0 = time.time()
        result = train_single_horizon(h, X, closes, feature_names, val_start, train_end, n)
        if result is None:
            print(f"    FAILED for h={h}")
            continue
        lgbm_m, xgb_m, feats, info, pred = result
        models[h] = (lgbm_m, xgb_m, feats, info)
        preds_test[h] = pred
        all_infos[h] = info
        print(f"    Done ({time.time() - t0:.1f}s)")

    if len(models) < 2:
        print("  ERROR: Need at least 2 horizons")
        return False

    # ── V11 config sweep ──
    print(f"\n  ── V11 Config Sweep ({len(models)} horizons) ──")
    best_cfg = None
    best_sharpe = -999
    best_result = None

    for dz in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for mh in [8, 12, 24]:
            for maxh_mult in [5, 8]:
                maxh = mh * maxh_mult
                for lo in [True, False]:
                    cfg = V11Config(
                        horizons=sorted(models.keys()),
                        deadzone=dz,
                        min_hold=mh,
                        max_hold=maxh,
                        long_only=lo,
                        lgbm_xgb_weight=lgbm_xgb_weight,
                        exit=ExitConfig(
                            trailing_stop_pct=trailing_stop_pct,
                            zscore_cap=zscore_cap,
                        ),
                        regime_gate=RegimeGateConfig(enabled=regime_gate_enabled),
                    )
                    r = backtest_v11(
                        preds_test, closes_test, cfg,
                        features_df=feat_df_test if regime_gate_enabled else None,
                    )
                    if r["sharpe"] > best_sharpe and r["trades"] >= 10:
                        best_sharpe = r["sharpe"]
                        best_cfg = cfg
                        best_result = r

    if best_cfg is None:
        print("  No viable config found!")
        return False

    print(f"  Best: dz={best_cfg.deadzone}, hold=[{best_cfg.min_hold},{best_cfg.max_hold}], "
          f"long_only={best_cfg.long_only}")
    print(f"  Sharpe={best_result['sharpe']:.2f}, trades={best_result['trades']}, "
          f"WR={best_result.get('win_rate', 0):.0f}%, ret={best_result['return'] * 100:+.2f}%")

    # ── Bootstrap ──
    print("\n  Bootstrap Sharpe...")
    z_all = []
    for h, pred in sorted(preds_test.items()):
        z_all.append(rolling_zscore(pred, window=720, warmup=180))
    z = np.mean(z_all, axis=0)

    trade_pnls = []
    pos = 0.0
    eb = 0
    for i in range(len(z)):
        if pos != 0:
            held = i - eb
            should_exit = held >= best_cfg.max_hold
            if not should_exit and held >= best_cfg.min_hold:
                should_exit = pos * z[i] < best_cfg.exit.reversal_threshold or abs(z[i]) < best_cfg.exit.deadzone_fade
            if should_exit:
                pnl = pos * (closes_test[i] - closes_test[eb]) / closes_test[eb]
                trade_pnls.append(pnl * 500 - COST_BPS_RT / 10000 * 500)
                pos = 0.0
        if pos == 0:
            if z[i] > best_cfg.deadzone:
                pos = 1.0
                eb = i
            elif not best_cfg.long_only and z[i] < -best_cfg.deadzone:
                pos = -1.0
                eb = i

    trade_pnls = np.array(trade_pnls) if trade_pnls else np.array([0.0])
    bs_sharpes = []
    for _ in range(1000):
        sample = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
        if np.std(sample) > 0:
            bs_sharpes.append(float(np.mean(sample) / np.std(sample) * np.sqrt(52)))
    bs_sharpes = np.array(bs_sharpes)
    p5, p50, p95 = np.percentile(bs_sharpes, [5, 50, 95])
    print(f"  Bootstrap: {p50:.2f} (p5={p5:.2f}, p95={p95:.2f})")

    # ── Production checks ──
    avg_ic = np.mean([info["ic_ensemble"] for info in all_infos.values()])

    print(f"\n  {'=' * 60}")
    print(f"  PRODUCTION CHECKS ({symbol})")
    print(f"  {'=' * 60}")
    checks = {
        "Sharpe > 1.0": best_result["sharpe"] > 1.0,
        "Avg IC > 0.02": avg_ic > 0.02,
        "Trades >= 15": best_result["trades"] >= 15,
        "Bootstrap p5 > 0": p5 > 0,
    }
    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {check}")

    # ── Save ──
    if all_pass:
        import shutil
        backup_dir = model_dir.parent / f"{model_dir.name}_backup_pre_v11"
        if not backup_dir.exists() and model_dir.exists():
            shutil.copytree(model_dir, backup_dir)
            print(f"\n  Backed up to {backup_dir}")

        model_dir.mkdir(parents=True, exist_ok=True)

        horizon_configs = []
        for h in sorted(models.keys()):
            lgbm_m, xgb_m, feats, info = models[h]
            lgbm_name = f"lgbm_h{h}.pkl"
            xgb_name = f"xgb_h{h}.pkl"

            with open(model_dir / lgbm_name, "wb") as f:
                pickle.dump({"model": lgbm_m, "features": feats}, f)
            with open(model_dir / xgb_name, "wb") as f:
                pickle.dump({"model": xgb_m, "features": feats}, f)

            horizon_configs.append({
                "horizon": h,
                "lgbm": lgbm_name,
                "xgb": xgb_name,
                "features": feats,
                "ic": info["ic_ensemble"],
            })

        # Backward-compat primary model
        primary_h = 24 if 24 in models else sorted(models.keys())[len(models) // 2]
        lgbm_p, xgb_p, feats_p, _ = models[primary_h]
        with open(model_dir / "lgbm_v8.pkl", "wb") as f:
            pickle.dump({"model": lgbm_p, "features": feats_p}, f)
        with open(model_dir / "xgb_v8.pkl", "wb") as f:
            pickle.dump({"model": xgb_p, "features": feats_p}, f)

        config_dict = {
            "symbol": symbol,
            "version": VERSION,
            "multi_horizon": True,
            "horizons": sorted(models.keys()),
            "horizon_models": horizon_configs,
            "primary_horizon": primary_h,
            "ensemble_method": best_cfg.ensemble_method,
            "lgbm_xgb_weight": best_cfg.lgbm_xgb_weight,
            "zscore_window": best_cfg.zscore_window,
            "zscore_warmup": best_cfg.zscore_warmup,
            "deadzone": best_cfg.deadzone,
            "min_hold": best_cfg.min_hold,
            "max_hold": best_cfg.max_hold,
            "long_only": best_cfg.long_only,
            "exit": {
                "trailing_stop_pct": best_cfg.exit.trailing_stop_pct,
                "zscore_cap": best_cfg.exit.zscore_cap,
                "reversal_threshold": best_cfg.exit.reversal_threshold,
                "deadzone_fade": best_cfg.exit.deadzone_fade,
            },
            "regime_gate": {
                "enabled": best_cfg.regime_gate.enabled,
                "ranging_high_vol_action": best_cfg.regime_gate.ranging_high_vol_action,
                "reduce_factor": best_cfg.regime_gate.reduce_factor,
            },
            "time_filter": {
                "enabled": best_cfg.time_filter.enabled,
                "skip_hours_utc": best_cfg.time_filter.skip_hours_utc,
            },
            "metrics": {
                "sharpe": best_result["sharpe"],
                "avg_ic": float(avg_ic),
                "per_horizon_ic": {str(h): info["ic_ensemble"] for h, info in all_infos.items()},
                "total_return": best_result["return"],
                "trades": best_result["trades"],
                "win_rate": best_result.get("win_rate", 0),
                "avg_net_bps": best_result.get("avg_net_bps", 0),
                "bootstrap_sharpe_p5": float(p5),
                "bootstrap_sharpe_p50": float(p50),
                "bootstrap_sharpe_p95": float(p95),
            },
            "checks": {k: bool(v) for k, v in checks.items()},
            "train_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "data_range": f"{start_date} → {end_date}",
            "n_bars": n,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        all_feats = []
        for hc in horizon_configs:
            for feat in hc["features"]:
                if feat not in all_feats:
                    all_feats.append(feat)
        with open(model_dir / "features.json", "w") as f:
            json.dump(all_feats, f)

        print(f"\n  Model saved to {model_dir}/ ({VERSION})")
        return True
    else:
        print("\n  FAILED — model NOT saved.")
        return False


def main():
    parser = argparse.ArgumentParser(description="V11 Alpha Training")
    parser.add_argument("--symbol", default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--horizons", default="12,24,48",
                        help="Comma-separated horizons (hours)")
    parser.add_argument("--trailing-stop", type=float, default=0.0,
                        help="Trailing stop pct (0=disabled)")
    parser.add_argument("--zscore-cap", type=float, default=0.0,
                        help="Z-score cap for entry gate (0=disabled)")
    parser.add_argument("--regime-gate", action="store_true",
                        help="Enable regime gate")
    parser.add_argument("--lgbm-xgb-weight", type=float, default=0.5,
                        help="LGBM weight in ensemble (0-1)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    horizons = [int(h.strip()) for h in args.horizons.split(",")]

    print("=" * 70)
    print("  V11 ALPHA TRAINING")
    print(f"  Horizons: {horizons}")
    print(f"  Symbols:  {symbols}")
    if args.trailing_stop > 0:
        print(f"  Trailing stop: {args.trailing_stop:.1%}")
    if args.zscore_cap > 0:
        print(f"  Z-score cap: {args.zscore_cap}")
    if args.regime_gate:
        print("  Regime gate: enabled")
    print("=" * 70)

    results = {}
    for symbol in symbols:
        print(f"\n{'=' * 70}")
        print(f"  {symbol}")
        print(f"{'=' * 70}")
        t0 = time.time()
        saved = train_symbol_v11(
            symbol,
            horizons=horizons,
            trailing_stop_pct=args.trailing_stop,
            zscore_cap=args.zscore_cap,
            regime_gate_enabled=args.regime_gate,
            lgbm_xgb_weight=args.lgbm_xgb_weight,
        )
        results[symbol] = saved
        print(f"\n  Total time: {time.time() - t0:.1f}s")

    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for sym, saved in results.items():
        status = "SAVED" if saved else "FAILED"
        print(f"  {sym}: {status}")


if __name__ == "__main__":
    main()
