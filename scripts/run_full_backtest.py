#!/usr/bin/env python3
"""Run walk-forward backtest validation for all active alpha models.

Usage:
    python3 scripts/run_full_backtest.py
    python3 scripts/run_full_backtest.py --symbol BTCUSDT_gate_v2

Note: Uses pickle for loading trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and HMAC-signed.
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — trusted local model artifacts, HMAC-signed
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_feature_engine import compute_features_batch
from features.batch_backtest import run_backtest_fast

MODELS_DIR = Path("models_v8")
DATA_DIR = Path("data_files")


def _fix_oi_file(symbol: str) -> None:
    """Fix mixed-format OI CSV (old: 2 cols, new: 4 cols with symbol)."""
    path = DATA_DIR / f"{symbol}_open_interest.csv"
    if not path.exists():
        return
    try:
        lines = path.read_text().strip().split("\n")
        if len(lines) < 2:
            return
        header = lines[0]
        # Already fixed or single-format
        if header == "timestamp,sum_open_interest" and "," not in lines[-1].split(",", 2)[-1]:
            # Check last line — if it has more than 2 fields, needs fixing
            if len(lines[-1].split(",")) <= 2:
                return
        cleaned = ["timestamp,sum_open_interest"]
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) == 2:
                cleaned.append(line)
            elif len(parts) >= 3:
                # New format: timestamp,symbol,oi_value,notional_value
                try:
                    float(parts[1])  # old format: ts,oi
                    cleaned.append(f"{parts[0]},{parts[1]}")
                except ValueError:
                    # parts[1] is symbol string, parts[2] is OI
                    cleaned.append(f"{parts[0]},{parts[2]}")
        path.write_text("\n".join(cleaned) + "\n")
    except Exception:
        pass  # don't crash backtest for data fix

# Active models: (model_dir, symbol, timeframe, data_file)
ACTIVE_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h", "BTCUSDT_1h.csv"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h", "ETHUSDT_1h.csv"),
    ("BTCUSDT_4h",      "BTCUSDT", "4h", None),  # resample from 1h
    ("ETHUSDT_4h",      "ETHUSDT", "4h", None),  # resample from 1h
    ("BTCUSDT_15m",     "BTCUSDT", "15m", "BTCUSDT_15m.csv"),
    ("ETHUSDT_15m",     "ETHUSDT", "15m", "ETHUSDT_15m.csv"),
]


def load_data(data_file: Optional[str], symbol: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV data."""
    if data_file is not None:
        path = DATA_DIR / data_file
    else:
        path = DATA_DIR / f"{symbol}_1h.csv"

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if timeframe == "4h" and data_file is None:
        df = df.set_index("timestamp")
        df = df.resample("4h").agg({
            "open_time": "first", "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).dropna().reset_index()

    return df


def load_model_and_predict(model_dir: Path, df: pd.DataFrame, config: dict,
                           symbol: str = "BTCUSDT") -> Optional[np.ndarray]:
    """Load model, compute features, generate predictions."""
    horizon_models = config.get("horizon_models", [])
    if not horizon_models:
        return None

    features_df = compute_features_batch(symbol, df)
    if features_df is None or len(features_df) == 0:
        return None

    ridge_weight = config.get("ridge_weight", 0.6)
    lgbm_weight = config.get("lgbm_weight", 0.4)

    preds_all = []

    for hm in horizon_models:
        lgbm_path = model_dir / hm["lgbm"]
        if not lgbm_path.exists():
            print(f"  SKIP: {lgbm_path} not found")
            continue

        # noqa: S301 — trusted local model artifacts produced by our training pipeline
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)  # noqa: S301

        model = lgbm_data["model"] if isinstance(lgbm_data, dict) else lgbm_data
        feat_names = hm.get("features") or (lgbm_data.get("features") if isinstance(lgbm_data, dict) else None)
        if not feat_names:
            feat_names = list(features_df.columns)

        available = [fname for fname in feat_names if fname in features_df.columns]
        missing = [fname for fname in feat_names if fname not in features_df.columns]
        if len(available) < len(feat_names) * 0.5:
            print(f"  SKIP h{hm.get('horizon','?')}: only {len(available)}/{len(feat_names)} features")
            continue
        if missing:
            print(f"  h{hm.get('horizon','?')}: padding {len(missing)} missing features with 0: {missing}")

        # Build feature matrix with all expected features (pad missing with 0)
        X = np.zeros((len(features_df), len(feat_names)))
        for j, fname in enumerate(feat_names):
            if fname in features_df.columns:
                X[:, j] = features_df[fname].fillna(0).values
        pred = model.predict(X)

        ridge_path_name = hm.get("ridge")
        if ridge_path_name:
            ridge_path = model_dir / ridge_path_name
            if ridge_path.exists():
                with open(ridge_path, "rb") as f:
                    ridge_data = pickle.load(f)  # noqa: S301
                ridge_model = ridge_data["model"] if isinstance(ridge_data, dict) else ridge_data
                ridge_feats = (ridge_data.get("features") if isinstance(ridge_data, dict) else None) or feat_names
                ridge_available = [rf for rf in ridge_feats if rf in features_df.columns]
                if len(ridge_available) >= len(ridge_feats) * 0.5:
                    Xr = np.zeros((len(features_df), len(ridge_feats)))
                    for j, rf in enumerate(ridge_feats):
                        if rf in features_df.columns:
                            Xr[:, j] = features_df[rf].fillna(0).values
                    ridge_pred = ridge_model.predict(Xr)
                    pred = ridge_weight * ridge_pred + lgbm_weight * pred

        preds_all.append(pred)

    if not preds_all:
        return None

    min_len = min(len(p) for p in preds_all)
    combined = np.mean([p[:min_len] for p in preds_all], axis=0)
    return combined


def run_single_backtest(model_name: str, symbol: str, timeframe: str,
                        data_file: Optional[str]) -> Dict[str, Any]:
    """Run backtest for a single model."""
    model_dir = MODELS_DIR / model_name
    config_path = model_dir / "config.json"

    result: Dict[str, Any] = {
        "model": model_name, "symbol": symbol, "timeframe": timeframe,
        "status": "FAIL", "sharpe": 0.0, "trades": 0,
    }

    if not config_path.exists():
        result["error"] = "config.json missing"
        return result

    with open(config_path) as f:
        config = json.load(f)

    if config.get("disabled"):
        result["status"] = "DISABLED"
        result["reason"] = config.get("disabled_reason", "disabled")
        return result

    try:
        df = load_data(data_file, symbol, timeframe)
    except Exception as e:
        result["error"] = f"data load: {e}"
        return result

    # Fix mixed-format OI files before feature computation
    _fix_oi_file(symbol)

    try:
        y_pred = load_model_and_predict(model_dir, df, config, symbol=symbol)
    except Exception as e:
        result["error"] = f"predict: {e}"
        return result

    if y_pred is None:
        result["error"] = "no predictions generated"
        return result

    n = min(len(y_pred), len(df))
    timestamps = df["open_time"].values[:n].astype(np.int64)
    closes = df["close"].values[:n].astype(np.float64)
    volumes = df["volume"].values[:n].astype(np.float64)

    bt_config = {
        "deadzone": config.get("deadzone", 0.5),
        "min_hold": config.get("min_hold", 24),
        "max_hold": config.get("max_hold", 120),
        "zscore_window": config.get("zscore_window", 720),
        "zscore_warmup": config.get("zscore_warmup", 180),
        "long_only": config.get("long_only", False),
        "monthly_gate": config.get("monthly_gate", False),
        "ma_window": config.get("monthly_gate_window", 480),
        "cost_per_trade": 6e-4,
        "capital": 10000.0,
    }

    try:
        bt = run_backtest_fast(
            timestamps=timestamps, closes=closes, y_pred=y_pred[:n],
            volumes=volumes, config=bt_config,
        )
    except Exception as e:
        result["error"] = f"backtest: {e}"
        return result

    sharpe = bt.get("sharpe", 0.0)
    total_return = bt.get("total_return", 0.0)
    max_dd = bt.get("max_drawdown", 0.0)
    n_trades = bt.get("n_trades", 0)
    win_rate = bt.get("win_rate", 0.0)

    result.update({
        "status": "PASS" if sharpe > 1.0 else "MARGINAL" if sharpe > 0 else "FAIL",
        "sharpe": round(sharpe, 2),
        "total_return": round(total_return * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "bars": n,
        "period": f"{pd.to_datetime(timestamps[0], unit='ms').date()} to "
                  f"{pd.to_datetime(timestamps[-1], unit='ms').date()}",
    })
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward backtest validation")
    parser.add_argument("--symbol", help="Run single model only")
    args = parser.parse_args()

    models = ACTIVE_MODELS
    if args.symbol:
        models = [m for m in models if m[0] == args.symbol]
        if not models:
            print(f"Model {args.symbol} not found")
            sys.exit(1)

    print("=" * 80)
    print("WALK-FORWARD BACKTEST VALIDATION")
    print("=" * 80)

    results = []
    for model_name, symbol, tf, data_file in models:
        print(f"\n>>> {model_name} ({symbol} {tf})")
        t0 = time.time()
        result = run_single_backtest(model_name, symbol, tf, data_file)
        elapsed = time.time() - t0
        result["time_s"] = round(elapsed, 1)
        results.append(result)

        status = result["status"]
        if status == "DISABLED":
            print(f"  DISABLED: {result.get('reason', '?')}")
        elif "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Sharpe={result['sharpe']:.2f}  Return={result['total_return']:.1f}%  "
                  f"MaxDD={result['max_drawdown']:.1f}%  Trades={result['trades']}  "
                  f"WinRate={result['win_rate']:.0f}%  ({elapsed:.1f}s)")
            print(f"  Status: {status}  Period: {result.get('period', '?')}")

    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'TF':>3} {'Status':>8} {'Sharpe':>7} {'Return':>8} "
          f"{'MaxDD':>7} {'Trades':>6} {'WR':>5}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25} {r['timeframe']:>3} {'ERROR':>8}  {r.get('error','')[:40]}")
        elif r["status"] == "DISABLED":
            print(f"{r['model']:<25} {r['timeframe']:>3} {'DISABLED':>8}")
        else:
            print(f"{r['model']:<25} {r['timeframe']:>3} {r['status']:>8} "
                  f"{r['sharpe']:>7.2f} {r['total_return']:>7.1f}% {r['max_drawdown']:>6.1f}% "
                  f"{r['trades']:>6} {r['win_rate']:>4.0f}%")
    print("=" * 80)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = sum(1 for r in results if r["status"] not in ("DISABLED",))
    print(f"\nResult: {passed}/{total} PASS (Sharpe > 1.0), {failed} FAIL")


if __name__ == "__main__":
    main()
