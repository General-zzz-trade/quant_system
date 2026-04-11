#!/usr/bin/env python3
"""OOS-only backtest: runs only on data AFTER model training date.

Usage:
    python3 scripts/run_oos_backtest.py
    python3 scripts/run_oos_backtest.py --months 3
    python3 scripts/run_oos_backtest.py --symbol BTCUSDT_gate_v2

Note: Uses pickle for loading trusted local ML model artifacts (lightgbm/sklearn).
These models are produced by our own training pipeline and HMAC-signed.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, "/quant_system")

from features.batch_backtest import run_backtest_fast
from scripts.run_full_backtest import (
    MODELS_DIR, _fix_oi_file, load_data, load_model_and_predict,
)

try:
    import json
except ImportError:
    raise

ACTIVE_MODELS = [
    ("BTCUSDT_gate_v2", "BTCUSDT", "1h", "BTCUSDT_1h.csv"),
    ("ETHUSDT_gate_v2", "ETHUSDT", "1h", "ETHUSDT_1h.csv"),
    ("BTCUSDT_4h",      "BTCUSDT", "4h", None),
    ("ETHUSDT_4h",      "ETHUSDT", "4h", None),
]


def run_oos_backtest(model_name: str, symbol: str, timeframe: str,
                     data_file: Optional[str], oos_months: int = 6,
                     overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one OOS backtest for a trained model.

    overrides: optional dict of ``bt_config`` keys to override the
        baked-in model config.  Used by the walk-forward grid search
        to sweep ``deadzone`` / ``min_hold`` / ``max_hold`` /
        ``zscore_window`` etc. without retraining the model.
    """
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
    if overrides:
        for k, v in overrides.items():
            config[k] = v

    # Determine OOS start: max(train_date, now - oos_months)
    train_date_str = config.get("train_date", "")
    if train_date_str:
        try:
            train_dt = datetime.strptime(train_date_str.split(" ")[0], "%Y-%m-%d")
        except ValueError:
            train_dt = datetime.now() - timedelta(days=oos_months * 30)
    else:
        train_dt = datetime.now() - timedelta(days=oos_months * 30)

    oos_start = max(train_dt - timedelta(days=30),  # 30-day buffer for z-score warmup
                    datetime.now() - timedelta(days=oos_months * 30))

    try:
        df = load_data(data_file, symbol, timeframe)
    except Exception as e:
        result["error"] = f"data load: {e}"
        return result

    _fix_oi_file(symbol)

    try:
        y_pred = load_model_and_predict(model_dir, df, config, symbol=symbol)
    except Exception as e:
        result["error"] = f"predict: {e}"
        return result

    if y_pred is None:
        result["error"] = "no predictions generated"
        return result

    # Filter to OOS period
    df_ts = pd.to_datetime(df["open_time"], unit="ms")
    oos_mask = df_ts >= pd.Timestamp(oos_start)
    oos_start_idx = int(oos_mask.values.argmax())

    # Need zscore_warmup bars before OOS start
    warmup = config.get("zscore_warmup", 180)
    start_idx = max(0, oos_start_idx - warmup)

    n = min(len(y_pred), len(df))
    timestamps = df["open_time"].values[start_idx:n].astype(np.int64)
    closes = df["close"].values[start_idx:n].astype(np.float64)
    volumes = df["volume"].values[start_idx:n].astype(np.float64)
    preds = y_pred[start_idx:n]

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
            timestamps=timestamps, closes=closes, y_pred=preds,
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

    actual_start = pd.to_datetime(
        timestamps[warmup] if len(timestamps) > warmup else timestamps[0], unit="ms")
    actual_end = pd.to_datetime(timestamps[-1], unit="ms")

    result.update({
        "status": "PASS" if sharpe > 0.5 else "MARGINAL" if sharpe > 0 else "FAIL",
        "sharpe": round(sharpe, 2),
        "total_return": round(total_return * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "bars": len(timestamps),
        "oos_bars": len(timestamps) - min(warmup, len(timestamps)),
        "period": f"{actual_start.date()} to {actual_end.date()}",
        "train_date": train_date_str,
    })
    return result


def _parse_grid(spec: str) -> Dict[str, list]:
    """Parse ``--grid`` spec into ``{param: [values...]}``.

    Format: space-separated ``KEY=V1,V2,V3`` pairs.

    >>> _parse_grid("deadzone=0.8,1.0,1.2 min_hold=4,6,8")
    {'deadzone': [0.8, 1.0, 1.2], 'min_hold': [4, 6, 8]}
    """
    out: Dict[str, list] = {}
    if not spec:
        return out
    for part in spec.split():
        if "=" not in part:
            continue
        k, raw_vals = part.split("=", 1)
        values: list = []
        for v in raw_vals.split(","):
            v = v.strip()
            if not v:
                continue
            # Try int → float → raw string
            try:
                values.append(int(v))
                continue
            except ValueError:
                pass
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)
        if values:
            out[k.strip()] = values
    return out


def _expand_grid(grid: Dict[str, list]) -> list[Dict[str, Any]]:
    """Cartesian product of a grid dict into a list of overrides."""
    from itertools import product
    if not grid:
        return [{}]
    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))
    return [dict(zip(keys, c)) for c in combos]


def run_grid_search(models: list, months: int, grid: Dict[str, list]) -> list[Dict[str, Any]]:
    """Sweep ``grid`` × ``models`` and return one row per (model, combo).

    Rows carry their override dict under ``"params"`` and the standard
    backtest metrics (sharpe / trades / ...).  Caller prints / sorts.
    """
    combos = _expand_grid(grid)
    rows: list[Dict[str, Any]] = []
    total = len(combos) * len(models)
    seen = 0
    for model_name, symbol, tf, data_file in models:
        for combo in combos:
            seen += 1
            t0 = time.time()
            r = run_oos_backtest(model_name, symbol, tf, data_file, months,
                                 overrides=combo)
            r["time_s"] = round(time.time() - t0, 1)
            r["params"] = combo
            rows.append(r)
            if "error" not in r:
                print(f"  [{seen:3d}/{total}] {model_name:<24} {combo}  "
                      f"→ Sharpe={r['sharpe']:+.2f} trades={r['trades']}")
            else:
                print(f"  [{seen:3d}/{total}] {model_name:<24} {combo}  "
                      f"→ ERROR: {r['error']}")
    return rows


def _print_grid_report(rows: list[Dict[str, Any]]) -> None:
    """Group by model, sort each group by Sharpe descending, print."""
    by_model: Dict[str, list[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model, group in by_model.items():
        group.sort(key=lambda x: x.get("sharpe", -999), reverse=True)
        print("\n" + "=" * 90)
        print(f"GRID RESULTS — {model}  ({len(group)} configs)")
        print("=" * 90)
        hdr_params = sorted({k for r in group for k in r.get("params", {})})
        header = " ".join(f"{k:<10}" for k in hdr_params)
        print(f"{'Rank':<5} {header} {'Sharpe':>8} {'Ret%':>8} {'DD%':>7} {'Trades':>7}")
        for i, r in enumerate(group[:20], 1):
            param_vals = " ".join(
                f"{str(r.get('params', {}).get(k, '')):<10}" for k in hdr_params
            )
            if "error" in r:
                print(f"{i:<5} {param_vals} ERROR: {r['error'][:40]}")
            else:
                print(f"{i:<5} {param_vals} "
                      f"{r['sharpe']:>+8.2f} {r['total_return']:>+7.1f}% "
                      f"{r['max_drawdown']:>+6.1f}% {r['trades']:>7d}")
        best = group[0]
        if "error" not in best:
            print(f"\n  BEST: {best.get('params', {})} → Sharpe {best['sharpe']:+.2f}")
            baseline = next((r for r in group if not r.get("params")), None)
            if baseline and "error" not in baseline:
                delta = best["sharpe"] - baseline["sharpe"]
                print(f"  vs baseline ({baseline.get('params', {})}): "
                      f"Sharpe Δ = {delta:+.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OOS-only backtest validation")
    parser.add_argument("--symbol", help="Run single model only")
    parser.add_argument("--months", type=int, default=6, help="OOS lookback months (default: 6)")
    parser.add_argument(
        "--grid", default=None,
        metavar="SPEC",
        help=(
            "Walk-forward grid search.  Space-separated KEY=V1,V2,V3 "
            "pairs, e.g. 'deadzone=0.8,1.0,1.2 min_hold=4,6,8'.  "
            "Cartesian product is swept against each selected model.  "
            "Supports any bt_config key (deadzone/min_hold/max_hold/"
            "zscore_window/long_only)."
        ),
    )
    args = parser.parse_args()

    models = ACTIVE_MODELS
    if args.symbol:
        models = [m for m in models if m[0] == args.symbol]
        if not models:
            print(f"Model {args.symbol} not found")
            sys.exit(1)

    # Grid-search mode: sweep parameters against all selected models and
    # print a ranked per-model table.  Use for quarterly parameter retuning.
    if args.grid:
        grid = _parse_grid(args.grid)
        if not grid:
            print(f"ERROR: could not parse --grid spec: {args.grid!r}")
            sys.exit(1)
        print("=" * 90)
        print(f"WALK-FORWARD GRID SEARCH (last {args.months} months)")
        print(f"  Grid: {grid}")
        print(f"  Models: {[m[0] for m in models]}")
        n_combos = 1
        for v in grid.values():
            n_combos *= len(v)
        print(f"  Configs per model: {n_combos}   Total runs: {n_combos * len(models)}")
        print("=" * 90)
        rows = run_grid_search(models, args.months, grid)
        _print_grid_report(rows)
        return

    print("=" * 90)
    print(f"OUT-OF-SAMPLE BACKTEST (last {args.months} months)")
    print("=" * 90)

    results = []
    for model_name, symbol, tf, data_file in models:
        print(f"\n>>> {model_name} ({symbol} {tf})")
        t0 = time.time()
        result = run_oos_backtest(model_name, symbol, tf, data_file, args.months)
        elapsed = time.time() - t0
        result["time_s"] = round(elapsed, 1)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Sharpe={result['sharpe']:.2f}  Return={result['total_return']:.1f}%  "
                  f"MaxDD={result['max_drawdown']:.1f}%  Trades={result['trades']}  "
                  f"WinRate={result['win_rate']:.0f}%  ({elapsed:.1f}s)")
            print(f"  Status: {result['status']}  Period: {result['period']}  "
                  f"Trained: {result.get('train_date', '?')}")

    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'TF':>3} {'Status':>8} {'Sharpe':>7} {'Return':>8} "
          f"{'MaxDD':>7} {'Trades':>6} {'WR':>5} {'OOS bars':>8}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['model']:<25} {r['timeframe']:>3} {'ERROR':>8}  {r.get('error','')[:45]}")
        else:
            print(f"{r['model']:<25} {r['timeframe']:>3} {r['status']:>8} "
                  f"{r['sharpe']:>7.2f} {r['total_return']:>7.1f}% {r['max_drawdown']:>6.1f}% "
                  f"{r['trades']:>6} {r['win_rate']:>4.0f}% {r['oos_bars']:>8}")
    print("=" * 90)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len([r for r in results if "error" not in r])
    print(f"\nResult: {passed}/{total} PASS (Sharpe > 0.5), {failed} FAIL")


if __name__ == "__main__":
    main()
