#!/usr/bin/env python3
"""Deadzone sweep — runs backtest_alpha_v8 across a range of dz values.

Usage:
    python3 scripts/dz_sweep.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

# Sweep configuration
DZ_VALUES = [0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
OOS_WINDOWS = {
    "6m": 4380,    # ~6 months
    "12m": 8760,   # ~12 months
}


def run_single(symbol: str, dz: float, oos_bars: int, window_name: str) -> dict | None:
    """Run a single backtest with modified dz, return summary dict."""
    model_dir = Path(f"models_v8/{symbol}_gate_v2")
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None

    # Create temp config with modified dz
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["deadzone"] = dz

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"dz_{symbol}_{dz}_"))
    tmp_config = tmp_dir / "config.json"
    with open(tmp_config, "w") as f:
        json.dump(cfg, f)

    out_dir = tmp_dir / "results"

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "scripts.backtest_alpha_v8",
                "--symbol", symbol,
                "--config", str(tmp_config),
                "--oos-bars", str(oos_bars),
                "--long-only",
            ],
            capture_output=True, text=True, timeout=300,
            cwd="/quant_system",
            env={**os.environ, "QUANT_ALLOW_UNSIGNED_MODELS": "1"},
        )
        # Parse output for key metrics
        output = result.stdout + result.stderr
        summary = _parse_output(output)
        summary["dz"] = dz
        summary["symbol"] = symbol
        summary["window"] = window_name
        summary["oos_bars"] = oos_bars
        return summary
    except Exception as e:
        return {"dz": dz, "symbol": symbol, "window": window_name, "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _parse_output(output: str) -> dict:
    """Extract metrics from backtest stdout."""
    d = {}
    for line in output.split("\n"):
        line = line.strip()
        # Sharpe ratio:    3.16
        if "Sharpe ratio:" in line:
            try:
                d["sharpe"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        # Max drawdown:    -18.79%
        if "Max drawdown:" in line:
            try:
                val = line.split(":")[-1].strip().replace("%", "")
                d["max_dd"] = float(val)
            except (ValueError, IndexError):
                pass
        # Position changes: 154
        if "Position changes:" in line:
            try:
                d["trades"] = int(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        # Win rate (bar):  49.9%
        if "Win rate" in line:
            try:
                val = line.split(":")[-1].strip().replace("%", "")
                d["win_rate"] = float(val)
            except (ValueError, IndexError):
                pass
        # Annual return:   +14.00%
        if "Annual return:" in line:
            try:
                val = line.split(":")[-1].strip().replace("%", "").replace("+", "")
                d["annual_return"] = float(val)
            except (ValueError, IndexError):
                pass
        # Profit factor:   1.08
        if "Profit factor:" in line:
            try:
                d["profit_factor"] = float(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        # Total return:    +6.67%
        if "Total return:" in line:
            try:
                val = line.split(":")[-1].strip().replace("%", "").replace("+", "")
                d["total_return"] = float(val)
            except (ValueError, IndexError):
                pass
    return d


def main():
    print("=" * 80)
    print("DEADZONE SWEEP — Backtest Parameter Optimization")
    print("=" * 80)

    all_results = []

    for symbol in SYMBOLS:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        for win_name, oos_bars in OOS_WINDOWS.items():
            print(f"\n  Window: {win_name} ({oos_bars} bars)")
            print(f"  {'dz':>5} | {'Sharpe':>8} | {'MaxDD':>8} | {'Trades':>6} | {'WinRate':>7} | {'AnnRet':>8} | {'PF':>6}")
            print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}")

            for dz in DZ_VALUES:
                r = run_single(symbol, dz, oos_bars, win_name)
                if r and "error" not in r:
                    all_results.append(r)
                    sharpe = r.get("sharpe", 0)
                    max_dd = r.get("max_dd", 0)
                    trades = r.get("trades", 0)
                    win_rate = r.get("win_rate", 0)
                    ann_ret = r.get("annual_return", 0)
                    pf = r.get("profit_factor", 0)
                    marker = " ◀" if dz in (1.5, 2.0) else ""
                    print(f"  {dz:5.1f} | {sharpe:8.2f} | {max_dd:7.1f}% | {trades:6d} | {win_rate:6.1f}% | {ann_ret:7.1f}% | {pf:6.2f}{marker}")
                else:
                    err = r.get("error", "unknown") if r else "failed"
                    print(f"  {dz:5.1f} | ERROR: {err[:40]}")

    # Summary: best dz per symbol
    print(f"\n{'='*60}")
    print("  OPTIMAL DZ SUMMARY")
    print(f"{'='*60}")
    for symbol in SYMBOLS:
        sym_results = [r for r in all_results if r["symbol"] == symbol and "sharpe" in r]
        if not sym_results:
            continue
        # Best by Sharpe (average across windows)
        from collections import defaultdict
        dz_sharpes = defaultdict(list)
        for r in sym_results:
            dz_sharpes[r["dz"]].append(r.get("sharpe", 0))
        best_dz = max(dz_sharpes, key=lambda d: sum(dz_sharpes[d]) / len(dz_sharpes[d]))
        avg_sharpe = sum(dz_sharpes[best_dz]) / len(dz_sharpes[best_dz])
        print(f"  {symbol}: best dz={best_dz:.1f} (avg Sharpe={avg_sharpe:.2f})")
        # Current
        cur_dz = 1.5 if "BTC" in symbol else 2.0
        if cur_dz in dz_sharpes:
            cur_avg = sum(dz_sharpes[cur_dz]) / len(dz_sharpes[cur_dz])
            print(f"    current dz={cur_dz:.1f} (avg Sharpe={cur_avg:.2f})")

    # Save raw results
    out_path = Path("results/dz_sweep_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Raw results saved to {out_path}")


if __name__ == "__main__":
    main()
