#!/usr/bin/env python3
"""Sweep 6 regime-gating configurations over walk-forward folds.

Runs walkforward_validate with different signal filter configs and
produces a comparison table.

Usage:
    python3 -m scripts.wf_regime_sweep --symbol BTCUSDT --out-dir results/regime_sweep
    python3 -m scripts.wf_regime_sweep --symbol BTCUSDT --no-hpo  # faster
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.walkforward_validate import (
    generate_wf_folds,
    run_fold,
    stitch_results,
    print_report,
    BLACKLIST,
)
from scripts.train_v7_alpha import _load_and_compute_features


@dataclass
class SweepConfig:
    name: str
    regime_gate: bool = False
    long_only: bool = False
    adaptive_sizing: Optional[Dict[str, float]] = None


CONFIGS: List[SweepConfig] = [
    SweepConfig(name="baseline"),
    SweepConfig(name="long_only", long_only=True),
    SweepConfig(name="regime_gate_strict", regime_gate=True),
    SweepConfig(name="long_only+regime_gate", regime_gate=True, long_only=True),
    SweepConfig(name="adaptive_sizing", adaptive_sizing={
        "trending": 1.0, "ranging": 0.3, "high_vol_flat": 0.0}),
    SweepConfig(name="long_only+adaptive", long_only=True, adaptive_sizing={
        "trending": 1.0, "ranging": 0.5, "high_vol_flat": 0.0}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime gating sweep")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--out-dir", default="results/regime_sweep")
    parser.add_argument("--fixed-features", nargs="*", default=None)
    parser.add_argument("--candidate-pool", nargs="*", default=None)
    parser.add_argument("--n-flexible", type=int, default=4)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    use_hpo = not args.no_hpo
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_bars = len(df)
    print(f"\n  Regime Sweep: {symbol}  ({n_bars:,} bars)")
    print(f"  HPO: {'ON' if use_hpo else 'OFF'}")

    folds = generate_wf_folds(n_bars)
    if not folds:
        print("  Not enough data")
        return
    print(f"  Folds: {len(folds)}")

    print("  Computing features...")
    feat_df = _load_and_compute_features(symbol, df)
    if feat_df is None:
        print("  Feature computation failed")
        return
    closes = (feat_df["close"].values.astype(np.float64)
              if "close" in feat_df.columns
              else df["close"].values.astype(np.float64))

    all_feature_names = [c for c in feat_df.columns
                         if c not in ("close", "timestamp", "open_time")
                         and c not in BLACKLIST]

    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    # Run each config
    all_summaries: List[Dict[str, Any]] = []
    for cfg in CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Config: {cfg.name}")
        print(f"  regime_gate={cfg.regime_gate}  long_only={cfg.long_only}"
              f"  adaptive_sizing={cfg.adaptive_sizing}")
        print(f"{'='*60}")

        fold_results = []
        t0 = time.time()
        for fold in folds:
            from datetime import datetime, timezone
            try:
                ts_s = datetime.fromtimestamp(
                    timestamps[fold.test_start] / 1000, tz=timezone.utc)
                ts_e = datetime.fromtimestamp(
                    timestamps[min(fold.test_end - 1, n_bars - 1)] / 1000,
                    tz=timezone.utc)
                period = f"{ts_s:%Y-%m}\u2192{ts_e:%Y-%m}"
            except (ValueError, OSError, IndexError):
                period = f"fold_{fold.idx}"

            result = run_fold(
                fold, feat_df, closes, all_feature_names,
                use_hpo=use_hpo,
                fixed_features=args.fixed_features,
                candidate_pool=args.candidate_pool,
                n_flexible=args.n_flexible,
                regime_gate=cfg.regime_gate,
                long_only=cfg.long_only,
                adaptive_sizing=cfg.adaptive_sizing,
            )
            result.period = period
            fold_results.append(result)

        elapsed = time.time() - t0
        summary = stitch_results(fold_results)
        summary["config"] = cfg.name
        summary["elapsed_s"] = elapsed
        all_summaries.append(summary)
        print_report(fold_results, summary)

        # Save individual config result
        cfg_path = out_dir / f"wf_{symbol}_{cfg.name}.json"
        with open(cfg_path, "w") as f:
            json.dump({
                "config": cfg.name,
                "regime_gate": cfg.regime_gate,
                "long_only": cfg.long_only,
                "adaptive_sizing": cfg.adaptive_sizing,
                "folds": [
                    {"idx": r.idx, "period": r.period, "ic": r.ic,
                     "sharpe": r.sharpe, "total_return": r.total_return,
                     "features": r.features}
                    for r in fold_results
                ],
                "summary": summary,
            }, f, indent=2, default=str)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"  {'Config':<28} {'Pass':>6} {'AvgSharpe':>10} {'Return':>10}"
          f"  {'F11-14 Ret':>10}  {'Time':>6}")
    print(f"  {'-'*78}")

    for s in all_summaries:
        pos = s["positive_sharpe"]
        n = s["n_folds"]
        # Folds 11-14 (0-indexed) loss
        f11_14_ret = sum(s["fold_returns"][i] for i in range(10, 14)
                         if i < len(s["fold_returns"])) if len(s["fold_returns"]) > 10 else 0.0
        elapsed = s.get("elapsed_s", 0)
        print(f"  {s['config']:<28} {pos:>2}/{n:<2}  {s['avg_sharpe']:>+9.2f}"
              f"  {s['total_return']*100:>+9.2f}%"
              f"  {f11_14_ret*100:>+9.2f}%"
              f"  {elapsed:>5.0f}s")

    # Save comparison
    comp_path = out_dir / f"sweep_{symbol}_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n  Comparison saved to {comp_path}")


if __name__ == "__main__":
    main()
