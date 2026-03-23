#!/usr/bin/env python3
"""3-stage signal parameter sweep over walk-forward folds.

Stage 1: Deadzone sweep (0.3, 0.5, 0.7, 1.0) — pick best
Stage 2: Min-hold sweep (12, 24, 36, 48) — using best deadzone
Stage 3: Continuous sizing — using best deadzone + min_hold

All runs: long-only + HPO, 10 fixed + 5 flexible features.

Usage:
    python3 -m scripts.wf_signal_sweep --symbol BTCUSDT \
        --fixed-features basis ret_24 fgi_normalized fgi_extreme parkinson_vol \
            atr_norm_14 rsi_14 tf4h_atr_norm_14 basis_zscore_24 cvd_20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.walkforward_validate import (
    generate_wf_folds,
    run_fold,
    stitch_results,
    print_report,
    MIN_HOLD,
    HPO_TRIALS,
)
from alpha.training.train_v7_alpha import _load_and_compute_features, BLACKLIST


def _run_config(
    folds,
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    timestamps: np.ndarray,
    n_bars: int,
    *,
    use_hpo: bool,
    fixed_features: Optional[List[str]],
    candidate_pool: Optional[List[str]],
    n_flexible: int,
    hpo_trials: int,
    deadzone: float,
    min_hold: int,
    continuous_sizing: bool,
    ensemble: bool = False,
    trend_follow: bool = False,
    trend_indicator: str = "tf4h_close_vs_ma20",
    trend_threshold: float = 0.0,
    max_hold: int = 120,
) -> Dict[str, Any]:
    """Run all folds with given signal params, return summary."""
    fold_results = []
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
            fold, feat_df, closes, feature_names,
            use_hpo=use_hpo,
            fixed_features=fixed_features,
            candidate_pool=candidate_pool,
            n_flexible=n_flexible,
            long_only=True,
            deadzone=deadzone,
            min_hold=min_hold,
            continuous_sizing=continuous_sizing,
            hpo_trials=hpo_trials,
            ensemble=ensemble,
            trend_follow=trend_follow,
            trend_indicator=trend_indicator,
            trend_threshold=trend_threshold,
            max_hold=max_hold,
        )
        result.period = period
        fold_results.append(result)

    summary = stitch_results(fold_results)
    return {"fold_results": fold_results, "summary": summary}


def _print_comparison(rows: List[Dict[str, Any]], title: str) -> None:
    """Print a comparison table for a sweep stage."""
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")
    print(f"  {'Config':<30} {'Pass':>6} {'AvgSharpe':>10} {'Return':>10}")
    print(f"  {'-'*60}")
    for r in rows:
        s = r["summary"]
        print(f"  {r['label']:<30} "
              f"{s['positive_sharpe']:>2}/{s['n_folds']:<2}  "
              f"{s['avg_sharpe']:>+9.2f}  "
              f"{s['total_return']*100:>+9.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal parameter sweep")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--out-dir", default="results/signal_sweep")
    parser.add_argument("--fixed-features", nargs="*", default=None)
    parser.add_argument("--candidate-pool", nargs="*", default=None)
    parser.add_argument("--n-flexible", type=int, default=5)
    parser.add_argument("--no-hpo", action="store_true")
    parser.add_argument("--hpo-trials", type=int, default=HPO_TRIALS)
    parser.add_argument("--ensemble", action="store_true",
                        help="Use LGBM+XGB ensemble")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="Skip Stage 4 trend hold sweep")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    use_hpo = not args.no_hpo
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hpo_trials = args.hpo_trials

    csv_path = Path(f"data_files/{symbol}_1h.csv")
    if not csv_path.exists():
        print(f"  Data not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_bars = len(df)
    print(f"\n  Signal Sweep: {symbol}  ({n_bars:,} bars)")
    print(f"  HPO: {'ON' if use_hpo else 'OFF'}"
          f"{f' ({hpo_trials} trials)' if use_hpo else ''}")
    print("  Mode: long-only")

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
    feature_names = [c for c in feat_df.columns
                     if c not in ("close", "timestamp", "open_time")
                     and c not in BLACKLIST]

    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    timestamps = df[ts_col].values.astype(np.int64)

    ensemble = args.ensemble
    skip_stage4 = args.skip_stage4

    common = dict(
        folds=folds, feat_df=feat_df, closes=closes,
        feature_names=feature_names, timestamps=timestamps, n_bars=n_bars,
        use_hpo=use_hpo, fixed_features=args.fixed_features,
        candidate_pool=args.candidate_pool, n_flexible=args.n_flexible,
        hpo_trials=hpo_trials, ensemble=ensemble,
    )

    # ── Stage 1: Deadzone sweep ──────────────────────────────
    DEADZONE_VALUES = [0.3, 0.5, 0.7, 1.0]
    print(f"\n  Stage 1: Deadzone sweep {DEADZONE_VALUES}")
    dz_rows = []
    for dz in DEADZONE_VALUES:
        label = f"deadzone={dz}"
        print(f"\n  Running {label}...")
        t0 = time.time()
        res = _run_config(**common, deadzone=dz, min_hold=MIN_HOLD,
                          continuous_sizing=False)
        elapsed = time.time() - t0
        res["label"] = label
        res["deadzone"] = dz
        dz_rows.append(res)
        s = res["summary"]
        print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
              f"Sharpe={s['avg_sharpe']:+.2f}, "
              f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

    _print_comparison(dz_rows, "STAGE 1: DEADZONE SWEEP")

    # Pick best by positive_sharpe, then avg_sharpe as tiebreak
    best_dz_row = max(dz_rows, key=lambda r: (
        r["summary"]["positive_sharpe"], r["summary"]["avg_sharpe"]))
    best_deadzone = best_dz_row["deadzone"]
    print(f"\n  Best deadzone: {best_deadzone}")

    # ── Stage 2: Min-hold sweep ──────────────────────────────
    MIN_HOLD_VALUES = [12, 24, 36, 48]
    print(f"\n  Stage 2: Min-hold sweep {MIN_HOLD_VALUES} (deadzone={best_deadzone})")
    mh_rows = []
    for mh in MIN_HOLD_VALUES:
        label = f"min_hold={mh}"
        print(f"\n  Running {label}...")
        t0 = time.time()
        res = _run_config(**common, deadzone=best_deadzone, min_hold=mh,
                          continuous_sizing=False)
        elapsed = time.time() - t0
        res["label"] = label
        res["min_hold"] = mh
        mh_rows.append(res)
        s = res["summary"]
        print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
              f"Sharpe={s['avg_sharpe']:+.2f}, "
              f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

    _print_comparison(mh_rows, "STAGE 2: MIN-HOLD SWEEP")

    best_mh_row = max(mh_rows, key=lambda r: (
        r["summary"]["positive_sharpe"], r["summary"]["avg_sharpe"]))
    best_min_hold = best_mh_row["min_hold"]
    print(f"\n  Best min_hold: {best_min_hold}")

    # ── Stage 3: Continuous sizing ───────────────────────────
    print(f"\n  Stage 3: Continuous sizing "
          f"(deadzone={best_deadzone}, min_hold={best_min_hold})")
    cs_rows = []
    for cs, label in [(False, "binary (baseline)"), (True, "continuous")]:
        print(f"\n  Running {label}...")
        t0 = time.time()
        res = _run_config(**common, deadzone=best_deadzone,
                          min_hold=best_min_hold, continuous_sizing=cs)
        elapsed = time.time() - t0
        res["label"] = label
        res["continuous_sizing"] = cs
        cs_rows.append(res)
        s = res["summary"]
        print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
              f"Sharpe={s['avg_sharpe']:+.2f}, "
              f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

    _print_comparison(cs_rows, "STAGE 3: CONTINUOUS SIZING")

    best_cs_row = max(cs_rows, key=lambda r: (
        r["summary"]["positive_sharpe"], r["summary"]["avg_sharpe"]))
    best_continuous = best_cs_row.get("continuous_sizing", False)

    # ── Stage 4: Trend hold sweep ─────────────────────────────
    best_trend_follow = False
    best_trend_indicator = "tf4h_close_vs_ma20"
    best_trend_threshold = 0.0
    best_max_hold = 120
    trend_rows = []

    if not skip_stage4:
        # Stage 4a: indicator × threshold sweep
        TREND_INDICATORS = ["tf4h_close_vs_ma20", "close_vs_ma50"]
        TREND_THRESHOLDS = [-0.005, 0.0, 0.005, 0.01, 0.02]
        print(f"\n  Stage 4a: Trend indicator × threshold sweep"
              f" (deadzone={best_deadzone}, min_hold={best_min_hold},"
              f" continuous={best_continuous})")
        for ind in TREND_INDICATORS:
            for thr in TREND_THRESHOLDS:
                label = f"{ind}>{thr}"
                print(f"\n  Running {label}...")
                t0 = time.time()
                res = _run_config(
                    **common, deadzone=best_deadzone,
                    min_hold=best_min_hold, continuous_sizing=best_continuous,
                    trend_follow=True, trend_indicator=ind,
                    trend_threshold=thr, max_hold=120)
                elapsed = time.time() - t0
                res["label"] = label
                res["trend_indicator"] = ind
                res["trend_threshold"] = thr
                res["max_hold"] = 120
                trend_rows.append(res)
                s = res["summary"]
                print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
                      f"Sharpe={s['avg_sharpe']:+.2f}, "
                      f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

        # Also run baseline (no trend hold) for comparison
        label_base = "no_trend_hold (baseline)"
        print(f"\n  Running {label_base}...")
        t0 = time.time()
        res_base = _run_config(
            **common, deadzone=best_deadzone,
            min_hold=best_min_hold, continuous_sizing=best_continuous,
            trend_follow=False)
        elapsed = time.time() - t0
        res_base["label"] = label_base
        res_base["trend_indicator"] = None
        res_base["trend_threshold"] = None
        res_base["max_hold"] = None
        trend_rows.append(res_base)
        s = res_base["summary"]
        print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
              f"Sharpe={s['avg_sharpe']:+.2f}, "
              f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

        _print_comparison(trend_rows, "STAGE 4a: TREND HOLD × THRESHOLD")

        best_4a = max(trend_rows, key=lambda r: (
            r["summary"]["positive_sharpe"], r["summary"]["total_return"]))

        if best_4a.get("trend_indicator") is not None:
            best_trend_follow = True
            best_trend_indicator = best_4a["trend_indicator"]
            best_trend_threshold = best_4a["trend_threshold"]
            print(f"\n  Best 4a: {best_trend_indicator} > {best_trend_threshold}")

            # Stage 4b: max_hold sweep with best indicator/threshold
            MAX_HOLD_VALUES = [72, 120, 168, 240]
            print(f"\n  Stage 4b: max_hold sweep {MAX_HOLD_VALUES}"
                  f" ({best_trend_indicator} > {best_trend_threshold})")
            mh_trend_rows = []
            for mh in MAX_HOLD_VALUES:
                label = f"max_hold={mh}"
                print(f"\n  Running {label}...")
                t0 = time.time()
                res = _run_config(
                    **common, deadzone=best_deadzone,
                    min_hold=best_min_hold, continuous_sizing=best_continuous,
                    trend_follow=True, trend_indicator=best_trend_indicator,
                    trend_threshold=best_trend_threshold, max_hold=mh)
                elapsed = time.time() - t0
                res["label"] = label
                res["max_hold"] = mh
                mh_trend_rows.append(res)
                s = res["summary"]
                print(f"    {s['positive_sharpe']}/{s['n_folds']} pass, "
                      f"Sharpe={s['avg_sharpe']:+.2f}, "
                      f"Return={s['total_return']*100:+.2f}% ({elapsed:.0f}s)")

            _print_comparison(mh_trend_rows, "STAGE 4b: MAX HOLD SWEEP")
            best_4b = max(mh_trend_rows, key=lambda r: (
                r["summary"]["positive_sharpe"], r["summary"]["total_return"]))
            best_max_hold = best_4b["max_hold"]
            print(f"\n  Best max_hold: {best_max_hold}")
        else:
            print("\n  No trend hold config beat baseline — trend hold disabled")
            best_trend_follow = False

    # ── Final summary ────────────────────────────────────────
    # Re-evaluate best overall config
    if best_trend_follow:
        final_res = _run_config(
            **common, deadzone=best_deadzone,
            min_hold=best_min_hold, continuous_sizing=best_continuous,
            trend_follow=True, trend_indicator=best_trend_indicator,
            trend_threshold=best_trend_threshold, max_hold=best_max_hold)
        best = final_res["summary"]
        best_fold_results = final_res["fold_results"]
    else:
        best = best_cs_row["summary"]
        best_fold_results = best_cs_row["fold_results"]
    print(f"\n{'='*75}")
    print("  BEST CONFIG")
    print(f"{'='*75}")
    print(f"  deadzone:          {best_deadzone}")
    print(f"  min_hold:          {best_min_hold}")
    print(f"  continuous_sizing: {best_continuous}")
    if best_trend_follow:
        print("  trend_follow:      True")
        print(f"  trend_indicator:   {best_trend_indicator}")
        print(f"  trend_threshold:   {best_trend_threshold}")
        print(f"  max_hold:          {best_max_hold}")
    else:
        print("  trend_follow:      False")
    print(f"  positive_sharpe:   {best['positive_sharpe']}/{best['n_folds']}")
    print(f"  avg_sharpe:        {best['avg_sharpe']:+.2f}")
    print(f"  total_return:      {best['total_return']*100:+.2f}%")

    target_pass = 16
    if best["positive_sharpe"] >= target_pass:
        print(f"\n  PASS: >= {target_pass}/{best['n_folds']} -> proceed to production training")
    else:
        print(f"\n  BELOW TARGET: {best['positive_sharpe']}/{best['n_folds']} "
              f"< {target_pass} -> review results")

    # Save all results
    sweep_result = {
        "symbol": symbol,
        "best_deadzone": best_deadzone,
        "best_min_hold": best_min_hold,
        "best_continuous_sizing": best_continuous,
        "best_trend_follow": best_trend_follow,
        "best_trend_indicator": best_trend_indicator if best_trend_follow else None,
        "best_trend_threshold": best_trend_threshold if best_trend_follow else None,
        "best_max_hold": best_max_hold if best_trend_follow else None,
        "ensemble": ensemble,
        "best_summary": best,
        "stages": {
            "deadzone": [{"value": r["deadzone"],
                          "summary": r["summary"]} for r in dz_rows],
            "min_hold": [{"value": r["min_hold"],
                          "summary": r["summary"]} for r in mh_rows],
            "continuous": [{"label": r["label"],
                            "summary": r["summary"]} for r in cs_rows],
            "trend_hold": [{"label": r["label"],
                            "trend_indicator": r.get("trend_indicator"),
                            "trend_threshold": r.get("trend_threshold"),
                            "max_hold": r.get("max_hold"),
                            "summary": r["summary"]} for r in trend_rows],
        },
    }
    out_path = out_dir / f"signal_sweep_{symbol}.json"
    with open(out_path, "w") as f:
        json.dump(sweep_result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    # Print detailed fold results for best config
    print_report(best_fold_results, best)


if __name__ == "__main__":
    main()
