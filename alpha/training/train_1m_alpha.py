#!/usr/bin/env python3
"""Train 1-minute Alpha — multi-resolution features on 1m bars.

Trains LGBM model for minute-level trading using:
  - Fast features: short-window microstructure on 1m bars
  - Slow features: hourly-resampled context features (forward-filled)
  - 4h features: 4-hourly regime context

IC analysis is run first as a GO/NO-GO gate.

Usage:
    # IC analysis only (go/no-go check)
    python3 -m scripts.train_1m_alpha --symbol BTCUSDT --ic-only

    # Full train
    python3 -m scripts.train_1m_alpha --symbol BTCUSDT

    # Custom horizon
    python3 -m scripts.train_1m_alpha --symbol BTCUSDT --horizon 3
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from features.multi_resolution import (
    compute_multi_resolution_features,
)
from features.dynamic_selector import greedy_ic_select, stable_icir_select, _rankdata, _spearman_ic

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────

V1M_DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.01,
    "num_leaves": 12,
    "min_child_samples": 200,
    "reg_alpha": 0.5,
    "reg_lambda": 5.0,
    "subsample": 0.5,
    "colsample_bytree": 0.6,
    "objective": "regression",
    "verbosity": -1,
}

V1M_SEARCH_SPACE_DICT = {
    "max_depth": (3, 6),
    "num_leaves": (8, 20),
    "min_child_samples": (100, 500),
    "learning_rate": (0.005, 0.02),
    "reg_alpha": (0.1, 2.0),
    "reg_lambda": (1.0, 10.0),
    "subsample": (0.3, 0.7),
    "colsample_bytree": (0.4, 0.8),
}

# Walk-forward constants (1m bars)
MIN_TRAIN_BARS = 525_600     # 12 months * 43800 bars/month ~ 365 * 1440
TEST_BARS = 131_400          # 3 months
STEP_BARS = 131_400          # 3-month step
WARMUP = 300                 # 5 hours of warmup

HORIZON_DEFAULT = 5          # 5-minute forward return
TARGET_MODE = "clipped"

# Cost model (more conservative for 1m frequency)
FEE_BPS = 4e-4
SLIPPAGE_BPS = 3e-4
COST_PER_TRADE = FEE_BPS + SLIPPAGE_BPS  # 7 bps total

# Signal parameters
DEADZONE = 0.3
MIN_HOLD = 3
ZSCORE_WINDOW = 720
MONTHLY_GATE_WINDOW = 28800  # 480h * 60 = 28800 1m bars

# Blacklisted features (known to be noisy at 1m)
BLACKLIST_1M: set = set()

# IC analysis thresholds
IC_PASS_THRESHOLD = 0.01
MIN_PASSING_FEATURES = 3


# ── Target computation ──────────────────────────────────────

def _compute_target(
    closes: np.ndarray,
    horizon: int,
    mode: str = "clipped",
) -> np.ndarray:
    """Compute forward return target (vectorized)."""
    n = len(closes)
    raw_ret = np.full(n, np.nan)
    raw_ret[:n - horizon] = closes[horizon:] / closes[:n - horizon] - 1.0

    if mode == "raw":
        return raw_ret
    if mode == "clipped":
        valid = raw_ret[~np.isnan(raw_ret)]
        if len(valid) < 10:
            return raw_ret
        p1, p99 = np.percentile(valid, [1, 99])
        return np.where(np.isnan(raw_ret), np.nan, np.clip(raw_ret, p1, p99))
    raise ValueError(f"Unknown target mode: {mode}")


# ── IC Analysis ──────────────────────────────────────────────

IC_SUBSAMPLE_SIZE = 500_000  # subsample for IC when data > this


def _fast_spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Fast Spearman rank IC using scipy if available, else numpy."""
    try:
        from scipy.stats import spearmanr
        r, _ = spearmanr(x, y)
        return float(r) if not np.isnan(r) else 0.0
    except ImportError:
        return float(_spearman_ic(_rankdata(x), _rankdata(y)))


def run_ic_analysis(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    horizons: List[int] = None,
) -> pd.DataFrame:
    """Compute Spearman IC for each feature at multiple horizons.

    For large datasets (>500K bars), subsamples to keep IC computation fast.
    Uses scipy.stats.spearmanr when available for C-level performance.

    Returns DataFrame: feature | horizon | IC | abs_IC | pass
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    results = []
    for h in horizons:
        y = _compute_target(closes, h, "clipped")
        valid = ~np.isnan(y)
        y_v = y[valid]
        if len(y_v) < 100:
            continue

        # Subsample if dataset is very large
        if len(y_v) > IC_SUBSAMPLE_SIZE:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(y_v), IC_SUBSAMPLE_SIZE, replace=False)
            idx.sort()
        else:
            idx = np.arange(len(y_v))

        y_sub = y_v[idx]

        for fname in feature_names:
            if fname not in feat_df.columns:
                continue
            x = feat_df[fname].values.astype(np.float64)
            x_v = x[valid]
            x_sub = x_v[idx]
            nan_mask = ~np.isnan(x_sub)
            if nan_mask.sum() < 100:
                continue
            x_clean = x_sub[nan_mask]
            y_clean = y_sub[nan_mask]
            if np.std(x_clean) < 1e-12:
                continue
            ic = _fast_spearman_ic(x_clean, y_clean)
            results.append({
                "feature": fname,
                "horizon": h,
                "IC": ic,
                "abs_IC": abs(ic),
                "pass": abs(ic) >= IC_PASS_THRESHOLD,
            })

    return pd.DataFrame(results)


def check_go_nogo(ic_df: pd.DataFrame, target_horizon: int = HORIZON_DEFAULT) -> bool:
    """Check if enough features pass IC threshold at target horizon."""
    if ic_df.empty:
        return False
    h_df = ic_df[ic_df["horizon"] == target_horizon]
    n_pass = h_df["pass"].sum()
    print(f"\n=== GO/NO-GO Check (horizon={target_horizon}) ===")
    print(f"Features with |IC| >= {IC_PASS_THRESHOLD}: {n_pass}/{len(h_df)}")
    if n_pass >= MIN_PASSING_FEATURES:
        print(">>> GO — proceeding with training")
        return True
    else:
        print(f">>> NO-GO — need at least {MIN_PASSING_FEATURES} passing features")
        return False


# ── Feature loading ──────────────────────────────────────────

def load_and_compute_features(
    symbol: str,
    data_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load 1m data and compute multi-resolution features.

    Returns (raw_df, feat_df).
    """
    if data_path is None:
        data_path = f"data_files/{symbol}_1m.csv"
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    print(f"Loading {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} bars loaded")

    print("Computing multi-resolution features...")
    feat_df = compute_multi_resolution_features(df, symbol)
    n_features = len([c for c in feat_df.columns if c != "close"])
    print(f"  {n_features} features computed")

    return df, feat_df


# ── Training ──────────────────────────────────────────────────

def train_model(
    feat_df: pd.DataFrame,
    closes: np.ndarray,
    feature_names: List[str],
    *,
    horizon: int = HORIZON_DEFAULT,
    target_mode: str = TARGET_MODE,
    params: Optional[Dict] = None,
    selector: str = "greedy",
    top_k: int = 15,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a single LGBM model on the full dataset.

    Returns dict with model, selected features, IC, etc.
    """
    import lightgbm as lgb

    if params is None:
        params = dict(V1M_DEFAULT_PARAMS)

    y = _compute_target(closes, horizon, target_mode)

    # Skip warmup
    X = feat_df[feature_names].values[WARMUP:].astype(np.float64)
    y_trimmed = y[WARMUP:]
    valid = ~np.isnan(y_trimmed)
    X_valid = X[valid]
    y_valid = y_trimmed[valid]

    print(f"Training samples: {len(X_valid):,}")

    # Feature selection
    if selector == "stable_icir":
        selected = stable_icir_select(X_valid, y_valid, feature_names, top_k=top_k)
    else:
        selected = greedy_ic_select(X_valid, y_valid, feature_names, top_k=top_k)

    print(f"Selected features ({len(selected)}): {selected}")

    sel_idx = [feature_names.index(f) for f in selected]
    X_sel = X_valid[:, sel_idx]

    # Train/val split (last 20% as validation for early stopping)
    n = len(X_sel)
    val_size = n // 5
    X_train, X_val = X_sel[:-val_size], X_sel[-val_size:]
    y_train, y_val = y_valid[:-val_size], y_valid[-val_size:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    bst = lgb.train(
        params, dtrain,
        num_boost_round=params.get("n_estimators", 300),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    # Evaluate IC on validation
    y_pred_val = bst.predict(X_val)
    val_valid = ~np.isnan(y_val)
    ic = 0.0
    if val_valid.sum() > 10:
        ic = float(_spearman_ic(
            _rankdata(y_pred_val[val_valid]),
            _rankdata(y_val[val_valid]),
        ))
    print(f"Validation IC: {ic:.4f}")

    # Save model
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bst.save_model(str(out / "model.txt"))
        with open(out / "features.json", "w") as f:
            json.dump({"features": selected, "horizon": horizon,
                       "target_mode": target_mode, "ic": ic}, f, indent=2)
        print(f"Model saved to {out}")

    return {
        "model": bst,
        "features": selected,
        "ic": ic,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 1-minute alpha model")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--data", help="Path to 1m CSV")
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT)
    parser.add_argument("--ic-only", action="store_true", help="Run IC analysis only")
    parser.add_argument("--selector", default="greedy", choices=["greedy", "stable_icir"])
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output-dir", help="Model output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load data and compute features
    df, feat_df = load_and_compute_features(args.symbol, args.data)
    closes = df["close"].values.astype(np.float64)

    # Feature names (exclude 'close')
    feature_names = [c for c in feat_df.columns if c != "close"]

    # IC analysis
    print("\n" + "=" * 60)
    print("IC ANALYSIS")
    print("=" * 60)
    ic_df = run_ic_analysis(feat_df, closes, feature_names)

    if not ic_df.empty:
        for h in sorted(ic_df["horizon"].unique()):
            h_df = ic_df[ic_df["horizon"] == h].sort_values("abs_IC", ascending=False)
            print(f"\n--- Horizon = {h} min ---")
            for _, row in h_df.head(15).iterrows():
                status = "PASS" if row["pass"] else "FAIL"
                print(f"  {row['feature']:30s}  IC={row['IC']:+.4f}  [{status}]")

    go = check_go_nogo(ic_df, args.horizon)

    if args.ic_only:
        if not ic_df.empty:
            out_path = Path(f"results/ic_analysis_1m_{args.symbol}.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ic_df.to_csv(out_path, index=False)
            print(f"\nIC results saved to {out_path}")
        return

    if not go:
        print("\nAborting: IC analysis did not pass GO/NO-GO gate.")
        return

    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    output_dir = args.output_dir or f"models_v8/{args.symbol}_1m_v1"
    result = train_model(
        feat_df, closes, feature_names,
        horizon=args.horizon,
        selector=args.selector,
        top_k=args.top_k,
        output_dir=output_dir,
    )
    print(f"\nDone. IC={result['ic']:.4f}, "
          f"features={result['features']}")


if __name__ == "__main__":
    main()
