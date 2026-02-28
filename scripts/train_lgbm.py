#!/usr/bin/env python3
"""Train LGBM alpha model via walk-forward cross-validation.

Usage:
    python3 -m scripts.train_lgbm --data data/btcusdt.csv --out models/
    python3 -m scripts.train_lgbm --config infra/config/examples/training.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from alpha.models.lgbm_alpha import LGBMAlphaModel
from alpha.training.trainer import ModelTrainer
from features.live_computer import LiveFeatureComputer

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES = ("ma_fast", "ma_slow", "vol", "momentum")
TARGET_HORIZON = 5  # bars forward


def compute_features_from_ohlcv(
    df: "pd.DataFrame",
    *,
    fast_ma: int = 10,
    slow_ma: int = 30,
    vol_window: int = 20,
) -> "pd.DataFrame":
    """Compute features using LiveFeatureComputer over historical OHLCV."""
    import pandas as pd

    computer = LiveFeatureComputer(fast_ma=fast_ma, slow_ma=slow_ma, vol_window=vol_window)

    records = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "BTCUSDT"))
        close = float(row["close"])
        volume = float(row.get("volume", 0))
        high = float(row.get("high", close))
        low = float(row.get("low", close))

        computer.on_bar(symbol, close=close, volume=volume, high=high, low=low)
        feats = computer.get_features_dict(symbol)
        feats["close"] = close
        records.append(feats)

    return pd.DataFrame(records)


def compute_target(closes: "pd.Series", horizon: int = 5) -> "pd.Series":
    """Forward return as target: (close[t+horizon] - close[t]) / close[t]."""
    return closes.shift(-horizon) / closes - 1.0


def load_config(path: Path) -> dict:
    import yaml  # type: ignore[import-untyped]
    with open(path) as f:
        result: dict = yaml.safe_load(f)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LGBM alpha model")
    parser.add_argument("--data", type=Path, help="Path to OHLCV CSV")
    parser.add_argument("--out", type=Path, default=Path("models/"), help="Output directory")
    parser.add_argument("--config", type=Path, help="Training config YAML")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward splits")
    parser.add_argument("--horizon", type=int, default=TARGET_HORIZON, help="Target horizon (bars)")
    parser.add_argument("--fast-ma", type=int, default=10)
    parser.add_argument("--slow-ma", type=int, default=30)
    parser.add_argument("--vol-window", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)
        features_cfg = cfg.get("features", {})
        training_cfg = cfg.get("training", {})
        args.fast_ma = features_cfg.get("fast_ma", args.fast_ma)
        args.slow_ma = features_cfg.get("slow_ma", args.slow_ma)
        args.vol_window = features_cfg.get("vol_window", args.vol_window)
        args.n_splits = training_cfg.get("n_splits", args.n_splits)
        args.horizon = training_cfg.get("horizon", args.horizon)
        if args.data is None:
            args.data = Path(cfg.get("data_path", "data/btcusdt.csv"))
        args.out = Path(training_cfg.get("out_dir", str(args.out)))

    if args.data is None:
        parser.error("--data or --config with data_path required")

    import numpy as np
    import pandas as pd

    logger.info("Loading data from %s", args.data)
    df = pd.read_csv(args.data)

    required = {"close"}
    if not required.issubset(df.columns):
        sys.exit(f"CSV must contain columns: {required}. Found: {list(df.columns)}")

    logger.info("Computing features (fast_ma=%d, slow_ma=%d, vol=%d)",
                args.fast_ma, args.slow_ma, args.vol_window)
    feat_df = compute_features_from_ohlcv(
        df, fast_ma=args.fast_ma, slow_ma=args.slow_ma, vol_window=args.vol_window,
    )

    # Target: forward returns
    target = compute_target(df["close"], horizon=args.horizon)

    # Align: drop rows where features or target are NaN
    X = feat_df[list(FEATURE_NAMES)]
    y = target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask].values
    y = y[mask].values

    logger.info("Training samples: %d (dropped %d warmup/tail rows)", len(X), len(df) - len(X))

    if len(X) < 200:
        sys.exit(f"Not enough data: {len(X)} samples (need >= 200)")

    # Train
    model = LGBMAlphaModel(name="lgbm_alpha", feature_names=FEATURE_NAMES)
    trainer = ModelTrainer(model=model, out_dir=args.out)
    results = trainer.walk_forward_train(X, y, n_splits=args.n_splits, expanding=True)

    # Summary
    metrics_summary = {
        "folds": len(results),
        "avg_val_mse": float(np.mean([r.metrics["val_mse"] for r in results])),
        "avg_direction_accuracy": float(np.mean([r.metrics["direction_accuracy"] for r in results])),
        "per_fold": [{"fold": r.fold, **r.metrics} for r in results],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Training complete. Avg direction_accuracy=%.4f", metrics_summary["avg_direction_accuracy"])
    logger.info("Model saved to %s", args.out / "lgbm_alpha_final.pkl")
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
