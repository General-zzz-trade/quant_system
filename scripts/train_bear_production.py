#!/usr/bin/env python3
"""Train production bear C model on ALL bear bars (full dataset).

Trains the same binary classifier as Strategy C in bear_alpha_research,
but on the entire dataset (not per-fold) for production deployment.

Output:
    models_v8/BTCUSDT_bear_c/lgbm_bear.pkl   — signed model artifact
    models_v8/BTCUSDT_bear_c/config.json      — features, metrics, threshold

Usage:
    python3 -m scripts.train_bear_production --symbol BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from alpha.models.lgbm_alpha import LGBMAlphaModel
from scripts.bear_alpha_research import (
    _load_features,
    _compute_bear_bull_mask,
    _compute_bear_target,
    V7_DEFAULT_PARAMS,
)

logger = logging.getLogger(__name__)

BEAR_DETECTOR_POOL = [
    "funding_zscore_24", "funding_momentum", "funding_extreme",
    "funding_sign_persist", "funding_cumulative_8",
    "basis", "basis_zscore_24", "basis_momentum",
    "vol_20", "vol_regime", "parkinson_vol", "atr_norm_14",
    "fgi_normalized", "fgi_extreme",
    "rsi_14", "bb_pctb_20",
    "oi_acceleration", "leverage_proxy",
]


def train_bear_model(symbol: str = "BTCUSDT") -> Path:
    """Train production bear detector and save to models_v8/{symbol}_bear_c/."""
    print(f"\n{'='*70}")
    print(f"  Training production bear model for {symbol}")
    print(f"{'='*70}")

    feat_df = _load_features(symbol)
    closes = feat_df["close"].values.astype(np.float64)

    bear_mask = _compute_bear_bull_mask(closes)
    target_bin = _compute_bear_target(closes, horizon=24, threshold=-0.02)

    features = [f for f in BEAR_DETECTOR_POOL if f in feat_df.columns]
    print(f"  Features ({len(features)}): {features}")

    X_all = feat_df[features].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, 0.0)

    valid_mask = bear_mask & ~np.isnan(target_bin)
    valid_mask[:65] = False

    valid_idx = np.where(valid_mask)[0]
    X_clean = X_all[valid_idx]
    y_clean = target_bin[valid_idx]

    n_pos = int(y_clean.sum())
    n_neg = len(y_clean) - n_pos
    print(f"  Bear samples: {len(X_clean)} (pos={n_pos}, neg={n_neg})")

    if len(X_clean) < 500:
        raise RuntimeError(f"Not enough bear samples: {len(X_clean)}")

    # Scale positive class weight to handle imbalance
    neg_ratio = n_neg / max(n_pos, 1)
    cls_params = V7_DEFAULT_PARAMS.copy()
    cls_params["objective"] = "binary"
    cls_params["metric"] = "binary_logloss"
    cls_params["scale_pos_weight"] = neg_ratio
    cls_params["n_estimators"] = 300
    cls_params["learning_rate"] = 0.02
    cls_params["min_child_samples"] = 50

    model = LGBMAlphaModel(name="bear_detector", feature_names=tuple(features))
    metrics = model.fit_classifier(
        X_clean, y_clean,
        params=cls_params,
        early_stopping_rounds=0,
        embargo_bars=26,
    )

    out_dir = Path(f"models_v8/{symbol}_bear_c")
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / "lgbm_bear.pkl"
    model.save(pkl_path)
    print(f"  Saved model → {pkl_path}")

    config = {
        "version": "bear_c_v1",
        "symbol": symbol,
        "is_classifier": True,
        "models": ["lgbm_bear.pkl"],
        "features": features,
        "threshold": -0.02,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "n_train": len(X_clean),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }
    config_path = out_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config → {config_path}")

    print(f"\n  Metrics: {metrics}")
    print(f"  Done. Model ready at {out_dir}/")
    return out_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train production bear model")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    args = parser.parse_args()
    train_bear_model(args.symbol)


if __name__ == "__main__":
    main()
