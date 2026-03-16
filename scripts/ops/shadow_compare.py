#!/usr/bin/env python3
"""Shadow comparison — candidate vs production model side-by-side.

Runs both models on the same OOS data and compares IC, direction accuracy,
and Sharpe ratio. Auto-promotes if candidate significantly outperforms.

Usage:
    python3 -m scripts.shadow_compare --symbol BTCUSDT
    python3 -m scripts.shadow_compare --symbol BTCUSDT --candidate-id <model_id>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compare_models(
    symbol: str,
    registry_db: str = "model_registry.db",
    candidate_id: Optional[str] = None,
    auto_promote: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compare candidate vs production model on OOS data."""
    from research.model_registry.registry import ModelRegistry
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from scripts.train_unified import (
        validate_oos_extended,
        _load_and_compute_features,
        _add_regime_feature,
    )

    registry = ModelRegistry(registry_db)
    reg_name = f"alpha_unified_{symbol}"

    # Get production model
    prod_mv = registry.get_production(reg_name)
    if prod_mv is None:
        logger.warning("No production model for %s", reg_name)
        return None

    # Get candidate model (latest non-production or specific ID)
    if candidate_id:
        cand_mv = registry.get(candidate_id)
    else:
        versions = registry.list_versions(reg_name)
        candidates = [v for v in versions if not v.is_production]
        if not candidates:
            logger.info("No candidate models to compare for %s", reg_name)
            return None
        cand_mv = candidates[-1]  # Latest non-production

    if cand_mv is None:
        logger.warning("Candidate model not found")
        return None

    logger.info("Comparing %s v%d (prod) vs v%d (candidate)",
                reg_name, prod_mv.version, cand_mv.version)

    # Load OOS data
    oos_path = Path(f"data_files/{symbol}_1h_oos.csv")
    if not oos_path.exists():
        logger.warning("No OOS file for %s", symbol)
        return None

    oos_df = pd.read_csv(oos_path)

    # Compute features
    feat_df = _load_and_compute_features(symbol, oos_df)
    if feat_df is None or len(feat_df) < 100:
        return None
    feat_df = _add_regime_feature(feat_df)

    ts_col = "open_time" if "open_time" in oos_df.columns else "timestamp"
    timestamps = oos_df[ts_col].values.astype(np.int64)

    # Validate production model
    prod_features = list(prod_mv.features)
    prod_model = LGBMAlphaModel(name="prod", feature_names=tuple(prod_features))
    prod_model_path = Path(f"models_unified/{symbol}/lgbm_unified.pkl")
    if prod_model_path.exists():
        prod_model.load(prod_model_path)
    else:
        logger.warning("Production model file not found")
        return None

    horizon = 6
    if prod_mv.tags:
        horizon_tags = [t for t in prod_mv.tags if t.startswith("horizon_")]
        if horizon_tags:
            try:
                horizon = int(horizon_tags[0].replace("horizon_", ""))
            except ValueError as e:
                logger.warning("Failed to parse horizon tag '%s': %s", horizon_tags[0], e)
    mode_tag = [t for t in prod_mv.tags if t.startswith("mode_")]
    target_mode = mode_tag[0].replace("mode_", "") if mode_tag else "clipped"

    prod_oos = validate_oos_extended(
        prod_model, symbol, prod_features,
        target_horizon=horizon,
        target_mode=target_mode,
        feat_df=feat_df.copy(),
        timestamps=timestamps,
    )

    # Validate candidate model
    cand_features = list(cand_mv.features)
    cand_model = LGBMAlphaModel(name="cand", feature_names=tuple(cand_features))
    # Load candidate model from version-specific path (falls back to staging dir)
    cand_model_path = Path(f"models_unified/{symbol}/lgbm_unified_v{cand_mv.version}.pkl")
    if not cand_model_path.exists():
        cand_model_path = Path(f"models_unified/{symbol}/lgbm_unified_candidate.pkl")
    if cand_model_path.exists():
        cand_model.load(cand_model_path)
    else:
        logger.warning("Candidate model file not found")
        return None

    cand_oos = validate_oos_extended(
        cand_model, symbol, cand_features,
        target_horizon=horizon,
        target_mode=target_mode,
        feat_df=feat_df.copy(),
        timestamps=timestamps,
    )

    if prod_oos is None or cand_oos is None:
        logger.warning("OOS validation failed for one or both models")
        return None

    # Compare metrics
    comparison = {
        "symbol": symbol,
        "production": {
            "model_id": prod_mv.model_id,
            "version": prod_mv.version,
            "oos_ic": prod_oos["overall"]["ic"],
            "h2_ic": prod_oos["h2"]["ic"],
            "oos_sharpe": prod_oos["overall"]["sharpe"],
            "stability": prod_oos["stability_score"],
            "ic_positive_ratio": prod_oos.get("ic_positive_ratio", 0),
        },
        "candidate": {
            "model_id": cand_mv.model_id,
            "version": cand_mv.version,
            "oos_ic": cand_oos["overall"]["ic"],
            "h2_ic": cand_oos["h2"]["ic"],
            "oos_sharpe": cand_oos["overall"]["sharpe"],
            "stability": cand_oos["stability_score"],
            "ic_positive_ratio": cand_oos.get("ic_positive_ratio", 0),
        },
    }

    # Deltas
    ic_delta = cand_oos["overall"]["ic"] - prod_oos["overall"]["ic"]
    sharpe_delta = cand_oos["overall"]["sharpe"] - prod_oos["overall"]["sharpe"]
    comparison["deltas"] = {
        "ic": ic_delta,
        "sharpe": sharpe_delta,
        "h2_ic": cand_oos["h2"]["ic"] - prod_oos["h2"]["ic"],
    }

    # Decision
    should_promote = (
        cand_oos["passed"]
        and cand_oos["overall"]["ic"] > prod_oos["overall"]["ic"]
        and cand_oos["h2"]["ic"] > 0
        and cand_oos["deflated_sharpe"] > 0
    )
    comparison["should_promote"] = should_promote
    comparison["reason"] = (
        "Candidate outperforms production" if should_promote
        else "Candidate does not meet promotion criteria"
    )

    # Print comparison
    print(f"\n{'='*60}")
    print(f"  Shadow Comparison: {symbol}")
    print(f"{'='*60}")
    print(f"\n  {'Metric':<20} {'Production':>12} {'Candidate':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    for metric in ["oos_ic", "h2_ic", "oos_sharpe", "stability", "ic_positive_ratio"]:
        prod_val = comparison["production"][metric]
        cand_val = comparison["candidate"][metric]
        delta = cand_val - prod_val
        print(f"  {metric:<20} {prod_val:>12.4f} {cand_val:>12.4f} {delta:>+12.4f}")

    print(f"\n  Decision: {'PROMOTE' if should_promote else 'KEEP PRODUCTION'}")
    print(f"  Reason: {comparison['reason']}")

    if auto_promote and should_promote:
        registry.promote(cand_mv.model_id)
        comparison["promoted"] = True
        print(f"  AUTO-PROMOTED: {reg_name} v{cand_mv.version}")
    else:
        comparison["promoted"] = False

    # Save comparison
    out_path = Path(f"models_unified/{symbol}/shadow_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow model comparison")
    parser.add_argument("--symbol", required=True, help="Symbol to compare")
    parser.add_argument("--registry-db", default="model_registry.db")
    parser.add_argument("--candidate-id", help="Specific candidate model ID")
    parser.add_argument("--auto-promote", action="store_true",
                        help="Auto-promote if candidate wins")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    compare_models(
        args.symbol.upper(),
        registry_db=args.registry_db,
        candidate_id=args.candidate_id,
        auto_promote=args.auto_promote,
    )


if __name__ == "__main__":
    main()
