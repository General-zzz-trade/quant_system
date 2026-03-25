"""Helper functions for auto_retrain: ensemble calibration, experiment metadata, model validation."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def calibrate_ensemble_weights(model_dir: Path, shrinkage: float = 0.3) -> Optional[Dict[str, float]]:
    """Calibrate Ridge/LGBM ensemble weights using recent IC and save to config.

    Loads recent prediction scores and returns from the model directory,
    calls rust_adaptive_ensemble_calibrate to compute optimal weights,
    and saves them to config.json under "ensemble_weights".

    Returns calibrated weights dict or None if calibration failed.
    Best-effort: never raises.
    """
    try:
        from _quant_hotpath import rust_adaptive_ensemble_calibrate  # type: ignore[import-untyped]

        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.debug("calibrate_ensemble_weights: no config.json in %s", model_dir)
            return None

        with open(config_path) as f:
            cfg = json.load(f)

        # Load score history (per-model predictions) and return history from
        # validation artifacts saved by train_v11
        scores_path = model_dir / "val_scores.json"
        returns_path = model_dir / "val_returns.json"

        if scores_path.exists() and returns_path.exists():
            with open(scores_path) as f:
                score_history = json.load(f)
            with open(returns_path) as f:
                return_history = json.load(f)
        else:
            # Fallback: construct from metrics if validation artifacts not available
            metrics = cfg.get("metrics", {})
            per_h_ic = metrics.get("per_horizon_ic", {})
            if not per_h_ic:
                logger.debug("calibrate_ensemble_weights: no score/return history for %s", model_dir)
                return None
            # Use per-horizon IC as a proxy for score quality
            score_history = {
                "ridge": [float(ic) for ic in per_h_ic.values()],
                "lgbm": [float(ic) * 0.9 for ic in per_h_ic.values()],
            }
            return_history = [float(ic) for ic in per_h_ic.values()]

        result = rust_adaptive_ensemble_calibrate(
            "ic_weighted", score_history, return_history, shrinkage,
        )
        if result is None:
            logger.debug("rust_adaptive_ensemble_calibrate returned None for %s", model_dir)
            return None

        ridge_w = result.get("ridge", 0.6)
        lgbm_w = result.get("lgbm", 0.4)
        total = ridge_w + lgbm_w
        if total <= 0:
            logger.warning("calibrate_ensemble_weights: invalid weights sum for %s", model_dir)
            return None
        weights = {"ridge": round(ridge_w / total, 4), "lgbm": round(lgbm_w / total, 4)}

        # Save to config.json
        cfg["ensemble_weights"] = weights
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        logger.info("Ensemble weights calibrated for %s: %s (shrinkage=%.2f)",
                     model_dir.name, weights, shrinkage)
        return weights

    except Exception as e:
        logger.warning("calibrate_ensemble_weights failed (non-blocking): %s", e)
        return None


def save_experiment_metadata(
    model_dir: Path,
    symbol: str,
    horizons: List[int],
    old_config: Optional[Dict[str, Any]],
    retrain_trigger: str = "scheduled",
) -> bool:
    """Save experiment tracking metadata alongside the trained model.

    Writes {model_dir}/experiment_meta.json with training details, metrics,
    and lineage info. Best-effort: never raises.

    Returns True if metadata was saved, False otherwise.
    """
    try:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.debug("save_experiment_metadata: no config.json in %s", model_dir)
            return False

        with open(config_path) as f:
            cfg = json.load(f)

        metrics = cfg.get("metrics", {})
        features = cfg.get("features", [])

        # Determine parent model name from backup or old config
        parent_model = None
        if old_config:
            old_train_date = old_config.get("train_date", "")
            if old_train_date:
                date_part = old_train_date.split(" ")[0].replace("-", "")
                parent_model = f"{model_dir.name}_backup_{date_part}"

        # Compute train/val row counts from metrics if available
        train_rows = metrics.get("train_rows", metrics.get("n_train", 0))
        val_rows = metrics.get("val_rows", metrics.get("n_val", metrics.get("oos_bars", 0)))

        meta = {
            "trained_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "symbol": symbol,
            "horizons": horizons,
            "n_features": len(features),
            "feature_names": list(features),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_ic": metrics.get("train_ic", metrics.get("is_avg_ic", 0)),
            "val_ic": metrics.get("avg_ic", 0),
            "val_sharpe": metrics.get("sharpe", 0),
            "ensemble_weights": cfg.get("ensemble_weights", {
                "ridge": cfg.get("ridge_weight", 0.6),
                "lgbm": cfg.get("lgbm_weight", 0.4),
            }),
            "parent_model": parent_model,
            "retrain_trigger": retrain_trigger,
        }

        meta_path = model_dir / "experiment_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("Experiment metadata saved to %s", meta_path)
        return True

    except Exception as e:
        logger.warning("save_experiment_metadata failed (non-blocking): %s", e)
        return False
