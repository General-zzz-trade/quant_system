"""Builder for RustTickProcessor — converts Python model config to Rust-native tick processor."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_tick_processor(
    symbols: list[str],
    currency: str,
    balance: float,
    model_dirs: dict[str, Path],  # runner_key -> model dir
    zscore_window: int = 720,
    zscore_warmup: int = 180,
) -> Any:
    """Build a RustTickProcessor from model directories.

    Models must have JSON exports (lgbm_*.json) alongside serialized files.
    Returns None if Rust tick processor is unavailable or models lack JSON exports.
    """
    try:
        from _quant_hotpath import RustTickProcessor
    except ImportError:
        logger.warning("RustTickProcessor not available")
        return None

    # Collect JSON model paths
    model_paths = []
    ensemble_weights = []

    for runner_key, model_dir in model_dirs.items():
        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.warning("No config.json in %s — skipping tick processor", model_dir)
            return None

        with open(config_path) as f:
            config = json.load(f)

        # Find JSON model exports (lgbm_*.json)
        horizon_models = config.get("horizon_models", [])
        if not horizon_models:
            # Legacy single model
            json_path = model_dir / "lgbm_v8.json"
            if json_path.exists():
                model_paths.append(str(json_path))
                ensemble_weights.append(1.0)
            else:
                logger.info("No JSON model export for %s — tick processor unavailable", runner_key)
                return None
        else:
            for hm in horizon_models:
                lgbm_name = hm.get("lgbm", "")
                json_name = lgbm_name.replace(".pkl", ".json")
                json_path = model_dir / json_name
                if json_path.exists():
                    model_paths.append(str(json_path))
                    ensemble_weights.append(hm.get("weight", 1.0))
                else:
                    logger.info("Missing JSON export %s — tick processor unavailable", json_path)
                    return None

    if not model_paths:
        return None

    try:
        processor = RustTickProcessor.create(
            symbols=symbols,
            currency=currency,
            balance=balance,
            model_paths=model_paths,
            ensemble_weights=ensemble_weights,
            zscore_window=zscore_window,
            zscore_warmup=zscore_warmup,
        )
        logger.info(
            "RustTickProcessor created: %d symbols, %d models",
            len(symbols), len(model_paths),
        )
        return processor
    except Exception:
        logger.exception("Failed to create RustTickProcessor")
        return None
