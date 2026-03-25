"""Model loading and adapter creation for the Bybit alpha runner."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from infra.model_signing import load_verified_pickle

logger = logging.getLogger(__name__)


def load_model(model_dir: Path) -> dict:
    """Load model config + all horizon models for IC-weighted ensemble.

    All model artifacts are HMAC-verified before deserialization.
    In live mode (BYBIT_BASE_URL=https://api.bybit.com), unsigned or
    tampered models are rejected with a clear error.
    """
    # Handle case where model_dir is a directory vs file path
    if model_dir.is_dir():
        config_path = model_dir / "config.json"
    elif model_dir.is_file():
        config_path = model_dir if model_dir.name == "config.json" else model_dir
        model_dir = model_dir.parent
    else:
        raise FileNotFoundError(f"Model path does not exist: {model_dir}")

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in model directory: {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    # Load ALL horizon models for ensemble (HMAC-verified)
    horizon_models = []
    for hm in config.get("horizon_models", []):
        lgbm_path = model_dir / hm["lgbm"]
        if not lgbm_path.exists():
            continue
        raw = load_verified_pickle(lgbm_path)
        model = raw["model"] if isinstance(raw, dict) else raw

        # Also load XGBoost if available
        xgb_model = None
        xgb_path = model_dir / hm.get("xgb", "")
        if xgb_path.exists() and xgb_path.is_file():
            xgb_raw = load_verified_pickle(xgb_path)
            xgb_model = xgb_raw["model"] if isinstance(xgb_raw, dict) else xgb_raw

        # Also load Ridge if available (walk-forward winner: 15/20 PASS)
        ridge_model = None
        ridge_features = None
        ridge_name = hm.get("ridge", "")
        ridge_path = model_dir / ridge_name if ridge_name else None
        if ridge_path and ridge_path.is_file():
            ridge_raw = load_verified_pickle(ridge_path)
            ridge_model = ridge_raw["model"] if isinstance(ridge_raw, dict) else ridge_raw
            ridge_features = ridge_raw.get("features") if isinstance(ridge_raw, dict) else None

        # Validate Ridge feature count matches model expectation
        if ridge_model is not None:
            expected = getattr(ridge_model, 'n_features_in_', None)
            effective_ridge_features = ridge_features or hm["features"]
            if expected and len(effective_ridge_features) != expected:
                logger.warning(
                    "Ridge expects %d features but config has %d — disabling Ridge for horizon %s",
                    expected, len(effective_ridge_features), hm["horizon"],
                )
                ridge_model = None
                ridge_features = None

        horizon_models.append({
            "horizon": hm["horizon"],
            "lgbm": model,
            "xgb": xgb_model,
            "ridge": ridge_model,
            "ridge_features": ridge_features,  # may differ from lgbm features
            "features": hm["features"],
            "ic": hm.get("ic", 0.01),
        })

    if not horizon_models:
        raise RuntimeError(f"No models found in {model_dir}")

    # Primary model = first horizon (for feature list compatibility)
    primary = horizon_models[0]

    return {
        "config": config,
        "model": primary["lgbm"],  # backward compat
        "features": primary["features"],
        "horizon_models": horizon_models,
        "lgbm_xgb_weight": config.get("lgbm_xgb_weight", 0.5),
        "deadzone": config.get("deadzone", 2.0),
        "min_hold": config.get("min_hold", 18),
        "max_hold": config.get("max_hold", 96),
        "zscore_window": config.get("zscore_window", 720),
        "zscore_warmup": config.get("zscore_warmup", 180),
        "long_only": config.get("long_only", False),
    }


def create_adapter():
    """Create Bybit adapter from environment variables."""
    from execution.adapters.bybit import BybitAdapter, BybitConfig

    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "BYBIT_API_KEY and BYBIT_API_SECRET environment variables are required. "
            "Set them in .env or export them. See .env.example for all required vars."
        )
    base_url = os.environ.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")

    config = BybitConfig(api_key=api_key, api_secret=api_secret,
                         base_url=base_url)
    adapter = BybitAdapter(config)
    if not adapter.connect():
        raise RuntimeError("Failed to connect to Bybit")
    return adapter
