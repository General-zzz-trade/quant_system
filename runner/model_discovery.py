# runner/model_discovery.py
"""Model auto-discovery and loading for production LiveRunner.

Centralizes model loading logic: scans models_v8/, loads only config.json-specified
models, and builds per-symbol LiveInferenceBridge instances with config.json overrides.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

_MODELS_DIR_DEFAULT = "models_v8"
_GATE_SUFFIX = "_gate_v2"


def discover_active_models(models_dir: str = _MODELS_DIR_DEFAULT) -> Dict[str, Dict[str, Any]]:
    """Scan models_v8/ for active model directories.

    Matches: {SYMBOL}_gate_v2/config.json
    Skips: *_backup_*, *_retired*, directories without config.json

    Returns:
        {symbol: {"dir": Path, "config": dict}}
    """
    base = Path(models_dir)
    if not base.exists():
        logger.warning("Models directory not found: %s", base)
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # Skip backups and retired
        if "_backup_" in name or "_retired" in name:
            continue
        # Must end with _gate_v2
        if not name.endswith(_GATE_SUFFIX):
            continue
        cfg_path = entry / "config.json"
        if not cfg_path.exists():
            logger.warning("No config.json in %s — skipping", entry)
            continue
        try:
            with open(cfg_path) as f:
                model_cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s — skipping", cfg_path, e)
            continue

        symbol = model_cfg.get("symbol", name.removesuffix(_GATE_SUFFIX))
        result[symbol] = {"dir": entry, "config": model_cfg}
        logger.info("Discovered model: %s in %s (v%s)", symbol, entry, model_cfg.get("version", "?"))

    return result


def load_symbol_models(
    symbol: str,
    model_dir: Path,
    model_cfg: Dict[str, Any],
) -> Tuple[List[Any], List[float]]:
    """Load models from config.json horizon_models array.

    Only loads files explicitly listed in config.json (lgbm/xgb fields).
    Does NOT glob *.pkl — avoids loading legacy/retired model files.

    Returns:
        (models_list, ensemble_weights)
    """
    from alpha.models.lgbm_alpha import LGBMAlphaModel
    from alpha.models.xgb_alpha import XGBAlphaModel

    horizon_models = model_cfg.get("horizon_models", [])
    if not horizon_models:
        logger.warning("No horizon_models in config for %s", symbol)
        return [], []

    lgbm_xgb_weight = model_cfg.get("lgbm_xgb_weight", 0.5)
    models: List[Any] = []
    raw_weights: List[float] = []

    for hm in horizon_models:
        horizon = hm.get("horizon", 0)
        features = hm.get("features", [])
        ic = abs(hm.get("ic", 0.0))

        # Load LightGBM model
        lgbm_file = hm.get("lgbm")
        if lgbm_file:
            pkl_path = model_dir / lgbm_file
            if pkl_path.exists():
                try:
                    m = LGBMAlphaModel(
                        name=f"{symbol}_h{horizon}_lgbm",
                        feature_names=tuple(features),
                    )
                    m.load(pkl_path)
                    models.append(m)
                    raw_weights.append(ic * lgbm_xgb_weight)
                    logger.info("Loaded %s", pkl_path)
                except Exception:
                    logger.warning("Failed to load %s", pkl_path, exc_info=True)
            else:
                logger.warning("LightGBM model not found: %s", pkl_path)

        # Load XGBoost model
        xgb_file = hm.get("xgb")
        if xgb_file:
            pkl_path = model_dir / xgb_file
            if pkl_path.exists():
                try:
                    m = XGBAlphaModel(
                        name=f"{symbol}_h{horizon}_xgb",
                        feature_names=tuple(features),
                    )
                    m.load(pkl_path)
                    models.append(m)
                    raw_weights.append(ic * (1.0 - lgbm_xgb_weight))
                    logger.info("Loaded %s", pkl_path)
                except Exception:
                    logger.warning("Failed to load %s", pkl_path, exc_info=True)
            else:
                logger.warning("XGBoost model not found: %s", pkl_path)

    # Normalize weights
    w_sum = sum(raw_weights) or 1.0
    ensemble_weights = [w / w_sum for w in raw_weights]

    logger.info(
        "%s: loaded %d model(s), weights=%s",
        symbol, len(models),
        [f"{w:.3f}" for w in ensemble_weights],
    )
    return models, ensemble_weights


def build_inference_bridge(
    symbol: str,
    models: List[Any],
    model_cfg: Dict[str, Any],
    runner_cfg: Any,
    *,
    metrics_exporter: Any = None,
    ensemble_weights: Optional[List[float]] = None,
) -> Any:
    """Build a LiveInferenceBridge for a single symbol.

    Priority: model_cfg (config.json) > runner_cfg (YAML).
    Logs overrides when config.json values differ from YAML defaults.
    """
    from alpha.inference.bridge import LiveInferenceBridge

    # Resolve per-field with config.json priority over YAML
    cfg_deadzone = model_cfg.get("deadzone")
    yaml_deadzone = runner_cfg.deadzone if hasattr(runner_cfg, "deadzone") else 0.5
    if isinstance(yaml_deadzone, dict):
        yaml_dz_val = yaml_deadzone.get(symbol, 0.5)
    else:
        yaml_dz_val = yaml_deadzone

    if cfg_deadzone is not None and cfg_deadzone != yaml_dz_val:
        logger.info(
            "%s deadzone=%.1f (config.json overrides YAML %.1f)",
            symbol, cfg_deadzone, yaml_dz_val,
        )

    deadzone_val = cfg_deadzone if cfg_deadzone is not None else yaml_dz_val
    deadzone: Dict[str, float] = {symbol: deadzone_val}

    cfg_min_hold = model_cfg.get("min_hold")
    yaml_min_hold = {}
    if hasattr(runner_cfg, "min_hold_bars") and runner_cfg.min_hold_bars:
        yaml_min_hold = runner_cfg.min_hold_bars
    yaml_mh_val = yaml_min_hold.get(symbol, 12) if isinstance(yaml_min_hold, dict) else 12

    if cfg_min_hold is not None and cfg_min_hold != yaml_mh_val:
        logger.info(
            "%s min_hold=%d (config.json overrides YAML %d)",
            symbol, cfg_min_hold, yaml_mh_val,
        )

    min_hold_val = cfg_min_hold if cfg_min_hold is not None else yaml_mh_val
    min_hold_bars: Dict[str, int] = {symbol: min_hold_val}

    cfg_max_hold = model_cfg.get("max_hold", getattr(runner_cfg, "max_hold", 120))

    long_only_symbols: Set[str] = set()
    if model_cfg.get("long_only"):
        long_only_symbols.add(symbol)
    elif hasattr(runner_cfg, "long_only_symbols") and runner_cfg.long_only_symbols:
        if symbol in runner_cfg.long_only_symbols:
            long_only_symbols.add(symbol)

    # zscore params: config.json > defaults (720/180)
    cfg_zscore_window = model_cfg.get("zscore_window", 720)
    cfg_zscore_warmup = model_cfg.get("zscore_warmup", 180)

    bridge = LiveInferenceBridge(
        models=models,
        metrics_exporter=metrics_exporter,
        min_hold_bars=min_hold_bars,
        deadzone=deadzone,
        max_hold=cfg_max_hold,
        long_only_symbols=long_only_symbols,
        ensemble_weights=ensemble_weights,
        monthly_gate=getattr(runner_cfg, "monthly_gate", False),
        monthly_gate_window=getattr(runner_cfg, "monthly_gate_window", 480),
        vol_target=getattr(runner_cfg, "vol_target", None),
        vol_feature=getattr(runner_cfg, "vol_feature", "atr_norm_14"),
        zscore_window=cfg_zscore_window,
        zscore_warmup=cfg_zscore_warmup,
    )

    logger.info(
        "%s bridge: deadzone=%.1f, min_hold=%d, max_hold=%d, zscore=%d/%d, models=%d",
        symbol, deadzone_val, min_hold_val, cfg_max_hold,
        cfg_zscore_window, cfg_zscore_warmup, len(models),
    )
    return bridge


def build_feature_computer() -> Any:
    """Create an EnrichedFeatureComputer instance."""
    from features.enriched_computer import EnrichedFeatureComputer
    computer = EnrichedFeatureComputer()
    logger.info("EnrichedFeatureComputer created")
    return computer
