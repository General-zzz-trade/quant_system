# runner/builders/inference.py
"""Inference subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

if TYPE_CHECKING:
    from runner.config import LiveRunnerConfig

logger = logging.getLogger(__name__)


def _build_multi_tf_ensemble(
    config: "LiveRunnerConfig",
    inference_bridge: Any,
    metrics_exporter: Any,
    report: Any,
) -> Any:
    """Build multi-timeframe ensemble bridges (Direction 13).

    Returns potentially modified inference_bridge (dict of per-symbol bridges).
    """
    try:
        from decision.ensemble_combiner import EnsembleCombiner
        from alpha.inference.bridge import LiveInferenceBridge as _LIB
        import json as _json

        multi_tf = config.multi_tf_models or {}
        combiners: Dict[str, Any] = {}

        for sym, tf_names in multi_tf.items():
            bridges_for_sym = []
            for tf_name in tf_names:
                model_dir = Path(f"models_v8/{sym}_{tf_name}")
                if not model_dir.exists():
                    logger.warning("Multi-TF model dir not found: %s", model_dir)
                    continue

                cfg_path = model_dir / "config.json"
                if not cfg_path.exists():
                    logger.warning("Multi-TF config not found: %s", cfg_path)
                    continue

                with open(cfg_path) as f:
                    model_cfg = _json.load(f)

                from alpha.models.lgbm_alpha import LGBMAlphaModel
                tf_models = []
                for pkl in sorted(model_dir.glob("*.pkl")):
                    try:
                        m = LGBMAlphaModel(name=f"{sym}_{tf_name}_{pkl.stem}")
                        m.load(pkl)
                        tf_models.append(m)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", pkl, e)
                if not tf_models:
                    logger.warning("No models loaded from %s", model_dir)
                    continue

                tf_min_hold = {sym: model_cfg.get("min_hold", 12)}
                tf_deadzone: Any = {sym: model_cfg.get("deadzone", 0.5)}
                tf_long_only = {sym} if model_cfg.get("long_only") else set()
                tf_bridge = _LIB(
                    models=tf_models,
                    metrics_exporter=metrics_exporter,
                    min_hold_bars=tf_min_hold,
                    deadzone=tf_deadzone,
                    max_hold=model_cfg.get("max_hold", 120),
                    long_only_symbols=tf_long_only,
                    ensemble_weights=model_cfg.get("ensemble_weights"),
                    monthly_gate=config.monthly_gate,
                    monthly_gate_window=config.monthly_gate_window,
                    vol_target=config.vol_target,
                    vol_feature=config.vol_feature,
                )
                bridges_for_sym.append((tf_name, tf_bridge))
                logger.info(
                    "Multi-TF bridge: %s/%s (min_hold=%d, deadzone=%.1f, max_hold=%d)",
                    sym, tf_name,
                    model_cfg.get("min_hold", 12),
                    model_cfg.get("deadzone", 0.5),
                    model_cfg.get("max_hold", 120),
                )

            if len(bridges_for_sym) > 1:
                combiner = EnsembleCombiner(
                    bridges=bridges_for_sym,
                    conflict_policy=config.ensemble_conflict_policy,
                )
                combiners[sym] = combiner
                logger.info(
                    "Ensemble combiner for %s: %d bridges (%s)",
                    sym, len(bridges_for_sym),
                    [n for n, _ in bridges_for_sym],
                )
            elif len(bridges_for_sym) == 1:
                combiners[sym] = bridges_for_sym[0][1]

        if combiners:
            if isinstance(inference_bridge, dict):
                inference_bridge.update(combiners)
            else:
                new_bridge: Dict[str, Any] = {}
                for sym in config.symbols:
                    if sym in combiners:
                        new_bridge[sym] = combiners[sym]
                    else:
                        new_bridge[sym] = inference_bridge
                inference_bridge = new_bridge
            logger.info("Multi-TF ensemble enabled for %d symbols", len(combiners))

        report.record("multi_tf_ensemble", True)
    except Exception as e:
        report.record("multi_tf_ensemble", False, str(e))
        logger.warning("Multi-TF ensemble setup failed - using single bridge", exc_info=True)

    return inference_bridge


@dataclass
class InferenceSubsystem:
    """Assembled inference subsystem components."""
    feature_hook: Optional[Any] = None
    inference_bridge: Optional[Any] = None
    model_loader: Optional[Any] = None


def build_inference_subsystem(
    config: Any,
    *,
    feature_computer: Any = None,
    alpha_models: Optional[Sequence[Any]] = None,
    inference_bridges: Optional[Dict[str, Any]] = None,
    unified_predictors: Optional[Dict[str, Any]] = None,
    tick_processors: Optional[Dict[str, Any]] = None,
    metrics_exporter: Any = None,
    bear_model: Any = None,
    funding_rate_source: Any = None,
    oi_source: Any = None,
    ls_ratio_source: Any = None,
    spot_close_source: Any = None,
    fgi_source: Any = None,
    implied_vol_source: Any = None,
    put_call_ratio_source: Any = None,
    onchain_source: Any = None,
    liquidation_source: Any = None,
    mempool_source: Any = None,
    macro_source: Any = None,
    sentiment_source: Any = None,
    report: Any = None,
) -> InferenceSubsystem:
    """Build feature computation and ML inference components."""
    from engine.feature_hook import FeatureComputeHook

    model_loader = None
    if config.model_registry_db and config.model_names:
        from research.model_registry.registry import ModelRegistry
        from research.model_registry.artifact import ArtifactStore
        from alpha.model_loader import ProductionModelLoader

        registry = ModelRegistry(config.model_registry_db)
        artifact_store = ArtifactStore(config.artifact_store_root or "artifacts")
        model_loader = ProductionModelLoader(registry, artifact_store)
        loaded = model_loader.load_production_models(config.model_names)
        if loaded:
            alpha_models = list(alpha_models or []) + loaded
            logger.info("Auto-loaded %d production model(s) from registry", len(loaded))

    inference_bridge = None
    feat_hook = None

    if feature_computer is not None:
        if inference_bridges is not None:
            inference_bridge = inference_bridges
        elif alpha_models:
            from alpha.inference.bridge import LiveInferenceBridge
            inference_bridge = LiveInferenceBridge(
                models=list(alpha_models),
                metrics_exporter=metrics_exporter,
                min_hold_bars=config.min_hold_bars,
                long_only_symbols=config.long_only_symbols,
                deadzone=config.deadzone,
                trend_follow=config.trend_follow,
                trend_indicator=config.trend_indicator,
                trend_threshold=config.trend_threshold,
                max_hold=config.max_hold,
                monthly_gate=config.monthly_gate,
                monthly_gate_window=config.monthly_gate_window,
                bear_model=bear_model,
                bear_thresholds=config.bear_thresholds,
                vol_target=config.vol_target,
                vol_feature=config.vol_feature,
                ensemble_weights=config.ensemble_weights,
            )

        # Multi-TF ensemble
        if config.enable_multi_tf_ensemble and inference_bridge is not None:
            inference_bridge = _build_multi_tf_ensemble(
                config, inference_bridge, metrics_exporter, report,
            )

        feat_hook = FeatureComputeHook(
            computer=feature_computer,
            inference_bridge=inference_bridge if unified_predictors is None else None,
            unified_predictor=unified_predictors,
            funding_rate_source=funding_rate_source,
            oi_source=oi_source,
            ls_ratio_source=ls_ratio_source,
            spot_close_source=spot_close_source,
            fgi_source=fgi_source,
            implied_vol_source=implied_vol_source,
            put_call_ratio_source=put_call_ratio_source,
            onchain_source=onchain_source,
            liquidation_source=liquidation_source,
            mempool_source=mempool_source,
            macro_source=macro_source,
            sentiment_source=sentiment_source,
        )

    return InferenceSubsystem(
        feature_hook=feat_hook,
        inference_bridge=inference_bridge,
        model_loader=model_loader,
    )
