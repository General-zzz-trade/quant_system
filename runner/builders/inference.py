# runner/builders/inference.py
"""Inference subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


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
            from runner.live_runner import _build_multi_tf_ensemble
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
