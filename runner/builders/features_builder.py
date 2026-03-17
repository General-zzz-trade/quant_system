# runner/builders/features_builder.py
"""Phase 5: features and inference — feature hook, inference bridge, multi-TF ensemble.

Extracted from LiveRunner._build_features_and_inference().
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from monitoring.engine_hook import EngineMonitoringHook

logger = logging.getLogger(__name__)


def build_features_and_inference(
    config: Any,
    feature_computer: Any,
    alpha_models: Any,
    inference_bridges: Optional[Dict[str, Any]],
    unified_predictors: Optional[Dict[str, Any]],
    metrics_exporter: Any,
    hook: Optional[EngineMonitoringHook],
    report: Any,
    bear_model: Any,
    funding_rate_source: Any,
    oi_source: Any,
    ls_ratio_source: Any,
    spot_close_source: Any,
    fgi_source: Any,
    implied_vol_source: Any,
    put_call_ratio_source: Any,
    onchain_source: Any,
    liquidation_source: Any,
    mempool_source: Any,
    macro_source: Any,
    sentiment_source: Any,
) -> tuple:
    """Phase 5: feature hook, inference bridge, multi-TF ensemble, decision recording.

    Returns (feat_hook, inference_bridge, dominance_computer).
    dominance_computer is a DominanceComputer instance when config.enable_dominance_features
    is True, otherwise None.  Phase 3 will migrate this to Rust.
    """
    from engine.feature_hook import FeatureComputeHook
    from runner.builders.inference import _build_multi_tf_ensemble

    # V14 dominance computer (Python path; Phase 3 migrates to Rust)
    dominance_computer = None
    if getattr(config, "enable_dominance_features", False):
        from features.dominance_computer import DominanceComputer
        dominance_computer = DominanceComputer()
        logger.info("V14 dominance features enabled (Python path)")

    feat_hook = None
    inference_bridge = None
    if feature_computer is not None:
        if inference_bridges is not None:
            # Multi-symbol: per-symbol bridges already constructed
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

        # ── Multi-timeframe Ensemble (Direction 13) ──────────────
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

    # Wire inference_bridge to monitoring hook
    _first_bridge = None
    if inference_bridge is not None:
        if isinstance(inference_bridge, dict):
            _first_bridge = next(iter(inference_bridge.values()), None)
        else:
            _first_bridge = inference_bridge
    if hook is not None and _first_bridge is not None:
        hook.inference_bridge = _first_bridge

    # Decision recording (replay support)
    if config.enable_decision_recording and hook is not None:
        from decision.persistence.decision_store import DecisionStore
        hook.decision_store = DecisionStore(path=config.decision_recording_path)
        logger.info("Decision recording enabled: %s", config.decision_recording_path)

    return feat_hook, inference_bridge, dominance_computer
