# runner/builders/rust_components_builder.py
"""Phase 1.5: Rust hot-path component initialization.

Creates the 9 stateful Rust components used by AlphaRunner:
- RustFeatureEngine, RustInferenceBridge, RustRiskEvaluator
- RustKillSwitch, RustOrderStateMachine, RustCircuitBreaker
- RustStateStore, RustUnifiedPredictor, RustTickProcessor

Only initialized when config.enable_rust_components is True.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional

_log = logging.getLogger(__name__)


class RustComponents(NamedTuple):
    feature_engine: Any  # RustFeatureEngine or None
    inference_bridges: Optional[Dict[str, Any]]  # symbol → RustInferenceBridge
    risk_evaluator: Any  # RustRiskEvaluator or None
    kill_switch_rust: Any  # RustKillSwitch or None
    order_state_machine: Any  # RustOrderStateMachine or None
    circuit_breaker: Any  # RustCircuitBreaker or None
    state_store: Any  # RustStateStore or None


def build_rust_components(config: Any, symbols: tuple) -> RustComponents:
    """Build Rust hot-path components if enabled.

    Args:
        config: LiveRunnerConfig (or SimpleNamespace in tests).
        symbols: Tuple of symbol strings to create per-symbol inference bridges for.

    Returns:
        RustComponents named tuple; all fields are None when disabled.
    """
    if not config.enable_rust_components:
        _log.info("Rust components disabled")
        return RustComponents(
            feature_engine=None,
            inference_bridges=None,
            risk_evaluator=None,
            kill_switch_rust=None,
            order_state_machine=None,
            circuit_breaker=None,
            state_store=None,
        )

    from _quant_hotpath import (
        RustFeatureEngine,
        RustInferenceBridge,
        RustRiskEvaluator,
        RustKillSwitch,
        RustOrderStateMachine,
        RustCircuitBreaker,
        RustStateStore,
    )

    _log.info("Initializing %d Rust components for %d symbols", 7, len(symbols))

    _SCALE = 100_000_000  # Fd8 fixed-point scale (10^8)

    feature_engine = RustFeatureEngine()
    risk_evaluator = RustRiskEvaluator()
    kill_switch_rust = RustKillSwitch()
    order_state_machine = RustOrderStateMachine()
    circuit_breaker = RustCircuitBreaker(
        failure_threshold=3, window_s=120, recovery_timeout_s=60,
    )

    # RustStateStore requires symbols list, currency, and initial balance (Fd8 i64)
    currency = getattr(config, "currency", "USDT")
    initial_equity = getattr(config, "initial_equity", 500.0)
    state_store = RustStateStore(list(symbols), currency, int(initial_equity * _SCALE))

    # Per-symbol inference bridges
    zscore_window = getattr(config, "zscore_window", 720)
    zscore_warmup = getattr(config, "zscore_warmup", 180)
    inference_bridges: Dict[str, Any] = {}
    for sym in symbols:
        inference_bridges[sym] = RustInferenceBridge(zscore_window, zscore_warmup)

    return RustComponents(
        feature_engine=feature_engine,
        inference_bridges=inference_bridges,
        risk_evaluator=risk_evaluator,
        kill_switch_rust=kill_switch_rust,
        order_state_machine=order_state_machine,
        circuit_breaker=circuit_breaker,
        state_store=state_store,
    )
