# runner/builders/order_infra_builder.py
"""Phase 4: order infrastructure — state machine, timeout tracker, model registry.

Extracted from LiveRunner._build_order_infra().
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_order_infra(config: Any, alpha_models: Any) -> tuple:
    """Phase 4: order state machine, timeout tracker, model registry.

    Returns (order_state_machine, timeout_tracker, model_loader_inst, alpha_models).
    """
    # ── OrderStateMachine (order lifecycle tracking) ────
    from execution.state_machine.machine import OrderStateMachine
    order_state_machine = OrderStateMachine()

    # ── TimeoutTracker (stale order detection) ──────────
    from execution.safety.timeout_tracker import OrderTimeoutTracker
    timeout_tracker = OrderTimeoutTracker(
        timeout_sec=config.pending_order_timeout_sec,
    )

    # ── ModelRegistry auto-loading (Phase 1) ──────────
    model_loader_inst = None
    if config.model_registry_db and config.model_names:
        from research.model_registry.registry import ModelRegistry
        from research.model_registry.artifact import ArtifactStore
        from alpha.model_loader import ProductionModelLoader

        registry = ModelRegistry(config.model_registry_db)
        artifact_store = ArtifactStore(config.artifact_store_root or "artifacts")
        model_loader_inst = ProductionModelLoader(registry, artifact_store)
        loaded = model_loader_inst.load_production_models(config.model_names)
        if loaded:
            alpha_models = list(alpha_models or []) + loaded
            logger.info("Auto-loaded %d production model(s) from registry", len(loaded))

    return order_state_machine, timeout_tracker, model_loader_inst, alpha_models
