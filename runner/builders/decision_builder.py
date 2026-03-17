# runner/builders/decision_builder.py
"""Phase 8: decision bridge and engine loop — regime wrapping, module reloader.

Extracted from LiveRunner._build_decision().
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from decision.regime_bridge import RegimeAwareDecisionModule
from decision.regime_policy import RegimePolicy
from engine.coordinator import EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.guards import build_basic_guard, GuardConfig
from engine.loop import EngineLoop, LoopConfig

logger = logging.getLogger(__name__)


def build_decision(
    config: Any,
    decision_modules: Sequence[Any] | None,
    _emit: Any,
    coordinator: EngineCoordinator,
) -> tuple:
    """Phase 8: decision bridge with regime wrapping, module reloader, engine loop.

    Returns (decision_bridge_inst, module_reloader, loop).
    """
    # ── 5) Decision bridge ────────────────────────────────
    modules = list(decision_modules or [])

    if config.enable_regime_gate and modules:
        composite_syms = getattr(config, "composite_regime_symbols", ())
        gated_modules = []
        for mod in modules:
            gated = RegimeAwareDecisionModule(
                inner=mod,
                policy=RegimePolicy(),
                composite_regime_symbols=tuple(composite_syms),
            )
            # Enable param routing when regime sizing is active
            if getattr(config, "enable_regime_sizing", False):
                gated.enable_param_routing = True
            gated_modules.append(gated)
        modules = gated_modules

    decision_bridge_inst = None
    if modules:
        decision_bridge_inst = DecisionBridge(
            dispatcher_emit=_emit, modules=modules,
        )
        coordinator.attach_decision_bridge(decision_bridge_inst)

    # ── 5a) ModuleReloader ───────────────────────────────
    from engine.module_reloader import ModuleReloader, ReloaderConfig
    module_reloader = ModuleReloader(
        config=ReloaderConfig(),
        on_reload=lambda trigger: logger.info("Module reload triggered: %s", trigger),
    )

    # ── 6) EngineLoop with guard ─────────────────────────
    guard = build_basic_guard(GuardConfig())
    loop = EngineLoop(coordinator=coordinator, guard=guard, cfg=LoopConfig())

    return decision_bridge_inst, module_reloader, loop
