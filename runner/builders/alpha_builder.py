"""Builder and balance helpers for alpha_main coordinator construction."""
from __future__ import annotations

import logging
from typing import Any

from _quant_hotpath import RustInferenceBridge

from decision.modules.alpha import AlphaDecisionModule
from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.feature_hook import FeatureComputeHook
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
from runner.strategy_config import SYMBOL_CONFIG

logger = logging.getLogger(__name__)


def get_initial_balance(adapter: Any) -> float:
    """Fetch USDT equity from Bybit adapter for tick processor initialization.

    Returns 0.0 on any failure (tick processor will be initialized with zero
    balance and updated from exchange on first bar).
    """
    try:
        snapshot = adapter.get_balances()
        bal = snapshot.get("USDT")
        if bal is not None:
            return float(bal.total)
    except Exception:
        logger.debug("Could not fetch initial balance for tick processor", exc_info=True)
    return 0.0


def build_coordinator(
    symbol: str,
    runner_key: str,
    model_info: dict,
    adapter: Any,
    dry_run: bool = False,
) -> tuple[EngineCoordinator, AlphaDecisionModule]:
    """Build a full coordinator pipeline for one runner.

    Returns (coordinator, alpha_module) so callers can wire consensus
    and warmup independently.
    """
    cfg = SYMBOL_CONFIG.get(runner_key, {})
    is_4h = "4h" in runner_key

    # Feature engine (per-symbol Rust instance created lazily by hook)
    feature_hook = FeatureComputeHook(
        computer=None,
        warmup_bars=cfg.get("warmup", 300 if is_4h else 800),
    )

    # Inference bridge for z-score normalization + constraints
    bridge = RustInferenceBridge(
        model_info["zscore_window"],
        model_info["zscore_warmup"],
    )

    # Signal pipeline components
    predictor = EnsemblePredictor(
        model_info["horizon_models"],
        model_info["config"],
    )
    discretizer = SignalDiscretizer(
        bridge,
        symbol=symbol,
        deadzone=model_info["deadzone"],
        min_hold=model_info["min_hold"],
        max_hold=model_info["max_hold"],
        long_only=model_info.get("long_only", False),
    )
    sizer = AdaptivePositionSizer(
        runner_key=runner_key,
        step_size=cfg.get("step", 0.001),
        min_size=cfg.get("size", 0.001),
        max_qty=cfg.get("max_qty", 0),
    )

    # Decision module
    alpha_module = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )

    # RustTickProcessor disabled until fill-qty-0 warmup bug is resolved.
    # The Python pipeline (feature_hook -> pipeline -> decision_bridge) is production-tested.
    tick_proc = None

    # Coordinator config
    coordinator_cfg = CoordinatorConfig(
        symbol_default=symbol,
        symbols=(symbol,),
        currency="USDT",
        feature_hook=feature_hook,
        tick_processor=tick_proc,
    )

    # Assemble coordinator
    coordinator = EngineCoordinator(cfg=coordinator_cfg)

    # Attach decision bridge
    decision_bridge = DecisionBridge(
        dispatcher_emit=coordinator.emit,
        modules=[alpha_module],
    )
    coordinator.attach_decision_bridge(decision_bridge)

    # Attach execution bridge (live only)
    if not dry_run:
        exec_adapter = BybitExecutionAdapter(adapter)
        execution_bridge = ExecutionBridge(
            adapter=exec_adapter,
            dispatcher_emit=coordinator.emit,
        )
        coordinator.attach_execution_bridge(execution_bridge)

    fast_path = "RustTickProcessor ENABLED" if tick_proc is not None else "Python pipeline (tick processor unavailable)"
    logger.info(
        "Built coordinator: runner_key=%s symbol=%s dry_run=%s warmup=%d path=%s",
        runner_key, symbol, dry_run,
        cfg.get("warmup", 300 if is_4h else 800),
        fast_path,
    )
    return coordinator, alpha_module
