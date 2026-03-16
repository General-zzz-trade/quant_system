# runner/builders/shutdown.py
"""Phase 12: graceful shutdown, health server, periodic checkpointer, event recorder.

Extracted from LiveRunner._build_shutdown().
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from engine.coordinator import EngineCoordinator
from monitoring.health import SystemHealthMonitor
from risk.kill_switch import KillMode, KillScope, KillSwitch
from runner.builders.monitoring import _build_health_server
from runner.graceful_shutdown import GracefulShutdown, ShutdownConfig
from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
    save_all_auxiliary_state,
)

logger = logging.getLogger(__name__)


def _find_module_attr(decision_bridge: Any, attr: str) -> Any:
    if decision_bridge is None:
        return None
    for mod in getattr(decision_bridge, 'modules', []):
        val = getattr(mod, attr, None)
        if val is not None:
            return val
        inner = getattr(mod, 'inner', None)
        if inner is not None:
            val = getattr(inner, attr, None)
            if val is not None:
                return val
    return None


def build_shutdown(
    config: Any,
    state_store: Any,
    coordinator: EngineCoordinator,
    kill_switch: KillSwitch,
    inference_bridge: Any,
    feat_hook: Any,
    decision_bridge_inst: Any,
    correlation_computer: Any,
    timeout_tracker: Any,
    reconcile_scheduler: Any,
    venue_client: Any,
    health: Optional[SystemHealthMonitor],
    alpha_health_monitor: Any,
    regime_sizer: Any,
    portfolio_allocator: Any,
    live_signal_tracker: Any,
    event_log: Any,
    report: Any,
) -> tuple:
    """Phase 12: graceful shutdown, health server, periodic checkpointer, event recorder."""
    # ── GracefulShutdown ──
    shutdown_cfg = ShutdownConfig(pending_order_timeout_sec=config.pending_order_timeout_sec)
    save_snapshot_fn = None
    if state_store is not None:
        def save_snapshot_fn(_path: str) -> None:
            data_dir_s = config.data_dir
            snapshot = coordinator.get_state_view().get("last_snapshot")
            if snapshot is not None:
                state_store.save(snapshot)
                logger.info("State snapshot saved on shutdown")
            save_all_auxiliary_state(
                kill_switch=kill_switch,
                inference_bridge=inference_bridge,
                feature_hook=feat_hook,
                exit_manager=_find_module_attr(decision_bridge_inst, '_exit_mgr'),
                regime_gate=_find_module_attr(decision_bridge_inst, '_regime_gate'),
                correlation_computer=correlation_computer,
                timeout_tracker=timeout_tracker,
                data_dir=data_dir_s,
            )

    def _wait_pending() -> bool:
        return timeout_tracker.pending_count == 0

    def _cancel_all() -> None:
        if hasattr(venue_client, "cancel_all_orders"):
            for sym in config.symbols:
                try:
                    venue_client.cancel_all_orders(sym)
                except Exception:
                    logger.warning("cancel_all_orders failed for %s", sym, exc_info=True)

    def _reconcile_once() -> None:
        if reconcile_scheduler is not None:
            try:
                reconcile_scheduler.run_once()
            except Exception:
                logger.warning("Shutdown reconciliation failed", exc_info=True)

    _runner_ref: List[Any] = []

    def _cleanup() -> None:
        if _runner_ref:
            _runner_ref[0]._running = False

    shutdown_handler = GracefulShutdown(
        config=shutdown_cfg,
        stop_new_orders=lambda: kill_switch.trigger(
            scope=KillScope.GLOBAL, key="*",
            mode=KillMode.HARD_KILL, reason="graceful_shutdown", source="shutdown",
        ),
        wait_pending=_wait_pending,
        cancel_all=_cancel_all,
        reconcile=_reconcile_once,
        save_snapshot=save_snapshot_fn,
        cleanup=_cleanup,
    )

    # ── Health HTTP endpoint ──
    health_server, control_plane = _build_health_server(
        config, health, alpha_health_monitor, regime_sizer,
        portfolio_allocator, live_signal_tracker, report,
    )

    # ── Periodic checkpointer ──
    periodic_checkpointer = None
    if state_store is not None:
        def _get_snapshot() -> Any:
            return coordinator.get_state_view().get("last_snapshot")

        periodic_checkpointer = PeriodicCheckpointer(
            state_store=state_store,
            get_snapshot=_get_snapshot,
            interval_sec=config.reconcile_interval_sec,
        )

    # ── Event recorder ──
    event_recorder = None
    if event_log is not None:
        event_recorder = EventRecorder(event_log)

    return (
        shutdown_handler, health_server, control_plane,
        periodic_checkpointer, event_recorder, _runner_ref,
    )
