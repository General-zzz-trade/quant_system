# runner/builders/persistence.py
"""Phase 11: persistence, recovery, reconcile, margin, alerts, data scheduler.

Extracted from LiveRunner._build_persistence_and_recovery().
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from engine.coordinator import EngineCoordinator
from execution.latency.tracker import LatencyTracker
from monitoring.alerts.manager import AlertManager
from monitoring.health import SystemHealthMonitor
from risk.kill_switch import KillSwitch
from risk.margin_monitor import MarginConfig, MarginMonitor
from runner.builders.monitoring import _build_alert_rules
from runner.recovery import (
    reconcile_and_heal,
    restore_all_auxiliary_state,
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


def _reconcile_startup(local_view: Any, venue_state: Dict, symbols: List[str]) -> List[str]:
    """Detect startup mismatches between local snapshot and venue state."""
    mismatches = []
    for sym in symbols:
        local_positions = local_view.get("positions", {})
        venue_positions = venue_state.get("positions", {})
        local_qty = getattr(local_positions.get(sym), "qty", 0)
        venue_qty = venue_positions.get(sym, {}).get("qty", 0)
        if str(local_qty) != str(venue_qty):
            mismatches.append(f"{sym} position: local={local_qty} venue={venue_qty}")
    return mismatches


def build_persistence_and_recovery(
    config: Any,
    coordinator: EngineCoordinator,
    kill_switch: KillSwitch,
    inference_bridge: Any,
    feat_hook: Any,
    correlation_computer: Any,
    timeout_tracker: Any,
    decision_bridge_inst: Any,
    fetch_venue_state: Optional[Callable],
    fetch_margin: Optional[Callable],
    alert_sink: Any,
    health: Optional[SystemHealthMonitor],
    latency_tracker: LatencyTracker,
    alpha_health_monitor: Any,
    report: Any,
) -> tuple:
    """Phase 11: reconcile, margin, alerts, persistent stores, recovery, data scheduler."""
    # ── ReconcileScheduler ──
    reconcile_scheduler = None
    if config.enable_reconcile and fetch_venue_state is not None:
        from execution.reconcile.controller import ReconcileController
        from execution.reconcile.scheduler import ReconcileScheduler, ReconcileSchedulerConfig

        reconcile_scheduler = ReconcileScheduler(
            controller=ReconcileController(),
            get_local_state=lambda: coordinator.get_state_view(),
            fetch_venue_state=fetch_venue_state,
            cfg=ReconcileSchedulerConfig(
                interval_sec=config.reconcile_interval_sec,
                venue=config.venue,
            ),
            on_halt=lambda report: coordinator.stop(),
        )

    # ── MarginMonitor ──
    margin_monitor = None
    if fetch_margin is not None:
        margin_monitor = MarginMonitor(
            config=MarginConfig(
                check_interval_sec=config.margin_check_interval_sec,
                warning_margin_ratio=config.margin_warning_ratio,
                critical_margin_ratio=config.margin_critical_ratio,
            ),
            fetch_margin=fetch_margin,
            kill_switch=kill_switch,
            alert_sink=alert_sink,
        )

    # ── AlertManager ──
    alert_manager = AlertManager(sink=alert_sink)
    _build_alert_rules(
        alert_manager, health, kill_switch, latency_tracker,
        alpha_health_monitor, correlation_computer, config, report,
    )

    # ── Persistent stores ──
    state_store = None
    event_log = None
    ack_store = None
    if config.enable_persistent_stores:
        from execution.store.ack_store import SQLiteAckStore
        from execution.store.event_log import SQLiteEventLog
        from state.store import SqliteStateStore

        data_dir = config.data_dir
        ack_store = SQLiteAckStore(path=os.path.join(data_dir, "ack_store.db"))
        logger.info("Persistent ack_store initialized at %s/ack_store.db", data_dir)
        event_log = SQLiteEventLog(path=os.path.join(data_dir, "event_log.db"))
        state_store = SqliteStateStore(path=os.path.join(data_dir, "state.db"))

        # State restoration
        for sym in config.symbols:
            checkpoint = state_store.latest(sym)
            if checkpoint is not None:
                coordinator.restore_from_snapshot(checkpoint.snapshot)
                logger.info("Restored state for %s from bar_index=%d", sym, checkpoint.bar_index)
                break

        # Auxiliary state recovery
        _exit_mgr = _find_module_attr(decision_bridge_inst, '_exit_mgr')
        _regime_gate = _find_module_attr(decision_bridge_inst, '_regime_gate')
        restore_results = restore_all_auxiliary_state(
            kill_switch=kill_switch,
            inference_bridge=inference_bridge,
            feature_hook=feat_hook,
            exit_manager=_exit_mgr,
            regime_gate=_regime_gate,
            correlation_computer=correlation_computer,
            timeout_tracker=timeout_tracker,
            data_dir=data_dir,
        )
        for comp_name, restored in restore_results.items():
            if restored:
                logger.info("Recovery: %s state restored from checkpoint", comp_name)

    # ── Startup reconciliation ──
    if config.reconcile_on_startup and fetch_venue_state is not None:
        try:
            venue_state = fetch_venue_state()
            local_view = coordinator.get_state_view()
            mismatches = _reconcile_startup(local_view, venue_state, config.symbols)
            for m in mismatches:
                logger.warning("Startup reconciliation mismatch: %s", m)
            if mismatches:
                actions = reconcile_and_heal(coordinator, venue_state, config.symbols)
                for a in actions:
                    logger.warning("Reconciliation action: %s", a)
        except Exception:
            logger.exception("Startup reconciliation failed — proceeding with local state")

    # ── DataScheduler + FreshnessMonitor ──
    data_scheduler = None
    freshness_monitor = None
    if config.enable_data_scheduler:
        try:
            from data.scheduler.data_scheduler import DataScheduler, DataSchedulerConfig
            from data.scheduler.freshness_monitor import FreshnessMonitor, FreshnessConfig

            data_scheduler = DataScheduler(DataSchedulerConfig(symbols=config.symbols))
            freshness_monitor = FreshnessMonitor(FreshnessConfig(
                data_dir=config.data_files_dir,
                symbols=config.symbols,
                on_alert=lambda a: logger.warning("Data stale: %s age=%.1fh", a.source, a.age_hours),
            ))
            logger.info("DataScheduler + FreshnessMonitor configured")
        except Exception as e:
            report.record("data_scheduler", False, str(e))
            logger.warning("DataScheduler setup failed — continuing without", exc_info=True)

    return (
        reconcile_scheduler, margin_monitor, alert_manager,
        state_store, event_log, data_scheduler, freshness_monitor,
        ack_store,
    )
