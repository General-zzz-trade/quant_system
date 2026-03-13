# runner/builders/recovery_builder.py
"""Recovery subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
    restore_feature_hook_state,
    restore_inference_bridge_state,
    restore_kill_switch_state,
    restore_exit_manager_state,
    restore_regime_gate_state,
    restore_correlation_state,
    restore_timeout_tracker_state,
    save_all_auxiliary_state,
)

logger = logging.getLogger(__name__)


@dataclass
class RecoverySubsystem:
    """Assembled recovery components."""
    state_store: Optional[Any] = None
    event_log: Optional[Any] = None
    periodic_checkpointer: Optional[PeriodicCheckpointer] = None
    event_recorder: Optional[EventRecorder] = None


def build_recovery_subsystem(
    config: Any,
    *,
    coordinator: Any,
    kill_switch: Any,
    inference_bridge: Any = None,
    feature_hook: Any = None,
    exit_manager: Any = None,
    regime_gate: Any = None,
    correlation_computer: Any = None,
    timeout_tracker: Any = None,
) -> RecoverySubsystem:
    """Build persistent stores, restore state, set up checkpointing."""
    state_store = None
    event_log = None

    if not config.enable_persistent_stores:
        return RecoverySubsystem()

    from execution.store.ack_store import SQLiteAckStore
    from execution.store.event_log import SQLiteEventLog
    from state.store import SqliteStateStore

    data_dir = config.data_dir
    SQLiteAckStore(path=os.path.join(data_dir, "ack_store.db"))
    event_log = SQLiteEventLog(path=os.path.join(data_dir, "event_log.db"))
    state_store = SqliteStateStore(path=os.path.join(data_dir, "state.db"))

    # Restore from latest checkpoint
    for sym in config.symbols:
        checkpoint = state_store.latest(sym)
        if checkpoint is not None:
            coordinator.restore_from_snapshot(checkpoint.snapshot)
            logger.info("Restored state for %s from bar_index=%d", sym, checkpoint.bar_index)
            break

    # Restore kill switch
    restored = restore_kill_switch_state(kill_switch, data_dir=data_dir)
    if restored:
        logger.warning("Restored %d kill switch record(s)", restored)

    # Restore inference bridge
    if inference_bridge is not None:
        if restore_inference_bridge_state(inference_bridge, data_dir=data_dir):
            logger.info("Inference bridge state restored")

    # Restore feature hook
    if feature_hook is not None:
        if restore_feature_hook_state(feature_hook, data_dir=data_dir):
            logger.info("Feature hook state restored")

    # Restore exit manager
    if exit_manager is not None:
        if restore_exit_manager_state(exit_manager, data_dir=data_dir):
            logger.info("Exit manager state restored")

    # Restore regime gate
    if regime_gate is not None:
        if restore_regime_gate_state(regime_gate, data_dir=data_dir):
            logger.info("Regime gate state restored")

    # Restore correlation computer
    if correlation_computer is not None:
        if restore_correlation_state(correlation_computer, data_dir=data_dir):
            logger.info("Correlation state restored")

    # Restore timeout tracker
    if timeout_tracker is not None:
        if restore_timeout_tracker_state(timeout_tracker, data_dir=data_dir):
            logger.info("Timeout tracker state restored")

    # Periodic checkpointer
    periodic_checkpointer = None
    if state_store is not None:
        def _get_snapshot() -> Any:
            return coordinator.get_state_view().get("last_snapshot")

        periodic_checkpointer = PeriodicCheckpointer(
            state_store=state_store,
            get_snapshot=_get_snapshot,
            interval_sec=config.reconcile_interval_sec,
        )

    # Event recorder
    event_recorder = None
    if event_log is not None:
        event_recorder = EventRecorder(event_log)

    return RecoverySubsystem(
        state_store=state_store,
        event_log=event_log,
        periodic_checkpointer=periodic_checkpointer,
        event_recorder=event_recorder,
    )
