# runner/builders/recovery_builder.py
"""Recovery subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
    restore_all_auxiliary_state,
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

    # Restore all auxiliary state atomically (bundle-first with individual fallback)
    restore_results = restore_all_auxiliary_state(
        kill_switch=kill_switch,
        inference_bridge=inference_bridge,
        feature_hook=feature_hook,
        exit_manager=exit_manager,
        regime_gate=regime_gate,
        correlation_computer=correlation_computer,
        timeout_tracker=timeout_tracker,
        data_dir=data_dir,
    )
    for component, was_restored in restore_results.items():
        if was_restored:
            logger.info("Restored %s state from checkpoint", component)

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
