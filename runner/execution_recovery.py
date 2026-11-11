"""Startup/runtime reconciliation for durable execution, with no order resubmission."""
from __future__ import annotations

import time
from typing import Any


def reconcile_execution(coordinator: Any, module: Any, *, restore: bool = False) -> list[str]:
    consumer = getattr(coordinator, "_durable_fill_consumer", None)
    if consumer is None:
        return []
    with coordinator._lock:
        # Timestamp the start, so a slow account query cannot make old evidence
        # look recent merely by finishing. Failed attempts never refresh a
        # successful reconciliation observation.
        observation = {"ready": False, "checked_at": time.time(), "monotonic_at": time.monotonic()}
        coordinator._execution_reconciliation = observation
        issues = _reconcile_locked(coordinator, module, consumer, restore=restore)
        coordinator._execution_reconciliation = {**observation, "ready": not issues}
        return issues


def _reconcile_locked(coordinator: Any, module: Any, consumer: Any, *, restore: bool) -> list[str]:
    module._execution_ready = False
    execution = consumer.executor
    execution._ready = False
    if restore:
        consumer.restore()
        orders = execution.journal.orders(execution.owner, unresolved=True)
        module._pending_orders = {r["intent_id"] for r in orders}
        module._pending_reductions = {r["intent_id"] for r in orders if r["reducing"]}
    for fill in execution.reconcile_pending():
        coordinator.emit(fill, actor="execution_reconcile")
    execution.notify_status()
    issues = execution.reconcile_account(module._symbol)
    module._execution_ready = not issues
    return issues
