"""Main-loop observations and freshness checks at the existing order boundary.

No worker refreshes these observations: a blocked main loop must close entry
permission even while a websocket callback remains active.
"""
from __future__ import annotations

from contextlib import closing
from decimal import Decimal
import math
import sqlite3
import time
from typing import Any, Callable

from shared.entry_risk import GateResult
from shared.runtime_status import HEARTBEAT_MAX_AGE, RECONCILIATION_MAX_AGE
from shared.runtime_state import _path


class RuntimeMonitor:
    def __init__(self, writer: Any, *, execution: bool, execution_runner_keys: list[str],
                 monotonic: Callable[[], float] = time.monotonic,
                 clock: Callable[[], float] = time.time) -> None:
        self.writer, self.execution = writer, execution
        self.execution_runner_keys = tuple(execution_runner_keys)
        self.monotonic = monotonic
        self.clock = clock
        self.coordinators: dict[str, Any] = {}
        self.modules: dict[str, Any] = {}
        self.portfolio = self.wallet = None
        self.phase = "starting"
        self._heartbeat_monotonic: float | None = None

    def bind(self, coordinators: dict[str, Any], modules: dict[str, Any], *,
             portfolio: Any = None, wallet: Any = None) -> None:
        self.coordinators, self.modules = coordinators, modules
        self.portfolio, self.wallet = portfolio, wallet

    def _execution_snapshot(self) -> dict:
        observations: dict[str, Any] = {"modules": {}, "reconciliations": {}}
        journals = {}
        for key in self.execution_runner_keys:
            coord, module = self.coordinators[key], self.modules[key]
            with coord._lock:
                consumer = getattr(coord, "_durable_fill_consumer", None)
                if consumer is None:
                    raise ValueError("Execution runner has no durable consumer")
                observations["modules"][key] = (
                    module._execution_ready is True and consumer.executor._ready is True
                )
                record = getattr(coord, "_execution_reconciliation", {})
                observations["reconciliations"][key] = {
                    "ready": record.get("ready") is True, "checked_at": record.get("checked_at"),
                }
                journal = consumer.executor.journal
                journals[(str(journal.path.absolute()), journal.scope)] = journal
        counts = {"unresolved": 0, "unknown": 0, "unconsumed": 0}
        for journal in journals.values():
            # Count the complete account, including unowned legacy records.
            # Query one shared journal once even when several runners use it.
            path = _path(journal.path)
            # Monitoring must never recreate a missing business database.
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as db:
                db.execute("BEGIN")
                counts["unresolved"] += db.execute(
                    "SELECT COUNT(*) FROM submissions WHERE scope=? AND settled=0", (journal.scope,)
                ).fetchone()[0]
                counts["unknown"] += db.execute(
                    "SELECT COUNT(*) FROM submissions WHERE scope=? AND settled=0 AND status='unknown'",
                    (journal.scope,),
                ).fetchone()[0]
                counts["unconsumed"] += db.execute(
                    "SELECT COUNT(*) FROM execution_fills WHERE scope=? AND delivered=0", (journal.scope,)
                ).fetchone()[0]
        # Capture before the query: slow snapshot collection must age existing
        # risk evidence rather than make it look fresh when publication ends.
        captured_at = self.clock()
        risk = self.portfolio.snapshot()
        wallet = self.wallet.snapshot()
        observations.update(
            risk={**{key: risk.get(key) for key in ("ready", "age_seconds")},
                  "captured_at": captured_at, "max_age_seconds": self.portfolio.max_age_seconds},
            wallet={**{key: wallet.get(key) for key in ("ready", "checked_at", "verified_through_ms")},
                    "max_age_seconds": self.wallet.max_age_seconds},
            journal=counts,
        )
        return observations

    def publish(self, phase: str) -> dict | None:
        # Withdraw entry permission before collecting potentially blocking state.
        self._heartbeat_monotonic = None
        self.phase = phase
        data, execution = {}, None
        if phase == "running":
            for key, coord in self.coordinators.items():
                with coord._lock:
                    readiness = coord._data_readiness
                    data[key] = {**readiness.snapshot(), "interval_ms": readiness.interval_ms}
            if self.execution:
                execution = self._execution_snapshot()
        result = self.writer.publish(phase, data=data, execution=execution) if self.writer is not None else None
        if phase == "running":
            self._heartbeat_monotonic = self.monotonic()
        return result

    def check(self) -> GateResult:
        now, heartbeat = self.monotonic(), self._heartbeat_monotonic
        if (self.phase != "running" or heartbeat is None or not math.isfinite(now)
                or not 0 <= now - heartbeat <= HEARTBEAT_MAX_AGE):
            return GateResult(False, "runtime heartbeat unavailable or stale")
        if not self.execution or not self.execution_runner_keys:
            return GateResult(False, "runtime execution observations unavailable")
        for key in self.execution_runner_keys:
            # Reconciliation publishes one immutable dict while holding the
            # coordinator lock. Do not take other coordinators' locks from an
            # order callback, which could deadlock simultaneous callbacks.
            coord, module = self.coordinators.get(key), self.modules.get(key)
            record = getattr(coord, "_execution_reconciliation", {})
            checked = record.get("monotonic_at")
            consumer = getattr(coord, "_durable_fill_consumer", None)
            if (record.get("ready") is not True or type(checked) not in (int, float)
                    or not math.isfinite(checked) or not 0 <= now - checked <= RECONCILIATION_MAX_AGE
                    or getattr(module, "_execution_ready", False) is not True
                    or getattr(getattr(consumer, "executor", None), "_ready", False) is not True):
                return GateResult(False, "runtime execution reconciliation unavailable or stale")
        return GateResult(True)


class RuntimeFreshnessGate:
    """Check freshness before any existing quota reservation can be persisted."""

    def __init__(self, monitor: RuntimeMonitor, risk_gate: Any) -> None:
        self.monitor, self.risk_gate = monitor, risk_gate

    def check(self, event: Any) -> GateResult:
        try:
            qty = Decimal(str(event.qty))
            if not qty.is_finite() or qty < 0:
                return GateResult(False, "invalid order quantity")
            reducing = qty == 0 or getattr(event, "reduce_only", False) is True
            if not reducing:
                freshness = self.monitor.check()
                if not freshness.allowed:
                    return freshness
            # Reductions still traverse the established validation/gate path.
            return self.risk_gate.check(event)
        except Exception as exc:
            return GateResult(False, "runtime gate failed: " + type(exc).__name__)
