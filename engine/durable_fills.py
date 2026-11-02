"""Crash-safe OKX fill consumption for one strategy/coordinator.

State mutation is rolled back on transaction failure. On process restart the
last committed state is restored before any unacknowledged fills are replayed.
Observers run after commit and are deliberately not part of the trading ledger.
"""
from __future__ import annotations

import logging
import json
from types import SimpleNamespace
from typing import Any

from engine.pipeline import PipelineInput
from shared.strategy_position import StrategyPosition
from state.store import _serialize_snapshot, _deserialize_snapshot

logger = logging.getLogger(__name__)


class DurableFillConsumer:
    def __init__(self, coordinator: Any, module: Any, executor: Any) -> None:
        self.coord, self.module, self.executor = coordinator, module, executor
        self.journal, self.owner = executor.journal, executor.owner
        self.position = StrategyPosition()
        self.restored = False
        self.revision = None
        self.observers: list[Any] = []
        self.restore_observers: list[Any] = []

    def restore(self) -> None:
        with self.coord._lock:
            saved = self.journal.checkpoint(self.owner)
            if saved:
                snap = _deserialize_snapshot(saved["engine"])
                # Preserve freshly warmed market/features; restore financial facts.
                current = self.coord._store.export_state()
                self.coord._store.load_exported(
                    current["markets"], dict(snap.positions), snap.account,
                    event_index=max(current["event_index"], snap.bar_index),
                    last_event_id=snap.event_id, last_ts=current["last_ts"],
                )
                self.position = StrategyPosition.from_dict(saved["position"])
                self.revision = saved["revision"]
                if saved.get("strategy"):
                    self.module.restore_execution_state(saved["strategy"], rebase=True)
            self.module.sync_confirmed_position(self.position)
            self.restored = True
            for observer in self.restore_observers:
                try:
                    observer(self.module._symbol, self.position)
                except Exception:
                    logger.exception("Position restored; auxiliary observer failed")

    def checkpoint_market(self) -> None:
        """Keep trailing stops/hold counters current between fills, under the engine lock."""
        if not self.restored or self.revision is None or self.coord._last_snapshot is None:
            return
        try:
            with self.journal.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT checkpoint FROM execution_consumers WHERE scope=? AND owner=?",
                                 (self.journal.scope, self.owner)).fetchone()
                if row is None or json.loads(row[0])["revision"] != self.revision:
                    raise RuntimeError("Consumer checkpoint changed in another process")
                data = {"engine": _serialize_snapshot(self.coord._last_snapshot),
                        "position": self.position.to_dict(), "revision": self.revision,
                        "strategy": self.module.execution_state()}
                db.execute("UPDATE execution_consumers SET checkpoint=? WHERE scope=? AND owner=?",
                           (json.dumps(data, allow_nan=False), self.journal.scope, self.owner))
        except Exception:
            self.module._execution_ready = False
            self.executor._ready = False
            raise

    def consume(self, event: Any) -> bool:
        if not self.restored:
            raise RuntimeError("Restore durable consumer before accepting fills")
        with self.coord._lock:
            fact = self.executor.fill_fact(event.fill_id)
            from decimal import Decimal
            if (str(event.order_id) != fact["intent_id"] or event.symbol != fact["symbol"]
                    or event.side != fact["side"] or event.qty != Decimal(fact["qty"])
                    or event.price != Decimal(fact["price"])):
                raise ValueError("Event differs from its durable exchange fact")
            before = self.coord._store.export_state()
            before_position = self.position.to_dict()
            before_strategy = self.module.execution_state()
            old_snapshot = self.coord._last_snapshot
            output = None

            def rollback() -> None:
                self.coord._store.load_exported(
                    before["markets"], before["positions"], before["account"],
                    event_index=before["event_index"], last_event_id=before["last_event_id"],
                    last_ts=before["last_ts"], portfolio=before["portfolio"], risk=before["risk"],
                )
                self.position = StrategyPosition.from_dict(before_position)
                self.module.restore_execution_state(before_strategy)
                self.module._confirmed_position = self.position
                self.coord._last_snapshot = old_snapshot

            def apply() -> dict[str, Any]:
                nonlocal output
                output = self.coord._pipeline.apply(PipelineInput(
                    event=event, event_index=before["event_index"], symbol_default=self.module._symbol,
                    markets=before["markets"], account=before["account"], positions=before["positions"],
                ))
                if not output.advanced or output.snapshot is None:
                    raise RuntimeError("Confirmed fill did not advance the state pipeline")
                self.position.apply(SimpleNamespace(**fact))
                parent = self.executor._parents[fact["intent_id"]]
                self.module.sync_confirmed_position(self.position, exit_reason=getattr(parent, "reason", None))
                return {"engine": _serialize_snapshot(output.snapshot), "position": self.position.to_dict(),
                        "revision": event.fill_id, "strategy": self.module.execution_state()}

            changed = self.journal.consume(self.owner, event.fill_id, apply, rollback,
                                           expected_revision=self.revision)
            if changed:
                self.revision = event.fill_id
                self.coord._last_snapshot = output.snapshot
                self.coord._cached_view = None
                self.module.sync_confirmed_position(self.position)
                # Auxiliary observers must not affect ledger commit or delivery.
                try:
                    for observer in self.observers:
                        observer(event)
                    if self.coord._cfg.on_pipeline_output:
                        self.coord._cfg.on_pipeline_output(output)
                    if self.coord._cfg.on_snapshot:
                        self.coord._cfg.on_snapshot(output.snapshot)
                except Exception:
                    logger.exception("Fill committed; auxiliary observer failed")
            return changed
