"""OKX execution boundary: emit only order-matched exchange trade facts.

Entries use one post-only order, then cancel the remainder. Automatic market
fallback and splitting are intentionally disabled until recovery is complete.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from event.header import EventHeader
from event.types import EventType, FillEvent
from monitoring.tca import TCALogger
from shared.confirmed_execution import ConfirmedExecutor

logger = logging.getLogger(__name__)


class OkxExecutionAdapter:
    def __init__(self, adapter: Any, circuit_breaker: Any = None, *,
                 journal_path: Path | None = None, scope: str | None = None) -> None:
        self._adapter, self._cb = adapter, circuit_breaker
        simulated = bool(getattr(getattr(adapter, "_config", None), "simulated", False))
        self._scope = scope or os.environ.get("RISK_SCOPE", f"okx-{'demo' if simulated else 'live'}-default")
        self._orders = ConfirmedExecutor(
            adapter, journal_path or Path(os.environ.get("ORDER_JOURNAL_PATH", "data/runtime/orders.sqlite")),
            self._scope, wait_seconds=float(os.environ.get("OKX_LIMIT_FILL_TIMEOUT_S", "3.0")),
        )
        self._parents: dict[str, Any] = {}
        self._submitted_at: dict[str, float] = {}
        self._lock = threading.RLock()
        if self._orders.unresolved():
            logger.critical("Unresolved OKX journal entries found; affected symbols remain blocked until reconciled")

    def _events(self, intent_id: str, fills: tuple[Any, ...]) -> tuple[FillEvent, ...]:
        parent = self._parents[intent_id]
        events = []
        for fill in fills:
            fill_id = f"{self._scope}:{fill.symbol}:{fill.order_id}:{fill.trade_id}"
            header = EventHeader.from_parent(parent=parent.header, event_type=EventType.FILL,
                                             version=1, source="okx")
            # Stable exchange identity also makes duplicate delivery detectable.
            header = replace(header, event_id=fill_id)
            events.append(FillEvent(
                header=header, fill_id=fill_id, order_id=parent.order_id,
                symbol=fill.symbol, qty=Decimal(str(fill.qty)),
                price=Decimal(str(fill.price)), side=fill.side,
            ))
            # Observability uses confirmed fills too. Reference is the decision
            # price, explicitly labelled so it is not confused with arrival mid.
            tca = None
            try:
                tca = TCALogger("okx", path=self._orders.path.parent / "tca_okx.jsonl")
                tca.record_fill(symbol=fill.symbol, side=fill.side, qty=float(fill.qty),
                                ref_price=float(parent.price or 0), fill_price=float(fill.price),
                                latency_ms=(time.monotonic() - self._submitted_at[intent_id]) * 1000,
                                order_id=fill.order_id, fill_id=fill_id, ref_source="decision_price")
            except Exception:
                logger.debug("TCA unavailable", exc_info=True)
            finally:
                if tca is not None:
                    tca.close()
        return tuple(events)

    def send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        with self._lock:
            return self._send_order(order_event)

    def _send_order(self, order_event: Any) -> tuple[FillEvent, ...]:
        try:
            qty = Decimal(str(order_event.qty))
            reducing = qty == 0 or getattr(order_event, "reduce_only", False) is True
            if not reducing and self._cb is not None and not self._cb.allow_request():
                return ()
            side = str(order_event.side).lower()
            if qty == 0:
                positions = self._adapter.get_positions(symbol=order_event.symbol)
                position = next((p for p in positions if p.symbol == order_event.symbol and not p.is_flat), None)
                if position is None:
                    return ()
                qty = abs(Decimal(str(position.qty)))
                side = "sell" if position.qty > 0 else "buy"
            intent_id = str(order_event.order_id)
            self._parents[intent_id] = order_event
            self._submitted_at[intent_id] = time.monotonic()
            if not reducing and datetime.now(timezone.utc).weekday() >= 5:
                factor = Decimal(os.environ.get("OKX_WEEKEND_QTY_FACTOR", "0.7"))
                if not factor.is_finite() or not 0 < factor <= 1:
                    raise ValueError("OKX_WEEKEND_QTY_FACTOR must be in (0, 1]")
                qty *= factor
            facts = self._orders.submit(intent_id=intent_id, symbol=order_event.symbol,
                                        side=side, qty=qty, reduce_only=reducing)
            events = self._events(intent_id, facts)
            if events and self._cb is not None:
                self._cb.record_success()
            return events
        except Exception:
            if self._cb is not None:
                self._cb.record_failure()
            logger.exception("OKX order unresolved or blocked; no automatic retry")
            return ()

    def reconcile_pending(self) -> tuple[FillEvent, ...]:
        with self._lock:
            events = tuple(event for intent_id, fill in self._orders.poll()
                           for event in self._events(intent_id, (fill,)))
            for intent_id in list(self._parents):
                if intent_id not in self._orders.pending:
                    self._parents.pop(intent_id, None)
                    self._submitted_at.pop(intent_id, None)
            return events
