"""Order lifecycle saga — tracks orders through states with compensating actions.

The Saga pattern ensures that every order has a well-defined lifecycle with
explicit state transitions, and that failures trigger compensating transactions
(e.g., cancel outstanding orders, flatten positions).

State Machine::

    PENDING ─→ SUBMITTED ─→ ACKED ─→ PARTIAL_FILL ─→ FILLED
                  │           │          │
                  │           ▼          │
                  │        CANCELLED     │
                  ▼                      ▼
               REJECTED              EXPIRED
                  │
                  ▼
              COMPENSATING ─→ COMPENSATED
                  │
                  ▼
               FAILED

Usage::

    manager = SagaManager()
    saga = manager.create("order-1", "intent-42", symbol="BTCUSDT", side="buy", qty=0.1)
    manager.transition("order-1", SagaState.SUBMITTED)
    manager.transition("order-1", SagaState.ACKED)
    manager.transition("order-1", SagaState.FILLED)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


# ── Saga States ──────────────────────────────────────────

class SagaState(Enum):
    """Order lifecycle states."""
    PENDING = auto()        # created, not yet submitted
    SUBMITTED = auto()      # sent to venue
    ACKED = auto()          # venue acknowledged
    PARTIAL_FILL = auto()   # partially filled
    FILLED = auto()         # fully filled (terminal)
    REJECTED = auto()       # venue rejected (terminal)
    CANCELLED = auto()      # cancelled (terminal)
    EXPIRED = auto()        # time-in-force expired (terminal)
    COMPENSATING = auto()   # executing compensating action
    COMPENSATED = auto()    # compensation complete (terminal)
    FAILED = auto()         # compensation failed (terminal, needs manual intervention)


# Valid state transitions
_TRANSITIONS: Dict[SagaState, frozenset[SagaState]] = {
    SagaState.PENDING: frozenset({SagaState.SUBMITTED, SagaState.REJECTED, SagaState.CANCELLED}),
    SagaState.SUBMITTED: frozenset({SagaState.ACKED, SagaState.REJECTED, SagaState.CANCELLED, SagaState.COMPENSATING}),
    SagaState.ACKED: frozenset({SagaState.PARTIAL_FILL, SagaState.FILLED, SagaState.CANCELLED, SagaState.EXPIRED, SagaState.COMPENSATING}),
    SagaState.PARTIAL_FILL: frozenset({SagaState.PARTIAL_FILL, SagaState.FILLED, SagaState.CANCELLED, SagaState.EXPIRED, SagaState.COMPENSATING}),
    SagaState.FILLED: frozenset(),
    SagaState.REJECTED: frozenset({SagaState.COMPENSATING}),
    SagaState.CANCELLED: frozenset({SagaState.COMPENSATING}),
    SagaState.EXPIRED: frozenset({SagaState.COMPENSATING}),
    SagaState.COMPENSATING: frozenset({SagaState.COMPENSATED, SagaState.FAILED}),
    SagaState.COMPENSATED: frozenset(),
    SagaState.FAILED: frozenset(),
}

TERMINAL_STATES = frozenset({
    SagaState.FILLED,
    SagaState.COMPENSATED,
    SagaState.FAILED,
})


# ── Saga Record ──────────────────────────────────────────

@dataclass
class SagaTransition:
    """A single state transition record for audit trail."""
    from_state: SagaState
    to_state: SagaState
    reason: str = ""
    timestamp_mono: float = 0.0


@dataclass
class OrderSaga:
    """Tracks a single order through its lifecycle.

    This is a mutable record — only the ``SagaManager`` should mutate it
    (under its lock).
    """
    order_id: str
    intent_id: str
    symbol: str
    side: str                  # "buy" | "sell"
    qty: float
    state: SagaState = SagaState.PENDING

    # Fills tracking
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fill_count: int = 0

    # Compensating action reference
    compensating_order_id: Optional[str] = None

    # Audit trail
    history: List[SagaTransition] = field(default_factory=list)

    # Arbitrary metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def remaining_qty(self) -> float:
        return max(0.0, self.qty - self.filled_qty)

    @property
    def fill_ratio(self) -> float:
        if self.qty <= 0:
            return 0.0
        return self.filled_qty / self.qty


class SagaError(RuntimeError):
    """Raised on invalid saga operations."""


# ── Compensating Action Protocol ─────────────────────────

class CompensatingAction:
    """Protocol-like base for compensating actions.

    Subclass and override ``execute`` to implement venue-specific
    compensation (e.g., cancel order, flatten position).
    """

    def execute(self, saga: OrderSaga) -> Optional[str]:
        """Execute compensating action.

        Returns a compensating order ID if applicable, else None.
        Raises on failure.
        """
        raise NotImplementedError


class CancelOrderAction(CompensatingAction):
    """Compensating action: cancel the order at the venue."""

    def __init__(self, cancel_fn: Callable[[str], None]) -> None:
        self._cancel = cancel_fn

    def execute(self, saga: OrderSaga) -> Optional[str]:
        self._cancel(saga.order_id)
        return None


# ── Saga Manager ─────────────────────────────────────────

class SagaManager:
    """Manages all active order sagas.

    Thread-safe.  Enforces valid state transitions and maintains
    an audit trail of all transitions.

    Parameters
    ----------
    on_terminal : callable, optional
        Called when a saga reaches a terminal state.
        Signature: ``(saga: OrderSaga) -> None``
    default_compensating_action : CompensatingAction, optional
        Default action to execute when a saga enters COMPENSATING state.
    max_completed : int
        Maximum number of completed sagas to retain (for audit).
        Oldest are evicted first.
    """

    def __init__(
        self,
        *,
        on_terminal: Optional[Callable[[OrderSaga], None]] = None,
        default_compensating_action: Optional[CompensatingAction] = None,
        max_completed: int = 10000,
    ) -> None:
        self._lock = threading.Lock()
        self._sagas: Dict[str, OrderSaga] = {}
        self._completed: Dict[str, OrderSaga] = {}
        self._on_terminal = on_terminal
        self._default_compensating = default_compensating_action
        self._max_completed = max_completed

    # ── Creation ─────────────────────────────────────────

    def create(
        self,
        order_id: str,
        intent_id: str,
        *,
        symbol: str,
        side: str,
        qty: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> OrderSaga:
        """Create a new order saga in PENDING state."""
        with self._lock:
            if order_id in self._sagas:
                raise SagaError(f"saga already exists: {order_id}")

            saga = OrderSaga(
                order_id=order_id,
                intent_id=intent_id,
                symbol=symbol,
                side=side,
                qty=qty,
                meta=dict(meta or {}),
            )
            self._sagas[order_id] = saga
            return saga

    # ── Transitions ──────────────────────────────────────

    def transition(
        self,
        order_id: str,
        to_state: SagaState,
        *,
        reason: str = "",
        timestamp_mono: float = 0.0,
    ) -> OrderSaga:
        """Transition a saga to a new state.

        Raises ``SagaError`` if the transition is invalid.
        """
        with self._lock:
            saga = self._sagas.get(order_id)
            if saga is None:
                raise SagaError(f"unknown saga: {order_id}")

            allowed = _TRANSITIONS.get(saga.state, frozenset())
            if to_state not in allowed:
                raise SagaError(
                    f"invalid transition: {saga.state.name} → {to_state.name} "
                    f"for order {order_id}"
                )

            saga.history.append(SagaTransition(
                from_state=saga.state,
                to_state=to_state,
                reason=reason,
                timestamp_mono=timestamp_mono,
            ))
            saga.state = to_state

            # Move to completed if terminal
            if saga.is_terminal:
                self._sagas.pop(order_id, None)
                self._completed[order_id] = saga
                self._evict_completed()

        # Terminal callback (outside lock)
        if saga.is_terminal and self._on_terminal is not None:
            self._on_terminal(saga)

        return saga

    def record_fill(
        self,
        order_id: str,
        fill_qty: float,
        fill_price: float,
    ) -> OrderSaga:
        """Record a partial or full fill."""
        with self._lock:
            saga = self._sagas.get(order_id)
            if saga is None:
                raise SagaError(f"unknown saga: {order_id}")

            if saga.state not in (SagaState.ACKED, SagaState.PARTIAL_FILL):
                raise SagaError(
                    f"cannot record fill in state {saga.state.name} for {order_id}"
                )

            # Weighted average price
            total_value = saga.avg_fill_price * saga.filled_qty + fill_price * fill_qty
            saga.filled_qty += fill_qty
            saga.fill_count += 1
            if saga.filled_qty > 0:
                saga.avg_fill_price = total_value / saga.filled_qty

        # Auto-transition
        if saga.filled_qty >= saga.qty:
            return self.transition(order_id, SagaState.FILLED, reason="fully filled")
        else:
            return self.transition(order_id, SagaState.PARTIAL_FILL, reason="partial fill")

    def compensate(
        self,
        order_id: str,
        *,
        action: Optional[CompensatingAction] = None,
        reason: str = "",
    ) -> OrderSaga:
        """Trigger compensating action for a saga."""
        act = action or self._default_compensating

        saga = self.transition(order_id, SagaState.COMPENSATING, reason=reason)

        if act is not None:
            try:
                comp_id = act.execute(saga)
                saga.compensating_order_id = comp_id
                return self.transition(order_id, SagaState.COMPENSATED, reason="compensation succeeded")
            except Exception as e:
                return self.transition(order_id, SagaState.FAILED, reason=f"compensation failed: {e}")

        # No action configured — mark as compensated (no-op compensation)
        return self.transition(order_id, SagaState.COMPENSATED, reason="no compensating action")

    # ── Queries ──────────────────────────────────────────

    def get(self, order_id: str) -> Optional[OrderSaga]:
        """Get saga by order ID (active or completed)."""
        with self._lock:
            saga = self._sagas.get(order_id)
            if saga is not None:
                return saga
            return self._completed.get(order_id)

    def active_sagas(self) -> Tuple[OrderSaga, ...]:
        """Return all non-terminal sagas."""
        with self._lock:
            return tuple(self._sagas.values())

    def active_count(self) -> int:
        with self._lock:
            return len(self._sagas)

    def by_intent(self, intent_id: str) -> Tuple[OrderSaga, ...]:
        """All sagas (active + completed) for a given intent."""
        with self._lock:
            result = []
            for saga in self._sagas.values():
                if saga.intent_id == intent_id:
                    result.append(saga)
            for saga in self._completed.values():
                if saga.intent_id == intent_id:
                    result.append(saga)
            return tuple(result)

    def by_symbol(self, symbol: str) -> Tuple[OrderSaga, ...]:
        """Active sagas for a given symbol."""
        with self._lock:
            return tuple(s for s in self._sagas.values() if s.symbol == symbol)

    # ── Internal ─────────────────────────────────────────

    def _evict_completed(self) -> None:
        """Evict oldest completed sagas if over limit. Must hold lock."""
        while len(self._completed) > self._max_completed:
            oldest_key = next(iter(self._completed))
            del self._completed[oldest_key]
