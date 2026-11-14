"""Order intent lifecycle: the statuses stored in the order journal and the transitions between them.

Unsettled statuses wait for the venue; settled ones are final and never change. The table is the
behaviour of shared.confirmed_execution as characterised in migration step 5:

    (new row)              -> reserved            intent reserved before anything is sent
    (new row)              -> submitted           venue-originated order adopted (triggered stop)
    reserved               -> submitted | rejected_unconfirmed | unknown     placement result
    unsettled              -> unknown             lookup failed or the result is ambiguous ("sticky")
    unknown                -> rejected_unconfirmed   a definitive rejection seen again after a failure
    rejected_unconfirmed   -> unknown
    unsettled              -> any settled status  venue terminal state with every trade recorded,
                                                  or the venue confirmed the order does not exist

Writing the current status again is a no-op, not a transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class OrderStatus(str, Enum):
    RESERVED = "reserved"
    SUBMITTED = "submitted"
    UNKNOWN = "unknown"
    REJECTED_UNCONFIRMED = "rejected_unconfirmed"
    FILLED = "filled"
    CANCELED = "canceled"
    MMP_CANCELED = "mmp_canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"


S = OrderStatus
UNSETTLED = frozenset({S.RESERVED, S.SUBMITTED, S.UNKNOWN, S.REJECTED_UNCONFIRMED})
VENUE_TERMINAL = frozenset({S.FILLED, S.CANCELED, S.MMP_CANCELED, S.REJECTED, S.EXPIRED})
ABSENT = frozenset({S.NOT_FOUND, S.REJECTED})         # settled without any exchange order
SETTLED = VENUE_TERMINAL | ABSENT
# Status a runtime may see in a status callback for an intent that was never stored.
BLOCKED = "blocked"

TRANSITIONS: dict[OrderStatus | None, frozenset[OrderStatus]] = {
    None: frozenset({S.RESERVED, S.SUBMITTED}),
    S.RESERVED: frozenset({S.SUBMITTED, S.REJECTED_UNCONFIRMED, S.UNKNOWN}) | SETTLED,
    S.SUBMITTED: frozenset({S.UNKNOWN}) | SETTLED,
    S.UNKNOWN: frozenset({S.REJECTED_UNCONFIRMED}) | SETTLED,
    S.REJECTED_UNCONFIRMED: frozenset({S.UNKNOWN}) | SETTLED,
}


class IllegalTransition(ValueError):
    pass


def check_transition(current: str | None, new: str) -> None:
    """Raise IllegalTransition unless ``current -> new`` is in the table (re-writing a status is allowed)."""
    try:
        target = OrderStatus(new)
        source = None if current is None else OrderStatus(current)
    except ValueError as exc:
        raise IllegalTransition(f"unknown order status in {current!r} -> {new!r}") from exc
    if source == target:
        return
    if target not in TRANSITIONS.get(source, frozenset()):
        raise IllegalTransition(f"order status {current!r} -> {new!r} is not allowed")


@dataclass(frozen=True)
class OrderIntent:
    """An order the strategy wants placed; ``order_id`` is the durable intent id (no exchange types)."""

    order_id: str
    source: str
    symbol: str
    side: str            # buy | sell
    qty: Decimal
    price: Decimal       # reference price at decision time
    reduce_only: bool
    reason: str

