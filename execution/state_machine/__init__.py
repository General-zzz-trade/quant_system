"""execution.state_machine — Order lifecycle state machine (Domain 3: core plumbing).

Tracks each order through its lifecycle (NEW -> SUBMITTED -> PARTIAL -> FILLED/CANCELLED).
Enforces valid transitions, detects invariant violations, projects expected state,
and provides reconciliation rules when venue state diverges from local state.
"""
from execution.state_machine.transitions import (
    OrderStatus, Transition, VALID_TRANSITIONS, TERMINAL_STATUSES,
)
from execution.state_machine.machine import (
    OrderStateMachine, OrderState, OrderStateMachineError, InvalidTransitionError,
)
from execution.state_machine.invariants import (
    check_order_invariants, assert_order_invariants, InvariantViolation,
)
from execution.state_machine.projection import project_order, OrderProjection
from execution.state_machine.reconciliation_rules import (
    reconcile_order, ReconciliationResult, DriftSeverity,
)

__all__ = [
    # Status & transitions
    "OrderStatus",
    "Transition",
    "VALID_TRANSITIONS",
    "TERMINAL_STATUSES",
    # State machine
    "OrderStateMachine",
    "OrderState",
    "OrderStateMachineError",
    "InvalidTransitionError",
    # Invariants
    "check_order_invariants",
    "assert_order_invariants",
    "InvariantViolation",
    # Projection
    "project_order",
    "OrderProjection",
    # Reconciliation
    "reconcile_order",
    "ReconciliationResult",
    "DriftSeverity",
]
