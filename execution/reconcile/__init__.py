"""execution.reconcile — Position/balance/order reconciliation (Domain 4: ops).

Detects and resolves drift between local state and venue state:
  - Drift detection (position, balance, fill, order level)
  - Policy-based resolution (auto-correct vs. alert vs. quarantine)
  - ReconcileController: orchestrates periodic reconciliation cycles
  - Scheduler: time-based reconciliation with backoff
"""
from execution.reconcile.drift import Drift, DriftSeverity, DriftType
from execution.reconcile.positions import reconcile_positions, PositionReconcileResult
from execution.reconcile.balances import reconcile_balances, BalanceReconcileResult
from execution.reconcile.fills import reconcile_fills, FillReconcileResult
from execution.reconcile.orders import reconcile_orders, OrderReconcileResult
from execution.reconcile.policies import ReconcilePolicy, ReconcileAction, PolicyDecision
from execution.reconcile.controller import ReconcileController, ReconcileReport

__all__ = [
    # Drift types
    "Drift",
    "DriftSeverity",
    "DriftType",
    # Reconcile functions
    "reconcile_positions",
    "PositionReconcileResult",
    "reconcile_balances",
    "BalanceReconcileResult",
    "reconcile_fills",
    "FillReconcileResult",
    "reconcile_orders",
    "OrderReconcileResult",
    # Policies
    "ReconcilePolicy",
    "ReconcileAction",
    "PolicyDecision",
    # Controller
    "ReconcileController",
    "ReconcileReport",
]
