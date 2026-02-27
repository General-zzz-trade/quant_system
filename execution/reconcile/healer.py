# execution/reconcile/healer.py
"""Reconciliation healer — auto-fix accepted drifts by synthesizing corrective events."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from execution.reconcile.controller import ReconcileReport
from execution.reconcile.drift import Drift, DriftType, DriftSeverity
from execution.reconcile.policies import PolicyDecision, ReconcileAction

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HealAction:
    """Record of an auto-heal action taken."""
    drift: Drift
    correction_type: str  # "position_adjust", "balance_adjust"
    symbol: str
    expected: str
    actual: str
    applied_at: float
    detail: str = ""


@dataclass
class ReconcileHealer:
    """Auto-fixes accepted drifts by emitting synthetic corrective events.

    For ACCEPT decisions on position/balance drifts, generates corrective
    actions that adjust local state to match venue state. All actions are
    recorded in an audit log.

    Usage:
        healer = ReconcileHealer(
            emit_position_adjust=lambda sym, qty: ...,
            emit_balance_adjust=lambda asset, amount: ...,
        )
        actions = healer.heal(report)
    """

    emit_position_adjust: Optional[Callable[[str, Decimal], None]] = None
    emit_balance_adjust: Optional[Callable[[str, Decimal], None]] = None
    max_auto_heal_per_cycle: int = 10

    _audit_log: List[HealAction] = field(default_factory=list, init=False)

    def heal(self, report: ReconcileReport) -> List[HealAction]:
        """Process a reconcile report and auto-fix accepted drifts.

        Only heals drifts where the policy decision is ACCEPT.
        Returns list of heal actions taken.
        """
        actions: List[HealAction] = []

        # Build mapping from drift to decision
        accept_drifts = [
            d.drift for d in report.decisions
            if d.action == ReconcileAction.ACCEPT
            and d.drift.severity != DriftSeverity.NONE
        ]

        for drift in accept_drifts[:self.max_auto_heal_per_cycle]:
            action = self._heal_drift(drift, report.venue)
            if action is not None:
                actions.append(action)
                self._audit_log.append(action)

        if actions:
            logger.info(
                "Healer applied %d corrections for venue %s",
                len(actions), report.venue,
            )

        return actions

    def _heal_drift(self, drift: Drift, venue: str) -> Optional[HealAction]:
        now = time.time()

        if drift.drift_type == DriftType.POSITION_QTY:
            return self._heal_position(drift, now)
        elif drift.drift_type == DriftType.BALANCE:
            return self._heal_balance(drift, now)

        return None

    def _heal_position(self, drift: Drift, ts: float) -> Optional[HealAction]:
        if self.emit_position_adjust is None:
            return None

        try:
            expected = Decimal(drift.expected)
            actual = Decimal(drift.actual)
        except Exception:
            logger.warning("Cannot parse position drift values: %s", drift)
            return None

        adjustment = actual - expected

        try:
            self.emit_position_adjust(drift.symbol, adjustment)
        except Exception as e:
            logger.error("Position heal failed for %s: %s", drift.symbol, e)
            return None

        action = HealAction(
            drift=drift,
            correction_type="position_adjust",
            symbol=drift.symbol,
            expected=drift.expected,
            actual=drift.actual,
            applied_at=ts,
            detail=f"adjusted by {adjustment}",
        )

        logger.info(
            "Healed position drift: %s %s -> %s (adj %s)",
            drift.symbol, drift.expected, drift.actual, adjustment,
        )
        return action

    def _heal_balance(self, drift: Drift, ts: float) -> Optional[HealAction]:
        if self.emit_balance_adjust is None:
            return None

        try:
            expected = Decimal(drift.expected)
            actual = Decimal(drift.actual)
        except Exception:
            logger.warning("Cannot parse balance drift values: %s", drift)
            return None

        adjustment = actual - expected

        try:
            self.emit_balance_adjust(drift.symbol, adjustment)
        except Exception as e:
            logger.error("Balance heal failed for %s: %s", drift.symbol, e)
            return None

        action = HealAction(
            drift=drift,
            correction_type="balance_adjust",
            symbol=drift.symbol,
            expected=drift.expected,
            actual=drift.actual,
            applied_at=ts,
            detail=f"adjusted by {adjustment}",
        )

        logger.info(
            "Healed balance drift: %s %s -> %s (adj %s)",
            drift.symbol, drift.expected, drift.actual, adjustment,
        )
        return action

    @property
    def audit_log(self) -> List[HealAction]:
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        self._audit_log.clear()
