# execution/reconcile/policies.py
"""Reconciliation policies — what to do when drift is detected."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from execution.reconcile.drift import Drift, DriftSeverity


class ReconcileAction(str, Enum):
    """对账后的处理动作。"""
    ACCEPT = "accept"            # 接受交易所数据
    ALERT = "alert"              # 告警但不自动修复
    HALT = "halt"                # 暂停交易
    MANUAL_REVIEW = "manual"     # 需要人工审核


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """对账策略决策。"""
    action: ReconcileAction
    reason: str
    drift: Drift


class ReconcilePolicy:
    """
    对账策略 — 根据漂移严重程度决定动作。
    """

    def __init__(
        self,
        *,
        auto_accept_info: bool = True,
        auto_accept_warning: bool = False,
        halt_on_critical: bool = True,
    ) -> None:
        self._auto_accept_info = auto_accept_info
        self._auto_accept_warning = auto_accept_warning
        self._halt_on_critical = halt_on_critical

    def decide(self, drift: Drift) -> PolicyDecision:
        """根据漂移决定处理动作。"""
        if drift.severity == DriftSeverity.NONE:
            return PolicyDecision(
                action=ReconcileAction.ACCEPT,
                reason="no drift", drift=drift,
            )

        if drift.severity == DriftSeverity.INFO:
            if self._auto_accept_info:
                return PolicyDecision(
                    action=ReconcileAction.ACCEPT,
                    reason="info-level drift auto-accepted",
                    drift=drift,
                )
            return PolicyDecision(
                action=ReconcileAction.ALERT,
                reason="info-level drift requires review",
                drift=drift,
            )

        if drift.severity == DriftSeverity.WARNING:
            if self._auto_accept_warning:
                return PolicyDecision(
                    action=ReconcileAction.ACCEPT,
                    reason="warning-level drift auto-accepted",
                    drift=drift,
                )
            return PolicyDecision(
                action=ReconcileAction.ALERT,
                reason=f"warning drift: {drift.detail}",
                drift=drift,
            )

        # CRITICAL
        if self._halt_on_critical:
            return PolicyDecision(
                action=ReconcileAction.HALT,
                reason=f"critical drift: {drift.detail}",
                drift=drift,
            )
        return PolicyDecision(
            action=ReconcileAction.MANUAL_REVIEW,
            reason=f"critical drift requires manual review: {drift.detail}",
            drift=drift,
        )

    def decide_batch(self, drifts: Sequence[Drift]) -> Sequence[PolicyDecision]:
        return [self.decide(d) for d in drifts]
