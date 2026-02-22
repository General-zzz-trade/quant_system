# execution/reconcile/controller.py
"""Reconciliation controller — orchestrates all reconciliation checks."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Mapping, Optional, Sequence, Set

from execution.reconcile.positions import reconcile_positions, PositionReconcileResult
from execution.reconcile.balances import reconcile_balances, BalanceReconcileResult
from execution.reconcile.fills import reconcile_fills, FillReconcileResult
from execution.reconcile.orders import reconcile_orders, OrderReconcileResult
from execution.reconcile.policies import ReconcilePolicy, PolicyDecision, ReconcileAction
from execution.reconcile.drift import Drift, DriftSeverity


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """完整对账报告。"""
    venue: str
    positions: Optional[PositionReconcileResult] = None
    balances: Optional[BalanceReconcileResult] = None
    fills: Optional[FillReconcileResult] = None
    orders: Optional[OrderReconcileResult] = None
    decisions: tuple[PolicyDecision, ...] = ()

    @property
    def ok(self) -> bool:
        parts = [self.positions, self.balances, self.fills, self.orders]
        return all(p is None or p.ok for p in parts)

    @property
    def all_drifts(self) -> Sequence[Drift]:
        drifts: list[Drift] = []
        for part in [self.positions, self.balances, self.fills, self.orders]:
            if part is not None:
                drifts.extend(part.drifts)
        return drifts

    @property
    def should_halt(self) -> bool:
        return any(d.action == ReconcileAction.HALT for d in self.decisions)


class ReconcileController:
    """
    对账控制器 — 编排所有对账检查。

    统一接口：传入本地/交易所数据，执行全量对账，返回报告。
    """

    def __init__(
        self,
        *,
        policy: Optional[ReconcilePolicy] = None,
        qty_tolerance: Decimal = Decimal("0.00001"),
        balance_tolerance: Decimal = Decimal("0.01"),
    ) -> None:
        self._policy = policy or ReconcilePolicy()
        self._qty_tolerance = qty_tolerance
        self._balance_tolerance = balance_tolerance

    def reconcile(
        self,
        *,
        venue: str,
        local_positions: Optional[Mapping[str, Decimal]] = None,
        venue_positions: Optional[Mapping[str, Decimal]] = None,
        local_balances: Optional[Mapping[str, Decimal]] = None,
        venue_balances: Optional[Mapping[str, Decimal]] = None,
        local_orders: Optional[Mapping[str, str]] = None,
        venue_orders: Optional[Mapping[str, str]] = None,
        local_fill_ids: Optional[Set[str]] = None,
        venue_fill_ids: Optional[Set[str]] = None,
        fill_symbol: str = "",
    ) -> ReconcileReport:
        """执行全量对账。"""
        pos_result = None
        bal_result = None
        fill_result = None
        order_result = None

        if local_positions is not None and venue_positions is not None:
            pos_result = reconcile_positions(
                venue=venue,
                local_positions=local_positions,
                venue_positions=venue_positions,
                tolerance=self._qty_tolerance,
            )

        if local_balances is not None and venue_balances is not None:
            bal_result = reconcile_balances(
                venue=venue,
                local_balances=local_balances,
                venue_balances=venue_balances,
                tolerance=self._balance_tolerance,
            )

        if local_fill_ids is not None and venue_fill_ids is not None:
            fill_result = reconcile_fills(
                venue=venue, symbol=fill_symbol,
                local_fill_ids=local_fill_ids,
                venue_fill_ids=venue_fill_ids,
            )

        if local_orders is not None and venue_orders is not None:
            order_result = reconcile_orders(
                venue=venue,
                local_orders=local_orders,
                venue_orders=venue_orders,
            )

        # 收集所有漂移，应用策略
        all_drifts: list[Drift] = []
        for part in [pos_result, bal_result, fill_result, order_result]:
            if part is not None:
                all_drifts.extend(part.drifts)

        decisions = tuple(self._policy.decide_batch(all_drifts))

        return ReconcileReport(
            venue=venue,
            positions=pos_result,
            balances=bal_result,
            fills=fill_result,
            orders=order_result,
            decisions=decisions,
        )
