# execution/reconcile/scheduler.py
"""ReconcileScheduler — periodic reconciliation between local and exchange state."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, Mapping, Optional, Set

from execution.reconcile.controller import ReconcileController, ReconcileReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcileSchedulerConfig:
    """Configuration for reconciliation scheduler."""
    interval_sec: float = 60.0
    venue: str = "binance"
    halt_on_critical: bool = True


@dataclass
class ReconcileScheduler:
    """Periodic reconciliation between local engine state and exchange.

    Reads local state from coordinator.get_state_view(), reads exchange state
    from venue_fetcher, and runs ReconcileController.reconcile() to detect drift.

    CRITICAL drift → coordinator.stop() + alert
    WARNING drift → alert but continue

    Usage:
        scheduler = ReconcileScheduler(
            controller=ReconcileController(),
            get_local_state=lambda: coordinator.get_state_view(),
            fetch_venue_state=my_venue_fetcher,
            on_halt=lambda report: coordinator.stop(),
        )
        scheduler.start()
    """

    controller: ReconcileController
    get_local_state: Callable[[], Mapping[str, Any]]
    fetch_venue_state: Callable[[], Dict[str, Any]]
    cfg: ReconcileSchedulerConfig = field(default_factory=ReconcileSchedulerConfig)
    on_halt: Optional[Callable[[ReconcileReport], None]] = None
    on_alert: Optional[Callable[[ReconcileReport], None]] = None

    _running: bool = field(default=False, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _last_report: Optional[ReconcileReport] = field(default=None, init=False)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="reconcile-scheduler", daemon=True,
        )
        self._thread.start()
        logger.info(
            "ReconcileScheduler started (interval=%ss, venue=%s)",
            self.cfg.interval_sec, self.cfg.venue,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=self.cfg.interval_sec * 2)
            self._thread = None
        logger.info("ReconcileScheduler stopped")

    @property
    def last_report(self) -> Optional[ReconcileReport]:
        return self._last_report

    def run_once(self) -> Optional[ReconcileReport]:
        """Run a single reconciliation check. Returns the report."""
        try:
            local = self.get_local_state()
            venue = self.fetch_venue_state()
        except Exception:
            logger.exception("Failed to fetch state for reconciliation")
            return None

        # Extract local positions as {symbol: qty}
        local_positions = self._extract_local_positions(local)
        venue_positions = self._extract_optional_position_map(venue)

        # Extract local balances as {asset: amount}
        local_balances = self._extract_local_balances(local)
        venue_balances = self._extract_optional_balance_map(venue)
        local_orders = self._extract_local_orders(local)
        venue_orders = self._extract_venue_orders(venue)
        local_fill_ids = self._extract_local_fill_ids(local)
        venue_fill_ids = self._extract_venue_fill_ids(venue)

        try:
            report = self.controller.reconcile(
                venue=self.cfg.venue,
                local_positions=local_positions,
                venue_positions=venue_positions,
                local_balances=local_balances,
                venue_balances=venue_balances,
                local_orders=local_orders,
                venue_orders=venue_orders,
                local_fill_ids=local_fill_ids,
                venue_fill_ids=venue_fill_ids,
                fill_symbol=self._infer_fill_symbol(local_positions, venue_positions),
            )
        except Exception:
            logger.exception("Reconciliation failed")
            return None

        self._last_report = report
        self._handle_report(report)
        return report

    def _run_loop(self) -> None:
        while self._running:
            self.run_once()
            if not self._running:
                break
            time.sleep(self.cfg.interval_sec)

    def _extract_local_positions(
        self, state: Mapping[str, Any],
    ) -> Dict[str, Decimal]:
        positions = state.get("positions", {})
        result: Dict[str, Decimal] = {}
        for sym, pos in positions.items():
            qty = getattr(pos, "qty", None) or getattr(pos, "quantity", Decimal("0"))
            result[sym] = Decimal(str(qty))
        return result

    def _extract_local_balances(
        self, state: Mapping[str, Any],
    ) -> Dict[str, Decimal]:
        account = state.get("account")
        if account is None:
            return {}
        balance = getattr(account, "balance", None)
        currency = getattr(account, "currency", "USDT")
        if balance is not None:
            return {str(currency): Decimal(str(balance))}
        return {}

    def _extract_optional_position_map(
        self, venue: Mapping[str, Any],
    ) -> Optional[Mapping[str, Decimal]]:
        if "positions" not in venue:
            return None
        raw = venue.get("positions")
        if raw is None:
            return None
        return {str(sym): Decimal(str(qty)) for sym, qty in dict(raw).items()}

    def _extract_optional_balance_map(
        self, venue: Mapping[str, Any],
    ) -> Optional[Mapping[str, Decimal]]:
        if "balances" not in venue:
            return None
        raw = venue.get("balances")
        if raw is None:
            return None
        return {str(asset): Decimal(str(amount)) for asset, amount in dict(raw).items()}

    def _extract_local_orders(
        self, state: Mapping[str, Any],
    ) -> Optional[Dict[str, str]]:
        if "orders" not in state:
            return None
        return _normalize_orders(state.get("orders"))

    def _extract_venue_orders(
        self, venue: Mapping[str, Any],
    ) -> Optional[Dict[str, str]]:
        if "orders" not in venue:
            return None
        return _normalize_orders(venue.get("orders"))

    def _extract_local_fill_ids(
        self, state: Mapping[str, Any],
    ) -> Optional[Set[str]]:
        if "fill_ids" in state:
            return {str(fill_id) for fill_id in state.get("fill_ids") or set()}
        if "fills" not in state:
            return None
        return _normalize_fill_ids(state.get("fills"))

    def _extract_venue_fill_ids(
        self, venue: Mapping[str, Any],
    ) -> Optional[Set[str]]:
        if "fill_ids" in venue:
            return {str(fill_id) for fill_id in venue.get("fill_ids") or set()}
        if "fills" not in venue:
            return None
        return _normalize_fill_ids(venue.get("fills"))

    @staticmethod
    def _infer_fill_symbol(
        local_positions: Mapping[str, Decimal],
        venue_positions: Optional[Mapping[str, Decimal]],
    ) -> str:
        if local_positions:
            return next(iter(local_positions))
        if venue_positions:
            return next(iter(venue_positions))
        return ""

    def _handle_report(self, report: ReconcileReport) -> None:
        if report.ok:
            logger.debug("Reconciliation OK (venue=%s)", report.venue)
            return

        drifts = report.all_drifts
        for d in drifts:
            logger.warning(
                "DRIFT [%s] %s/%s: expected=%s actual=%s (%s)",
                d.severity, d.venue, d.symbol, d.expected, d.actual, d.detail,
            )

        if report.should_halt and self.cfg.halt_on_critical:
            logger.error("CRITICAL drift detected — halting system")
            if self.on_halt is not None:
                try:
                    self.on_halt(report)
                except Exception:
                    logger.exception("on_halt callback failed")
        elif not report.ok:
            if self.on_alert is not None:
                try:
                    self.on_alert(report)
                except Exception:
                    logger.exception("on_alert callback failed")


def _normalize_orders(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        result: Dict[str, str] = {}
        for order_id, order in raw.items():
            if isinstance(order, Mapping):
                status = order.get("status")
            else:
                status = getattr(order, "status", order)
            if status is not None:
                result[str(order_id)] = str(status)
        return result

    result = {}
    for order in raw:
        if isinstance(order, Mapping):
            order_id = order.get("order_id") or order.get("id")
            status = order.get("status")
        else:
            order_id = getattr(order, "order_id", None) or getattr(order, "id", None)
            status = getattr(order, "status", None)
        if order_id is not None and status is not None:
            result[str(order_id)] = str(status)
    return result


def _normalize_fill_ids(raw: Any) -> Set[str]:
    if raw is None:
        return set()
    if isinstance(raw, Mapping):
        return {str(fill_id) for fill_id in raw.keys()}

    result: Set[str] = set()
    for fill in raw:
        if isinstance(fill, Mapping):
            fill_id = fill.get("fill_id") or fill.get("trade_id")
        else:
            fill_id = getattr(fill, "fill_id", None) or getattr(fill, "trade_id", None)
        if fill_id is not None:
            result.add(str(fill_id))
    return result
