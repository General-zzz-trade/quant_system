# execution/reconcile/scheduler.py
"""ReconcileScheduler — periodic reconciliation between local and exchange state."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, Mapping, Optional

from execution.reconcile.controller import ReconcileController, ReconcileReport
from execution.reconcile.policies import ReconcileAction

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
        venue_positions = venue.get("positions", {})

        # Extract local balances as {asset: amount}
        local_balances = self._extract_local_balances(local)
        venue_balances = venue.get("balances", {})

        try:
            report = self.controller.reconcile(
                venue=self.cfg.venue,
                local_positions=local_positions,
                venue_positions=venue_positions if venue_positions else None,
                local_balances=local_balances,
                venue_balances=venue_balances if venue_balances else None,
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
