"""Ingestion orchestrator — manages data collector lifecycles and health monitoring."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence

from data.ingestion.config import IngestionConfig

logger = logging.getLogger(__name__)


class CollectorProtocol(Protocol):
    """Minimal interface for a data collector."""

    @property
    def name(self) -> str: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    @property
    def is_running(self) -> bool: ...

    @property
    def last_active_ts(self) -> datetime | None: ...


class BackfillerProtocol(Protocol):
    """Minimal interface for a backfiller."""

    def backfill(self, symbol: str, start: datetime, end: datetime) -> int: ...


class IngestionOrchestrator:
    """Manages all data collector lifecycles.

    Responsibilities:
    - On start: optionally run backfill for configured symbols
    - Start / stop all registered collectors
    - Run periodic health checks
    """

    def __init__(
        self,
        *,
        config: IngestionConfig,
        bar_store: Any,
        tick_store: Optional[Any] = None,
        backfiller: Optional[Any] = None,
        collectors: Sequence[Any] = (),
    ) -> None:
        self._config = config
        self._bar_store = bar_store
        self._tick_store = tick_store
        self._backfiller = backfiller
        self._collectors: List[Any] = list(collectors)
        self._health_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the orchestrator: backfill, start collectors, begin health checks."""
        if self._running:
            logger.warning("IngestionOrchestrator already running")
            return

        logger.info("Starting IngestionOrchestrator")
        self._stop_event.clear()
        self._running = True

        # 1. Backfill if configured
        if self._config.backfill_on_start and self._backfiller is not None:
            self._run_backfill()

        # 2. Start all collectors
        for collector in self._collectors:
            try:
                collector.start()
                logger.info("Started collector: %s", collector.name)
            except Exception:
                logger.exception("Failed to start collector: %s", collector.name)

        # 3. Start health check loop
        self._health_thread = threading.Thread(
            target=self._health_loop,
            name="ingestion-health",
            daemon=True,
        )
        self._health_thread.start()
        logger.info(
            "IngestionOrchestrator started with %d collectors",
            len(self._collectors),
        )

    def stop(self) -> None:
        """Stop all collectors and the health check loop."""
        if not self._running:
            return

        logger.info("Stopping IngestionOrchestrator")
        self._stop_event.set()

        for collector in self._collectors:
            try:
                collector.stop()
                logger.info("Stopped collector: %s", collector.name)
            except Exception:
                logger.exception("Failed to stop collector: %s", collector.name)

        if self._health_thread is not None:
            self._health_thread.join(timeout=5.0)
            self._health_thread = None

        self._running = False
        logger.info("IngestionOrchestrator stopped")

    def health_report(self) -> Dict[str, Any]:
        """Return health status for all collectors."""
        now = datetime.now(timezone.utc)
        collector_reports: list[dict[str, Any]] = []

        for collector in self._collectors:
            last_ts = getattr(collector, "last_active_ts", None)
            lag_seconds: float | None = None
            if last_ts is not None:
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                lag_seconds = (now - last_ts).total_seconds()

            collector_reports.append(
                {
                    "name": collector.name,
                    "is_running": getattr(collector, "is_running", False),
                    "last_active_ts": last_ts.isoformat() if last_ts else None,
                    "lag_seconds": lag_seconds,
                }
            )

        return {
            "orchestrator_running": self._running,
            "collector_count": len(self._collectors),
            "collectors": collector_reports,
            "checked_at": now.isoformat(),
        }

    def _run_backfill(self) -> None:
        """Execute backfill for each configured symbol."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self._config.backfill_days)

        for sym_cfg in self._config.symbols:
            if not sym_cfg.collect_bars:
                continue
            try:
                count = self._backfiller.backfill(sym_cfg.symbol, start, now)
                logger.info(
                    "Backfilled %d bars for %s",
                    count,
                    sym_cfg.symbol,
                )
            except Exception:
                logger.exception("Backfill failed for %s", sym_cfg.symbol)

    def _health_loop(self) -> None:
        """Periodically log health report until stop is signaled."""
        interval = self._config.health_check_interval_sec
        while not self._stop_event.wait(timeout=interval):
            report = self.health_report()
            for cr in report["collectors"]:
                if not cr["is_running"]:
                    logger.warning("Collector %s is not running", cr["name"])
                elif cr["lag_seconds"] is not None and cr["lag_seconds"] > 300:
                    logger.warning(
                        "Collector %s lag %.1fs",
                        cr["name"],
                        cr["lag_seconds"],
                    )
