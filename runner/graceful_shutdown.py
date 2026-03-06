# runner/graceful_shutdown.py
"""GracefulShutdown — orderly shutdown sequence for live trading systems.

Handles SIGTERM/SIGINT with a multi-step shutdown:
  1. Stop new orders (trigger KillSwitch GLOBAL HARD_KILL)
  2. Wait for pending orders (with timeout)
  3. Cancel remaining orders
  4. Final reconciliation
  5. Save state snapshot
  6. Cleanup
"""
from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShutdownConfig:
    pending_order_timeout_sec: float = 30.0
    save_state: bool = True
    state_file: str = "state_snapshot.json"
    poll_interval_sec: float = 0.5


class GracefulShutdown:
    """Handles SIGTERM/SIGINT with orderly shutdown sequence.

    Each callback is optional. Steps with no callback are skipped.

    Usage:
        shutdown = GracefulShutdown(
            stop_new_orders=lambda: kill_switch.trigger(scope=KillScope.GLOBAL, key="*", mode=KillMode.HARD_KILL),
            wait_pending=lambda: len(pending_orders) == 0,
            cancel_all=venue_client.cancel_all_orders,
            reconcile=reconcile_scheduler.run_once,
            save_snapshot=lambda path: state_store.save(path),
            cleanup=runner.stop,
        )
        shutdown.install_handlers()
    """

    def __init__(
        self,
        *,
        config: ShutdownConfig = ShutdownConfig(),
        stop_new_orders: Optional[Callable[[], None]] = None,
        wait_pending: Optional[Callable[[], bool]] = None,
        cancel_all: Optional[Callable[[], None]] = None,
        reconcile: Optional[Callable[[], None]] = None,
        save_snapshot: Optional[Callable[[str], None]] = None,
        cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        self._config = config
        self._stop_new_orders = stop_new_orders
        self._wait_pending = wait_pending
        self._cancel_all = cancel_all
        self._reconcile = reconcile
        self._save_snapshot = save_snapshot
        self._cleanup = cleanup
        self._shutting_down = False

    def install_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers."""
        if threading.current_thread() is not threading.main_thread():
            logger.warning("Skipping signal handler install: not running in main thread")
            return
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum: int, frame: object) -> None:
        sig_name = signal.Signals(signum).name
        if self._shutting_down:
            logger.warning("Force shutdown requested (second %s)", sig_name)
            sys.exit(1)
        logger.info("Received %s, initiating graceful shutdown", sig_name)
        self._shutting_down = True
        self.execute()

    def execute(self) -> None:
        """Execute the full shutdown sequence."""
        logger.info("Starting graceful shutdown sequence...")

        # Step 1: Stop new orders
        self._run_step("stop_new_orders", self._stop_new_orders)

        # Step 2: Wait for pending orders with timeout
        if self._wait_pending is not None:
            logger.info("Waiting for pending orders (timeout=%ss)...", self._config.pending_order_timeout_sec)
            deadline = time.monotonic() + self._config.pending_order_timeout_sec
            drained = False
            while time.monotonic() < deadline:
                try:
                    if self._wait_pending():
                        drained = True
                        break
                except Exception:
                    logger.exception("wait_pending callback failed")
                    continue
                time.sleep(self._config.poll_interval_sec)
            if drained:
                logger.info("All pending orders resolved")
            else:
                logger.warning("Pending order timeout reached, proceeding to cancel")

        # Step 3: Cancel remaining orders
        self._run_step("cancel_all", self._cancel_all)

        # Step 4: Final reconciliation
        self._run_step("reconcile", self._reconcile)

        # Step 5: Save state snapshot
        if self._save_snapshot is not None and self._config.save_state:
            logger.info("Saving state to %s", self._config.state_file)
            try:
                self._save_snapshot(self._config.state_file)
            except Exception:
                logger.exception("save_snapshot failed")

        # Step 6: Cleanup
        self._run_step("cleanup", self._cleanup)

        logger.info("Graceful shutdown complete")

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def _run_step(self, name: str, fn: Optional[Callable[[], None]]) -> None:
        if fn is None:
            return
        logger.info("Shutdown step: %s", name)
        try:
            fn()
        except Exception:
            logger.exception("Shutdown step '%s' failed", name)
