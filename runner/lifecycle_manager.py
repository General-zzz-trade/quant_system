"""LifecycleManager — startup/shutdown sequencing + signal handling.

Owns the start() and stop() sequences, SIGTERM/SIGINT/SIGHUP handlers,
and optional systemd watchdog notification.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
from typing import Any

logger = logging.getLogger(__name__)


class LifecycleManager:
    """Manages startup sequence, shutdown ordering, and OS signals."""

    def __init__(
        self,
        engine: Any,
        executor: Any,
        recovery: Any,
        loop: Any,
        enable_watchdog: bool = False,
    ) -> None:
        self._engine = engine
        self._executor = executor
        self._recovery = recovery
        self._loop = loop
        self._enable_watchdog = enable_watchdog
        self._running = False

    def install_signal_handlers(self) -> None:
        """Install OS signal handlers: SIGTERM/SIGINT → stop, SIGHUP → reload."""
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())
        signal.signal(signal.SIGINT, lambda s, f: self.stop())
        signal.signal(signal.SIGHUP, lambda s, f: self._handle_sighup())
        logger.info("Signal handlers installed (SIGTERM/SIGINT→stop, SIGHUP→reload)")

    def start(self) -> None:
        """Execute startup sequence, then enter event loop.

        Sequence:
        1. Apply performance tuning (optional)
        2. Restore from checkpoint
        3. Start user stream
        4. Install signal handlers
        5. Enter main loop (blocks)
        """
        self._running = True
        self._apply_perf_tuning()

        # Restore state from last checkpoint
        restored = self._recovery.restore()
        if restored:
            logger.info("State restored from checkpoint")

        # Start user stream for fill notifications
        self._executor.start_user_stream()

        # Install signal handlers
        self.install_signal_handlers()

        # Notify systemd if watchdog enabled
        self._sd_notify("READY=1")

        logger.info("Lifecycle started, entering event loop")

        # Enter main event loop (blocks)
        try:
            self._loop.start()
        finally:
            if self._running:
                self.stop()

    def stop(self) -> None:
        """Execute shutdown sequence in correct order.

        Order:
        1. User stream
        2. Checkpoint save
        3. Main loop
        """
        if not self._running:
            return
        self._running = False
        logger.info("Shutdown initiated")

        self._sd_notify("STOPPING=1")

        # 1. Stop user stream first (no more fill events)
        try:
            self._executor.stop_user_stream()
        except Exception as e:
            logger.error("Error stopping user stream: %s", e)

        # 2. Save final checkpoint
        try:
            self._recovery.save()
        except Exception as e:
            logger.error("Error saving checkpoint: %s", e)

        # 3. Stop main loop
        try:
            self._loop.stop()
        except Exception as e:
            logger.error("Error stopping loop: %s", e)

        logger.info("Shutdown complete")

    def _handle_sighup(self) -> None:
        """Handle SIGHUP: reload models."""
        logger.info("SIGHUP received, reloading models")
        try:
            results = self._engine.reload_models()
            for symbol, status in results.items():
                logger.info("Model reload: %s → %s", symbol, status)
        except Exception as e:
            logger.error("Model reload failed: %s", e)

    def _apply_perf_tuning(self) -> None:
        """Optional OS-level performance tuning."""
        try:
            os.nice(-5)
        except PermissionError:
            pass

    def _sd_notify(self, msg: str) -> None:
        """Send systemd notification if watchdog enabled."""
        if not self._enable_watchdog:
            return
        try:
            addr = os.environ.get("NOTIFY_SOCKET")
            if addr:
                import socket
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                sock.connect(addr)
                sock.send(msg.encode())
                sock.close()
        except Exception:
            pass
