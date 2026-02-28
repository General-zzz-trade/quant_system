"""Strategy hot reloader — reload decision modules on SIGHUP or file change.

Allows updating strategy code without restarting the live trading process.
"""
from __future__ import annotations

import importlib
import logging
import os
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReloaderConfig:
    """Configuration for module reloader."""

    watch_paths: tuple[str, ...] = ()
    poll_interval: float = 5.0  # seconds between file checks
    enable_sighup: bool = True


class ModuleReloader:
    """Watches for SIGHUP signals or file changes, then triggers module swap.

    Usage:
        reloader = ModuleReloader(config, on_reload=bridge.swap_modules)
        reloader.start()
        # ... runs in background ...
        reloader.stop()
    """

    def __init__(
        self,
        config: ReloaderConfig,
        on_reload: Callable[[str], None],
        module_names: Sequence[str] = (),
    ) -> None:
        self._config = config
        self._on_reload = on_reload
        self._module_names = list(module_names)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sighup_received = threading.Event()
        self._mtimes: dict[str, float] = {}
        self._prev_handler: Any = None

    def start(self) -> None:
        """Start watching for reload triggers."""
        if self._running:
            return

        self._running = True

        # Register SIGHUP handler
        if self._config.enable_sighup and hasattr(signal, "SIGHUP"):
            self._prev_handler = signal.getsignal(signal.SIGHUP)
            signal.signal(signal.SIGHUP, self._handle_sighup)

        # Initialize file mtimes
        for path_str in self._config.watch_paths:
            path = Path(path_str)
            if path.exists():
                self._mtimes[path_str] = path.stat().st_mtime

        # Start file watcher thread if watch paths configured
        if self._config.watch_paths:
            self._thread = threading.Thread(
                target=self._watch_loop, daemon=True, name="module-reloader",
            )
            self._thread.start()

        logger.info("ModuleReloader started (modules=%s, watch_paths=%s)",
                     self._module_names, self._config.watch_paths)

    def stop(self) -> None:
        """Stop the reloader."""
        self._running = False
        self._sighup_received.set()  # unblock watcher

        if self._config.enable_sighup and hasattr(signal, "SIGHUP"):
            if self._prev_handler is not None:
                try:
                    signal.signal(signal.SIGHUP, self._prev_handler)
                except (OSError, ValueError):
                    pass

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("ModuleReloader stopped")

    def trigger_reload(self) -> None:
        """Manually trigger a reload (for testing or programmatic use)."""
        self._do_reload("manual")

    @property
    def is_running(self) -> bool:
        return self._running

    def _handle_sighup(self, signum: int, frame: Any) -> None:
        """Signal handler for SIGHUP."""
        logger.info("SIGHUP received, scheduling reload")
        self._sighup_received.set()

    def _watch_loop(self) -> None:
        """Background thread: check files periodically for changes."""
        while self._running:
            # Check for SIGHUP
            if self._sighup_received.wait(timeout=self._config.poll_interval):
                if not self._running:
                    break
                self._sighup_received.clear()
                self._do_reload("sighup")

            if not self._running:
                break

            # Check file modifications
            for path_str in self._config.watch_paths:
                path = Path(path_str)
                if not path.exists():
                    continue
                mtime = path.stat().st_mtime
                prev = self._mtimes.get(path_str, 0.0)
                if mtime > prev:
                    self._mtimes[path_str] = mtime
                    self._do_reload(f"file_changed:{path_str}")

    def _do_reload(self, trigger: str) -> None:
        """Execute the reload: reimport modules and call on_reload callback."""
        logger.info("Reloading modules (trigger=%s)", trigger)

        for mod_name in self._module_names:
            try:
                mod = importlib.import_module(mod_name)
                importlib.reload(mod)
                logger.info("Reloaded module: %s", mod_name)
            except Exception:
                logger.exception("Failed to reload module: %s", mod_name)

        try:
            self._on_reload(trigger)
        except Exception:
            logger.exception("on_reload callback failed (trigger=%s)", trigger)
