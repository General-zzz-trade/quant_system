"""Tests for module hot reloader — signal handling, file watching, atomic swap."""
from __future__ import annotations

import os
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from engine.module_reloader import ModuleReloader, ReloaderConfig


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestReloaderLifecycle:

    def test_start_stop(self):
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=MagicMock(),
        )
        reloader.start()
        assert reloader.is_running
        reloader.stop()
        assert not reloader.is_running

    def test_double_start(self):
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=MagicMock(),
        )
        reloader.start()
        reloader.start()  # should not crash
        assert reloader.is_running
        reloader.stop()

    def test_stop_without_start(self):
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=MagicMock(),
        )
        reloader.stop()  # should not crash


# ---------------------------------------------------------------------------
# Manual trigger
# ---------------------------------------------------------------------------

class TestManualTrigger:

    def test_trigger_reload_calls_callback(self):
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=callback,
        )
        reloader.trigger_reload()
        callback.assert_called_once_with("manual")

    def test_trigger_with_module_names(self):
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=callback,
            module_names=["json"],  # safe stdlib module
        )
        reloader.trigger_reload()
        callback.assert_called_once()

    def test_callback_error_does_not_crash(self):
        def bad_callback(trigger):
            raise RuntimeError("boom")

        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=bad_callback,
        )
        # Should not raise
        reloader.trigger_reload()


# ---------------------------------------------------------------------------
# SIGHUP handling
# ---------------------------------------------------------------------------

class TestSighupHandling:

    @pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="No SIGHUP on this platform")
    def test_sighup_triggers_reload(self):
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(
                enable_sighup=True,
                poll_interval=0.1,
                watch_paths=("/tmp/_sighup_test_sentinel",),
            ),
            on_reload=callback,
        )
        reloader.start()
        try:
            time.sleep(0.2)  # let watcher thread start
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(1.0)  # give time for watcher to process
        finally:
            reloader.stop()

        callback.assert_called()

    @pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="No SIGHUP on this platform")
    def test_sighup_handler_restored_on_stop(self):
        original = signal.getsignal(signal.SIGHUP)
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=True),
            on_reload=callback,
        )
        reloader.start()
        reloader.stop()

        current = signal.getsignal(signal.SIGHUP)
        assert current == original


# ---------------------------------------------------------------------------
# File watching
# ---------------------------------------------------------------------------

class TestFileWatching:

    def test_file_change_triggers_reload(self):
        callback = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# original")
            f.flush()
            watch_path = f.name

        try:
            reloader = ModuleReloader(
                ReloaderConfig(
                    watch_paths=(watch_path,),
                    poll_interval=0.1,
                    enable_sighup=False,
                ),
                on_reload=callback,
            )
            reloader.start()

            # Modify the file
            time.sleep(0.3)
            Path(watch_path).write_text("# modified")

            time.sleep(0.5)
            reloader.stop()

            # Should have been called at least once
            assert callback.call_count >= 1
        finally:
            os.unlink(watch_path)

    def test_nonexistent_watch_path(self):
        """Non-existent watch path should not crash."""
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(
                watch_paths=("/nonexistent/path/foo.py",),
                poll_interval=0.1,
                enable_sighup=False,
            ),
            on_reload=callback,
        )
        reloader.start()
        time.sleep(0.3)
        reloader.stop()
        # No crash, no callback
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Module reload
# ---------------------------------------------------------------------------

class TestModuleReload:

    def test_reload_stdlib_module(self):
        """Reloading a safe stdlib module succeeds."""
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=callback,
            module_names=["json"],
        )
        reloader.trigger_reload()
        callback.assert_called_once()

    def test_reload_nonexistent_module(self):
        """Reloading a non-existent module logs error but doesn't crash."""
        callback = MagicMock()
        reloader = ModuleReloader(
            ReloaderConfig(enable_sighup=False),
            on_reload=callback,
            module_names=["nonexistent_module_xyz"],
        )
        reloader.trigger_reload()
        callback.assert_called_once()
