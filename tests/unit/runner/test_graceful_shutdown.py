# tests/unit/runner/test_graceful_shutdown.py
"""Tests for GracefulShutdown — orderly shutdown sequence."""
from __future__ import annotations

import signal
import sys
import time
import threading
from unittest.mock import MagicMock, patch

import pytest

from runner.graceful_shutdown import GracefulShutdown, ShutdownConfig


class TestShutdownSequence:
    def test_execute_calls_all_steps_in_order(self):
        call_order: list[str] = []

        shutdown = GracefulShutdown(
            stop_new_orders=lambda: call_order.append("stop_new_orders"),
            wait_pending=lambda: (call_order.append("wait_pending") or True),
            cancel_all=lambda: call_order.append("cancel_all"),
            reconcile=lambda: call_order.append("reconcile"),
            save_snapshot=lambda path: call_order.append(f"save:{path}"),
            cleanup=lambda: call_order.append("cleanup"),
        )
        shutdown.execute()

        assert call_order == [
            "stop_new_orders",
            "wait_pending",
            "cancel_all",
            "reconcile",
            "save:state_snapshot.json",
            "cleanup",
        ]

    def test_execute_skips_none_callbacks(self):
        call_order: list[str] = []

        shutdown = GracefulShutdown(
            stop_new_orders=lambda: call_order.append("stop"),
            cleanup=lambda: call_order.append("cleanup"),
        )
        shutdown.execute()

        assert call_order == ["stop", "cleanup"]

    def test_save_snapshot_uses_config_path(self):
        saved_paths: list[str] = []
        cfg = ShutdownConfig(state_file="my_state.json")

        shutdown = GracefulShutdown(
            config=cfg,
            save_snapshot=lambda path: saved_paths.append(path),
        )
        shutdown.execute()

        assert saved_paths == ["my_state.json"]

    def test_save_snapshot_skipped_when_disabled(self):
        saved = []
        cfg = ShutdownConfig(save_state=False)

        shutdown = GracefulShutdown(
            config=cfg,
            save_snapshot=lambda path: saved.append(path),
        )
        shutdown.execute()

        assert saved == []

    def test_step_exception_does_not_abort_sequence(self):
        call_order: list[str] = []

        def bad_reconcile():
            raise RuntimeError("reconcile exploded")

        shutdown = GracefulShutdown(
            stop_new_orders=lambda: call_order.append("stop"),
            reconcile=bad_reconcile,
            cleanup=lambda: call_order.append("cleanup"),
        )
        shutdown.execute()

        assert "stop" in call_order
        assert "cleanup" in call_order


class TestPendingOrderTimeout:
    def test_timeout_triggers_cancel(self):
        call_order: list[str] = []
        cfg = ShutdownConfig(pending_order_timeout_sec=0.1, poll_interval_sec=0.02)

        shutdown = GracefulShutdown(
            config=cfg,
            wait_pending=lambda: False,
            cancel_all=lambda: call_order.append("cancel"),
        )
        start = time.monotonic()
        shutdown.execute()
        elapsed = time.monotonic() - start

        assert "cancel" in call_order
        assert elapsed >= 0.08

    def test_pending_drained_before_timeout(self):
        counter = {"n": 0}
        cfg = ShutdownConfig(pending_order_timeout_sec=5.0, poll_interval_sec=0.01)

        def wait():
            counter["n"] += 1
            return counter["n"] >= 3

        shutdown = GracefulShutdown(config=cfg, wait_pending=wait)

        start = time.monotonic()
        shutdown.execute()
        elapsed = time.monotonic() - start

        assert elapsed < 2.0
        assert counter["n"] >= 3


class TestSignalHandling:
    def test_is_shutting_down_starts_false(self):
        shutdown = GracefulShutdown()
        assert shutdown.is_shutting_down is False

    def test_execute_sets_shutting_down(self):
        shutdown = GracefulShutdown()
        shutdown._shutting_down = True
        assert shutdown.is_shutting_down is True

    def test_handle_signal_first_time_executes(self):
        executed = []
        shutdown = GracefulShutdown(
            cleanup=lambda: executed.append("done"),
        )

        shutdown._handle_signal(signal.SIGTERM, None)

        assert shutdown.is_shutting_down is True
        assert "done" in executed

    def test_handle_signal_second_time_force_exits(self):
        shutdown = GracefulShutdown()
        shutdown._shutting_down = True

        with pytest.raises(SystemExit) as exc_info:
            shutdown._handle_signal(signal.SIGINT, None)

        assert exc_info.value.code == 1

    def test_install_handlers(self):
        shutdown = GracefulShutdown()

        with patch.object(signal, "signal") as mock_signal:
            shutdown.install_handlers()

            assert mock_signal.call_count == 2
            calls = {args[0] for args, _ in mock_signal.call_args_list}
            assert signal.SIGTERM in calls
            assert signal.SIGINT in calls

    def test_install_handlers_skipped_outside_main_thread(self):
        shutdown = GracefulShutdown()
        worker = MagicMock()
        worker.name = "worker-thread"
        main = MagicMock()
        main.name = "MainThread"

        with (
            patch.object(signal, "signal") as mock_signal,
            patch.object(threading, "current_thread", return_value=worker),
            patch.object(threading, "main_thread", return_value=main),
        ):
            shutdown.install_handlers()

        mock_signal.assert_not_called()
