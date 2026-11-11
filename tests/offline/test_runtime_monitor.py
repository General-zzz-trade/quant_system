"""Hung-loop, failed reconciliation and durable-account monitoring regressions."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from runner.execution_recovery import reconcile_execution
from runner.runtime_monitor import RuntimeFreshnessGate, RuntimeMonitor
from shared.entry_risk import GateResult
from shared.execution_journal import ExecutionJournal
from shared.runtime_status import RuntimeStatusWriter, evaluate_status, read_json


class RuntimeMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now, self.monotonic = 1_800_000_000.0, 1000.0
        self.status_path = self.root / "runtime-status.json"
        writer = RuntimeStatusWriter(
            self.status_path, scope="account", state_id="a" * 32, mode="execution",
            runner_keys=["BTCUSDT", "BTCUSDT_4h"], execution_runner_keys=["BTCUSDT"],
            clock=lambda: self.now,
        )
        self.journal = ExecutionJournal(self.root / "orders.sqlite", "account")
        self.executor = NS(_ready=True, journal=self.journal)
        self.coord = NS(
            _lock=threading.RLock(), _durable_fill_consumer=NS(executor=self.executor),
            _execution_reconciliation={"ready": True, "checked_at": self.now, "monotonic_at": self.monotonic},
            _data_readiness=NS(snapshot=self.data, interval_ms=3_600_000),
        )
        self.module = NS(_execution_ready=True)
        self.coordinators = {
            "BTCUSDT": self.coord,
            "BTCUSDT_4h": NS(_lock=threading.RLock(), _data_readiness=self.coord._data_readiness),
        }
        self.monitor = RuntimeMonitor(
            writer, execution=True, execution_runner_keys=["BTCUSDT"], monotonic=lambda: self.monotonic,
            clock=lambda: self.now,
        )
        self.wallet = NS(max_age_seconds=120, snapshot=lambda: {
            "ready": True, "checked_at": self.now, "verified_through_ms": int(self.now * 1000),
            "account_secret": "excluded-wallet-detail",
        })
        self.monitor.bind(
            self.coordinators, {"BTCUSDT": self.module, "BTCUSDT_4h": NS()},
            portfolio=NS(max_age_seconds=300, snapshot=lambda: {
                "ready": True, "age_seconds": 1, "reservations": "excluded-detail",
            }),
            wallet=self.wallet,
        )
        self.downstream = Mock()
        self.downstream.check.return_value = GateResult(True)
        self.gate = RuntimeFreshnessGate(self.monitor, self.downstream)
        self.order = NS(qty="1", price="100", symbol="BTCUSDT", side="buy", reduce_only=False)

    def data(self):
        return {"status": "ready", "ready": True, "mode": "live", "bar_ts_ms": int(self.now * 1000),
                "required_warmup_bars": 1, "valid_warmup_bars": 1, "open_interest": 100.0, "ls_ratio": 1.0,
                "oi_event_ts_ms": int(self.now * 1000), "ls_event_ts_ms": int(self.now * 1000),
                "received_at_ms": int(self.now * 1000), "external_conditions": {"batch_inputs": {"ready": True}}}

    def test_current_actual_observations_allow_existing_gate_and_ready_health(self):
        self.assertFalse(self.gate.check(self.order).allowed)
        result = self.monitor.publish("running")
        self.assertTrue(result["execution_eligible"])
        self.assertTrue(self.gate.check(self.order).allowed)
        self.downstream.check.assert_called_once_with(self.order)
        status = read_json(self.status_path)
        self.assertEqual(status["execution_runner_keys"], ["BTCUSDT"])
        self.assertEqual(set(status["data"]), set(self.coordinators))
        self.assertNotIn("excluded", self.status_path.read_text())

    def test_websocket_entry_is_blocked_when_mainloop_heartbeat_stalls(self):
        self.monitor.publish("running")
        self.monotonic += 31
        # Even a still-active callback refreshing other observations cannot
        # renew the heartbeat owned by the main loop.
        self.coord._execution_reconciliation["monotonic_at"] = self.monotonic
        self.assertFalse(self.gate.check(self.order).allowed)
        self.downstream.check.assert_not_called()
        self.now += 31
        self.assertFalse(evaluate_status(read_json(self.status_path), now=self.now)["live"])

    def test_collection_preserves_stricter_expiry_and_original_risk_capture_time(self):
        self.monitor.portfolio.max_age_seconds = 10
        self.wallet.max_age_seconds = 5
        result = self.monitor.publish("running")
        self.assertTrue(result["execution_eligible"])
        status = read_json(self.status_path)
        self.assertEqual(status["execution"]["risk"]["max_age_seconds"], 10)
        self.assertEqual(status["execution"]["wallet"]["max_age_seconds"], 5)
        self.assertFalse(evaluate_status(status, now=self.now + 6)["execution_eligible"])

        def slow_risk():
            self.now += 11
            return {"ready": True, "age_seconds": 0}
        self.monitor.portfolio.snapshot = slow_risk
        self.assertFalse(self.monitor.publish("running")["execution_eligible"])

    def test_recent_heartbeat_cannot_hide_stale_or_failed_reconciliation(self):
        self.monotonic += 121
        self.monitor.publish("running")
        self.assertFalse(self.gate.check(self.order).allowed)
        self.coord._execution_reconciliation = {
            "ready": False, "checked_at": self.now, "monotonic_at": self.monotonic,
        }
        self.assertFalse(self.gate.check(self.order).allowed)
        self.coord._execution_reconciliation["ready"] = True
        self.module._execution_ready = False
        self.assertFalse(self.gate.check(self.order).allowed)
        self.downstream.check.assert_not_called()

    def test_stale_runtime_preserves_reduction_validation_and_never_reserves_entries(self):
        for qty, reduce_only in (("0", False), ("1", True)):
            self.order.qty, self.order.reduce_only = qty, reduce_only
            self.assertTrue(self.gate.check(self.order).allowed)
        self.assertEqual(self.downstream.check.call_count, 2)
        self.downstream.check.return_value = GateResult(False, "underlying rejection")
        self.assertFalse(self.gate.check(self.order).allowed)
        self.order.qty = "NaN"
        self.assertFalse(self.gate.check(self.order).allowed)

    def test_failed_snapshot_or_publication_withdraws_entry_permission(self):
        self.monitor.publish("running")
        self.wallet.snapshot = Mock(side_effect=OSError("state unavailable"))
        with self.assertRaises(OSError):
            self.monitor.publish("running")
        self.assertFalse(self.gate.check(self.order).allowed)
        self.wallet.snapshot = lambda: {"ready": False, "checked_at": None, "verified_through_ms": None}
        with patch.object(self.monitor.writer, "publish", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.monitor.publish("running")
        self.assertFalse(self.gate.check(self.order).allowed)

    def test_lifecycle_and_backwards_monotonic_clock_revoke_entries(self):
        self.monitor.publish("running")
        self.monotonic -= 1
        self.assertFalse(self.gate.check(self.order).allowed)
        for phase in ("stopping", "stopped", "starting"):
            result = self.monitor.publish(phase)
            self.assertFalse(result["ready"])
            self.assertFalse(self.gate.check(self.order).allowed)

    def test_account_counts_include_unowned_unknowns_and_do_not_double_count(self):
        second = NS(**vars(self.coord))
        self.coordinators["BTCUSDT_4h"] = second
        self.monitor.execution_runner_keys = ("BTCUSDT", "BTCUSDT_4h")
        self.monitor.modules["BTCUSDT_4h"] = self.module
        with self.journal.connect() as db:
            db.execute("INSERT INTO submissions(scope,intent_id,settled,status,owner) VALUES(?,?,?,?,?)",
                       ("account", "hidden-order-id", 0, "unknown", ""))
            db.execute("INSERT INTO execution_fills VALUES(?,?,?,?,?,0)",
                       ("account", "hidden-fill-id", "hidden-order-id", "unowned", json.dumps({})))
            db.execute("INSERT INTO submissions(scope,intent_id,settled,status) VALUES(?,?,?,?)",
                       ("other-account", "unrelated", 0, "unknown"))
        evidence = self.monitor._execution_snapshot()
        self.assertEqual(evidence["journal"], {"unresolved": 1, "unknown": 1, "unconsumed": 1})
        self.assertNotIn("hidden", json.dumps(evidence))

    def test_missing_journal_is_not_recreated_by_monitoring(self):
        self.journal.path.unlink()
        with self.assertRaises(sqlite3.OperationalError):
            self.monitor._execution_snapshot()
        self.assertFalse(self.journal.path.exists())


class ReconciliationObservationTests(unittest.TestCase):
    def setUp(self):
        self.executor = NS(_ready=True, reconcile_pending=lambda: [], notify_status=lambda: None)
        self.coord = NS(_lock=threading.RLock(), _durable_fill_consumer=NS(executor=self.executor))
        self.module = NS(_execution_ready=True, _symbol="BTCUSDT")

    def test_success_uses_query_start_time_and_failure_clears_previous_success(self):
        def reconcile(_):
            self.executor._ready = True
            return []
        self.executor.reconcile_account = reconcile
        with patch("runner.execution_recovery.time.time", return_value=100), \
                patch("runner.execution_recovery.time.monotonic", return_value=200):
            self.assertEqual(reconcile_execution(self.coord, self.module), [])
        self.assertEqual(self.coord._execution_reconciliation,
                         {"ready": True, "checked_at": 100, "monotonic_at": 200})
        self.executor.reconcile_account = Mock(side_effect=TimeoutError("account unavailable"))
        with self.assertRaises(TimeoutError):
            reconcile_execution(self.coord, self.module)
        self.assertFalse(self.coord._execution_reconciliation["ready"])
        self.assertFalse(self.module._execution_ready)
        self.assertFalse(self.executor._ready)

    def test_reconciliation_issues_never_report_success(self):
        self.executor.reconcile_account = lambda _: ["unknown order"]
        self.assertEqual(reconcile_execution(self.coord, self.module), ["unknown order"])
        self.assertFalse(self.coord._execution_reconciliation["ready"])
        self.assertFalse(self.module._execution_ready)


if __name__ == "__main__":
    unittest.main()
