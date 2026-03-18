# tests/unit/execution/test_reconcile_scheduler.py
"""Tests for ReconcileScheduler."""
from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List


from execution.reconcile.controller import ReconcileController
from execution.reconcile.controller import ReconcileReport
from execution.reconcile.scheduler import (
    ReconcileScheduler,
    ReconcileSchedulerConfig,
)


# ── Stubs ────────────────────────────────────────────────────

def _local_state(
    *,
    positions: Dict[str, Any] | None = None,
    balance: str = "10000",
) -> Dict[str, Any]:
    pos = {}
    if positions:
        for sym, qty in positions.items():
            pos[sym] = SimpleNamespace(qty=Decimal(qty))
    return {
        "positions": pos,
        "account": SimpleNamespace(balance=Decimal(balance), currency="USDT"),
    }


def _venue_state(
    *,
    positions: Dict[str, str] | None = None,
    balances: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if positions:
        result["positions"] = {k: Decimal(v) for k, v in positions.items()}
    if balances:
        result["balances"] = {k: Decimal(v) for k, v in balances.items()}
    return result


# ── Tests: Basic reconciliation ──────────────────────────────

class TestRunOnce:
    def test_matching_state_ok(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(
                positions={"BTCUSDT": "1.0"}, balance="10000",
            ),
            fetch_venue_state=lambda: _venue_state(
                positions={"BTCUSDT": "1.0"}, balances={"USDT": "10000"},
            ),
        )
        report = scheduler.run_once()
        assert report is not None
        assert report.ok

    def test_position_drift_detected(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "1.0"}),
            fetch_venue_state=lambda: _venue_state(positions={"BTCUSDT": "0.5"}),
        )
        report = scheduler.run_once()
        assert report is not None
        assert not report.ok
        assert len(report.all_drifts) > 0

    def test_empty_venue_positions_are_authoritative(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "1.0"}),
            fetch_venue_state=lambda: {"positions": {}},
        )
        report = scheduler.run_once()
        assert report is not None
        assert not report.ok
        assert any("not on venue" in drift.detail for drift in report.all_drifts)

    def test_balance_drift_detected(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(balance="10000"),
            fetch_venue_state=lambda: _venue_state(balances={"USDT": "9000"}),
        )
        report = scheduler.run_once()
        assert report is not None
        assert not report.ok

    def test_no_venue_data_returns_ok(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(),
            fetch_venue_state=lambda: {},  # no positions/balances
        )
        report = scheduler.run_once()
        assert report is not None
        # No data to compare → ok
        assert report.ok

    def test_fetch_error_returns_none(self):
        def _fail():
            raise ConnectionError("network error")

        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(),
            fetch_venue_state=_fail,
        )
        report = scheduler.run_once()
        assert report is None


# ── Tests: Halt/Alert callbacks ──────────────────────────────

class TestCallbacks:
    def test_halt_callback_on_critical(self):
        halted: List[Any] = []

        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "10.0"}),
            fetch_venue_state=lambda: _venue_state(positions={"BTCUSDT": "0.0"}),
            cfg=ReconcileSchedulerConfig(halt_on_critical=True),
            on_halt=lambda r: halted.append(r),
        )
        report = scheduler.run_once()
        assert report is not None
        # The drift is critical (expected=10, actual=0, diff >> 10% of expected)
        assert report.should_halt
        assert len(halted) == 1

    def test_alert_callback_on_warning(self):
        alerts: List[Any] = []

        controller = ReconcileController(qty_tolerance=Decimal("0.00001"))
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "1.0"}),
            fetch_venue_state=lambda: _venue_state(positions={"BTCUSDT": "0.999"}),
            on_alert=lambda r: alerts.append(r),
        )
        report = scheduler.run_once()
        assert report is not None
        if not report.ok and not report.should_halt:
            assert len(alerts) == 1

    def test_no_callback_on_ok(self):
        halted: List[Any] = []
        alerts: List[Any] = []

        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "1.0"}),
            fetch_venue_state=lambda: _venue_state(positions={"BTCUSDT": "1.0"}),
            on_halt=lambda r: halted.append(r),
            on_alert=lambda r: alerts.append(r),
        )
        scheduler.run_once()
        assert len(halted) == 0
        assert len(alerts) == 0


# ── Tests: Periodic scheduling ───────────────────────────────

class TestPeriodicScheduling:
    def test_start_stop_lifecycle(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(),
            fetch_venue_state=lambda: {},
            cfg=ReconcileSchedulerConfig(interval_sec=0.02),
        )
        scheduler.start()
        assert scheduler._running is True
        time.sleep(0.05)
        scheduler.stop()
        assert scheduler._running is False

    def test_periodic_run(self):
        call_count = 0
        orig_local = _local_state(positions={"BTCUSDT": "1.0"})

        def _fetch():
            nonlocal call_count
            call_count += 1
            return _venue_state(positions={"BTCUSDT": "1.0"})

        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: orig_local,
            fetch_venue_state=_fetch,
            cfg=ReconcileSchedulerConfig(interval_sec=0.02),
        )
        scheduler.start()
        time.sleep(0.1)
        scheduler.stop()
        assert call_count >= 2  # at least 2 runs in 0.1s with 0.02s interval

    def test_last_report_updated(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: _local_state(positions={"BTCUSDT": "1.0"}),
            fetch_venue_state=lambda: _venue_state(positions={"BTCUSDT": "1.0"}),
            cfg=ReconcileSchedulerConfig(interval_sec=0.02),
        )
        assert scheduler.last_report is None
        scheduler.start()
        deadline = time.monotonic() + 1.0
        while scheduler.last_report is None and time.monotonic() < deadline:
            time.sleep(0.01)
        scheduler.stop()
        assert scheduler.last_report is not None


# ── Tests: State extraction ──────────────────────────────────

class TestStateExtraction:
    def test_extract_positions(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: {},
            fetch_venue_state=lambda: {},
        )
        state = _local_state(positions={"BTCUSDT": "1.5", "ETHUSDT": "10.0"})
        result = scheduler._extract_local_positions(state)
        assert result == {"BTCUSDT": Decimal("1.5"), "ETHUSDT": Decimal("10.0")}

    def test_extract_balances(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: {},
            fetch_venue_state=lambda: {},
        )
        state = _local_state(balance="9500")
        result = scheduler._extract_local_balances(state)
        assert result == {"USDT": Decimal("9500")}

    def test_extract_empty_positions(self):
        controller = ReconcileController()
        scheduler = ReconcileScheduler(
            controller=controller,
            get_local_state=lambda: {},
            fetch_venue_state=lambda: {},
        )
        result = scheduler._extract_local_positions({})
        assert result == {}

    def test_run_once_passes_orders_and_fill_ids_when_present(self):
        captured = {}

        class _CapturingController:
            def reconcile(self, **kwargs):
                captured.update(kwargs)
                return ReconcileReport(venue=kwargs["venue"])

        scheduler = ReconcileScheduler(
            controller=_CapturingController(),
            get_local_state=lambda: {
                **_local_state(positions={"BTCUSDT": "1.0"}),
                "orders": {"ord-1": SimpleNamespace(status="new")},
                "fills": [SimpleNamespace(fill_id="fill-1")],
            },
            fetch_venue_state=lambda: {
                "positions": {},
                "orders": {"ord-1": "filled"},
                "fill_ids": {"fill-1"},
            },
        )

        report = scheduler.run_once()

        assert report is not None
        assert captured["local_orders"] == {"ord-1": "new"}
        assert captured["venue_orders"] == {"ord-1": "filled"}
        assert captured["local_fill_ids"] == {"fill-1"}
        assert captured["venue_fill_ids"] == {"fill-1"}
