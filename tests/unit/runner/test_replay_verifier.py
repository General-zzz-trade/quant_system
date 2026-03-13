"""Tests for runner/replay_verifier.py and replay adapter balance tracking."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from runner.replay_verifier import ReplayVerifier, Violation, VerificationResult
from runner.replay_runner import ReplayResult


# ── Verifier tests ──────────────────────────────────────────


class TestMarginVerification:
    def test_margin_violation_detected(self):
        """Snapshot with leverage > max triggers a violation."""
        verifier = ReplayVerifier()
        snapshots = [
            {"ts": 1.0, "balance": 1000.0, "position_notional": 15_000.0},
        ]
        result = verifier.verify_margin_never_exceeded(snapshots, max_leverage=10.0)
        assert not result.passed
        assert len(result.violations) == 1
        assert "15.0x" in result.violations[0].message

    def test_margin_ok(self):
        """Normal snapshots within leverage limit pass."""
        verifier = ReplayVerifier()
        snapshots = [
            {"ts": 1.0, "balance": 10_000.0, "position_notional": 50_000.0},
            {"ts": 2.0, "balance": 10_000.0, "position_notional": 80_000.0},
        ]
        result = verifier.verify_margin_never_exceeded(snapshots, max_leverage=10.0)
        assert result.passed
        assert len(result.violations) == 0


class TestBalanceTracking:
    def test_balance_tracking(self):
        """ReplayExecutionAdapter deducts fees from starting balance."""
        from execution.sim.replay_adapter import ReplayExecutionAdapter

        adapter = ReplayExecutionAdapter(
            price_source=lambda sym: Decimal("50000"),
            starting_balance=10_000.0,
        )

        assert adapter._balance == 10_000.0

        # Create a mock order event
        order = SimpleNamespace(
            symbol="BTCUSDT",
            side="BUY",
            qty=Decimal("0.01"),
            order_id="test-001",
        )

        fills = adapter.send_order(order)
        assert len(fills) == 1

        # Balance should decrease by fee amount
        assert adapter._balance < 10_000.0

        # Fee = fill_price * qty * fee_bps / 10000
        # fill_price ~ 50000 * (1 + 2/10000) = 50010
        # fee ~ 50010 * 0.01 * 4/10000 ~ 0.20004
        snap = adapter.get_account_snapshot()
        assert snap["balance"] < 10_000.0
        assert snap["num_fills"] == 1
        assert snap["starting_balance"] == 10_000.0

        # Account snapshots list should have one entry
        assert len(adapter.account_snapshots) == 1
        assert adapter.account_snapshots[0]["symbol"] == "BTCUSDT"


class TestDeterminism:
    def test_deterministic_replay(self):
        """Two identical ReplayResults are detected as deterministic."""
        verifier = ReplayVerifier()
        orders = [
            {"symbol": "BTCUSDT", "side": "BUY", "qty": "0.01"},
            {"symbol": "ETHUSDT", "side": "SELL", "qty": "0.1"},
        ]
        a = ReplayResult(events_processed=100, order_log=orders)
        b = ReplayResult(events_processed=100, order_log=list(orders))
        assert verifier.verify_deterministic(a, b) is True

    def test_non_deterministic_replay(self):
        """Two different ReplayResults are detected as non-deterministic."""
        verifier = ReplayVerifier()
        a = ReplayResult(
            events_processed=100,
            order_log=[{"symbol": "BTCUSDT", "side": "BUY", "qty": "0.01"}],
        )
        b = ReplayResult(
            events_processed=100,
            order_log=[{"symbol": "BTCUSDT", "side": "SELL", "qty": "0.01"}],
        )
        assert verifier.verify_deterministic(a, b) is False

    def test_non_deterministic_event_count(self):
        """Different event counts are non-deterministic."""
        verifier = ReplayVerifier()
        a = ReplayResult(events_processed=100, order_log=[])
        b = ReplayResult(events_processed=99, order_log=[])
        assert verifier.verify_deterministic(a, b) is False
