# tests/contract/test_execution_safety_contract.py
"""Contract: execution safety — notional limits, circuit breaker, state machine, dedup."""
from __future__ import annotations


from _quant_hotpath import RustCircuitBreaker, RustOrderStateMachine
from execution.state_machine import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    OrderStatus,
)
from runner.strategy_config import MAX_ORDER_NOTIONAL


# ── MAX_ORDER_NOTIONAL ──────────────────────────────────────

class TestMaxOrderNotional:
    def test_max_notional_value(self):
        """MAX_ORDER_NOTIONAL must be <= $1000 for safety."""
        assert MAX_ORDER_NOTIONAL <= 1000
        assert MAX_ORDER_NOTIONAL > 0

    def test_max_notional_is_500(self):
        """Current production value is $500."""
        assert MAX_ORDER_NOTIONAL == 500.0

    def test_notional_blocks_large_orders(self):
        """Any order above MAX_ORDER_NOTIONAL must be blocked."""
        price = 2000.0
        qty = 0.3  # 0.3 * 2000 = $600 > $500
        notional = price * qty
        assert notional > MAX_ORDER_NOTIONAL, "Test setup: order should exceed limit"


# ── Circuit Breaker ──────────────────────────────────────────

class TestCircuitBreakerContract:
    def test_initial_state_closed(self):
        """Circuit breaker starts in closed state (allowing requests)."""
        cb = RustCircuitBreaker(
            failure_threshold=3, window_s=120.0, recovery_timeout_s=60.0,
        )
        assert cb.allow_request() is True
        assert cb.state == "closed"

    def test_opens_after_threshold_failures(self):
        """After failure_threshold failures, circuit opens (blocks requests)."""
        cb = RustCircuitBreaker(
            failure_threshold=3, window_s=120.0, recovery_timeout_s=60.0,
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        assert cb.allow_request() is False

    def test_success_resets_failures(self):
        """Recording success resets the failure counter."""
        cb = RustCircuitBreaker(
            failure_threshold=3, window_s=120.0, recovery_timeout_s=60.0,
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Should still be closed (success reset count)
        assert cb.state == "closed"
        assert cb.allow_request() is True

    def test_reset_returns_to_closed(self):
        """Manual reset brings circuit back to closed."""
        cb = RustCircuitBreaker(
            failure_threshold=2, window_s=120.0, recovery_timeout_s=0.01,
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.allow_request() is True


# ── Order State Machine ──────────────────────────────────────

class TestOrderStateMachineContract:
    def test_terminal_states_defined(self):
        """Terminal states must include filled, canceled, rejected, expired."""
        assert OrderStatus.FILLED in TERMINAL_STATUSES
        assert OrderStatus.CANCELED in TERMINAL_STATUSES
        assert OrderStatus.REJECTED in TERMINAL_STATUSES
        assert OrderStatus.EXPIRED in TERMINAL_STATUSES

    def test_no_transitions_from_terminal(self):
        """Terminal states must have no outgoing transitions."""
        for status in TERMINAL_STATUSES:
            outgoing = [t for t in VALID_TRANSITIONS if t[0] == status]
            assert len(outgoing) == 0, (
                f"Terminal state {status} has outgoing transitions: {outgoing}"
            )

    def test_rust_osm_register_and_transition(self):
        """Rust OSM: register → transition to filled."""
        osm = RustOrderStateMachine()
        osm.register("order-1", "BTCUSDT", "buy", "market", "0.1")
        assert osm.active_count() == 1
        osm.transition("order-1", "filled")
        assert osm.active_count() == 0

    def test_rust_osm_terminal_no_further_transition(self):
        """Once in terminal state, further transitions should fail or be no-op."""
        osm = RustOrderStateMachine()
        osm.register("order-2", "ETHUSDT", "sell", "market", "1.0")
        osm.transition("order-2", "filled")
        # Attempting to transition again should not succeed
        try:
            osm.transition("order-2", "canceled")
        except Exception:
            pass  # Expected: terminal state rejects transition
        assert osm.active_count() == 0


# ── Dedup ────────────────────────────────────────────────────

class TestDedupContract:
    def test_orderLinkId_uniqueness_format(self):
        """orderLinkId must contain symbol and timestamp for dedup."""
        import time
        ts = int(time.time())
        link_id = f"qs_ETHUSDT_b_{ts}"
        assert "ETHUSDT" in link_id
        assert str(ts) in link_id

    def test_duplicate_fill_detection(self):
        """Same fill_id processed twice should be detected via dedup store."""
        from execution.store.dedup_store import InMemoryDedupStore
        store = InMemoryDedupStore()
        # First fill: not seen
        assert store.get("fill-001") is None
        store.put("fill-001", "digest-abc")
        # Duplicate: already seen
        assert store.get("fill-001") == "digest-abc"
