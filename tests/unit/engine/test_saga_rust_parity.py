"""Parity tests: RustSagaManager vs Python SagaManager."""
import pytest

try:
    from _quant_hotpath import RustSagaManager
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestSagaParity:
    def test_create_and_get(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        assert sm.get_state("ord1") == "pending"
        assert sm.active_count() == 1

    def test_transition_valid(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        result = sm.transition("ord1", "submitted", "sent", 1001.0)
        assert result == "submitted"

    def test_transition_invalid_raises(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        sm.transition("ord1", "submitted", "sent", 1001.0)
        sm.transition("ord1", "filled", "done", 1002.0)
        # FILLED is terminal -- no transitions allowed
        with pytest.raises(Exception):
            sm.transition("ord1", "submitted", "retry", 1003.0)

    def test_record_fill_partial(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 2.0, None, 1000.0)
        sm.transition("ord1", "submitted", "sent", 1001.0)
        sm.transition("ord1", "acked", "ack", 1002.0)
        state = sm.record_fill("ord1", 1.0, 3000.0, 1003.0)
        assert state == "partial_fill"

    def test_record_fill_complete(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 2.0, None, 1000.0)
        sm.transition("ord1", "submitted", "sent", 1001.0)
        sm.transition("ord1", "acked", "ack", 1002.0)
        sm.record_fill("ord1", 1.0, 3000.0, 1003.0)
        state = sm.record_fill("ord1", 1.0, 3100.0, 1004.0)
        assert state == "filled"

    def test_check_timeouts(self):
        sm = RustSagaManager(default_ttl_sec=10.0)
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        sm.transition("ord1", "submitted", "sent", 1001.0)
        # At t=1020, TTL has expired (submitted_at=1001, ttl=10, 1020-1001=19 > 10)
        expired = sm.check_timeouts(1020.0)
        assert "ord1" in expired

    def test_no_timeout_before_ttl(self):
        sm = RustSagaManager(default_ttl_sec=60.0)
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        sm.transition("ord1", "submitted", "sent", 1001.0)
        expired = sm.check_timeouts(1010.0)
        assert len(expired) == 0

    def test_by_symbol(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        sm.create("ord2", "int2", "BTCUSDT", "sell", 0.5, None, 1000.0)
        eth_orders = sm.by_symbol("ETHUSDT")
        assert "ord1" in eth_orders
        assert "ord2" not in eth_orders

    def test_duplicate_create_raises(self):
        sm = RustSagaManager()
        sm.create("ord1", "int1", "ETHUSDT", "buy", 1.0, None, 1000.0)
        with pytest.raises(Exception):
            sm.create("ord1", "int2", "ETHUSDT", "sell", 2.0, None, 1001.0)

    def test_completed_eviction(self):
        sm = RustSagaManager(max_completed=2)
        for i in range(5):
            oid = f"ord{i}"
            sm.create(oid, f"int{i}", "ETHUSDT", "buy", 1.0, None, float(1000 + i))
            sm.transition(oid, "submitted", "sent", float(1001 + i))
            sm.transition(oid, "acked", "ack", float(1002 + i))
            sm.record_fill(oid, 1.0, 3000.0, float(1003 + i))
        assert sm.completed_count() <= 2
