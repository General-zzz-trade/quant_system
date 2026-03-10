"""Tests for RustOrderStateMachine — order lifecycle state transitions."""
import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import RustOrderStateMachine


@pytest.fixture
def sm():
    return RustOrderStateMachine()


# ── Registration ──

class TestRegister:
    def test_register_basic(self, sm):
        state = sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01", price="50000")
        assert state.order_id == "o1"
        assert state.symbol == "BTCUSDT"
        assert state.side == "BUY"
        assert state.order_type == "LIMIT"
        assert state.qty == "0.01"
        assert state.price == "50000"
        assert state.status == "pending_new"
        assert state.filled_qty == "0"
        assert state.avg_price is None

    def test_register_market_order_no_price(self, sm):
        state = sm.register("o1", "ETHUSDT", "SELL", "MARKET", "1.0")
        assert state.price is None
        assert state.order_type == "MARKET"

    def test_register_with_client_id(self, sm):
        state = sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01",
                            client_order_id="my_client_1")
        assert state.client_order_id == "my_client_1"

    def test_register_duplicate_raises(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        with pytest.raises(RuntimeError, match="already registered"):
            sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")

    def test_active_count(self, sm):
        assert sm.active_count() == 0
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        assert sm.active_count() == 1
        sm.register("o2", "ETHUSDT", "SELL", "MARKET", "1.0")
        assert sm.active_count() == 2


# ── Happy-path transitions ──

class TestHappyPath:
    def test_pending_to_new(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        state = sm.transition("o1", "new", ts_ms=1000)
        assert state.status == "new"
        assert state.last_update_ts == 1000

    def test_new_to_filled(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        state = sm.transition("o1", "filled", filled_qty="0.01",
                              avg_price="50100", ts_ms=2000)
        assert state.status == "filled"
        assert state.filled_qty == "0.01"
        assert state.avg_price == "50100"

    def test_partial_fill_then_filled(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "1.0")
        sm.transition("o1", "new")
        s1 = sm.transition("o1", "partially_filled", filled_qty="0.5",
                           avg_price="50000", ts_ms=1000)
        assert s1.status == "partially_filled"
        assert s1.filled_qty == "0.5"
        s2 = sm.transition("o1", "filled", filled_qty="1.0",
                           avg_price="50050", ts_ms=2000)
        assert s2.status == "filled"
        assert s2.filled_qty == "1.0"

    def test_new_to_canceled(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        state = sm.transition("o1", "canceled", ts_ms=3000, reason="user_cancel")
        assert state.status == "canceled"

    def test_pending_to_rejected(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        state = sm.transition("o1", "rejected", reason="insufficient_margin")
        assert state.status == "rejected"

    def test_new_to_expired(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        state = sm.transition("o1", "expired")
        assert state.status == "expired"

    def test_pending_cancel_to_canceled(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "pending_cancel")
        state = sm.transition("o1", "canceled")
        assert state.status == "canceled"

    def test_pending_cancel_to_filled(self, sm):
        """Cancel request arrives too late, order already filled."""
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "pending_cancel")
        state = sm.transition("o1", "filled", filled_qty="0.01", avg_price="50000")
        assert state.status == "filled"


# ── Terminal state archival ──

class TestTerminalArchival:
    def test_filled_moves_to_archive(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "filled", filled_qty="0.01")
        assert sm.active_count() == 0
        state = sm.get("o1")
        assert state is not None
        assert state.status == "filled"

    def test_canceled_moves_to_archive(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "canceled")
        assert sm.active_count() == 0
        assert sm.get("o1").status == "canceled"

    def test_rejected_moves_to_archive(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "rejected")
        assert sm.active_count() == 0

    def test_expired_moves_to_archive(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "expired")
        assert sm.active_count() == 0

    def test_active_orders_excludes_archived(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.register("o2", "ETHUSDT", "SELL", "MARKET", "1.0")
        sm.transition("o1", "rejected")
        active = sm.active_orders()
        assert len(active) == 1
        assert active[0].order_id == "o2"


# ── Illegal transitions ──

class TestIllegalTransitions:
    def test_filled_is_terminal(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new")
        sm.transition("o1", "filled", filled_qty="0.01")
        with pytest.raises(RuntimeError, match="not allowed|unknown order"):
            sm.transition("o1", "canceled")

    def test_pending_to_expired_not_allowed(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        with pytest.raises(RuntimeError, match="not allowed"):
            sm.transition("o1", "expired")

    def test_pending_to_canceled_not_allowed(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        with pytest.raises(RuntimeError, match="not allowed"):
            sm.transition("o1", "canceled")

    def test_unknown_order_raises(self, sm):
        with pytest.raises(RuntimeError, match="unknown order"):
            sm.transition("nonexistent", "new")

    def test_unknown_status_raises(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        with pytest.raises(RuntimeError, match="unknown order status"):
            sm.transition("o1", "BOGUS_STATUS")

    def test_pending_to_pending_cancel_not_allowed(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        with pytest.raises(RuntimeError, match="not allowed"):
            sm.transition("o1", "pending_cancel")


# ── Transition history ──

class TestTransitionHistory:
    def test_transitions_recorded(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.transition("o1", "new", ts_ms=100)
        sm.transition("o1", "partially_filled", filled_qty="0.005", ts_ms=200)
        state = sm.transition("o1", "filled", filled_qty="0.01", ts_ms=300)
        transitions = state.transitions
        assert len(transitions) == 3
        assert transitions[0].from_status == "pending_new"
        assert transitions[0].to_status == "new"
        assert transitions[0].ts_ms == 100
        assert transitions[1].from_status == "new"
        assert transitions[1].to_status == "partially_filled"
        assert transitions[2].from_status == "partially_filled"
        assert transitions[2].to_status == "filled"

    def test_transition_reason(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        state = sm.transition("o1", "rejected", reason="margin_insufficient")
        assert state.transitions[0].reason == "margin_insufficient"

    def test_status_normalization(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        state = sm.transition("o1", "NEW")  # uppercase
        assert state.status == "new"

    def test_repr(self, sm):
        state = sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        r = repr(state)
        assert "o1" in r
        assert "pending_new" in r
        assert "BTCUSDT" in r


# ── Concurrent orders ──

class TestConcurrentOrders:
    def test_multiple_symbols(self, sm):
        sm.register("o1", "BTCUSDT", "BUY", "LIMIT", "0.01")
        sm.register("o2", "ETHUSDT", "SELL", "MARKET", "1.0")
        sm.register("o3", "SOLUSDT", "BUY", "LIMIT", "10")
        assert sm.active_count() == 3
        sm.transition("o1", "new")
        sm.transition("o1", "filled", filled_qty="0.01")
        assert sm.active_count() == 2
        sm.transition("o2", "rejected")
        assert sm.active_count() == 1
