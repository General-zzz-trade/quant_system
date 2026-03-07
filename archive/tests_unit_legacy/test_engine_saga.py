"""Tests for engine.saga — OrderSaga, SagaManager, state machine."""
from __future__ import annotations

from engine.saga import (
    CancelOrderAction,
    CompensatingAction,
    OrderSaga,
    SagaError,
    SagaManager,
    SagaState,
    TERMINAL_STATES,
)


# ── OrderSaga Tests ──────────────────────────────────────


class TestOrderSaga:
    def test_initial_state(self) -> None:
        saga = OrderSaga(
            order_id="o1",
            intent_id="i1",
            symbol="BTCUSDT",
            side="buy",
            qty=1.0,
        )
        assert saga.state == SagaState.PENDING
        assert saga.is_terminal is False
        assert saga.remaining_qty == 1.0
        assert saga.fill_ratio == 0.0

    def test_fill_ratio(self) -> None:
        saga = OrderSaga(
            order_id="o1", intent_id="i1", symbol="BTCUSDT",
            side="buy", qty=2.0, filled_qty=1.0,
        )
        assert saga.fill_ratio == 0.5
        assert saga.remaining_qty == 1.0

    def test_zero_qty_fill_ratio(self) -> None:
        saga = OrderSaga(
            order_id="o1", intent_id="i1", symbol="BTCUSDT",
            side="buy", qty=0.0,
        )
        assert saga.fill_ratio == 0.0

    def test_terminal_states(self) -> None:
        for s in TERMINAL_STATES:
            saga = OrderSaga(
                order_id="o1", intent_id="i1", symbol="BTCUSDT",
                side="buy", qty=1.0, state=s,
            )
            assert saga.is_terminal is True


# ── SagaManager Tests ────────────────────────────────────


class TestSagaManager:
    def test_create_and_get(self) -> None:
        mgr = SagaManager()
        saga = mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=0.1)
        assert saga.order_id == "o1"
        assert saga.intent_id == "i1"
        assert saga.state == SagaState.PENDING

        found = mgr.get("o1")
        assert found is saga

    def test_create_duplicate_raises(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=0.1)
        try:
            mgr.create("o1", "i2", symbol="BTCUSDT", side="buy", qty=0.1)
            assert False, "Should raise"
        except SagaError:
            pass

    def test_full_lifecycle(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)
        mgr.transition("o1", SagaState.FILLED, reason="full fill")
        saga = mgr.get("o1")
        assert saga is not None
        assert saga.state == SagaState.FILLED
        assert saga.is_terminal is True
        assert len(saga.history) == 3

    def test_invalid_transition_raises(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        try:
            mgr.transition("o1", SagaState.FILLED)  # PENDING → FILLED is invalid
            assert False, "Should raise"
        except SagaError as e:
            assert "invalid transition" in str(e)

    def test_unknown_saga_raises(self) -> None:
        mgr = SagaManager()
        try:
            mgr.transition("unknown", SagaState.SUBMITTED)
            assert False, "Should raise"
        except SagaError:
            pass

    def test_record_fill_partial(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=2.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)

        saga = mgr.record_fill("o1", fill_qty=0.5, fill_price=50000.0)
        assert saga.state == SagaState.PARTIAL_FILL
        assert saga.filled_qty == 0.5
        assert saga.fill_count == 1

    def test_record_fill_completes(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)

        saga = mgr.record_fill("o1", fill_qty=1.0, fill_price=50000.0)
        assert saga.state == SagaState.FILLED
        assert saga.fill_ratio == 1.0

    def test_record_fill_avg_price(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=2.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)

        mgr.record_fill("o1", fill_qty=1.0, fill_price=50000.0)
        saga = mgr.record_fill("o1", fill_qty=1.0, fill_price=52000.0)
        assert saga.state == SagaState.FILLED
        assert saga.avg_fill_price == 51000.0

    def test_reject_then_compensate(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.REJECTED, reason="price moved")

        saga = mgr.compensate("o1", reason="cleanup")
        assert saga.state == SagaState.COMPENSATED

    def test_compensate_with_action(self) -> None:
        cancelled = []

        class MockCancel(CompensatingAction):
            def execute(self, saga: OrderSaga) -> None:
                cancelled.append(saga.order_id)
                return None

        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)

        saga = mgr.compensate("o1", action=MockCancel(), reason="manual cancel")
        assert saga.state == SagaState.COMPENSATED
        assert cancelled == ["o1"]

    def test_compensate_failure(self) -> None:
        class FailAction(CompensatingAction):
            def execute(self, saga: OrderSaga) -> None:
                raise RuntimeError("venue down")

        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)

        saga = mgr.compensate("o1", action=FailAction(), reason="try cancel")
        assert saga.state == SagaState.FAILED

    def test_terminal_callback(self) -> None:
        terminals = []
        mgr = SagaManager(on_terminal=lambda s: terminals.append(s.order_id))
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.REJECTED)

        # REJECTED is not a terminal state by itself (can compensate)
        # Let's do a full terminal path
        mgr.create("o2", "i2", symbol="ETHUSDT", side="sell", qty=5.0)
        mgr.transition("o2", SagaState.SUBMITTED)
        mgr.transition("o2", SagaState.ACKED)
        mgr.transition("o2", SagaState.FILLED)

        assert "o2" in terminals

    def test_active_sagas(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.create("o2", "i2", symbol="ETHUSDT", side="sell", qty=2.0)
        mgr.transition("o1", SagaState.SUBMITTED)
        mgr.transition("o1", SagaState.ACKED)
        mgr.transition("o1", SagaState.FILLED)

        active = mgr.active_sagas()
        assert len(active) == 1
        assert active[0].order_id == "o2"

    def test_by_intent(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.create("o2", "i1", symbol="BTCUSDT", side="buy", qty=0.5)
        mgr.create("o3", "i2", symbol="ETHUSDT", side="sell", qty=2.0)

        sagas = mgr.by_intent("i1")
        assert len(sagas) == 2
        assert {s.order_id for s in sagas} == {"o1", "o2"}

    def test_by_symbol(self) -> None:
        mgr = SagaManager()
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        mgr.create("o2", "i2", symbol="ETHUSDT", side="sell", qty=2.0)

        btc = mgr.by_symbol("BTCUSDT")
        assert len(btc) == 1
        assert btc[0].order_id == "o1"

    def test_max_completed_eviction(self) -> None:
        mgr = SagaManager(max_completed=2)

        for i in range(5):
            oid = f"o{i}"
            mgr.create(oid, "i1", symbol="BTCUSDT", side="buy", qty=1.0)
            mgr.transition(oid, SagaState.SUBMITTED)
            mgr.transition(oid, SagaState.REJECTED)
            mgr.compensate(oid, reason="cleanup")

        # Only last 2 completed sagas retained
        assert mgr.get("o0") is None
        assert mgr.get("o1") is None
        assert mgr.get("o2") is None
        assert mgr.get("o3") is not None
        assert mgr.get("o4") is not None

    def test_active_count(self) -> None:
        mgr = SagaManager()
        assert mgr.active_count() == 0
        mgr.create("o1", "i1", symbol="BTCUSDT", side="buy", qty=1.0)
        assert mgr.active_count() == 1
        mgr.create("o2", "i2", symbol="ETHUSDT", side="sell", qty=2.0)
        assert mgr.active_count() == 2

    def test_cancel_order_action(self) -> None:
        cancelled = []
        action = CancelOrderAction(cancel_fn=lambda oid: cancelled.append(oid))
        saga = OrderSaga(
            order_id="o1", intent_id="i1", symbol="BTCUSDT",
            side="buy", qty=1.0, state=SagaState.COMPENSATING,
        )
        action.execute(saga)
        assert cancelled == ["o1"]
