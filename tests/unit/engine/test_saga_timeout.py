"""Test OrderSaga TTL and auto-cancellation."""
import time
from unittest.mock import MagicMock

class TestSagaTimeout:
    def test_submitted_saga_expires_after_ttl(self):
        from engine.saga import SagaManager, SagaState
        terminal_cb = MagicMock()
        mgr = SagaManager(on_terminal=terminal_cb, saga_ttl_seconds=1.0)
        mgr.create("order-1", intent_id="i1", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-1", SagaState.SUBMITTED, reason="sent")
        expired = mgr.tick()
        assert len(expired) == 0
        time.sleep(1.1)
        expired = mgr.tick()
        assert len(expired) == 1
        assert expired[0] == "order-1"
        saga = mgr.get("order-1")
        assert saga is None  # moved to _completed

    def test_filled_saga_not_expired(self):
        from engine.saga import SagaManager, SagaState
        mgr = SagaManager(saga_ttl_seconds=0.1)
        mgr.create("order-2", intent_id="i2", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-2", SagaState.SUBMITTED, reason="sent")
        mgr.transition("order-2", SagaState.FILLED, reason="filled")
        time.sleep(0.2)
        expired = mgr.tick()
        assert len(expired) == 0

    def test_tick_without_ttl_is_noop(self):
        from engine.saga import SagaManager, SagaState
        mgr = SagaManager()
        mgr.create("order-3", intent_id="i3", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-3", SagaState.SUBMITTED, reason="sent")
        expired = mgr.tick()
        assert len(expired) == 0
