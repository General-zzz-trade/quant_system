# tests/unit/execution/test_algo_adapter.py
"""Tests for AlgoExecutionAdapter."""
from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, List, Optional


from execution.algo_adapter import AlgoConfig, AlgoExecutionAdapter, _make_fill_event


# ── Stubs ────────────────────────────────────────────────────

def _order_event(
    *,
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: str = "1.0",
    price: str | None = "40000",
    order_id: str = "ord-001",
) -> SimpleNamespace:
    return SimpleNamespace(
        event_type="order",
        order_id=order_id,
        intent_id="int-001",
        symbol=symbol,
        side=side,
        qty=Decimal(qty),
        price=Decimal(price) if price else None,
    )


class _FakeSubmitFn:
    """Records submissions and returns configurable price."""

    def __init__(self, fill_price: Decimal = Decimal("40000")):
        self.calls: List[tuple] = []
        self.fill_price = fill_price

    def __call__(self, symbol: str, side: str, qty: Decimal) -> Optional[Decimal]:
        self.calls.append((symbol, side, qty))
        return self.fill_price


class _RaisingIncidentLogger:
    def __call__(self, _alert: Any) -> None:
        raise RuntimeError("boom")


# ── Tests: Routing ───────────────────────────────────────────

class TestOrderRouting:
    def test_small_order_direct_execution(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        emitted: List[Any] = []
        cfg = AlgoConfig(large_order_notional=Decimal("50000"))

        adapter = AlgoExecutionAdapter(
            submit_fn=submit, dispatcher_emit=emitted.append, cfg=cfg,
        )

        # 1 BTC * 40000 = 40000 < 50000 threshold → direct
        order = _order_event(qty="1.0", price="40000")
        results = list(adapter.send_order(order))

        assert len(results) == 1
        assert results[0].event_type == "FILL"
        assert results[0].qty == 1.0
        assert results[0].quantity == 1.0
        assert results[0].price == 40000.0
        assert len(submit.calls) == 1

    def test_large_order_routed_to_algo(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        emitted: List[Any] = []
        cfg = AlgoConfig(large_order_notional=Decimal("10000"))

        adapter = AlgoExecutionAdapter(
            submit_fn=submit, dispatcher_emit=emitted.append, cfg=cfg,
        )

        # 1 BTC * 40000 = 40000 > 10000 threshold → algo
        order = _order_event(qty="1.0", price="40000")
        results = list(adapter.send_order(order))

        # No immediate fills (async via algo)
        assert len(results) == 0
        # But algo order should be active
        assert len(adapter._active_orders) == 1
        thread = adapter._thread
        adapter.stop()
        assert adapter._thread is None
        assert thread is not None
        assert not thread.is_alive()

    def test_no_price_uses_qty_as_notional(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        emitted: List[Any] = []
        cfg = AlgoConfig(large_order_notional=Decimal("5"))

        adapter = AlgoExecutionAdapter(
            submit_fn=submit, dispatcher_emit=emitted.append, cfg=cfg,
        )

        # No price → notional = qty (1.0) < 5 → direct
        order = _order_event(qty="1.0", price=None)
        results = list(adapter.send_order(order))
        assert len(results) == 1

    def test_submit_returns_none_no_fill(self):
        def fail_submit(sym, side, qty):
            return None

        adapter = AlgoExecutionAdapter(
            submit_fn=fail_submit,
            dispatcher_emit=lambda e: None,
            cfg=AlgoConfig(large_order_notional=Decimal("999999")),
        )
        results = list(adapter.send_order(_order_event()))
        assert len(results) == 0

    def test_small_order_emits_synthetic_fill_to_incident_logger(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        incidents: List[Any] = []
        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=lambda e: None,
            incident_logger=incidents.append,
            cfg=AlgoConfig(large_order_notional=Decimal("50000")),
        )

        results = list(adapter.send_order(_order_event(qty="1.0", price="40000")))

        assert len(results) == 1
        assert len(incidents) == 1
        assert incidents[0].title == "execution-synthetic-fill"
        assert incidents[0].meta["category"] == "execution_fill"

    def test_incident_logger_failure_does_not_break_direct_fill(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=lambda e: None,
            incident_logger=_RaisingIncidentLogger(),
            cfg=AlgoConfig(large_order_notional=Decimal("50000")),
        )

        results = list(adapter.send_order(_order_event(qty="1.0", price="40000")))

        assert len(results) == 1


# ── Tests: Fill event structure ──────────────────────────────

class TestFillEventStructure:
    def test_fill_event_fields(self):
        order = _order_event(symbol="ETHUSDT", side="sell", order_id="o-123")
        fill = _make_fill_event(order, Decimal("3000"), Decimal("5.0"))

        assert fill.event_type == "FILL"
        assert fill.symbol == "ETHUSDT"
        assert fill.side == "sell"
        assert fill.qty == 5.0
        assert fill.quantity == 5.0
        assert fill.price == 3000.0
        assert fill.order_id == "o-123"
        assert fill.fill_id.startswith("algo-fill-")
        assert fill.payload_digest
        assert len(fill.payload_digest) == 16
        assert fill.fee == 0.0
        assert fill.header.event_type == "FILL"
        assert fill.header.event_id is None

    def test_fill_event_identity_uses_fill_sequence(self):
        order = _order_event(symbol="ETHUSDT", side="sell", order_id="o-123")

        fill1 = _make_fill_event(order, Decimal("3000"), Decimal("5.0"), fill_seq=1)
        fill2 = _make_fill_event(order, Decimal("3000"), Decimal("5.0"), fill_seq=1)
        fill3 = _make_fill_event(order, Decimal("3000"), Decimal("5.0"), fill_seq=2)

        assert fill1.fill_id == fill2.fill_id
        assert fill1.payload_digest == fill2.payload_digest
        assert fill3.fill_id != fill1.fill_id


# ── Tests: Algo creation ────────────────────────────────────

class TestAlgoCreation:
    def test_twap_algo_created(self):
        submit = _FakeSubmitFn()
        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=lambda e: None,
            cfg=AlgoConfig(
                large_order_notional=Decimal("100"),
                default_algo="twap",
                twap_slices=5,
                twap_duration_sec=10,
                tick_interval_sec=0.01,
            ),
        )
        order = _order_event(qty="1.0", price="1000")
        adapter.send_order(order)

        assert len(adapter._active_orders) == 1
        _, algo_order = list(adapter._active_orders.values())[0]
        assert len(algo_order.slices) == 5
        adapter.stop()

    def test_vwap_algo_created(self):
        submit = _FakeSubmitFn()
        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=lambda e: None,
            cfg=AlgoConfig(
                large_order_notional=Decimal("100"),
                default_algo="vwap",
                vwap_slices=3,
                tick_interval_sec=0.01,
            ),
        )
        order = _order_event(qty="1.0", price="1000")
        adapter.send_order(order)

        _, algo_order = list(adapter._active_orders.values())[0]
        assert len(algo_order.slices) == 3
        adapter.stop()

    def test_iceberg_algo_created(self):
        submit = _FakeSubmitFn()
        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=lambda e: None,
            cfg=AlgoConfig(
                large_order_notional=Decimal("100"),
                default_algo="iceberg",
                iceberg_clip_fraction=0.25,
                tick_interval_sec=0.01,
            ),
        )
        order = _order_event(qty="1.0", price="1000")
        adapter.send_order(order)

        _, algo_order = list(adapter._active_orders.values())[0]
        assert algo_order.clip_size == Decimal("0.25")
        adapter.stop()


# ── Tests: Async tick loop ───────────────────────────────────

class TestTickLoop:
    def test_twap_fills_emitted_asynchronously(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        emitted: List[Any] = []
        cfg = AlgoConfig(
            large_order_notional=Decimal("100"),
            default_algo="twap",
            twap_slices=3,
            twap_duration_sec=0.01,  # near-instant scheduling
            tick_interval_sec=0.05,
        )

        adapter = AlgoExecutionAdapter(
            submit_fn=submit, dispatcher_emit=emitted.append, cfg=cfg,
        )

        order = _order_event(qty="3.0", price="1000")
        adapter.send_order(order)

        # Wait for ticker to process slices
        deadline = time.monotonic() + 3.0
        while len(emitted) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)

        adapter.stop()

        assert len(emitted) == 3
        for fill in emitted:
            assert fill.event_type == "FILL"
            assert fill.price == 40000.0

    def test_completed_orders_cleaned_up(self):
        submit = _FakeSubmitFn(Decimal("100"))
        emitted: List[Any] = []
        cfg = AlgoConfig(
            large_order_notional=Decimal("1"),
            default_algo="twap",
            twap_slices=2,
            twap_duration_sec=0.01,
            tick_interval_sec=0.05,
        )

        adapter = AlgoExecutionAdapter(
            submit_fn=submit, dispatcher_emit=emitted.append, cfg=cfg,
        )

        adapter.send_order(_order_event(qty="1.0", price="100"))

        deadline = time.monotonic() + 3.0
        while len(adapter._active_orders) > 0 and time.monotonic() < deadline:
            time.sleep(0.05)

        adapter.stop()
        assert len(adapter._active_orders) == 0

    def test_async_algo_fill_emits_incident_logger(self):
        submit = _FakeSubmitFn(Decimal("40000"))
        emitted: List[Any] = []
        incidents: List[Any] = []
        cfg = AlgoConfig(
            large_order_notional=Decimal("100"),
            default_algo="twap",
            twap_slices=2,
            twap_duration_sec=0.01,
            tick_interval_sec=0.05,
        )

        adapter = AlgoExecutionAdapter(
            submit_fn=submit,
            dispatcher_emit=emitted.append,
            incident_logger=incidents.append,
            cfg=cfg,
        )

        adapter.send_order(_order_event(qty="2.0", price="1000"))

        deadline = time.monotonic() + 3.0
        while len(incidents) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)

        adapter.stop()

        assert len(incidents) == 2
        assert all(alert.title == "execution-synthetic-fill" for alert in incidents)


class TestStopLifecycle:
    def test_stop_idempotent(self):
        adapter = AlgoExecutionAdapter(
            submit_fn=lambda s, si, q: None,
            dispatcher_emit=lambda e: None,
        )
        adapter.stop()
        adapter.stop()  # should not raise
