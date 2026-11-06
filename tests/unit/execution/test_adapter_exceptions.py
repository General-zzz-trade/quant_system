"""Ambiguous Bybit submissions stay unresolved without retries or fabricated fills."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from event.header import EventHeader
from event.types import EventType
from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter


def _make_order_event(symbol="BTCUSDT", side="buy", qty=Decimal("0.01")):
    return SimpleNamespace(
        header=EventHeader.new_root(event_type=EventType.ORDER, version=1, source="test"),
        order_id="ord-001", symbol=symbol, side=side, qty=qty, price=Decimal("65000"),
    )


def _adapter(*, side="buy", qty="0.01"):
    adapter = MagicMock()
    adapter.send_market_order.return_value = {"status": "submitted", "orderId": "exchange-1"}
    adapter.get_order.side_effect = lambda symbol, **kwargs: SimpleNamespace(
        client_order_id=adapter.send_market_order.call_args.kwargs["client_order_id"],
        order_id="exchange-1", symbol=symbol, side=side, status="filled", filled_qty=Decimal(qty),
    )
    adapter.get_recent_fills.return_value = (SimpleNamespace(
        symbol="BTCUSDT", order_id="exchange-1", trade_id="trade-1", side=side,
        qty=Decimal(qty), price=Decimal("65000"), fee=Decimal("0"), fee_asset="USDT",
    ),)
    return adapter


def _wrap(adapter, tmp_path):
    return BybitExecutionAdapter(
        adapter, journal_path=tmp_path / "orders.sqlite", scope="test-demo", wait_seconds=0,
    )


class TestAdapterExceptions:
    def test_timeout_after_acceptance_is_confirmed_without_second_submission(self, tmp_path):
        adapter = _adapter()
        adapter.send_market_order.side_effect = [
            ConnectionError("response lost after exchange acceptance"),
            AssertionError("must not submit again"),
        ]
        execution = _wrap(adapter, tmp_path)

        result = execution.send_order(_make_order_event())

        assert len(result) == 1
        assert result[0].price == Decimal("65000")
        adapter.send_market_order.assert_called_once()
        adapter.get_order.assert_called_once_with(
            "BTCUSDT", order_id="", client_order_id=execution.journal.orders()[0]["client_id"],
        )

    def test_timeout_and_unknown_order_are_never_resubmitted(self, tmp_path):
        adapter = _adapter()
        adapter.send_market_order.side_effect = ConnectionError("timeout")
        adapter.get_order.side_effect = None
        adapter.get_order.return_value = None
        execution = _wrap(adapter, tmp_path)
        order = _make_order_event()

        assert execution.send_order(order) == ()
        assert execution.send_order(order) == ()
        assert execution.reconcile_pending() == ()
        restarted = _wrap(adapter, tmp_path)
        assert restarted.send_order(order) == ()
        assert restarted.reconcile_pending() == ()

        adapter.send_market_order.assert_called_once()
        assert execution.journal.orders(unresolved=True)[0]["status"] == "unknown"
        assert not execution.journal.fills("")

    @pytest.mark.parametrize("query", ["get_order", "get_recent_fills"])
    def test_confirmation_query_exception_never_fabricates_fill(self, tmp_path, query):
        adapter = _adapter()
        getattr(adapter, query).side_effect = RuntimeError("API error")
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event()) == ()
        assert not execution.journal.fills("")
        assert execution.journal.orders(unresolved=True)[0]["status"] == "unknown"
        adapter.send_market_order.assert_called_once()

    def test_empty_fills_do_not_create_zero_price_or_full_quantity_fill(self, tmp_path):
        adapter = _adapter()
        adapter.get_recent_fills.return_value = []
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event()) == ()
        assert not execution.journal.fills("")
        assert len(execution.journal.orders(unresolved=True)) == 1
        adapter.send_market_order.assert_called_once()

    def test_ack_without_execution_does_not_create_fill(self, tmp_path):
        adapter = _adapter()
        adapter.get_order.side_effect = lambda symbol, **kwargs: SimpleNamespace(
            client_order_id=kwargs["client_order_id"], order_id="exchange-1", symbol=symbol,
            side="buy", status="new", filled_qty=Decimal("0"),
        )
        adapter.get_recent_fills.return_value = ()
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event()) == ()
        assert not execution.journal.fills("")
        assert len(execution.journal.orders(unresolved=True)) == 1

    def test_close_position_query_failure_blocks_placement(self, tmp_path):
        adapter = _adapter()
        adapter.get_positions.side_effect = TimeoutError("position unavailable")
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event(qty=Decimal("0"))) == ()
        adapter.get_positions.assert_called_once_with(symbol="BTCUSDT", strict=True)
        adapter.send_market_order.assert_not_called()
        adapter.reliable_close_position.assert_not_called()
        assert not execution.journal.orders()

    @pytest.mark.parametrize("quantities", [[], ["0"], ["0.1", "-0.1"]])
    def test_zero_quantity_requires_one_unambiguous_position(self, tmp_path, quantities):
        adapter = _adapter()
        adapter.get_positions.return_value = [
            SimpleNamespace(symbol="BTCUSDT", qty=Decimal(qty)) for qty in quantities
        ]
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event(qty=Decimal("0"))) == ()
        adapter.send_market_order.assert_not_called()
        assert not execution.journal.fills("")

    @pytest.mark.parametrize("response", [{"retCode": 0}, {"status": "failed", "msg": "margin"}])
    def test_incomplete_or_failed_ack_without_confirmation_stays_unknown(self, tmp_path, response):
        adapter = _adapter()
        adapter.send_market_order.return_value = response
        adapter.get_order.side_effect = None
        adapter.get_order.return_value = None
        execution = _wrap(adapter, tmp_path)

        assert execution.send_order(_make_order_event()) == ()
        assert execution.journal.orders(unresolved=True)[0]["status"] == "unknown"
        assert not execution.journal.fills("")
        adapter.send_market_order.assert_called_once()

    @pytest.mark.parametrize("missing", ["order_id", "header", "symbol", "side", "qty", "price"])
    def test_order_event_missing_attribute_never_places_order(self, tmp_path, missing):
        adapter = _adapter()
        execution = _wrap(adapter, tmp_path)
        event = _make_order_event()
        delattr(event, missing)

        assert execution.send_order(event) == ()
        adapter.send_market_order.assert_not_called()
        assert not execution.journal.orders()

    def test_sell_side_and_confirmed_quantity_are_preserved(self, tmp_path):
        adapter = _adapter(side="sell", qty="0.05")
        execution = _wrap(adapter, tmp_path)

        result = execution.send_order(_make_order_event(side="sell", qty=Decimal("0.05")))

        assert len(result) == 1
        assert (result[0].side, result[0].qty, result[0].price) == (
            "sell", Decimal("0.05"), Decimal("65000"),
        )
        adapter.send_market_order.assert_called_once_with(
            "BTCUSDT", "sell", 0.05, reduce_only=False,
            client_order_id=execution.journal.orders()[0]["client_id"],
        )

    def test_zero_wait_timeout_has_no_retry_delay(self, tmp_path):
        adapter = _adapter()
        adapter.send_market_order.side_effect = ConnectionError("timeout")
        adapter.get_order.side_effect = None
        adapter.get_order.return_value = None
        execution = _wrap(adapter, tmp_path)

        with patch("shared.confirmed_execution.time.sleep") as sleep:
            assert execution.send_order(_make_order_event()) == ()
        sleep.assert_not_called()
        adapter.send_market_order.assert_called_once()

    def test_large_quantity_is_not_scaled_or_replaced_by_ack(self, tmp_path):
        adapter = _adapter(qty="999.999")
        execution = _wrap(adapter, tmp_path)

        result = execution.send_order(_make_order_event(qty=Decimal("999.999")))

        assert len(result) == 1
        assert result[0].qty == Decimal("999.999")
        adapter.send_market_order.assert_called_once_with(
            "BTCUSDT", "buy", 999.999, reduce_only=False,
            client_order_id=execution.journal.orders()[0]["client_id"],
        )
