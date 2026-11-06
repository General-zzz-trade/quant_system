"""Bybit market orders emit only matching, confirmed exchange trade facts."""
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


def _trade(*, price="65000", qty="0.01", side="buy", order_id="exchange-1", trade_id="trade-1"):
    return SimpleNamespace(
        symbol="BTCUSDT", order_id=order_id, trade_id=trade_id, side=side,
        qty=Decimal(qty), price=Decimal(price), fee=Decimal("0"), fee_asset="USDT",
    )


def _confirmed_adapter(*, side="buy", qty="0.01", price="65000"):
    adapter = MagicMock()
    adapter.send_market_order.return_value = {"status": "submitted", "orderId": "exchange-1"}
    adapter.get_order.side_effect = lambda symbol, **kwargs: SimpleNamespace(
        client_order_id=adapter.send_market_order.call_args.kwargs["client_order_id"],
        order_id="exchange-1", symbol=symbol, side=side, status="filled", filled_qty=Decimal(qty),
    )
    adapter.get_recent_fills.return_value = (_trade(side=side, qty=qty, price=price),)
    return adapter


def _wrap(adapter, tmp_path):
    return BybitExecutionAdapter(
        adapter, journal_path=tmp_path / "orders.sqlite", scope="test-demo", wait_seconds=0,
    )


def test_send_order_buy(tmp_path):
    adapter = _confirmed_adapter()
    execution = _wrap(adapter, tmp_path)

    result = execution.send_order(_make_order_event())

    assert len(result) == 1
    assert (result[0].symbol, result[0].side, result[0].qty, result[0].price) == (
        "BTCUSDT", "buy", Decimal("0.01"), Decimal("65000"),
    )
    client_id = execution.journal.orders()[0]["client_id"]
    assert client_id
    adapter.send_market_order.assert_called_once_with(
        "BTCUSDT", "buy", 0.01, reduce_only=False, client_order_id=client_id,
    )
    adapter.get_order.assert_called_once_with(
        "BTCUSDT", order_id="exchange-1", client_order_id=client_id,
    )


@pytest.mark.parametrize("position_qty,side", [("0.025", "sell"), ("-0.025", "buy")])
def test_send_order_close_position_uses_actual_quantity_and_reduce_only(tmp_path, position_qty, side):
    adapter = _confirmed_adapter(side=side, qty="0.025")
    adapter.get_positions.return_value = (
        SimpleNamespace(symbol="BTCUSDT", qty=Decimal(position_qty)),
    )
    execution = _wrap(adapter, tmp_path)

    result = execution.send_order(_make_order_event(qty=Decimal("0")))

    assert len(result) == 1
    assert (result[0].qty, result[0].side) == (Decimal("0.025"), side)
    adapter.get_positions.assert_called_once_with(symbol="BTCUSDT", strict=True)
    adapter.send_market_order.assert_called_once_with(
        "BTCUSDT", side, 0.025, reduce_only=True,
        client_order_id=execution.journal.orders()[0]["client_id"],
    )
    adapter.reliable_close_position.assert_not_called()


def test_send_order_failure_returns_empty_and_remains_unresolved(tmp_path):
    adapter = _confirmed_adapter()
    adapter.send_market_order.return_value = {"status": "error", "msg": "insufficient"}
    adapter.get_order.side_effect = None
    adapter.get_order.return_value = None
    execution = _wrap(adapter, tmp_path)

    assert execution.send_order(_make_order_event()) == ()
    assert execution.journal.orders(unresolved=True)[0]["status"] == "unknown"
    assert not execution.journal.fills("")
    adapter.send_market_order.assert_called_once()
    adapter.get_recent_fills.assert_not_called()


def test_fill_price_and_quantity_come_from_matching_exchange_trade(tmp_path):
    adapter = _confirmed_adapter(qty="0.004", price="42123.45")
    adapter.get_recent_fills.return_value = (
        _trade(order_id="another-order", price="1", qty="99"),
        _trade(order_id="another-order", side="sell", price="2", qty="88"),
        _trade(qty="0.004", price="42123.45"),
    )
    execution = _wrap(adapter, tmp_path)

    result = execution.send_order(_make_order_event())

    assert len(result) == 1
    assert result[0].price == Decimal("42123.45")
    assert result[0].qty == Decimal("0.004")
    assert result[0].side == "buy"
    adapter.get_recent_fills.assert_called_once_with(
        symbol="BTCUSDT", order_id="exchange-1", limit=100,
    )


def test_scope_is_required_before_creating_journal(tmp_path, monkeypatch):
    monkeypatch.delenv("RISK_SCOPE", raising=False)
    with pytest.raises(ValueError, match="RISK_SCOPE"):
        BybitExecutionAdapter(MagicMock(), journal_path=tmp_path / "orders.sqlite", wait_seconds=0)
    assert not (tmp_path / "orders.sqlite").exists()


def test_tca_close_failure_cannot_hide_a_confirmed_fill(tmp_path):
    execution = _wrap(_confirmed_adapter(), tmp_path)
    with patch("execution.adapters.bybit.execution_adapter.TCALogger") as tca:
        tca.return_value.close.side_effect = OSError("mock log close failed")
        result = execution.send_order(_make_order_event())
    assert len(result) == 1
    assert result[0].qty == Decimal("0.01")
    assert len(execution.journal.fills("")) == 1


def test_whitespace_scope_cannot_bypass_account_lease_identity(tmp_path):
    with pytest.raises(ValueError, match="RISK_SCOPE"):
        BybitExecutionAdapter(MagicMock(), journal_path=tmp_path / "orders.sqlite", scope=" account ")
    assert not (tmp_path / "orders.sqlite").exists()
