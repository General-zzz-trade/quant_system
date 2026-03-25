"""Tests for BybitExecutionAdapter."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter


def _make_order_event(symbol="BTCUSDT", side="buy", qty=Decimal("0.01")):
    header = SimpleNamespace(
        event_id="evt-001",
        root_event_id="root-001",
        run_id=None,
        correlation_id=None,
    )
    return SimpleNamespace(
        header=header,
        order_id="ord-001",
        symbol=symbol,
        side=side,
        qty=qty,
    )


@patch("execution.adapters.bybit.execution_adapter.time")
def test_send_order_buy(mock_time):
    adapter = MagicMock()
    adapter.send_market_order.return_value = {"status": "ok"}
    fill_obj = SimpleNamespace(price=65000.0)
    adapter.get_recent_fills.return_value = (fill_obj,)

    exa = BybitExecutionAdapter(adapter)
    result = tuple(exa.send_order(_make_order_event()))

    assert len(result) == 1
    assert result[0].symbol == "BTCUSDT"
    assert result[0].price == Decimal("65000.0")
    adapter.send_market_order.assert_called_once_with("BTCUSDT", "buy", 0.01)


@patch("execution.adapters.bybit.execution_adapter.reliable_close_position")
@patch("execution.adapters.bybit.execution_adapter.time")
def test_send_order_close_position(mock_time, mock_close):
    mock_close.return_value = {"status": "closed"}
    adapter = MagicMock()
    fill_obj = SimpleNamespace(price=65000.0)
    adapter.get_recent_fills.return_value = (fill_obj,)

    exa = BybitExecutionAdapter(adapter)
    order = _make_order_event(qty=Decimal("0"))
    result = tuple(exa.send_order(order))

    assert len(result) == 1
    mock_close.assert_called_once_with(adapter, "BTCUSDT")
    adapter.send_market_order.assert_not_called()


@patch("execution.adapters.bybit.execution_adapter.time")
def test_send_order_failure_returns_empty(mock_time):
    adapter = MagicMock()
    adapter.send_market_order.return_value = {"status": "error", "msg": "insufficient"}

    exa = BybitExecutionAdapter(adapter)
    result = tuple(exa.send_order(_make_order_event()))

    assert result == ()


@patch("execution.adapters.bybit.execution_adapter.time")
def test_fill_price_from_exchange(mock_time):
    adapter = MagicMock()
    adapter.send_market_order.return_value = {"status": "ok"}
    fill_obj = SimpleNamespace(price=42123.45)
    adapter.get_recent_fills.return_value = (fill_obj,)

    exa = BybitExecutionAdapter(adapter)
    result = tuple(exa.send_order(_make_order_event()))

    assert len(result) == 1
    assert result[0].price == Decimal("42123.45")
    adapter.get_recent_fills.assert_called_once_with(symbol="BTCUSDT")
