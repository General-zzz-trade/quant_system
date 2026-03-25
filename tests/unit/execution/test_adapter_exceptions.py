"""Exception and boundary-condition tests for BybitExecutionAdapter.

Covers: send_market_order retries, get_recent_fills failure, reliable_close
failure, qty=0 dispatch, missing resp keys.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter


# ── helpers ────────────────────────────────────────────────────────


def _make_order_event(
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: Decimal = Decimal("0.01"),
) -> SimpleNamespace:
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


# ── tests ──────────────────────────────────────────────────────────


@patch("execution.adapters.bybit.execution_adapter.time")
class TestAdapterExceptions:
    """Edge cases for BybitExecutionAdapter.send_order()."""

    def test_send_market_order_retry_then_success(self, mock_time):
        """First attempt fails, second succeeds → returns fill."""
        adapter = MagicMock()
        adapter.send_market_order.side_effect = [
            ConnectionError("timeout"),
            {"status": "ok"},
        ]
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert len(result) == 1
        assert result[0].price == Decimal("65000.0")
        assert adapter.send_market_order.call_count == 2

    def test_send_market_order_all_retries_fail(self, mock_time):
        """All 3 retries fail → returns error dict → empty result."""
        adapter = MagicMock()
        adapter.send_market_order.side_effect = ConnectionError("timeout")

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert result == ()
        assert adapter.send_market_order.call_count == 3

    def test_get_recent_fills_exception_fill_price_zero(self, mock_time):
        """get_recent_fills raises → fill_price = 0.0."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"status": "ok"}
        adapter.get_recent_fills.side_effect = RuntimeError("API error")

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert len(result) == 1
        assert result[0].price == Decimal("0.0")

    def test_get_recent_fills_empty_list(self, mock_time):
        """get_recent_fills returns [] → fill_price = 0.0."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"status": "ok"}
        adapter.get_recent_fills.return_value = []

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert len(result) == 1
        assert result[0].price == Decimal("0.0")

    @patch("execution.adapters.bybit.execution_adapter.reliable_close_position")
    def test_reliable_close_all_fail(self, mock_close, mock_time):
        """reliable_close_position returns failed → empty result."""
        mock_close.return_value = {"status": "failed"}

        adapter = MagicMock()
        exa = BybitExecutionAdapter(adapter)
        order = _make_order_event(qty=Decimal("0"))
        result = tuple(exa.send_order(order))
        assert result == ()
        mock_close.assert_called_once()

    @patch("execution.adapters.bybit.execution_adapter.reliable_close_position")
    def test_qty_zero_calls_reliable_close(self, mock_close, mock_time):
        """qty=0 → dispatches to reliable_close_position, not send_market_order."""
        mock_close.return_value = {"status": "closed"}
        adapter = MagicMock()
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        order = _make_order_event(qty=Decimal("0"))
        result = tuple(exa.send_order(order))
        assert len(result) == 1
        mock_close.assert_called_once_with(adapter, "BTCUSDT")
        adapter.send_market_order.assert_not_called()

    def test_resp_missing_status_key(self, mock_time):
        """Response dict missing 'status' key → treated as non-error."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"retCode": 0}  # no 'status'
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        # resp.get("status", "") returns "" which is not in ("error", "failed")
        assert len(result) == 1

    def test_resp_status_failed(self, mock_time):
        """Response status='failed' → returns empty."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"status": "failed", "msg": "margin"}

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert result == ()

    def test_order_event_missing_attr_returns_empty(self, mock_time):
        """Order event with missing attribute → outer except catches → empty."""
        exa = BybitExecutionAdapter(MagicMock())
        # SimpleNamespace without required attrs
        bad_event = SimpleNamespace()
        result = tuple(exa.send_order(bad_event))
        assert result == ()

    def test_sell_side_order(self, mock_time):
        """Sell order passes correct side to adapter."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"status": "ok"}
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        order = _make_order_event(side="sell", qty=Decimal("0.05"))
        result = tuple(exa.send_order(order))
        assert len(result) == 1
        adapter.send_market_order.assert_called_once_with("BTCUSDT", "sell", 0.05)
        assert result[0].side == "sell"

    def test_retry_delay_called_on_failure(self, mock_time):
        """time.sleep called between retries."""
        adapter = MagicMock()
        adapter.send_market_order.side_effect = [
            ConnectionError("fail1"),
            ConnectionError("fail2"),
            {"status": "ok"},
        ]
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        result = tuple(exa.send_order(_make_order_event()))
        assert len(result) == 1
        # sleep called twice (after attempt 1 and 2, not after 3)
        sleep_calls = [c for c in mock_time.sleep.call_args_list if c == call(0.5)]
        assert len(sleep_calls) == 2

    def test_large_qty_passes_through(self, mock_time):
        """Large qty value passes through without modification."""
        adapter = MagicMock()
        adapter.send_market_order.return_value = {"status": "ok"}
        fill_obj = SimpleNamespace(price=65000.0)
        adapter.get_recent_fills.return_value = [fill_obj]

        exa = BybitExecutionAdapter(adapter)
        order = _make_order_event(qty=Decimal("999.999"))
        result = tuple(exa.send_order(order))
        assert len(result) == 1
        adapter.send_market_order.assert_called_once_with("BTCUSDT", "buy", 999.999)
