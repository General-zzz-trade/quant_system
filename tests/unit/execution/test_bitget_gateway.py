# tests/unit/execution/test_bitget_gateway.py
"""Tests for Bitget order gateway — mock REST, verify submit/cancel params."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from unittest.mock import MagicMock

import pytest

from execution.adapters.bitget.order_gateway import BitgetFuturesOrderGateway
from execution.adapters.bitget.venue_client import BitgetFuturesVenueClient
from execution.adapters.bitget.rest import BitgetRetryableError, BitgetNonRetryableError


@dataclass
class MockCmd:
    symbol: str = "BTCUSDT"
    side: str = "buy"
    order_type: str = "limit"
    qty: str = "0.01"
    price: Optional[str] = "50000"
    time_in_force: Optional[str] = "gtc"
    reduce_only: bool = False
    client_order_id: Optional[str] = "test-123"
    order_id: Optional[str] = None


class TestBitgetOrderGateway:
    def setup_method(self):
        self.mock_rest = MagicMock()
        self.mock_rest.request_signed.return_value = {"orderId": "99999", "clientOid": "test-123"}
        self.gw = BitgetFuturesOrderGateway(rest=self.mock_rest)

    def test_submit_order_params(self):
        cmd = MockCmd()
        result = self.gw.submit_order(cmd)

        self.mock_rest.request_signed.assert_called_once()
        call_kwargs = self.mock_rest.request_signed.call_args[1]

        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "/api/v2/mix/order/place-order"

        body = call_kwargs["body"]
        assert body["symbol"] == "BTCUSDT"
        assert body["productType"] == "USDT-FUTURES"
        assert body["marginMode"] == "crossed"
        assert body["marginCoin"] == "USDT"
        assert body["side"] == "buy"
        assert body["tradeSide"] == "open"
        assert body["orderType"] == "limit"
        assert body["force"] == "gtc"
        assert body["size"] == "0.01"
        assert body["clientOid"] == "test-123"
        assert body["price"] == "50000"

    def test_submit_order_reduce_only(self):
        cmd = MockCmd(reduce_only=True)
        self.gw.submit_order(cmd)

        body = self.mock_rest.request_signed.call_args[1]["body"]
        assert body["tradeSide"] == "close"

    def test_submit_order_market_no_price(self):
        cmd = MockCmd(order_type="market", price=None)
        self.gw.submit_order(cmd)

        body = self.mock_rest.request_signed.call_args[1]["body"]
        assert body["orderType"] == "market"
        assert "price" not in body

    def test_submit_order_no_client_id(self):
        cmd = MockCmd(client_order_id=None)
        self.gw.submit_order(cmd)

        body = self.mock_rest.request_signed.call_args[1]["body"]
        assert "clientOid" not in body

    def test_cancel_order_by_order_id(self):
        cmd = MockCmd(order_id="12345")
        self.gw.cancel_order(cmd)

        call_kwargs = self.mock_rest.request_signed.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "/api/v2/mix/order/cancel-order"

        body = call_kwargs["body"]
        assert body["symbol"] == "BTCUSDT"
        assert body["productType"] == "USDT-FUTURES"
        assert body["orderId"] == "12345"

    def test_cancel_order_by_client_oid(self):
        cmd = MockCmd(order_id=None, client_order_id="my-oid")
        self.gw.cancel_order(cmd)

        body = self.mock_rest.request_signed.call_args[1]["body"]
        assert body["clientOid"] == "my-oid"
        assert "orderId" not in body

    def test_cancel_order_no_id_raises(self):
        cmd = MockCmd(order_id=None, client_order_id=None)
        with pytest.raises(ValueError, match="cancel_order requires"):
            self.gw.cancel_order(cmd)

    def test_sell_side_mapping(self):
        cmd = MockCmd(side="sell")
        self.gw.submit_order(cmd)

        body = self.mock_rest.request_signed.call_args[1]["body"]
        assert body["side"] == "sell"


class TestBitgetVenueClient:
    def setup_method(self):
        self.mock_gw = MagicMock()
        self.client = BitgetFuturesVenueClient(gw=self.mock_gw)

    def test_submit_success(self):
        self.mock_gw.submit_order.return_value = {"orderId": "123"}
        result = self.client.submit_order(MockCmd())
        assert result["orderId"] == "123"

    def test_submit_retryable_error(self):
        self.mock_gw.submit_order.side_effect = BitgetRetryableError("rate limit")
        from execution.bridge.execution_bridge import RetryableVenueError
        with pytest.raises(RetryableVenueError):
            self.client.submit_order(MockCmd())

    def test_submit_non_retryable_error(self):
        self.mock_gw.submit_order.side_effect = BitgetNonRetryableError("bad param")
        from execution.bridge.execution_bridge import NonRetryableVenueError
        with pytest.raises(NonRetryableVenueError):
            self.client.submit_order(MockCmd())

    def test_cancel_success(self):
        self.mock_gw.cancel_order.return_value = {"orderId": "456"}
        result = self.client.cancel_order(MockCmd(order_id="456"))
        assert result["orderId"] == "456"

    def test_cancel_retryable_error(self):
        self.mock_gw.cancel_order.side_effect = BitgetRetryableError("timeout")
        from execution.bridge.execution_bridge import RetryableVenueError
        with pytest.raises(RetryableVenueError):
            self.client.cancel_order(MockCmd(order_id="1"))

    def test_cancel_non_retryable_error(self):
        self.mock_gw.cancel_order.side_effect = BitgetNonRetryableError("not found")
        from execution.bridge.execution_bridge import NonRetryableVenueError
        with pytest.raises(NonRetryableVenueError):
            self.client.cancel_order(MockCmd(order_id="1"))
