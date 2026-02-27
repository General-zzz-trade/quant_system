# tests/unit/execution/adapters/bitget/test_bitget_e2e.py
"""End-to-end integration tests for the Bitget adapter stack.

Covers: VenueClient -> OrderGateway -> REST (mocked) -> Mappers -> Canonical models.
"""
from __future__ import annotations

import pytest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from execution.adapters.bitget.rest import (
    BitgetRestClient,
    BitgetRetryableError,
    BitgetNonRetryableError,
)
from execution.adapters.bitget.order_gateway import BitgetFuturesOrderGateway
from execution.adapters.bitget.venue_client import BitgetFuturesVenueClient
from execution.adapters.bitget.mapper_order import BitgetOrderMapper
from execution.adapters.bitget.mapper_fill import BitgetFillMapper
from execution.bridge.execution_bridge import RetryableVenueError, NonRetryableVenueError
from execution.models.orders import CanonicalOrder
from execution.models.fills import CanonicalFill


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_client() -> tuple[BitgetFuturesVenueClient, MagicMock]:
    mock_rest = MagicMock(spec=BitgetRestClient)
    gw = BitgetFuturesOrderGateway(rest=mock_rest)
    client = BitgetFuturesVenueClient(gw=gw)
    return client, mock_rest


def _buy_cmd(**overrides) -> SimpleNamespace:
    defaults = dict(
        symbol="BTCUSDT",
        side="buy",
        order_type="limit",
        qty=0.5,
        price=60000,
        reduce_only=False,
        time_in_force="gtc",
        request_id="client-001",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _cancel_cmd(**overrides) -> SimpleNamespace:
    defaults = dict(
        symbol="BTCUSDT",
        order_id="bg-order-123",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


RAW_REST_ORDER = {
    "orderId": "bg-order-123",
    "clientOid": "client-001",
    "symbol": "BTCUSDT",
    "side": "buy",
    "orderType": "limit",
    "status": "new",
    "force": "gtc",
    "size": "0.5",
    "price": "60000",
    "baseVolume": "0",
    "priceAvg": "0",
    "cTime": "1700000000000",
}

RAW_REST_FILL = {
    "tradeId": "fill-777",
    "orderId": "bg-order-123",
    "symbol": "BTCUSDT",
    "side": "buy",
    "baseVolume": "0.25",
    "price": "59950",
    "fee": "-0.003",
    "feeCoin": "USDT",
    "cTime": "1700000001000",
    "tradeScope": "taker",
}


# ---------------------------------------------------------------------------
# 1. Submit order E2E
# ---------------------------------------------------------------------------

class TestSubmitOrderE2E:
    def test_submit_order_calls_rest_and_returns_data(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.return_value = RAW_REST_ORDER

        result = client.submit_order(_buy_cmd())

        mock_rest.request_signed.assert_called_once()
        call_kw = mock_rest.request_signed.call_args
        assert call_kw.kwargs["method"] == "POST"
        assert call_kw.kwargs["path"] == "/api/v2/mix/order/place-order"
        body = call_kw.kwargs["body"]
        assert body["symbol"] == "BTCUSDT"
        assert body["side"] == "buy"
        assert body["orderType"] == "limit"
        assert body["size"] == "0.5"
        assert body["price"] == "60000"
        assert body["clientOid"] == "client-001"
        assert body["tradeSide"] == "open"
        assert body["force"] == "gtc"
        assert body["productType"] == "USDT-FUTURES"
        assert body["marginMode"] == "crossed"
        assert body["marginCoin"] == "USDT"

        assert result == RAW_REST_ORDER

    def test_submit_reduce_only_sets_trade_side_close(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.return_value = RAW_REST_ORDER

        client.submit_order(_buy_cmd(reduce_only=True))

        body = mock_rest.request_signed.call_args.kwargs["body"]
        assert body["tradeSide"] == "close"

    def test_submit_then_map_produces_canonical_order(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.return_value = RAW_REST_ORDER

        raw = client.submit_order(_buy_cmd())
        order = BitgetOrderMapper().map_order(raw)

        assert isinstance(order, CanonicalOrder)
        assert order.venue == "bitget"
        assert order.symbol == "BTCUSDT"
        assert order.order_id == "bg-order-123"
        assert order.client_order_id == "client-001"
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.status == "new"
        assert order.tif == "gtc"
        assert order.qty == Decimal("0.5")
        assert order.price == Decimal("60000")
        assert order.filled_qty == Decimal("0")
        assert order.ts_ms == 1700000000000
        assert order.order_key == "bitget:BTCUSDT:order:bg-order-123"
        assert order.payload_digest  # non-empty hash
        assert order.raw is raw


# ---------------------------------------------------------------------------
# 2. Cancel order E2E
# ---------------------------------------------------------------------------

class TestCancelOrderE2E:
    def test_cancel_order_by_order_id(self):
        client, mock_rest = _make_client()
        cancel_resp = {"orderId": "bg-order-123", "clientOid": "", "status": "cancelled"}
        mock_rest.request_signed.return_value = cancel_resp

        result = client.cancel_order(_cancel_cmd())

        call_kw = mock_rest.request_signed.call_args
        assert call_kw.kwargs["method"] == "POST"
        assert call_kw.kwargs["path"] == "/api/v2/mix/order/cancel-order"
        body = call_kw.kwargs["body"]
        assert body["orderId"] == "bg-order-123"
        assert body["symbol"] == "BTCUSDT"
        assert body["productType"] == "USDT-FUTURES"
        assert result == cancel_resp

    def test_cancel_order_by_client_order_id(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.return_value = {}

        cmd = SimpleNamespace(symbol="ETHUSDT", client_order_id="my-cloid-99")
        client.cancel_order(cmd)

        body = mock_rest.request_signed.call_args.kwargs["body"]
        assert body["clientOid"] == "my-cloid-99"
        assert "orderId" not in body

    def test_cancel_without_ids_raises(self):
        client, _ = _make_client()
        cmd = SimpleNamespace(symbol="BTCUSDT")
        with pytest.raises(ValueError, match="cancel_order requires"):
            client.cancel_order(cmd)


# ---------------------------------------------------------------------------
# 3. Retryable error flow
# ---------------------------------------------------------------------------

class TestRetryableErrorFlow:
    def test_submit_retryable_wraps_to_venue_error(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.side_effect = BitgetRetryableError("rate limited")

        with pytest.raises(RetryableVenueError, match="rate limited"):
            client.submit_order(_buy_cmd())

    def test_cancel_retryable_wraps_to_venue_error(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.side_effect = BitgetRetryableError("timeout")

        with pytest.raises(RetryableVenueError, match="timeout"):
            client.cancel_order(_cancel_cmd())


# ---------------------------------------------------------------------------
# 4. Non-retryable error flow
# ---------------------------------------------------------------------------

class TestNonRetryableErrorFlow:
    def test_submit_non_retryable_wraps_to_venue_error(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.side_effect = BitgetNonRetryableError("invalid param")

        with pytest.raises(NonRetryableVenueError, match="invalid param"):
            client.submit_order(_buy_cmd())

    def test_cancel_non_retryable_wraps_to_venue_error(self):
        client, mock_rest = _make_client()
        mock_rest.request_signed.side_effect = BitgetNonRetryableError("auth failed")

        with pytest.raises(NonRetryableVenueError, match="auth failed"):
            client.cancel_order(_cancel_cmd())


# ---------------------------------------------------------------------------
# 5. Mapper roundtrip — order & fill
# ---------------------------------------------------------------------------

class TestMapperRoundtrip:
    def test_map_rest_order_all_fields(self):
        raw = {
            "orderId": "99001",
            "clientOid": "cloid-42",
            "symbol": "ethusdt",
            "side": "Sell",
            "orderType": "market",
            "status": "filled",
            "force": "ioc",
            "size": "1.25",
            "price": "3100.50",
            "baseVolume": "1.25",
            "priceAvg": "3100.00",
            "cTime": "1700000099000",
        }
        order = BitgetOrderMapper().map_order(raw)

        assert order.venue == "bitget"
        assert order.symbol == "ETHUSDT"
        assert order.order_id == "99001"
        assert order.client_order_id == "cloid-42"
        assert order.side == "sell"
        assert order.order_type == "market"
        assert order.status == "filled"
        assert order.tif == "ioc"
        assert order.qty == Decimal("1.25")
        assert order.price == Decimal("3100.50")
        assert order.filled_qty == Decimal("1.25")
        assert order.avg_price == Decimal("3100.00")
        assert order.ts_ms == 1700000099000
        assert order.order_key
        assert order.payload_digest

    def test_map_fill_all_fields(self):
        raw = {
            "tradeId": "t-5001",
            "orderId": "o-9001",
            "symbol": "BTCUSDT",
            "side": "buy",
            "baseVolume": "0.1",
            "price": "62000",
            "feeDetail": [{"totalFee": "-0.62", "feeCoin": "USDT"}],
            "cTime": "1700000050000",
            "tradeScope": "maker",
        }
        fill = BitgetFillMapper().map_fill(raw)

        assert isinstance(fill, CanonicalFill)
        assert fill.venue == "bitget"
        assert fill.symbol == "BTCUSDT"
        assert fill.order_id == "o-9001"
        assert fill.trade_id == "t-5001"
        assert fill.fill_id == "bitget:BTCUSDT:t-5001"
        assert fill.side == "buy"
        assert fill.qty == Decimal("0.1")
        assert fill.price == Decimal("62000")
        assert fill.fee == Decimal("0.62")
        assert fill.fee_asset == "USDT"
        assert fill.liquidity == "maker"
        assert fill.ts_ms == 1700000050000
        assert fill.payload_digest

    def test_map_fill_flat_fee_fields(self):
        raw = {
            "tradeId": "t-5002",
            "orderId": "o-9002",
            "symbol": "SOLUSDT",
            "side": "sell",
            "baseVolume": "10",
            "price": "150.5",
            "fee": "-0.015",
            "feeCoin": "USDT",
            "cTime": "1700000060000",
            "tradeScope": "taker",
        }
        fill = BitgetFillMapper().map_fill(raw)

        assert fill.symbol == "SOLUSDT"
        assert fill.side == "sell"
        assert fill.qty == Decimal("10")
        assert fill.price == Decimal("150.5")
        assert fill.fee == Decimal("0.015")
        assert fill.fee_asset == "USDT"
        assert fill.liquidity == "taker"

    def test_map_order_unsupported_payload_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            BitgetOrderMapper().map_order({"random_key": "value"})

    def test_map_fill_unsupported_payload_raises(self):
        with pytest.raises(ValueError, match="unsupported"):
            BitgetFillMapper().map_fill({"random_key": "value"})
