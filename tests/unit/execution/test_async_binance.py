# tests/unit/execution/test_async_binance.py
"""Tests for async Binance REST client, WS transport, and gateway."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from execution.adapters.binance.async_rest import AsyncBinanceRestClient
from execution.adapters.binance.async_ws_transport import AsyncWsTransport
from execution.adapters.binance.async_gateway import AsyncBinanceUmFuturesOrderGateway
from execution.adapters.binance.rest import BinanceRestConfig


def _run(coro):
    return asyncio.run(coro)


# ── AsyncBinanceRestClient tests ────────────────────────────


class TestAsyncBinanceRestClient:
    @pytest.fixture
    def cfg(self):
        return BinanceRestConfig(
            base_url="https://testnet.binancefuture.com",
            api_key="test_key",
            api_secret="test_secret",
        )

    def test_init(self, cfg):
        client = AsyncBinanceRestClient(cfg)
        assert client._session is None

    def test_close_when_no_session(self, cfg):
        client = AsyncBinanceRestClient(cfg)
        _run(client.close())  # should not raise

    def test_signed_request_constructs_signature(self, cfg):
        client = AsyncBinanceRestClient(cfg)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value='{"orderId": 123}')
        mock_resp.json = AsyncMock(return_value={"orderId": 123})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = _run(client.request_signed(
            method="POST", path="/fapi/v1/order",
            params={"symbol": "BTCUSDT", "side": "BUY"},
        ))
        assert result["orderId"] == 123
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[0][0] == "POST"
        assert "signature=" in call_args[1]["data"].decode()

        _run(client.close())

    def test_api_key_request_get(self, cfg):
        client = AsyncBinanceRestClient(cfg)
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value='{"listenKey": "abc"}')
        mock_resp.json = AsyncMock(return_value={"listenKey": "abc"})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.closed = False
        client._session = mock_session

        result = _run(client.request_api_key(
            method="GET", path="/fapi/v1/listenKey",
        ))
        assert result["listenKey"] == "abc"
        _run(client.close())


# ── AsyncWsTransport tests ──────────────────────────────────


class TestAsyncWsTransport:
    def test_init(self):
        t = AsyncWsTransport()
        assert not t.connected

    def test_not_connected_by_default(self):
        t = AsyncWsTransport()
        assert t._ws is None


# ── AsyncBinanceUmFuturesOrderGateway tests ─────────────────


class TestAsyncBinanceUmFuturesOrderGateway:
    @pytest.fixture
    def mock_rest(self):
        rest = AsyncMock(spec=AsyncBinanceRestClient)
        rest.request_signed = AsyncMock(return_value={"orderId": 456})
        return rest

    def test_submit_order(self, mock_rest):
        gw = AsyncBinanceUmFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTCUSDT", side="buy", order_type="MARKET",
            qty=Decimal("0.1"), price=None,
            request_id="req_001",
        )
        result = _run(gw.submit_order(cmd))
        assert result["orderId"] == 456
        mock_rest.request_signed.assert_called_once()
        call_kwargs = mock_rest.request_signed.call_args[1]
        assert call_kwargs["path"] == "/fapi/v1/order"
        assert call_kwargs["params"]["symbol"] == "BTCUSDT"

    def test_cancel_order_by_id(self, mock_rest):
        mock_rest.request_signed = AsyncMock(return_value={"status": "CANCELED"})
        gw = AsyncBinanceUmFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(symbol="BTCUSDT", order_id="12345")
        result = _run(gw.cancel_order(cmd))
        assert result["status"] == "CANCELED"

    def test_cancel_order_requires_id(self, mock_rest):
        gw = AsyncBinanceUmFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(symbol="BTCUSDT")
        with pytest.raises(ValueError, match="requires order_id"):
            _run(gw.cancel_order(cmd))

    def test_close(self, mock_rest):
        gw = AsyncBinanceUmFuturesOrderGateway(rest=mock_rest)
        _run(gw.close())
        mock_rest.close.assert_called_once()
