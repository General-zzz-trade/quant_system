# tests/unit/execution/test_okx_gateway.py
"""Tests for OKX REST client and order gateway."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from execution.adapters.okx.rest import (
    OkxRestClient,
    OkxRestConfig,
    OkxRestError,
    OkxRetryableError,
    OkxNonRetryableError,
    _sign,
    _iso_ts,
)
from execution.adapters.okx.gateway import OkxFuturesOrderGateway


class TestOkxRestHelpers:
    def test_iso_ts_format(self):
        ts = _iso_ts()
        assert ts.endswith("Z")
        assert "T" in ts

    def test_sign_produces_base64(self):
        sig = _sign("secret", "2024-01-01T00:00:00.000Z", "GET", "/api/v5/test")
        import base64
        # Should be valid base64
        base64.b64decode(sig)

    def test_sign_deterministic(self):
        s1 = _sign("sec", "ts", "GET", "/path", "body")
        s2 = _sign("sec", "ts", "GET", "/path", "body")
        assert s1 == s2

    def test_sign_different_for_different_input(self):
        s1 = _sign("sec", "ts", "GET", "/path1")
        s2 = _sign("sec", "ts", "GET", "/path2")
        assert s1 != s2


class TestOkxRestConfig:
    def test_defaults(self):
        cfg = OkxRestConfig()
        assert cfg.base_url == "https://www.okx.com"
        assert cfg.timeout_s == 10.0
        assert not cfg.simulated

    def test_simulated_mode(self):
        cfg = OkxRestConfig(simulated=True)
        assert cfg.simulated


class TestOkxFuturesOrderGateway:
    @pytest.fixture
    def mock_rest(self):
        rest = MagicMock(spec=OkxRestClient)
        rest.request_signed = MagicMock(return_value={
            "code": "0",
            "data": [{"ordId": "12345", "clOrdId": "req_001"}],
        })
        return rest

    def test_submit_market_order(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            side="buy",
            order_type="market",
            qty=Decimal("0.1"),
            price=None,
            request_id="req_001",
        )
        result = gw.submit_order(cmd)
        assert result["code"] == "0"

        call_kwargs = mock_rest.request_signed.call_args[1]
        assert call_kwargs["path"] == "/api/v5/trade/order"
        body = call_kwargs["body"]
        assert body["instId"] == "BTC-USDT-SWAP"
        assert body["side"] == "buy"
        assert body["ordType"] == "market"
        assert body["tdMode"] == "cross"

    def test_submit_limit_order(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            side="sell",
            order_type="limit",
            qty=Decimal("0.5"),
            price=Decimal("50000"),
            request_id="req_002",
        )
        result = gw.submit_order(cmd)
        body = mock_rest.request_signed.call_args[1]["body"]
        assert body["px"] == "50000"
        assert body["ordType"] == "limit"

    def test_submit_reduce_only(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            side="sell",
            order_type="market",
            qty=Decimal("0.1"),
            reduce_only=True,
        )
        gw.submit_order(cmd)
        body = mock_rest.request_signed.call_args[1]["body"]
        assert body["reduceOnly"] is True

    def test_cancel_by_order_id(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            order_id="12345",
        )
        gw.cancel_order(cmd)
        body = mock_rest.request_signed.call_args[1]["body"]
        assert body["ordId"] == "12345"

    def test_cancel_by_client_id(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            client_order_id="cl_001",
        )
        gw.cancel_order(cmd)
        body = mock_rest.request_signed.call_args[1]["body"]
        assert body["clOrdId"] == "cl_001"

    def test_cancel_requires_id(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        cmd = SimpleNamespace(symbol="BTC-USDT-SWAP")
        with pytest.raises(ValueError, match="requires order_id"):
            gw.cancel_order(cmd)

    def test_get_positions(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        gw.get_positions()
        call_kwargs = mock_rest.request_signed.call_args[1]
        assert call_kwargs["path"] == "/api/v5/account/positions"

    def test_get_balance(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest)
        gw.get_balance()
        call_kwargs = mock_rest.request_signed.call_args[1]
        assert call_kwargs["path"] == "/api/v5/account/balance"

    def test_td_mode_isolated(self, mock_rest):
        gw = OkxFuturesOrderGateway(rest=mock_rest, td_mode="isolated")
        cmd = SimpleNamespace(
            symbol="BTC-USDT-SWAP",
            side="buy",
            order_type="market",
            qty=Decimal("0.1"),
        )
        gw.submit_order(cmd)
        body = mock_rest.request_signed.call_args[1]["body"]
        assert body["tdMode"] == "isolated"
