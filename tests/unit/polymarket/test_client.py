"""Tests for Polymarket REST client with mocked HTTP responses."""
from __future__ import annotations
import json
from unittest.mock import patch, MagicMock
from io import BytesIO

from execution.adapters.polymarket.client import PolymarketRestClient, PolymarketRestError
from execution.adapters.polymarket.auth import PolymarketAuth


def _make_client() -> PolymarketRestClient:
    auth = PolymarketAuth(api_key="testkey", api_secret="testsecret")
    return PolymarketRestClient(auth=auth, base_url="https://clob.polymarket.com", timeout_s=5.0)


def _mock_response(data, status=200):
    resp = MagicMock()
    raw = json.dumps(data).encode("utf-8")
    resp.read.return_value = raw
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestGetMarkets:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_markets_returns_list(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"condition_id": "0x1", "question": "BTC above 100k?",
             "tokens": [{"token_id": "t1", "outcome": "Yes"}, {"token_id": "t2", "outcome": "No"}],
             "end_date_iso": "2026-12-31T00:00:00Z", "active": True,
             "volume_24h": "50000", "slug": "btc-100k", "description": "", "category": "crypto"},
        ])
        client = _make_client()
        result = client.get_markets()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["condition_id"] == "0x1"

    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_markets_with_params(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([])
        client = _make_client()
        result = client.get_markets(limit=10, offset=0)
        assert result == []
        # Verify URL contains params
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "limit=10" in req.full_url


class TestGetOrderbook:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_orderbook(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({
            "bids": [{"price": "0.65", "size": "100"}],
            "asks": [{"price": "0.70", "size": "150"}],
            "timestamp": 1700000000000,
        })
        client = _make_client()
        result = client.get_orderbook(token_id="t1")
        assert "bids" in result
        assert "asks" in result


class TestGetTrades:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_trades(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"id": "trade1", "price": "0.65", "size": "10", "side": "buy",
             "timestamp": 1700000000000},
        ])
        client = _make_client()
        result = client.get_trades(token_id="t1")
        assert len(result) == 1
        assert result[0]["id"] == "trade1"


class TestCreateOrder:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_create_order(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({
            "id": "order123", "status": "LIVE",
        })
        client = _make_client()
        result = client.create_order(
            token_id="t1", side="buy", price="0.65", size="10",
        )
        assert result["id"] == "order123"
        assert result["status"] == "LIVE"


class TestCancelOrder:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_cancel_order(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"id": "order123", "status": "CANCELED"})
        client = _make_client()
        result = client.cancel_order(order_id="order123")
        assert result["status"] == "CANCELED"


class TestGetPositions:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_positions(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response([
            {"token_id": "t1", "size": "50", "avg_price": "0.60"},
        ])
        client = _make_client()
        result = client.get_positions()
        assert len(result) == 1
        assert result[0]["token_id"] == "t1"


class TestGetBalances:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_get_balances(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response({"usdc": "10000.00"})
        client = _make_client()
        result = client.get_balances()
        assert result["usdc"] == "10000.00"


class TestErrorHandling:
    @patch("execution.adapters.polymarket.client.urlopen")
    def test_http_error_raises(self, mock_urlopen):
        from urllib.error import HTTPError
        mock_urlopen.side_effect = HTTPError(
            url="https://clob.polymarket.com/markets",
            code=500, msg="Internal Server Error",
            hdrs=MagicMock(), fp=BytesIO(b'{"error": "server error"}'),
        )
        client = _make_client()
        try:
            client.get_markets()
            assert False, "Should have raised"
        except PolymarketRestError as e:
            assert "500" in str(e)
