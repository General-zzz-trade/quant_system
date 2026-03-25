"""Coverage wave 3 — Binance REST, Bybit client.

Targets:
- execution/adapters/binance/rest.py
- execution/adapters/bybit/client.py
"""
from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

class TestBinanceRestClient:
    def _make_client(self, rate_policy=None):
        from execution.adapters.binance.rest import BinanceRestConfig, BinanceRestClient
        cfg = BinanceRestConfig(
            base_url="https://fapi.binance.com",
            api_key="testkey",
            api_secret="testsecret",
        )
        return BinanceRestClient(cfg, rate_policy=rate_policy)

    def _mock_urlopen(self, response_data: dict):
        """Returns a context manager mock for urlopen."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.getheader.return_value = None
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_config_repr_hides_secrets(self):
        from execution.adapters.binance.rest import BinanceRestConfig
        cfg = BinanceRestConfig(
            base_url="https://fapi.binance.com",
            api_key="mykey",
            api_secret="mysecret",
        )
        r = repr(cfg)
        assert "mykey" not in r
        assert "mysecret" not in r
        assert "***" in r

    def test_encode_params_bool_conversion(self):
        from execution.adapters.binance.rest import _encode_params
        qs = _encode_params({"reduceOnly": True, "foo": False, "bar": None, "val": 123})
        assert "reduceOnly=true" in qs
        assert "foo=false" in qs
        assert "bar" not in qs
        assert "val=123" in qs

    def test_hmac_sha256_hex(self):
        from execution.adapters.binance.rest import _hmac_sha256_hex
        result = _hmac_sha256_hex("secret", "payload")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_request_signed_get(self):
        client = self._make_client()
        resp = self._mock_urlopen({"orderId": 123})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_signed(method="GET", path="/fapi/v1/order", params={"symbol": "BTCUSDT"})
        assert result == {"orderId": 123}

    def test_request_signed_post(self):
        client = self._make_client()
        resp = self._mock_urlopen({"orderId": 456})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_signed(method="POST", path="/fapi/v1/order", params={"symbol": "BTCUSDT"})
        assert result == {"orderId": 456}

    def test_request_signed_rate_limited(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        rate_policy = MagicMock()
        rate_policy.check.return_value = False
        client = self._make_client(rate_policy=rate_policy)
        with pytest.raises(BinanceRetryableError, match="Rate limited"):
            client.request_signed(method="GET", path="/fapi/v1/order")

    def test_request_public_get(self):
        client = self._make_client()
        resp = self._mock_urlopen({"price": "50000"})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_public(method="GET", path="/fapi/v1/ticker/price", params={"symbol": "BTCUSDT"})
        assert result == {"price": "50000"}

    def test_request_public_post(self):
        client = self._make_client()
        resp = self._mock_urlopen({"result": "ok"})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_public(method="POST", path="/fapi/v1/something", params={"x": "1"})
        assert result == {"result": "ok"}

    def test_send_http_error_retryable_429(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 429, "Too Many Requests", {}, BytesIO(b"rate limit"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceRetryableError, match="HTTP 429"):
                client.request_signed(method="GET", path="/fapi/v1/order")

    def test_send_http_error_retryable_500(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 500, "Internal Server Error", {}, BytesIO(b"server error"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceRetryableError, match="HTTP 500"):
                client.request_signed(method="GET", path="/fapi/v1/order")

    def test_send_http_error_non_retryable_400(self):
        from execution.adapters.binance.rest import BinanceNonRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 400, "Bad Request", {}, BytesIO(b"bad params"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceNonRetryableError, match="HTTP 400"):
                client.request_signed(method="GET", path="/fapi/v1/order")

    def test_send_network_error(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        from urllib.error import URLError
        client = self._make_client()
        with patch("execution.adapters.binance.rest.urlopen", side_effect=URLError("timeout")):
            with pytest.raises(BinanceRetryableError, match="Network error"):
                client.request_signed(method="GET", path="/fapi/v1/order")

    def test_send_invalid_json_raises(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not-json{"
        mock_resp.getheader.return_value = None
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("execution.adapters.binance.rest.urlopen", return_value=mock_resp):
            with pytest.raises(BinanceRetryableError, match="Invalid JSON"):
                client.request_signed(method="GET", path="/fapi/v1/order")

    def test_send_empty_response_returns_empty_dict(self):
        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"   "
        mock_resp.getheader.return_value = None
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("execution.adapters.binance.rest.urlopen", return_value=mock_resp):
            result = client.request_signed(method="GET", path="/fapi/v1/order")
        assert result == {}

    def test_sync_rate_limit_headers_weight(self):
        client = self._make_client()
        rate_policy = MagicMock()
        client._rate_policy = rate_policy
        mock_resp = MagicMock()
        mock_resp.getheader.side_effect = lambda h: "150" if h == "X-MBX-USED-WEIGHT-1M" else None
        client._sync_rate_limit_headers(mock_resp)
        rate_policy.sync_used_weight.assert_called_once_with(150)

    def test_sync_rate_limit_headers_no_policy(self):
        client = self._make_client()
        # No rate policy — should not raise
        mock_resp = MagicMock()
        client._sync_rate_limit_headers(mock_resp)

    def test_request_api_key_get(self):
        client = self._make_client()
        resp = self._mock_urlopen({"listenKey": "abc123"})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_api_key(method="GET", path="/fapi/v1/listenKey", params={"x": "1"})
        assert result == {"listenKey": "abc123"}

    def test_request_api_key_post(self):
        client = self._make_client()
        resp = self._mock_urlopen({"listenKey": "newkey"})
        with patch("execution.adapters.binance.rest.urlopen", return_value=resp):
            result = client.request_api_key(method="POST", path="/fapi/v1/listenKey")
        assert result == {"listenKey": "newkey"}

    def test_send_api_key_http_error_429(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 429, "Rate limit", {}, BytesIO(b"rate"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceRetryableError):
                client.request_api_key(method="POST", path="/fapi/v1/listenKey")

    def test_send_api_key_non_retryable(self):
        from execution.adapters.binance.rest import BinanceNonRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 401, "Unauthorized", {}, BytesIO(b"unauth"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceNonRetryableError):
                client.request_api_key(method="POST", path="/fapi/v1/listenKey")

    def test_send_public_retryable_418(self):
        from execution.adapters.binance.rest import BinanceRetryableError
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 418, "I'm a teapot", {}, BytesIO(b"teapot"))
        with patch("execution.adapters.binance.rest.urlopen", side_effect=err):
            with pytest.raises(BinanceRetryableError):
                client.request_public(method="GET", path="/fapi/v1/ping")


# ─────────────────────────────────────────────────────────────
# bybit/client.py
# ─────────────────────────────────────────────────────────────

class TestBybitRestClient:
    def _make_client(self):
        from execution.adapters.bybit.client import BybitRestClient
        from execution.adapters.bybit.config import BybitConfig
        cfg = BybitConfig(api_key="testkey", api_secret="testsecret")
        return BybitRestClient(cfg)

    def _mock_urlopen(self, data: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_sign_produces_hex_string(self):
        client = self._make_client()
        sig = client._sign("12345", "testparams")
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)

    def test_headers_contain_required_fields(self):
        client = self._make_client()
        headers = client._headers("12345", "sig123")
        assert headers["X-BAPI-API-KEY"] == "testkey"
        assert headers["X-BAPI-SIGN"] == "sig123"
        assert headers["X-BAPI-TIMESTAMP"] == "12345"
        assert headers["Content-Type"] == "application/json"

    def test_get_success(self):
        client = self._make_client()
        resp = self._mock_urlopen({"retCode": 0, "result": {"price": "50000"}})
        with patch("execution.adapters.bybit.client.urlopen", return_value=resp):
            result = client.get("/v5/market/tickers", {"symbol": "BTCUSDT"})
        assert result["retCode"] == 0

    def test_get_no_params(self):
        client = self._make_client()
        resp = self._mock_urlopen({"retCode": 0})
        with patch("execution.adapters.bybit.client.urlopen", return_value=resp):
            result = client.get("/v5/market/time")
        assert result["retCode"] == 0

    def test_post_success(self):
        client = self._make_client()
        resp = self._mock_urlopen({"retCode": 0, "result": {"orderId": "abc"}})
        with patch("execution.adapters.bybit.client.urlopen", return_value=resp):
            result = client.post("/v5/order/create", {"symbol": "BTCUSDT", "qty": "0.001"})
        assert result["retCode"] == 0

    def test_api_error_non_zero_ret_code_logs_warning(self):
        client = self._make_client()
        resp = self._mock_urlopen({"retCode": 10001, "retMsg": "param error"})
        with patch("execution.adapters.bybit.client.urlopen", return_value=resp):
            result = client.get("/v5/order/create")
        assert result["retCode"] == 10001

    def test_http_error_retryable_429(self):
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 429, "Too Many Requests", {}, BytesIO(b"rate limit"))
        with patch("execution.adapters.bybit.client.urlopen", side_effect=err):
            result = client.get("/v5/order/create")
        assert result["retCode"] == 429
        assert result["retryable"] is True

    def test_http_error_retryable_500(self):
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 500, "Internal Server Error", {}, BytesIO(b"err"))
        with patch("execution.adapters.bybit.client.urlopen", side_effect=err):
            result = client.post("/v5/order/create", {})
        assert result["retCode"] == 500
        assert result["retryable"] is True

    def test_http_error_non_retryable_400(self):
        from urllib.error import HTTPError
        client = self._make_client()
        err = HTTPError("http://x.com", 400, "Bad Request", {}, BytesIO(b"bad"))
        with patch("execution.adapters.bybit.client.urlopen", side_effect=err):
            result = client.get("/v5/order")
        assert result["retCode"] == 400
        assert result["retryable"] is False

    def test_network_exception_retryable(self):
        client = self._make_client()
        with patch("execution.adapters.bybit.client.urlopen", side_effect=Exception("network error")):
            result = client.get("/v5/order")
        assert result["retCode"] == -1
        assert result["retryable"] is True



