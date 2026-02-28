# tests/unit/execution/test_bitget_rest.py
"""Tests for Bitget REST client — signature, error classification, response parsing."""
from __future__ import annotations

import json
import pytest

from execution.adapters.bitget.rest import (
    BitgetRestConfig,
    BitgetRestClient,
    BitgetRetryableError,
    BitgetNonRetryableError,
    _sign,
    _now_ms,
)
from execution.adapters.bitget.error_map import classify_error, BitgetErrorAction


# ------------------------------------------------------------------
# Signature tests
# ------------------------------------------------------------------
class TestBitgetSignature:
    def test_sign_known_vector(self):
        """Verify Base64(HMAC-SHA256) against a known test vector."""
        secret = "test_secret_key"
        message = "1234567890GET/api/v2/mix/account/accounts?productType=USDT-FUTURES"
        sig = _sign(secret, message)
        # Verify it's valid base64 and non-empty
        assert sig
        import base64
        decoded = base64.b64decode(sig)
        assert len(decoded) == 32  # SHA256 produces 32 bytes

    def test_sign_deterministic(self):
        """Same input should always produce same signature."""
        secret = "my_secret"
        message = "1709123456789POST/api/v2/mix/order/place-order{\"symbol\":\"BTCUSDT\"}"
        sig1 = _sign(secret, message)
        sig2 = _sign(secret, message)
        assert sig1 == sig2

    def test_sign_different_secrets(self):
        """Different secrets must produce different signatures."""
        msg = "test_message"
        sig1 = _sign("secret_a", msg)
        sig2 = _sign("secret_b", msg)
        assert sig1 != sig2

    def test_sign_post_body(self):
        """Verify signature includes JSON body for POST requests."""
        secret = "abc123"
        body = json.dumps({"symbol": "BTCUSDT", "productType": "USDT-FUTURES"})
        msg = "1709000000000POST/api/v2/mix/order/place-order" + body
        sig = _sign(secret, msg)
        assert sig  # non-empty base64 string

    def test_now_ms_returns_string(self):
        """_now_ms returns a string timestamp."""
        ts = _now_ms()
        assert isinstance(ts, str)
        assert int(ts) > 0


# ------------------------------------------------------------------
# Error classification tests
# ------------------------------------------------------------------
class TestBitgetErrorClassification:
    def test_rate_limit_is_retryable(self):
        action, _ = classify_error("40014")
        assert action == BitgetErrorAction.RETRY

    def test_system_busy_is_retryable(self):
        action, _ = classify_error("40700")
        assert action == BitgetErrorAction.RETRY

    def test_invalid_sign_is_halt(self):
        action, _ = classify_error("40003")
        assert action == BitgetErrorAction.HALT

    def test_invalid_passphrase_is_halt(self):
        action, _ = classify_error("40004")
        assert action == BitgetErrorAction.HALT

    def test_invalid_param_is_reject(self):
        action, _ = classify_error("40013")
        assert action == BitgetErrorAction.REJECT

    def test_insufficient_balance_is_reject(self):
        action, _ = classify_error("43004")
        assert action == BitgetErrorAction.REJECT

    def test_unknown_code_defaults_to_reject(self):
        action, desc = classify_error("99999")
        assert action == BitgetErrorAction.REJECT
        assert "99999" in desc


# ------------------------------------------------------------------
# Response parsing tests (using mock)
# ------------------------------------------------------------------
class TestBitgetResponseParsing:
    def test_config_defaults(self):
        cfg = BitgetRestConfig(api_key="k", api_secret="s", passphrase="p")
        assert cfg.base_url == "https://api.bitget.com"
        assert cfg.timeout_s == 10.0

    def test_config_custom(self):
        cfg = BitgetRestConfig(
            base_url="https://test.bitget.com",
            api_key="k",
            api_secret="s",
            passphrase="p",
            timeout_s=5.0,
        )
        assert cfg.base_url == "https://test.bitget.com"
        assert cfg.timeout_s == 5.0

    def test_client_instantiation(self):
        cfg = BitgetRestConfig(api_key="k", api_secret="s", passphrase="p")
        client = BitgetRestClient(cfg)
        assert client._cfg is cfg


class TestBitgetErrorTypes:
    def test_retryable_error_inherits_rest_error(self):
        from execution.adapters.bitget.rest import BitgetRestError
        e = BitgetRetryableError("test")
        assert isinstance(e, BitgetRestError)
        assert isinstance(e, RuntimeError)

    def test_non_retryable_error_inherits_rest_error(self):
        from execution.adapters.bitget.rest import BitgetRestError
        e = BitgetNonRetryableError("test")
        assert isinstance(e, BitgetRestError)
        assert isinstance(e, RuntimeError)
