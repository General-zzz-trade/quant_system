import hmac
import hashlib
from unittest.mock import patch

from execution.adapters.polymarket.auth import PolymarketAuth
from execution.adapters.polymarket.config import PolymarketConfig


def test_sign_request_returns_required_headers():
    auth = PolymarketAuth(api_key="mykey", api_secret="mysecret")
    with patch("execution.adapters.polymarket.auth.time") as mock_time:
        mock_time.time.return_value = 1700000000
        headers = auth.sign_request("GET", "/markets")
    assert headers["POLY_ADDRESS"] == "mykey"
    assert headers["POLY_TIMESTAMP"] == "1700000000"
    assert headers["POLY_NONCE"] == "1700000000"
    # verify signature
    expected_msg = "1700000000GET/markets"
    expected_sig = hmac.new(b"mysecret", expected_msg.encode(), hashlib.sha256).hexdigest()
    assert headers["POLY_SIGNATURE"] == expected_sig


def test_sign_request_with_body():
    auth = PolymarketAuth(api_key="k", api_secret="s")
    with patch("execution.adapters.polymarket.auth.time") as mock_time:
        mock_time.time.return_value = 1700000001
        headers = auth.sign_request("POST", "/order", body='{"side":"buy"}')
    expected_msg = '1700000001POST/order{"side":"buy"}'
    expected_sig = hmac.new(b"s", expected_msg.encode(), hashlib.sha256).hexdigest()
    assert headers["POLY_SIGNATURE"] == expected_sig


def test_polymarket_config_defaults():
    cfg = PolymarketConfig()
    assert cfg.base_url == "https://clob.polymarket.com"
    assert cfg.scan_interval_sec == 3600
    assert cfg.min_liquidity_usd == 10_000
    assert cfg.max_position_pct == 0.10
    assert cfg.kelly_fraction == 0.5
    assert "BTC" in cfg.crypto_keywords
    assert "Ethereum" in cfg.crypto_keywords


def test_polymarket_config_custom():
    cfg = PolymarketConfig(api_key="abc", api_secret="xyz", max_position_pct=0.05)
    assert cfg.api_key == "abc"
    assert cfg.api_secret == "xyz"
    assert cfg.max_position_pct == 0.05
