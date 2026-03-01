# tests/unit/execution/test_testnet_wiring.py
"""Tests for testnet config wiring — VenueConfig.testnet consumed correctly."""
import os
from unittest.mock import patch

import pytest

from execution.config.venue_config import VenueConfig
from execution.config.load import build_rest_config_from_venue


def _venue(testnet: bool = False, rest_url: str = "") -> VenueConfig:
    return VenueConfig(
        name="binance",
        testnet=testnet,
        rest_url=rest_url,
        api_key_env="TEST_API_KEY",
        api_secret_env="TEST_API_SECRET",
    )


@pytest.fixture(autouse=True)
def _fake_creds():
    with patch.dict(os.environ, {
        "TEST_API_KEY": "fake-key",
        "TEST_API_SECRET": "fake-secret",
    }):
        yield


class TestBuildRestConfigTestnet:
    def test_testnet_true_uses_testnet_url(self):
        cfg = build_rest_config_from_venue(_venue(testnet=True))
        assert cfg.base_url == "https://testnet.binancefuture.com"

    def test_testnet_false_uses_production_url(self):
        cfg = build_rest_config_from_venue(_venue(testnet=False))
        assert cfg.base_url == "https://fapi.binance.com"

    def test_explicit_url_overrides_testnet(self):
        cfg = build_rest_config_from_venue(
            _venue(testnet=True, rest_url="https://custom.example.com")
        )
        assert cfg.base_url == "https://custom.example.com"

    def test_default_venue_is_production(self):
        cfg = build_rest_config_from_venue(_venue())
        assert cfg.base_url == "https://fapi.binance.com"

    def test_missing_credentials_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing credentials"):
                build_rest_config_from_venue(_venue(testnet=True))
