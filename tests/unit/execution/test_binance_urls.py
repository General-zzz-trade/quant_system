# tests/unit/execution/test_binance_urls.py
"""Tests for Binance URL registry."""
from execution.adapters.binance.urls import BinanceURLs, resolve_binance_urls


class TestBinanceURLs:
    def test_production_urls(self):
        urls = resolve_binance_urls(testnet=False)
        assert urls.rest_base == "https://fapi.binance.com"
        assert urls.ws_market_stream == "wss://fstream.binance.com/stream"
        assert urls.ws_user_stream == "wss://fstream.binance.com/ws"

    def test_testnet_urls(self):
        urls = resolve_binance_urls(testnet=True)
        assert urls.rest_base == "https://testnet.binancefuture.com"
        assert urls.ws_market_stream == "wss://stream.binancefuture.com/stream"
        assert urls.ws_user_stream == "wss://stream.binancefuture.com/ws"

    def test_default_is_production(self):
        urls = resolve_binance_urls()
        assert urls.rest_base == "https://fapi.binance.com"

    def test_urls_are_frozen(self):
        urls = resolve_binance_urls()
        assert isinstance(urls, BinanceURLs)
        import dataclasses
        assert dataclasses.is_dataclass(urls)
