"""Integration test: PolymarketRunner end-to-end flow."""
from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock
from polymarket.runner import PolymarketRunner
from polymarket.config import PolymarketConfig


def _ensure_market_discovery_stub():
    """Create a stub market_discovery module if the real one doesn't exist yet."""
    mod_name = "execution.adapters.polymarket.market_discovery"
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except (ImportError, ModuleNotFoundError):
            mod = types.ModuleType(mod_name)
            mod.filter_crypto_markets = lambda markets, keywords, min_volume=0: []  # type: ignore[attr-defined]
            sys.modules[mod_name] = mod


def test_runner_run_once_with_no_markets():
    _ensure_market_discovery_stub()
    config = PolymarketConfig(api_key="test", api_secret="secret")
    runner = PolymarketRunner(config)
    runner._client = MagicMock()
    runner._client.get_markets.return_value = []
    runner.run_once()
    runner._client.get_markets.assert_called_once()


def test_runner_stop():
    config = PolymarketConfig()
    runner = PolymarketRunner(config)
    runner._running = True
    runner.stop()
    assert not runner._running
