from __future__ import annotations

from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
from execution.adapters.binance.ws_order_adapter import WsOrderAdapter
from runner.builders.user_stream_builder import _resolve_binance_rest_client


def _rest_client() -> BinanceRestClient:
    return BinanceRestClient(
        BinanceRestConfig(
            base_url="https://testnet.binancefuture.com",
            api_key="key",
            api_secret="secret",
        )
    )


def test_resolve_binance_rest_client_accepts_rest_client_directly() -> None:
    rest = _rest_client()
    assert _resolve_binance_rest_client(rest) is rest


def test_resolve_binance_rest_client_unwraps_ws_order_adapter() -> None:
    rest = _rest_client()
    adapter = WsOrderAdapter(rest_adapter=rest, api_key="key", api_secret="secret", testnet=True)
    assert _resolve_binance_rest_client(adapter) is rest


def test_resolve_binance_rest_client_rejects_unrelated_clients() -> None:
    assert _resolve_binance_rest_client(object()) is None

