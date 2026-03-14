# execution/adapters/binance/rest_client.py
"""High-level Binance REST client — wraps rest.py with typed methods."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from execution.adapters.binance.rest import BinanceRestClient


class BinanceUmRestClient:
    """Binance USDT-M Futures typed REST client。"""

    def __init__(self, client: BinanceRestClient) -> None:
        self._client = client

    def get_exchange_info(self) -> Dict[str, Any]:
        return self._client.request_signed(method="GET", path="/fapi/v1/exchangeInfo")

    def get_account(self) -> Dict[str, Any]:
        return self._client.request_signed(method="GET", path="/fapi/v2/account")

    def get_balances(self) -> Sequence[Dict[str, Any]]:
        return self._client.request_signed(method="GET", path="/fapi/v2/balance")

    def get_positions(self) -> Sequence[Dict[str, Any]]:
        acct = self.get_account()
        return acct.get("positions", [])

    def get_open_orders(self, symbol: str = "") -> Sequence[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol
        return self._client.request_signed(method="GET", path="/fapi/v1/openOrders", params=params)

    def get_recent_trades(self, symbol: str, limit: int = 100) -> Sequence[Dict[str, Any]]:
        return self._client.request_signed(
            method="GET",
            path="/fapi/v1/userTrades",
            params={"symbol": symbol, "limit": limit},
        )
