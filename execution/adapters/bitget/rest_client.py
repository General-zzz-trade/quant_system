# execution/adapters/bitget/rest_client.py
"""High-level Bitget REST client — wraps rest.py with typed methods."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from execution.adapters.bitget.rest import BitgetRestClient
from execution.adapters.bitget.schemas import PRODUCT_TYPE


class BitgetFuturesRestClient:
    """Bitget USDT-M Futures typed REST client."""

    def __init__(self, client: BitgetRestClient) -> None:
        self._client = client

    def get_contracts(self) -> Sequence[Dict[str, Any]]:
        result = self._client.request_signed(
            method="GET",
            path="/api/v2/mix/market/contracts",
            params={"productType": PRODUCT_TYPE},
        )
        return result if isinstance(result, list) else []

    def get_accounts(self) -> Sequence[Dict[str, Any]]:
        result = self._client.request_signed(
            method="GET",
            path="/api/v2/mix/account/accounts",
            params={"productType": PRODUCT_TYPE},
        )
        return result if isinstance(result, list) else []

    def get_positions(self) -> Sequence[Dict[str, Any]]:
        result = self._client.request_signed(
            method="GET",
            path="/api/v2/mix/position/all-position",
            params={"productType": PRODUCT_TYPE, "marginCoin": "USDT"},
        )
        return result if isinstance(result, list) else []

    def get_pending_orders(self, symbol: str = "") -> Sequence[Dict[str, Any]]:
        params: Dict[str, Any] = {"productType": PRODUCT_TYPE}
        if symbol:
            params["symbol"] = symbol
        result = self._client.request_signed(
            method="GET",
            path="/api/v2/mix/order/orders-pending",
            params=params,
        )
        return result if isinstance(result, list) else []

    def get_fills(self, symbol: str, limit: int = 100) -> Sequence[Dict[str, Any]]:
        result = self._client.request_signed(
            method="GET",
            path="/api/v2/mix/order/fills",
            params={"productType": PRODUCT_TYPE, "symbol": symbol, "limit": limit},
        )
        return result if isinstance(result, list) else []
