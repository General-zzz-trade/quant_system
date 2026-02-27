# execution/adapters/binance/async_gateway.py
"""Async Binance UM Futures order gateway."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.adapters.binance.async_rest import AsyncBinanceRestClient


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _upper(x: Optional[str]) -> Optional[str]:
    return x.upper() if isinstance(x, str) else x


@dataclass(slots=True)
class AsyncBinanceUmFuturesOrderGateway:
    """Async counterpart to BinanceUmFuturesOrderGateway.

    Maps canonical commands to Binance UM Futures REST calls, using
    the async REST client for non-blocking IO.
    """
    rest: AsyncBinanceRestClient

    async def submit_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        side = _upper(_get(cmd, "side"))
        typ = _upper(_get(cmd, "order_type", "type"))
        qty = _get(cmd, "qty", "quantity")
        price = _get(cmd, "price")
        tif = _upper(_get(cmd, "time_in_force", "tif", default=None))
        reduce_only = _get(cmd, "reduce_only", default=None)
        client_id = _get(cmd, "request_id", "command_id", "client_order_id", default=None)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": typ,
            "quantity": qty,
            "newClientOrderId": client_id,
        }

        if price is not None:
            params["price"] = price
        if tif is not None:
            params["timeInForce"] = tif
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)

        return await self.rest.request_signed(
            method="POST", path="/fapi/v1/order", params=params,
        )

    async def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        order_id = _get(cmd, "order_id", "binance_order_id", default=None)
        orig_client_order_id = _get(cmd, "client_order_id", "orig_client_order_id", default=None)

        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("cancel_order requires order_id or client_order_id")

        return await self.rest.request_signed(
            method="DELETE", path="/fapi/v1/order", params=params,
        )

    async def close(self) -> None:
        await self.rest.close()
