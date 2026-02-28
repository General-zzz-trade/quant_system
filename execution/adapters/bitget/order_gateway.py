# execution/adapters/bitget/order_gateway.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.adapters.bitget.rest import BitgetRestClient
from execution.adapters.bitget.schemas import PRODUCT_TYPE


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


@dataclass(slots=True)
class BitgetFuturesOrderGateway:
    """
    Map canonical commands to Bitget USDT-M Futures REST calls.

    Handles:
    - Parameter mapping (tradeSide, marginMode, marginCoin, force, etc.)
    - clientOid binding from request_id / command_id / client_order_id
    - Delegates to BitgetRestClient for signing and sending
    """
    rest: BitgetRestClient

    def submit_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        side = _get(cmd, "side")
        order_type = _get(cmd, "order_type", "type")
        qty = _get(cmd, "qty", "quantity")
        price = _get(cmd, "price")
        reduce_only = _get(cmd, "reduce_only", default=False)
        tif = _get(cmd, "time_in_force", "tif", default=None)
        client_id = _get(cmd, "request_id", "command_id", "client_order_id", default=None)

        body: Dict[str, Any] = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "side": str(side).lower(),
            "tradeSide": "close" if reduce_only else "open",
            "orderType": str(order_type).lower(),
            "force": str(tif).lower() if tif else "gtc",
            "size": str(qty),
        }

        if client_id is not None:
            body["clientOid"] = str(client_id)

        if price is not None:
            body["price"] = str(price)

        return self.rest.request_signed(
            method="POST",
            path="/api/v2/mix/order/place-order",
            body=body,
        )

    def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        order_id = _get(cmd, "order_id", "bitget_order_id", default=None)
        client_order_id = _get(cmd, "client_order_id", "orig_client_order_id", default=None)

        body: Dict[str, Any] = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
        }

        if order_id is not None:
            body["orderId"] = str(order_id)
        elif client_order_id is not None:
            body["clientOid"] = str(client_order_id)
        else:
            raise ValueError("cancel_order requires order_id or client_order_id")

        return self.rest.request_signed(
            method="POST",
            path="/api/v2/mix/order/cancel-order",
            body=body,
        )
