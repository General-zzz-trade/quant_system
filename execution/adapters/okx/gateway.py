# execution/adapters/okx/gateway.py
"""OKX order gateway — maps canonical commands to OKX API v5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from execution.adapters.okx.rest import OkxRestClient


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


_SIDE_MAP = {"buy": "buy", "sell": "sell"}
_ORD_TYPE_MAP = {
    "market": "market",
    "limit": "limit",
    "stop_market": "trigger",
}


@dataclass(slots=True)
class OkxFuturesOrderGateway:
    """Maps canonical commands to OKX Futures API.

    OKX v5 uses a different API structure from Binance:
    - POST /api/v5/trade/order for new orders
    - POST /api/v5/trade/cancel-order for cancellation
    - instId instead of symbol
    - tdMode (trade mode) required: "cross" or "isolated"
    """

    rest: OkxRestClient
    venue: str = "okx"
    td_mode: str = "cross"  # "cross" or "isolated"

    def submit_order(self, cmd: Any) -> Dict[str, Any]:
        inst_id = _get(cmd, "symbol", "inst_id", "sym")
        side = _get(cmd, "side")
        ord_type = _get(cmd, "order_type", "type", default="market")
        qty = str(_get(cmd, "qty", "quantity"))
        price = _get(cmd, "price")
        client_id = _get(cmd, "request_id", "command_id", "client_order_id", default="")
        reduce_only = _get(cmd, "reduce_only", default=False)

        body: Dict[str, Any] = {
            "instId": inst_id,
            "tdMode": self.td_mode,
            "side": _SIDE_MAP.get(side, side),
            "ordType": _ORD_TYPE_MAP.get(ord_type, ord_type),
            "sz": qty,
        }

        if client_id:
            body["clOrdId"] = str(client_id)
        if price is not None:
            body["px"] = str(price)
        if reduce_only:
            body["reduceOnly"] = True

        # OKX futures needs posSide for hedge mode
        # Default to "net" for one-way mode
        body.setdefault("posSide", "net")

        return self.rest.request_signed(
            method="POST",
            path="/api/v5/trade/order",
            body=body,
        )

    def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        inst_id = _get(cmd, "symbol", "inst_id", "sym")
        order_id = _get(cmd, "order_id", default=None)
        client_order_id = _get(cmd, "client_order_id", "orig_client_order_id", default=None)

        body: Dict[str, Any] = {"instId": inst_id}

        if order_id is not None:
            body["ordId"] = str(order_id)
        elif client_order_id is not None:
            body["clOrdId"] = str(client_order_id)
        else:
            raise ValueError("cancel_order requires order_id or client_order_id")

        return self.rest.request_signed(
            method="POST",
            path="/api/v5/trade/cancel-order",
            body=body,
        )

    def get_positions(self) -> Dict[str, Any]:
        return self.rest.request_signed(
            method="GET",
            path="/api/v5/account/positions",
        )

    def get_balance(self) -> Dict[str, Any]:
        return self.rest.request_signed(
            method="GET",
            path="/api/v5/account/balance",
        )
