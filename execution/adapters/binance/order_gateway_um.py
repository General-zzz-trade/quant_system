# execution/adapters/binance/order_gateway_um.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.adapters.binance.rest import BinanceRestClient

logger = logging.getLogger(__name__)


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
class BinanceUmFuturesOrderGateway:
    """
    把你的 canonical commands 转成 Binance UM Futures REST 调用。

    只做：
    - 参数映射
    - request_id / command_id 绑定 newClientOrderId
    - 交给 BinanceRestClient 签名发送
    """
    rest: BinanceRestClient

    def submit_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        side = _upper(_get(cmd, "side"))
        typ = _upper(_get(cmd, "order_type", "type"))
        qty = _get(cmd, "qty", "quantity")
        price = _get(cmd, "price")

        tif = _upper(_get(cmd, "time_in_force", "tif", default=None))
        reduce_only = _get(cmd, "reduce_only", default=None)

        # 用你的 request_id/command_id/client_order_id 绑定 Binance 的 newClientOrderId
        client_id = _get(cmd, "request_id", "command_id", "client_order_id", default=None)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": typ,
            "quantity": qty,
            "newClientOrderId": client_id,
        }

        # LIMIT/STOP 等通常需要 price + timeInForce
        if price is not None:
            params["price"] = price
        if tif is not None:
            params["timeInForce"] = tif

        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)

        return self.rest.request_signed(method="POST", path="/fapi/v1/order", params=params)

    def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        order_id = _get(cmd, "order_id", "binance_order_id", default=None)
        orig_client_order_id = _get(cmd, "client_order_id", "orig_client_order_id", default=None)

        params: Dict[str, Any] = {"symbol": symbol}

        # Binance 要求 orderId / origClientOrderId 二选一
        if order_id is not None:
            params["orderId"] = order_id
        elif orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("cancel_order requires order_id or client_order_id")

        return self.rest.request_signed(method="DELETE", path="/fapi/v1/order", params=params)


@dataclass(slots=True)
class BinanceUmFuturesWsOrderGateway:
    """WS-first order gateway: uses WebSocket API (~4ms) with REST fallback (~30ms).

    Same interface as BinanceUmFuturesOrderGateway.submit_order/cancel_order.
    """
    rest: BinanceRestClient
    ws_gateway: Any = None  # BinanceWsOrderGateway (optional)

    def submit_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        side = _upper(_get(cmd, "side"))
        typ = _upper(_get(cmd, "order_type", "type"))
        qty = _get(cmd, "qty", "quantity")
        price = _get(cmd, "price")
        tif = _upper(_get(cmd, "time_in_force", "tif", default=None))
        reduce_only = _get(cmd, "reduce_only", default=None)
        client_id = _get(cmd, "request_id", "command_id", "client_order_id", default=None)

        # Try WS-API first (fast path: ~4ms)
        if self.ws_gateway is not None and self.ws_gateway.is_running:
            try:
                req_id = self.ws_gateway.submit_order(
                    symbol=symbol, side=side, order_type=typ,
                    quantity=str(qty) if qty is not None else None,
                    price=str(price) if price is not None else None,
                    time_in_force=tif,
                    reduce_only=reduce_only,
                    client_order_id=client_id,
                )
                return {"status": "ws_submitted", "request_id": req_id}
            except Exception as e:
                logger.warning("WS order failed, falling back to REST: %s", e)

        # REST fallback (slow path: ~30-200ms)
        params: Dict[str, Any] = {
            "symbol": symbol, "side": side, "type": typ,
            "quantity": qty, "newClientOrderId": client_id,
        }
        if price is not None:
            params["price"] = price
        if tif is not None:
            params["timeInForce"] = tif
        if reduce_only is not None:
            params["reduceOnly"] = bool(reduce_only)
        return self.rest.request_signed(method="POST", path="/fapi/v1/order", params=params)

    def cancel_order(self, cmd: Any) -> Dict[str, Any]:
        symbol = _get(cmd, "symbol", "sym")
        order_id = _get(cmd, "order_id", "binance_order_id", default=None)
        orig_client_order_id = _get(cmd, "client_order_id", "orig_client_order_id", default=None)

        # Try WS-API first
        if self.ws_gateway is not None and self.ws_gateway.is_running:
            try:
                req_id = self.ws_gateway.cancel_order(
                    symbol=symbol,
                    order_id=order_id,
                    orig_client_order_id=orig_client_order_id,
                )
                return {"status": "ws_submitted", "request_id": req_id}
            except Exception as e:
                logger.warning("WS cancel failed, falling back to REST: %s", e)

        params: Dict[str, Any] = {"symbol": symbol}
        if order_id is not None:
            params["orderId"] = order_id
        elif orig_client_order_id is not None:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("cancel_order requires order_id or client_order_id")
        return self.rest.request_signed(method="DELETE", path="/fapi/v1/order", params=params)
