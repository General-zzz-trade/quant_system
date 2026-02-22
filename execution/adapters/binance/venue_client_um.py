# execution/adapters/binance/venue_client_um.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from execution.adapters.binance.order_gateway_um import BinanceUmFuturesOrderGateway
from execution.adapters.binance.rest import (
    BinanceRetryableError,
    BinanceNonRetryableError,
)
from execution.bridge.execution_bridge import RetryableVenueError, NonRetryableVenueError


@dataclass(slots=True)
class BinanceUmFuturesVenueClient:
    """
    ExecutionBridge 只认 VenueClient 接口（submit_order/cancel_order）。
    这里负责：
    - 调用 gateway
    - 把 Binance 错误分成 retryable / non-retryable
    """
    gw: BinanceUmFuturesOrderGateway

    def submit_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.gw.submit_order(cmd)
        except BinanceRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BinanceNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e

    def cancel_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.gw.cancel_order(cmd)
        except BinanceRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BinanceNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e
