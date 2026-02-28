# execution/adapters/bitget/venue_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from execution.adapters.bitget.order_gateway import BitgetFuturesOrderGateway
from execution.adapters.bitget.rest import (
    BitgetRetryableError,
    BitgetNonRetryableError,
)
from execution.bridge.execution_bridge import RetryableVenueError, NonRetryableVenueError


@dataclass(slots=True)
class BitgetFuturesVenueClient:
    """
    ExecutionBridge VenueClient interface for Bitget.

    Translates Bitget-specific errors to bridge-level
    RetryableVenueError / NonRetryableVenueError.
    """
    gw: BitgetFuturesOrderGateway

    def submit_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.gw.submit_order(cmd)
        except BitgetRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BitgetNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e

    def cancel_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.gw.cancel_order(cmd)
        except BitgetRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BitgetNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e
