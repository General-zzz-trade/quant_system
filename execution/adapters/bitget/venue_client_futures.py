# execution/adapters/bitget/venue_client_futures.py
"""Full VenueAdapter implementation for Bitget USDT-M futures."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from execution.adapters.bitget.mapper_balance import map_balances
from execution.adapters.bitget.mapper_fill import BitgetFillMapper
from execution.adapters.bitget.mapper_instrument import map_instruments
from execution.adapters.bitget.mapper_order import BitgetOrderMapper
from execution.adapters.bitget.mapper_position import map_positions
from execution.adapters.bitget.order_gateway import BitgetFuturesOrderGateway
from execution.adapters.bitget.rest import BitgetRetryableError, BitgetNonRetryableError
from execution.adapters.bitget.rest_client import BitgetFuturesRestClient
from execution.bridge.execution_bridge import RetryableVenueError, NonRetryableVenueError
from execution.models.balances import BalanceSnapshot, CanonicalBalance
from execution.models.fills import CanonicalFill
from execution.models.instruments import InstrumentInfo
from execution.models.orders import CanonicalOrder
from execution.models.positions import VenuePosition

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BitgetFuturesFullVenueClient:
    """Full VenueAdapter for Bitget USDT-M futures.

    Provides:
    - Instrument listing (contracts)
    - Balance/position/order/fill queries via REST
    - Order submission/cancellation via OrderGateway

    Existing venue_client.py is preserved for backward compatibility.
    This class extends it with read-side queries using rest_client + mappers.
    """

    venue: str = "bitget"

    rest_client: BitgetFuturesRestClient = None  # type: ignore[assignment]
    order_gateway: BitgetFuturesOrderGateway = None  # type: ignore[assignment]

    _order_mapper: BitgetOrderMapper = None  # type: ignore[assignment]
    _fill_mapper: BitgetFillMapper = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._order_mapper is None:
            object.__setattr__(self, "_order_mapper", BitgetOrderMapper(venue=self.venue))
        if self._fill_mapper is None:
            object.__setattr__(self, "_fill_mapper", BitgetFillMapper(venue=self.venue))

    def list_instruments(self) -> Tuple[InstrumentInfo, ...]:
        raws = self.rest_client.get_contracts()
        return tuple(map_instruments(list(raws)))

    def get_balances(self) -> BalanceSnapshot:
        raws = self.rest_client.get_accounts()
        balances = map_balances(list(raws))
        ts_ms = int(time.time() * 1000)
        return BalanceSnapshot(
            venue=self.venue,
            balances=tuple(balances),
            ts_ms=ts_ms,
        )

    def get_positions(self) -> Tuple[VenuePosition, ...]:
        raws = self.rest_client.get_positions()
        return tuple(map_positions(list(raws)))

    def get_open_orders(self, *, symbol: str = "") -> Tuple[CanonicalOrder, ...]:
        raws = self.rest_client.get_pending_orders(symbol=symbol)
        result = []
        for raw in raws:
            try:
                result.append(self._order_mapper.map_order(raw))
            except (ValueError, TypeError) as e:
                logger.warning("Skipping unmappable order: %s", e)
        return tuple(result)

    def get_recent_fills(self, *, symbol: str = "", since_ms: int = 0) -> Tuple[CanonicalFill, ...]:
        if not symbol:
            return ()
        raws = self.rest_client.get_fills(symbol=symbol)
        result = []
        for raw in raws:
            try:
                fill = self._fill_mapper.map_fill(raw)
                if fill.ts_ms >= since_ms:
                    result.append(fill)
            except (ValueError, TypeError) as e:
                logger.warning("Skipping unmappable fill: %s", e)
        return tuple(result)

    def submit_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.order_gateway.submit_order(cmd)
        except BitgetRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BitgetNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e

    def cancel_order(self, cmd: Any) -> Mapping[str, Any]:
        try:
            return self.order_gateway.cancel_order(cmd)
        except BitgetRetryableError as e:
            raise RetryableVenueError(str(e)) from e
        except BitgetNonRetryableError as e:
            raise NonRetryableVenueError(str(e)) from e
