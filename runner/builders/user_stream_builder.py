# runner/builders/user_stream_builder.py
"""Phase 10: user stream — private fill/order feed.

Extracted from LiveRunner._build_user_stream().
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

from engine.coordinator import EngineCoordinator

logger = logging.getLogger(__name__)


def build_user_stream(
    config: Any,
    venue_client: Any,
    coordinator: EngineCoordinator,
    binance_urls: Any,
    user_stream_transport: Any,
    report: Any,
) -> Optional[Any]:
    """Phase 10: user stream (private fill/order feed).

    Returns user_stream_client or None.
    """
    if not config.shadow_mode:
        from execution.adapters.binance.rest import BinanceRestClient as _BRC2
        if isinstance(venue_client, _BRC2):
            try:
                from execution.adapters.binance.listen_key_um import BinanceUmListenKeyClient
                from execution.adapters.binance.listen_key_manager import (
                    BinanceUmListenKeyManager, ListenKeyManagerConfig,
                )
                from execution.adapters.binance.ws_user_stream_um import (
                    BinanceUmUserStreamWsClient, UserStreamWsConfig,
                )
                from execution.adapters.binance.user_stream_processor_um import (
                    BinanceUmUserStreamProcessor,
                )
                from execution.adapters.binance.mapper_fill import BinanceFillMapper
                from execution.adapters.binance.mapper_order import BinanceOrderMapper
                from execution.ingress.router import FillIngressRouter
                from execution.ingress.order_router import OrderIngressRouter

                class _TimeClock:
                    def now(self) -> float:
                        return time.time()

                fill_router = FillIngressRouter(
                    coordinator=coordinator, default_actor="venue:binance",
                )
                order_router = OrderIngressRouter(
                    coordinator=coordinator, default_actor="venue:binance",
                )
                us_processor = BinanceUmUserStreamProcessor(
                    order_router=order_router,
                    fill_router=fill_router,
                    order_mapper=BinanceOrderMapper(),
                    fill_mapper=BinanceFillMapper(),
                    default_actor="venue:binance",
                )
                lk_client = BinanceUmListenKeyClient(rest=venue_client)
                lk_mgr = BinanceUmListenKeyManager(
                    client=lk_client,
                    clock=_TimeClock(),
                    cfg=ListenKeyManagerConfig(validity_sec=3600, renew_margin_sec=300),
                )

                us_transport = user_stream_transport
                if us_transport is None:
                    from execution.adapters.binance.transport_factory import create_ws_transport as _cwt
                    us_transport = _cwt()

                user_stream_client = BinanceUmUserStreamWsClient(
                    transport=us_transport,
                    listen_key_mgr=lk_mgr,
                    processor=us_processor,
                    cfg=UserStreamWsConfig(
                        ws_base_url=binance_urls.ws_user_stream,
                    ),
                )
                logger.info(
                    "User stream wired (url_base=%s)", binance_urls.ws_user_stream,
                )
                return user_stream_client
            except Exception as e:
                report.record("user_stream", False, str(e))
                logger.warning("User stream setup failed — continuing without", exc_info=True)
                return None
        else:
            return None
    else:
        return None
