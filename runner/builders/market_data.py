# runner/builders/market_data.py
"""Market data subsystem builder — extracted from LiveRunner.build()."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MarketDataSubsystem:
    """Assembled market data components."""
    loop: Any
    runtime: Any
    user_stream: Optional[Any] = None


def build_market_data_subsystem(
    config: Any,
    *,
    coordinator: Any,
    transport: Any = None,
    user_stream_transport: Any = None,
    venue_client: Any = None,
    report: Any = None,
) -> MarketDataSubsystem:
    """Build EngineLoop, WS runtime, and optional user stream."""
    from engine.loop import EngineLoop, LoopConfig
    from engine.guards import build_basic_guard, GuardConfig

    loop_cfg = LoopConfig(max_events_per_tick=100)
    guard = build_basic_guard(config=GuardConfig())
    loop = EngineLoop(
        coordinator=coordinator,
        config=loop_cfg,
        guard=guard,
    )

    # WS + REST runtime
    from execution.adapters.binance.urls import BinanceUrls
    binance_urls = BinanceUrls.testnet() if config.testnet else BinanceUrls.production()

    from execution.adapters.binance.ws_market import BinanceWsMarketClient
    from execution.adapters.binance.rest import BinanceRestClient
    from execution.adapters.binance.runtime import BinanceMarketDataRuntime

    ws_client = BinanceWsMarketClient(
        ws_base_url=binance_urls.ws_market,
        transport=transport,
    )
    rest_fallback = None
    if isinstance(venue_client, BinanceRestClient):
        rest_fallback = venue_client

    runtime = BinanceMarketDataRuntime(
        ws_client=ws_client,
        rest_fallback=rest_fallback,
        symbols=config.symbols,
        kline_interval=config.kline_interval,
    )
    loop.attach_runtime(runtime)

    return MarketDataSubsystem(
        loop=loop,
        runtime=runtime,
        user_stream=None,
    )
