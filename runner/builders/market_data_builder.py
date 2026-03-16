# runner/builders/market_data_builder.py
"""Phase 9: market data runtime — WS + REST fallback.

Extracted from LiveRunner._build_market_data().
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_market_data(
    config: Any,
    transport: Any,
    venue_client: Any,
    loop: Any,
) -> tuple:
    """Phase 9: market data runtime (WS + REST fallback).

    Returns (runtime, binance_urls).
    """
    from execution.adapters.binance.kline_processor import KlineProcessor
    from execution.adapters.binance.ws_market_stream_um import (
        BinanceUmMarketStreamWsClient,
        MarketStreamConfig,
    )
    from execution.adapters.binance.market_data_runtime import BinanceMarketDataRuntime
    from execution.adapters.binance.urls import resolve_binance_urls

    if config.testnet:
        logger.warning("*** TESTNET MODE — NOT PRODUCTION ***")

    binance_urls = resolve_binance_urls(config.testnet)

    if transport is None:
        from execution.adapters.binance.transport_factory import create_ws_transport
        transport = create_ws_transport()

    ws_url = config.ws_base_url
    if config.testnet:
        ws_url = binance_urls.ws_market_stream

    streams = tuple(
        f"{sym.lower()}@kline_{config.kline_interval}"
        for sym in config.symbols
    )
    processor = KlineProcessor(source="binance.ws.kline")
    ws_client = BinanceUmMarketStreamWsClient(
        transport=transport,
        processor=processor,
        streams=streams,
        cfg=MarketStreamConfig(ws_base_url=ws_url),
    )
    from execution.adapters.binance.rest_kline_source import RestKlineSource
    rest_base = (
        getattr(venue_client, '_cfg', None) and venue_client._cfg.base_url
        or binance_urls.rest_base
    )
    rest_fallback = RestKlineSource(
        base_url=rest_base,
        source="binance.rest.kline",
    )
    runtime = BinanceMarketDataRuntime(
        ws_client=ws_client,
        rest_fallback=rest_fallback,
        symbols=config.symbols,
        kline_interval=config.kline_interval,
    )
    loop.attach_runtime(runtime)

    return runtime, binance_urls
