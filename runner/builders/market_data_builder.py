# runner/builders/market_data_builder.py
"""Phase 9: market data runtime — WS + REST fallback.

Extracted from LiveRunner._build_market_data().
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_subscriptions(
    symbols: tuple,
    kline_interval: str,
    multi_interval_symbols: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, str]]:
    """Build a deduplicated list of WS subscription configs.

    Each entry is a dict with keys:
      - ``symbol``   — uppercase symbol string, e.g. ``"ETHUSDT"``
      - ``interval`` — kline interval string, e.g. ``"60"`` or ``"15"``
      - ``stream``   — Binance stream name, e.g. ``"ethusdt@kline_60"``

    For symbols present in *multi_interval_symbols*, one entry is emitted per
    requested interval.  All other symbols receive a single entry using
    *kline_interval*.  Duplicate (symbol, interval) pairs are silently dropped
    so callers can safely merge the default interval with the multi-interval
    list without creating duplicates.
    """
    seen: set[tuple[str, str]] = set()
    result: List[Dict[str, str]] = []

    for sym in symbols:
        intervals: List[str]
        if multi_interval_symbols and sym in multi_interval_symbols:
            intervals = list(multi_interval_symbols[sym])
        else:
            intervals = [kline_interval]

        for ivl in intervals:
            key = (sym, ivl)
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "symbol": sym,
                    "interval": ivl,
                    "stream": f"{sym.lower()}@kline_{ivl}",
                }
            )

    return result


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
