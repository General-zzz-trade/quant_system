"""Factory for choosing the best available WS transport."""
from __future__ import annotations

import logging
from execution.adapters.binance.ws_transport import WsTransport

logger = logging.getLogger(__name__)


def create_ws_transport() -> WsTransport:
    """Create the best available WS transport.

    Prefers Rust (GIL-free recv) over Python websocket-client.
    """
    try:
        from execution.adapters.binance.ws_transport_rust import RustWsTransport
        logger.info("Using Rust WS transport (GIL-free recv)")
        return RustWsTransport()
    except ImportError:
        pass

    from execution.adapters.binance.ws_transport_websocket_client import (
        WebsocketClientTransport,
    )
    logger.info("Using Python websocket-client transport")
    return WebsocketClientTransport()
