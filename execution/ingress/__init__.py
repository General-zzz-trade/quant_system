"""execution.ingress — Inbound message routing and health (Domain 3: core plumbing).

Handles the inbound path from venue WebSocket/REST into the execution pipeline:
  - EventSink: typed callback sink for processed events
  - QuarantineStore: holds malformed or suspicious messages for review
  - SequenceBuffer: reorders out-of-sequence messages
  - StreamHealthMonitor: detects stale/broken venue connections
  - FillIngressRouter / OrderIngressRouter: dedup + route inbound fills/orders
"""
from execution.ingress.sink import EventSink
from execution.ingress.quarantine import QuarantineStore, QuarantineReason, QuarantinedMessage
from execution.ingress.sequence_buffer import SequenceBuffer
from execution.ingress.stream_health import StreamHealthMonitor, StreamStatus, StreamHealthSnapshot
from execution.ingress.router import FillIngressRouter, FillDeduplicator
from execution.ingress.order_router import OrderIngressRouter

__all__ = [
    # Sink
    "EventSink",
    # Quarantine
    "QuarantineStore",
    "QuarantineReason",
    "QuarantinedMessage",
    # Sequence buffer
    "SequenceBuffer",
    # Stream health
    "StreamHealthMonitor",
    "StreamStatus",
    "StreamHealthSnapshot",
    # Routers
    "FillIngressRouter",
    "FillDeduplicator",
    "OrderIngressRouter",
]
