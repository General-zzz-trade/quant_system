"""In-process pub/sub message bus.

Provides a ``MessageBus`` protocol and a simple ``InProcessMessageBus``
implementation for single-process deployments.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class MessageBus(Protocol):
    """Protocol for publish/subscribe message buses."""

    def publish(self, topic: str, msg: bytes) -> None: ...
    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None: ...
    def unsubscribe(self, topic: str) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class InProcessMessageBus:
    """Simple in-process pub/sub message bus for single-process deployment.

    Thread-safe. Handlers are invoked synchronously on the publishing
    thread. Handler exceptions are logged and do not prevent delivery
    to remaining handlers.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[bytes], None]]] = {}
        self._lock = threading.Lock()
        self._started = False

    def publish(self, topic: str, msg: bytes) -> None:
        """Publish a message to all handlers subscribed to *topic*."""
        with self._lock:
            handlers = list(self._handlers.get(topic, []))

        for h in handlers:
            try:
                h(msg)
            except Exception:
                logger.exception("Handler error on topic %s", topic)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        """Register a handler for *topic*."""
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str) -> None:
        """Remove all handlers for *topic*."""
        with self._lock:
            self._handlers.pop(topic, None)

    def unsubscribe_handler(
        self, topic: str, handler: Callable[[bytes], None]
    ) -> None:
        """Remove a specific handler from *topic*."""
        with self._lock:
            handlers = self._handlers.get(topic, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    @property
    def topics(self) -> list[str]:
        with self._lock:
            return list(self._handlers.keys())
