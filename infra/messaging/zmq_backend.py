"""ZeroMQ PUB/SUB message bus for inter-process communication.

Optional dependency: pyzmq. Raises ImportError at instantiation if not
available.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ZmqMessageBus:
    """ZeroMQ PUB/SUB message bus for inter-process communication.

    Publishes on *pub_addr* and subscribes on *sub_addr*. Typically a
    ZMQ proxy (XSUB/XPUB) sits between publishers and subscribers.
    """

    def __init__(
        self,
        *,
        pub_addr: str = "tcp://127.0.0.1:5555",
        sub_addr: str = "tcp://127.0.0.1:5556",
    ) -> None:
        self._pub_addr = pub_addr
        self._sub_addr = sub_addr
        self._handlers: dict[str, list[Callable[[bytes], None]]] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        try:
            import zmq  # type: ignore[import-untyped]
            self._zmq: Any = zmq
            self._context: Any = zmq.Context()
        except ImportError:
            raise ImportError(
                "pyzmq is required for ZmqMessageBus. Install with: pip install pyzmq"
            )

        self._pub_socket: Any = None
        self._sub_socket: Any = None

    def publish(self, topic: str, msg: bytes) -> None:
        """Publish message on topic via ZMQ PUB socket."""
        if self._pub_socket is None:
            return
        frame = topic.encode("utf-8") + b" " + msg
        self._pub_socket.send(frame)

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        """Register a handler for *topic*."""
        with self._lock:
            self._handlers.setdefault(topic, []).append(handler)
        if self._sub_socket is not None:
            self._sub_socket.setsockopt(
                self._zmq.SUBSCRIBE, topic.encode("utf-8")
            )

    def unsubscribe(self, topic: str) -> None:
        """Remove all handlers for *topic*."""
        with self._lock:
            self._handlers.pop(topic, None)
        if self._sub_socket is not None:
            self._sub_socket.setsockopt(
                self._zmq.UNSUBSCRIBE, topic.encode("utf-8")
            )

    def start(self) -> None:
        """Start PUB/SUB sockets and receiver thread."""
        if self._running:
            return

        self._pub_socket = self._context.socket(self._zmq.PUB)
        self._pub_socket.connect(self._pub_addr)

        self._sub_socket = self._context.socket(self._zmq.SUB)
        self._sub_socket.connect(self._sub_addr)

        with self._lock:
            for topic in self._handlers:
                self._sub_socket.setsockopt(
                    self._zmq.SUBSCRIBE, topic.encode("utf-8")
                )

        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop,
            name="zmq-sub-recv",
            daemon=True,
        )
        self._thread.start()
        logger.info("ZmqMessageBus started (pub=%s, sub=%s)", self._pub_addr, self._sub_addr)

    def stop(self) -> None:
        """Stop sockets and receiver thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pub_socket is not None:
            self._pub_socket.close()
            self._pub_socket = None
        if self._sub_socket is not None:
            self._sub_socket.close()
            self._sub_socket = None
        logger.info("ZmqMessageBus stopped")

    def _recv_loop(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._sub_socket, self._zmq.POLLIN)

        while self._running:
            try:
                socks = dict(poller.poll(timeout=500))
            except Exception:
                break

            if self._sub_socket in socks:
                frame = self._sub_socket.recv()
                parts = frame.split(b" ", 1)
                if len(parts) == 2:
                    topic = parts[0].decode("utf-8", errors="replace")
                    payload = parts[1]
                    with self._lock:
                        handlers = list(self._handlers.get(topic, []))
                    for h in handlers:
                        try:
                            h(payload)
                        except Exception:
                            logger.exception("Handler error on topic %s", topic)
