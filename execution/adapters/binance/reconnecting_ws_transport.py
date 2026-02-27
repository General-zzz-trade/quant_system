"""WebSocket transport with exponential backoff reconnection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from execution.adapters.binance.ws_transport import WsTransport


@dataclass
class ReconnectingWsTransport:
    """WsTransport wrapper that auto-reconnects with exponential backoff.

    Implements WsTransport protocol. On disconnect, reconnects and
    re-sends any saved subscription messages.
    """

    inner: WsTransport
    max_retries: int = 10
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    on_reconnect: Optional[Callable[[], None]] = None

    _url: Optional[str] = field(default=None, repr=False)
    _subscriptions: List[str] = field(default_factory=list, repr=False)
    _connected: bool = field(default=False, repr=False)
    _attempt: int = field(default=0, repr=False)

    def connect(self, url: str) -> None:
        self._url = url
        self.inner.connect(url)
        self._connected = True
        self._attempt = 0

    def send_subscribe(self, message: str) -> None:
        """Send a subscription message and save it for re-subscription on reconnect."""
        if message not in self._subscriptions:
            self._subscriptions.append(message)

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        try:
            msg = self.inner.recv(timeout_s=timeout_s)
            if msg:
                self._attempt = 0
                return msg
            # Empty string may indicate disconnection for some transports
            if self._connected:
                return msg
            raise ConnectionError("disconnected")
        except Exception:
            return self._reconnect_and_recv(timeout_s=timeout_s)

    def close(self) -> None:
        self._connected = False
        self.inner.close()

    def _reconnect_and_recv(self, *, timeout_s: Optional[float] = None) -> str:
        """Attempt reconnection with exponential backoff."""
        self._connected = False
        for attempt in range(self.max_retries):
            delay = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
            time.sleep(delay)
            try:
                self.inner.close()
                if self._url is not None:
                    self.inner.connect(self._url)
                self._connected = True
                self._attempt = 0

                # Re-subscribe
                for sub in self._subscriptions:
                    try:
                        # Attempt to send via inner's send method if available
                        send_fn = getattr(self.inner, "send", None)
                        if send_fn is not None:
                            send_fn(sub)
                    except Exception:
                        pass

                if self.on_reconnect is not None:
                    self.on_reconnect()

                return self.inner.recv(timeout_s=timeout_s)
            except Exception:
                continue

        raise ConnectionError(
            f"Failed to reconnect after {self.max_retries} attempts"
        )
