"""WebSocket transport with exponential backoff reconnection and state machine."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from execution.adapters.binance.ws_transport import WsTransport


class WsConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


@dataclass
class ReconnectingWsTransport:
    """WsTransport wrapper that auto-reconnects with exponential backoff.

    Implements WsTransport protocol. On disconnect, reconnects and
    re-sends any saved subscription messages.

    State machine: DISCONNECTED -> CONNECTING -> CONNECTED -> RECONNECTING -> CONNECTED
                                                           -> CLOSED (terminal)
    """

    inner: WsTransport
    max_retries: int = 10
    base_delay_s: float = 1.0
    max_delay_s: float = 60.0
    on_reconnect: Optional[Callable[[], None]] = None
    on_state_change: Optional[Callable[[WsConnectionState, WsConnectionState], None]] = None

    _url: Optional[str] = field(default=None, repr=False)
    _subscriptions: List[str] = field(default_factory=list, repr=False)
    _state: WsConnectionState = field(default=WsConnectionState.DISCONNECTED, repr=False)
    _attempt: int = field(default=0, repr=False)

    @property
    def state(self) -> WsConnectionState:
        return self._state

    @property
    def connected(self) -> bool:
        return self._state == WsConnectionState.CONNECTED

    def _set_state(self, new_state: WsConnectionState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        if self.on_state_change is not None:
            try:
                self.on_state_change(old, new_state)
            except Exception:
                pass

    def connect(self, url: str) -> None:
        self._url = url
        self._set_state(WsConnectionState.CONNECTING)
        self.inner.connect(url)
        self._set_state(WsConnectionState.CONNECTED)
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
            if self._state == WsConnectionState.CONNECTED:
                return msg
            raise ConnectionError("disconnected")
        except Exception:
            return self._reconnect_and_recv(timeout_s=timeout_s)

    def close(self) -> None:
        self._set_state(WsConnectionState.CLOSED)
        self.inner.close()

    def _reconnect_and_recv(self, *, timeout_s: Optional[float] = None) -> str:
        """Attempt reconnection with exponential backoff + jitter."""
        self._set_state(WsConnectionState.RECONNECTING)
        for attempt in range(self.max_retries):
            delay = min(self.base_delay_s * (2 ** attempt), self.max_delay_s)
            delay += random.uniform(0, delay * 0.1)  # 10% jitter
            time.sleep(delay)
            try:
                self.inner.close()
                if self._url is not None:
                    self.inner.connect(self._url)
                self._set_state(WsConnectionState.CONNECTED)
                self._attempt = 0

                # Re-subscribe
                for sub in self._subscriptions:
                    try:
                        send_fn = getattr(self.inner, "send", None)
                        if send_fn is not None:
                            send_fn(sub)
                    except Exception:
                        pass

                if self.on_reconnect is not None:
                    self.on_reconnect()

                return self.inner.recv(timeout_s=timeout_s)
            except Exception:
                self._set_state(WsConnectionState.RECONNECTING)
                continue

        self._set_state(WsConnectionState.CLOSED)
        raise ConnectionError(
            f"Failed to reconnect after {self.max_retries} attempts"
        )
