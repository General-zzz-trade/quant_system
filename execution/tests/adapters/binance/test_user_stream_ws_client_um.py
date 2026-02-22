from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from execution.adapters.binance.ws_user_stream_um import BinanceUmUserStreamWsClient, UserStreamWsConfig
from execution.adapters.binance.user_stream_processor_um import BinanceUmUserStreamProcessor


@dataclass
class FakeListenKeyMgr:
    lk: str = "LK-1"
    # tick 触发一次更新
    next_lk: Optional[str] = None

    def ensure(self) -> str:
        return self.lk

    def tick(self) -> Optional[str]:
        if self.next_lk is None:
            return None
        x = self.next_lk
        self.lk = x
        self.next_lk = None
        return x


@dataclass
class FakeTransport:
    connects: List[str]
    closes: int
    inbox: List[str]

    def connect(self, url: str) -> None:
        self.connects.append(url)

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self.inbox:
            return self.inbox.pop(0)
        return ""  # timeout/no msg

    def close(self) -> None:
        self.closes += 1


@dataclass
class SpyProcessor(BinanceUmUserStreamProcessor):
    raws: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.raws is None:
            self.raws = []

    def process_raw(self, raw: str) -> None:
        self.raws.append(raw)


@dataclass
class _NoopOrderRouter:
    def ingest_canonical_order(self, order, *, actor=None) -> bool:
        return True


@dataclass
class _NoopFillRouter:
    def ingest_canonical_fill(self, fill, *, actor=None) -> bool:
        return True


def test_ws_client_connects_and_reconnects_on_listen_key_change():
    transport = FakeTransport(connects=[], closes=0, inbox=["{}", "{}"])
    lk = FakeListenKeyMgr(lk="LK-1", next_lk=None)

    proc = SpyProcessor(
        order_router=_NoopOrderRouter(),
        fill_router=_NoopFillRouter(),
        order_mapper=lambda _: None,
        fill_mapper=lambda _: None,
    )

    c = BinanceUmUserStreamWsClient(
        transport=transport,
        listen_key_mgr=lk,
        processor=proc,
        cfg=UserStreamWsConfig(ws_base_url="wss://fstream.binance.com/ws", recv_timeout_s=0.0),
    )

    c.step()
    assert transport.connects == ["wss://fstream.binance.com/ws/LK-1"]

    lk.next_lk = "LK-2"
    c.step()

    # listenKey 更新 -> close + reconnect
    assert transport.closes == 1
    assert transport.connects[-1] == "wss://fstream.binance.com/ws/LK-2"
