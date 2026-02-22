# execution/tests/adapters/binance/test_listen_key_manager_um.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from execution.adapters.binance.listen_key_manager import (
    BinanceUmListenKeyManager,
    ListenKeyManagerConfig,
)


@dataclass
class FakeClock:
    t: float = 0.0
    def now(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


class FakeListenKeyClient:
    def __init__(self) -> None:
        self.created = 0
        self.pinged = 0
        self._lk = "LK-1"
        self._keepalive_raises: Optional[Exception] = None

    def create(self) -> str:
        self.created += 1
        self._lk = f"LK-{self.created}"
        return self._lk

    def keepalive(self, listen_key: str) -> str:
        self.pinged += 1
        if self._keepalive_raises:
            raise self._keepalive_raises
        return listen_key

    def set_keepalive_raises(self, e: Exception) -> None:
        self._keepalive_raises = e

    @staticmethod
    def is_listen_key_missing_error(e: BaseException) -> bool:
        return "-1125" in str(e) or "listenKey does not exist" in str(e)


def test_manager_creates_on_first_tick():
    clk = FakeClock()
    c = FakeListenKeyClient()
    m = BinanceUmListenKeyManager(client=c, clock=clk)

    assert m.listen_key is None
    lk = m.tick()
    assert lk is not None
    assert m.listen_key is not None
    assert c.created == 1


def test_manager_keepalive_near_expiry():
    clk = FakeClock()
    c = FakeListenKeyClient()
    m = BinanceUmListenKeyManager(
        client=c,
        clock=clk,
        cfg=ListenKeyManagerConfig(validity_sec=3600, renew_margin_sec=300),
    )

    m.tick()  # create
    assert c.created == 1
    assert c.pinged == 0

    # 距离过期还很远 → 不 ping
    clk.advance(3600 - 301)
    assert m.tick() is None
    assert c.pinged == 0

    # 进入 renew 窗口 → ping
    clk.advance(2)
    m.tick()
    assert c.pinged == 1


def test_manager_recreates_on_missing_listenkey():
    clk = FakeClock()
    c = FakeListenKeyClient()
    m = BinanceUmListenKeyManager(
        client=c,
        clock=clk,
        cfg=ListenKeyManagerConfig(validity_sec=3600, renew_margin_sec=300),
    )

    m.tick()  # create LK-1
    clk.advance(3600 - 299)  # 进入 renew 窗口

    c.set_keepalive_raises(RuntimeError("(-1125) This listenKey does not exist."))
    m.tick()  # keepalive 失败 → recreate
    assert c.created >= 2
