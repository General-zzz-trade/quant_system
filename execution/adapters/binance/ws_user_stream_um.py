from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from execution.adapters.binance.ws_transport import WsTransport
from execution.adapters.binance.user_stream_processor_um import BinanceUmUserStreamProcessor


class ListenKeyManager(Protocol):
    """
    兼容你的 BinanceUmListenKeyManager：ensure()/tick()
    """
    def ensure(self) -> str: ...
    def tick(self) -> Optional[str]: ...


@dataclass(frozen=True, slots=True)
class UserStreamWsConfig:
    ws_base_url: str = "wss://fstream.binance.com/ws"  # 只拼接，不在 tests 做网络验证
    recv_timeout_s: float = 5.0
    reconnect_backoff_s: float = 0.25


@dataclass(slots=True)
class BinanceUmUserStreamWsClient:
    """
    tick 驱动的 WS client：
    - ensure() 获取 listenKey，并连接
    - recv() 收消息交给 processor
    - tick() 若 listenKey 更新/过期 -> 触发重连
    """
    transport: WsTransport
    listen_key_mgr: ListenKeyManager
    processor: BinanceUmUserStreamProcessor
    cfg: UserStreamWsConfig = UserStreamWsConfig()

    _connected_url: Optional[str] = None

    def _url_for(self, listen_key: str) -> str:
        base = self.cfg.ws_base_url.rstrip("/")
        lk = listen_key.strip()
        return f"{base}/{lk}"

    def connect(self) -> str:
        lk = self.listen_key_mgr.ensure()
        url = self._url_for(lk)
        self.transport.connect(url)
        self._connected_url = url
        return url

    def close(self) -> None:
        try:
            self.transport.close()
        finally:
            self._connected_url = None

    def step(self) -> None:
        """
        执行一个 step：
        - 若未连接，先 connect
        - 尝试 recv 一条消息（可能超时）
        - 处理消息
        - tick 检查 listenKey 是否更新，必要时重连
        """
        if not self._connected_url:
            self.connect()

        # 先收消息（即使 timeout 也不报错）
        raw = self.transport.recv(timeout_s=self.cfg.recv_timeout_s)
        if raw:
            # processor 内部 invalid/mismatch 会直接抛错 -> fail-fast
            self.processor.process_raw(raw)

        # 检查 listenKey 是否需要更新
        new_lk = self.listen_key_mgr.tick()
        if new_lk:
            new_url = self._url_for(new_lk)
            if new_url != self._connected_url:
                self.close()
                self.transport.connect(new_url)
                self._connected_url = new_url
