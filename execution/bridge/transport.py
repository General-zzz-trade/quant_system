# execution/bridge/transport.py
"""Transport layer abstraction for venue communication."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """HTTP 传输层响应。"""
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str]
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class Transport(Protocol):
    """
    传输层协议 — 抽象 HTTP/WS 通信。

    实现可以是真实 HTTP 客户端或测试 mock。
    """

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> TransportResponse:
        """发送 HTTP 请求。"""
        ...


class InMemoryTransport:
    """
    内存传输层 — 用于测试。

    预设响应队列，按顺序返回。
    """

    def __init__(self) -> None:
        self._responses: list[TransportResponse] = []
        self._requests: list[Dict[str, Any]] = []

    def enqueue(self, response: TransportResponse) -> None:
        """添加预设响应。"""
        self._responses.append(response)

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> TransportResponse:
        self._requests.append({
            "method": method,
            "path": path,
            "params": dict(params) if params else {},
            "body": dict(body) if body else {},
        })
        if self._responses:
            return self._responses.pop(0)
        return TransportResponse(status_code=200, body={}, headers={})

    @property
    def request_history(self) -> list[Dict[str, Any]]:
        return list(self._requests)
