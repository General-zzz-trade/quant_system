# execution/adapters/binance/async_rest.py
"""Async Binance REST client using aiohttp."""
from __future__ import annotations

import hmac
import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import aiohttp

from execution.adapters.binance.rest import (
    BinanceRestConfig,
    BinanceRestError,
    BinanceRetryableError,
    BinanceNonRetryableError,
    _encode_params,
    _hmac_sha256_hex,
    _now_ms,
)


class AsyncBinanceRestClient:
    """Async counterpart to BinanceRestClient using aiohttp."""

    def __init__(self, cfg: BinanceRestConfig) -> None:
        self._cfg = cfg
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._cfg.timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def request_signed(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        p: Dict[str, Any] = dict(params or {})
        p.setdefault("recvWindow", self._cfg.recv_window)
        p.setdefault("timestamp", _now_ms())

        qs = _encode_params(p)
        sig = _hmac_sha256_hex(self._cfg.api_secret, qs)
        signed_qs = f"{qs}&signature={sig}" if qs else f"signature={sig}"

        url = f"{self._cfg.base_url}{path}"
        return await self._send(method=method, url=url, body_qs=signed_qs)

    async def request_api_key(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        p = dict(params or {})
        qs = _encode_params(p)
        url = f"{self._cfg.base_url}{path}"
        m = method.upper()

        if m == "GET" and qs:
            url = f"{url}?{qs}"
            body_qs = ""
        else:
            body_qs = qs

        return await self._send_api_key(method=m, url=url, body_qs=body_qs)

    async def _send(self, *, method: str, url: str, body_qs: str) -> Dict[str, Any]:
        session = await self._ensure_session()
        headers = {
            "X-MBX-APIKEY": self._cfg.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        try:
            async with session.request(
                method.upper(), url, data=body_qs.encode("utf-8"), headers=headers,
            ) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    if resp.status in (418, 429) or 500 <= resp.status <= 599:
                        raise BinanceRetryableError(f"HTTP {resp.status}: {raw}")
                    raise BinanceNonRetryableError(f"HTTP {resp.status}: {raw}")
                if not raw.strip():
                    return {}
                return await resp.json()
        except aiohttp.ClientError as e:
            raise BinanceRetryableError(f"Network error: {e}") from e

    async def _send_api_key(self, *, method: str, url: str, body_qs: str) -> Dict[str, Any]:
        session = await self._ensure_session()
        headers = {
            "X-MBX-APIKEY": self._cfg.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = body_qs.encode("utf-8") if body_qs else None
        try:
            async with session.request(method, url, data=data, headers=headers) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    if resp.status in (418, 429) or 500 <= resp.status <= 599:
                        raise BinanceRetryableError(f"HTTP {resp.status}: {raw}")
                    raise BinanceNonRetryableError(f"HTTP {resp.status}: {raw}")
                if not raw.strip():
                    return {}
                return await resp.json()
        except aiohttp.ClientError as e:
            raise BinanceRetryableError(f"Network error: {e}") from e
