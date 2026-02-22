# execution/adapters/binance/rest.py
from __future__ import annotations

import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class BinanceRestError(RuntimeError):
    pass


class BinanceRetryableError(BinanceRestError):
    """网络/限流/5xx 等可重试错误"""
    pass


class BinanceNonRetryableError(BinanceRestError):
    """参数/权限/逻辑错误等不可重试"""
    pass


@dataclass(frozen=True, slots=True)
class BinanceRestConfig:
    base_url: str                 # e.g. https://fapi.binance.com
    api_key: str
    api_secret: str
    recv_window: int = 5000
    timeout_s: float = 10.0


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hmac_sha256_hex(secret: str, payload: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_params(params: Mapping[str, Any]) -> str:
    # Binance 期望 application/x-www-form-urlencoded 风格
    # 注意：bool 要转 "true"/"false"（尤其 reduceOnly 等）
    normalized: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            normalized[k] = "true" if v else "false"
        else:
            normalized[k] = v
    return urlencode(normalized, doseq=True)


class BinanceRestClient:
    def __init__(self, cfg: BinanceRestConfig) -> None:
        self._cfg = cfg

    def request_signed(
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
        return self._send(method=method, url=url, body_qs=signed_qs)

    def _send(self, *, method: str, url: str, body_qs: str) -> Dict[str, Any]:
        data = body_qs.encode("utf-8")
        headers = {
            "X-MBX-APIKEY": self._cfg.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        req = Request(url=url, data=data, headers=headers, method=method.upper())

        try:
            with urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                return json.loads(raw)
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            # 429/418/5xx 视为可重试；其余默认不可重试
            if e.code in (418, 429) or 500 <= e.code <= 599:
                raise BinanceRetryableError(f"HTTP {e.code}: {raw}") from e
            raise BinanceNonRetryableError(f"HTTP {e.code}: {raw}") from e
        except URLError as e:
            raise BinanceRetryableError(f"Network error: {e}") from e

    # execution/adapters/binance/rest.py  (在 BinanceRestClient 内新增)
    from urllib.parse import urlencode

    def request_api_key(
            self,
            *,
            method: str,
            path: str,
            params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        仅携带 APIKEY，不签名。
        适用于 listenKey / userDataStream 等端点。
        """
        p = dict(params or {})
        qs = _encode_params(p)

        url = f"{self._cfg.base_url}{path}"
        m = method.upper()

        # GET 参数必须在 query string（Binance 通用规范）:contentReference[oaicite:3]{index=3}
        if m == "GET" and qs:
            url = f"{url}?{qs}"
            body_qs = ""
        else:
            body_qs = qs

        return self._send_api_key(method=m, url=url, body_qs=body_qs)

    def _send_api_key(self, *, method: str, url: str, body_qs: str) -> Dict[str, Any]:
        data = body_qs.encode("utf-8") if body_qs else None
        headers = {
            "X-MBX-APIKEY": self._cfg.api_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        req = Request(url=url, data=data, headers=headers, method=method.upper())

        try:
            with urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                return json.loads(raw)
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            if e.code in (418, 429) or 500 <= e.code <= 599:
                raise BinanceRetryableError(f"HTTP {e.code}: {raw}") from e
            raise BinanceNonRetryableError(f"HTTP {e.code}: {raw}") from e
        except URLError as e:
            raise BinanceRetryableError(f"Network error: {e}") from e


