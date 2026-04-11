"""OKX REST client (API v5).

Auth scheme (different from Binance/Bybit):
    sign = base64( HMAC-SHA256(secret, timestamp + method + path + body) )

Headers:
    OK-ACCESS-KEY        — api key
    OK-ACCESS-SIGN       — base64 sign
    OK-ACCESS-TIMESTAMP  — ISO 8601 UTC millisecond, e.g. 2026-04-11T14:00:00.000Z
    OK-ACCESS-PASSPHRASE — passphrase set when creating the API key
    x-simulated-trading  — "1" for demo, omit for live

Body is a JSON string for POST; empty for GET. For GET, query params are
appended to `path` BEFORE signing: path = "/api/v5/foo?a=1&b=2".
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from execution.adapters.okx.urls import OKX_REST_BASE


class OkxRestError(RuntimeError):
    """Generic OKX REST error."""


class OkxRetryableError(OkxRestError):
    """Network / rate-limit / 5xx — caller may retry."""


class OkxNonRetryableError(OkxRestError):
    """Parameter / permission / business logic — do NOT retry."""


@dataclass(frozen=True, slots=True)
class OkxRestConfig:
    api_key: str
    api_secret: str
    passphrase: str
    base_url: str = OKX_REST_BASE
    simulated: bool = False  # True → adds x-simulated-trading: 1 header
    timeout_s: float = 10.0

    def __repr__(self) -> str:
        return (
            f"OkxRestConfig(base_url={self.base_url!r}, "
            f"api_key='***', api_secret='***', passphrase='***', "
            f"simulated={self.simulated}, timeout_s={self.timeout_s})"
        )


def _iso_ts_ms() -> str:
    """OKX timestamp format: 2026-04-11T14:00:00.000Z (UTC, millisecond)."""
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{ms:03d}Z"


def _sign(secret: str, timestamp: str, method: str, path: str, body: str) -> str:
    """HMAC-SHA256 base64 signature per OKX v5 spec."""
    msg = f"{timestamp}{method.upper()}{path}{body}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


class OkxRestClient:
    """Thin REST client.

    Two entry points:
    - `request_public(method, path, params)` — no auth
    - `request_signed(method, path, params=None, body=None)` — full auth

    For GET requests, pass params via `params` (added to query string before
    signing). For POST, pass a dict via `body` — it's serialised to JSON.
    """

    def __init__(self, cfg: OkxRestConfig) -> None:
        self._cfg = cfg

    # ── Public endpoints (no auth) ────────────────────────────────
    def request_public(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        m = method.upper()
        full_path = path
        if m == "GET" and params:
            qs = urlencode({k: v for k, v in params.items() if v is not None})
            if qs:
                full_path = f"{path}?{qs}"
        url = f"{self._cfg.base_url}{full_path}"
        return self._send(method=m, url=url, headers={}, body="")

    # ── Signed endpoints ──────────────────────────────────────────
    def request_signed(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        m = method.upper()
        full_path = path
        body_str = ""

        if m == "GET":
            if params:
                qs = urlencode({k: v for k, v in params.items() if v is not None})
                if qs:
                    full_path = f"{path}?{qs}"
        else:
            if body is not None:
                # OKX expects a JSON body for POST; empty dict → empty string
                body_str = json.dumps(body, separators=(",", ":")) if body else ""

        ts = _iso_ts_ms()
        sig = _sign(self._cfg.api_secret, ts, m, full_path, body_str)

        headers = {
            "OK-ACCESS-KEY": self._cfg.api_key,
            "OK-ACCESS-SIGN": sig,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._cfg.passphrase,
            "Content-Type": "application/json",
        }
        if self._cfg.simulated:
            headers["x-simulated-trading"] = "1"

        url = f"{self._cfg.base_url}{full_path}"
        return self._send(method=m, url=url, headers=headers, body=body_str)

    # ── Transport ─────────────────────────────────────────────────
    def _send(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: str,
    ) -> Dict[str, Any]:
        data = body.encode("utf-8") if body else None
        # OKX sits behind Cloudflare which blocks the default urllib UA.
        # Use a browser-like UA to avoid 403 from edge WAF.
        merged = dict(headers)
        merged.setdefault("User-Agent", "Mozilla/5.0 (quant-system/okx-adapter)")
        req = Request(url=url, data=data, headers=merged, method=method)
        try:
            with urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise OkxRetryableError(f"Invalid JSON response: {e}") from e
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            if e.code in (418, 429) or 500 <= e.code <= 599:
                raise OkxRetryableError(f"HTTP {e.code}: {raw}") from e
            raise OkxNonRetryableError(f"HTTP {e.code}: {raw}") from e
        except URLError as e:
            raise OkxRetryableError(f"Network error: {e}") from e

        # OKX wraps responses with {code, msg, data}. code=="0" = success.
        if isinstance(parsed, dict) and "code" in parsed:
            code = parsed.get("code")
            if code != "0":
                msg = parsed.get("msg", "")
                # Rate limit / busy → retryable
                if code in ("50011", "50013", "50014", "50026"):
                    raise OkxRetryableError(f"OKX {code}: {msg}")
                raise OkxNonRetryableError(f"OKX {code}: {msg} | resp={parsed}")
        return parsed
