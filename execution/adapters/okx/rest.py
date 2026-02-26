# execution/adapters/okx/rest.py
"""OKX REST client — sync HTTP client for OKX API v5."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class OkxRestError(RuntimeError):
    pass


class OkxRetryableError(OkxRestError):
    pass


class OkxNonRetryableError(OkxRestError):
    pass


@dataclass(frozen=True, slots=True)
class OkxRestConfig:
    base_url: str = "https://www.okx.com"
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    timeout_s: float = 10.0
    simulated: bool = False  # x-simulated-trading header


def _iso_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _sign(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    message = timestamp + method.upper() + path + body
    mac = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    )
    return base64.b64encode(mac.digest()).decode("utf-8")


class OkxRestClient:
    """Sync REST client for OKX API v5."""

    def __init__(self, cfg: OkxRestConfig) -> None:
        self._cfg = cfg

    def request_signed(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        ts = _iso_ts()
        m = method.upper()

        url = f"{self._cfg.base_url}{path}"
        body_str = ""

        if m == "GET" and params:
            qs = urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{qs}"
            sign_path = f"{path}?{qs}"
        else:
            sign_path = path
            if body:
                body_str = json.dumps(body)

        signature = _sign(self._cfg.secret_key, ts, m, sign_path, body_str)

        headers = {
            "OK-ACCESS-KEY": self._cfg.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self._cfg.passphrase,
            "Content-Type": "application/json",
        }

        if self._cfg.simulated:
            headers["x-simulated-trading"] = "1"

        return self._send(method=m, url=url, body=body_str, headers=headers)

    def request_public(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self._cfg.base_url}{path}"
        if params:
            qs = urlencode({k: v for k, v in params.items() if v is not None})
            url = f"{url}?{qs}"

        headers = {"Content-Type": "application/json"}
        return self._send(method=method.upper(), url=url, body="", headers=headers)

    def _send(
        self,
        *,
        method: str,
        url: str,
        body: str,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        data = body.encode("utf-8") if body else None
        req = Request(url=url, data=data, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                result = json.loads(raw)
                # OKX uses {"code": "0", "data": [...]} format
                if isinstance(result, dict) and result.get("code") != "0":
                    code = result.get("code", "")
                    msg = result.get("msg", "")
                    raise OkxNonRetryableError(f"OKX error {code}: {msg}")
                return result
        except json.JSONDecodeError as e:
            raise OkxRetryableError(f"Invalid JSON: {e}") from e
        except HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            if e.code in (429,) or 500 <= e.code <= 599:
                raise OkxRetryableError(f"HTTP {e.code}: {raw}") from e
            raise OkxNonRetryableError(f"HTTP {e.code}: {raw}") from e
        except URLError as e:
            raise OkxRetryableError(f"Network error: {e}") from e
