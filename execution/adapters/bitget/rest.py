# execution/adapters/bitget/rest.py
from __future__ import annotations

import base64
import hmac
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from execution.adapters.bitget.error_map import classify_error, BitgetErrorAction


class BitgetRestError(RuntimeError):
    pass


class BitgetRetryableError(BitgetRestError):
    """Network / rate-limit / 5xx retryable errors."""
    pass


class BitgetNonRetryableError(BitgetRestError):
    """Parameter / auth / logic non-retryable errors."""
    pass


@dataclass(frozen=True, slots=True)
class BitgetRestConfig:
    base_url: str = "https://api.bitget.com"
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    timeout_s: float = 10.0


def _now_ms() -> str:
    return str(int(time.time() * 1000))


def _sign(secret: str, message: str) -> str:
    """Base64(HMAC-SHA256(secret, message))."""
    mac = hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode("utf-8")


class BitgetRestClient:
    def __init__(self, cfg: BitgetRestConfig) -> None:
        self._cfg = cfg

    def request_signed(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        timestamp = _now_ms()
        m = method.upper()

        # Build sign string: timestamp + METHOD + path[?query] + body
        if m == "GET" and params:
            query_string = urlencode(params, doseq=True)
            sign_path = f"{path}?{query_string}"
            body_str = ""
        else:
            query_string = ""
            sign_path = path
            body_str = json.dumps(body) if body else ""

        sign_message = timestamp + m + sign_path + body_str
        signature = _sign(self._cfg.api_secret, sign_message)

        url = f"{self._cfg.base_url}{path}"
        if query_string:
            url = f"{url}?{query_string}"

        headers = {
            "ACCESS-KEY": self._cfg.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self._cfg.passphrase,
            "Content-Type": "application/json",
        }

        data = body_str.encode("utf-8") if body_str else None
        if m == "GET":
            data = None

        return self._send(method=m, url=url, headers=headers, data=data)

    def _send(
        self,
        *,
        method: str,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
    ) -> Dict[str, Any]:
        req = Request(url=url, data=data, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self._cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise BitgetRetryableError(f"Invalid JSON response: {e}") from e
        except HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 or 500 <= e.code <= 599:
                raise BitgetRetryableError(f"HTTP {e.code}: {raw_body}") from e
            raise BitgetNonRetryableError(f"HTTP {e.code}: {raw_body}") from e
        except URLError as e:
            raise BitgetRetryableError(f"Network error: {e}") from e

        # Bitget envelope: {"code": "00000", "msg": "success", "data": ...}
        code = str(parsed.get("code", ""))
        if code == "00000":
            return parsed.get("data", parsed)

        # Classify error by code
        action, desc = classify_error(code)
        msg = f"Bitget API error {code}: {parsed.get('msg', desc)}"
        if action in (BitgetErrorAction.RETRY,):
            raise BitgetRetryableError(msg)
        raise BitgetNonRetryableError(msg)
