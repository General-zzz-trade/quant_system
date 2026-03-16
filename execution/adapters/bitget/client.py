"""Bitget V2 REST API client with HMAC-SHA256 signing."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from execution.adapters.bitget.config import BitgetConfig

logger = logging.getLogger(__name__)


class BitgetRestClient:
    """Low-level Bitget V2 REST client.

    Signing spec (Bitget V2):
      sign_string = timestamp + METHOD + requestPath + queryString + body
      signature   = base64(hmac_sha256(secret, sign_string))

    Headers:
      ACCESS-KEY, ACCESS-SIGN, ACCESS-TIMESTAMP, ACCESS-PASSPHRASE
      Content-Type: application/json
    """

    def __init__(self, config: BitgetConfig) -> None:
        self._config = config

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """HMAC-SHA256 + base64 per Bitget V2 spec.

        message = timestamp + METHOD(upper) + requestPath + body
        For GET with query string: path includes '?k=v&...'
        For POST: body is the JSON string
        """
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self._config.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode("utf-8")

    def _headers(self, timestamp: str, signature: str) -> dict[str, str]:
        return {
            "ACCESS-KEY": self._config.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self._config.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    def get(self, path: str, params: dict[str, str] | None = None) -> dict:
        """Authenticated GET request."""
        ts = str(int(time.time() * 1000))
        full_path = path
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_path = f"{path}?{query}"
        signature = self._sign(ts, "GET", full_path)
        url = f"{self._config.base_url}{full_path}"
        req = Request(url, headers=self._headers(ts, signature))
        return self._send(req)

    def post(self, path: str, body: dict[str, Any]) -> dict:
        """Authenticated POST request."""
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body)
        signature = self._sign(ts, "POST", path, body_str)
        url = f"{self._config.base_url}{path}"
        req = Request(
            url, data=body_str.encode("utf-8"),
            headers=self._headers(ts, signature),
            method="POST",
        )
        return self._send(req)

    def _send(self, req: Request) -> dict:
        """Execute HTTP request, parse JSON response."""
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                code = data.get("code", "")
                if str(code) != "00000":
                    logger.warning(
                        "Bitget API error: code=%s msg=%s",
                        code, data.get("msg"),
                    )
                return data
        except HTTPError as e:
            body = e.read().decode(errors="ignore")
            logger.error("Bitget HTTP %d: %s", e.code, body[:300])
            return {"code": str(e.code), "msg": body[:300]}
        except Exception as e:
            logger.error("Bitget request failed: %s", e)
            return {"code": "-1", "msg": str(e)}
