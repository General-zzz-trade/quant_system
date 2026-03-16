# execution/adapters/bybit/client.py
"""Bybit V5 REST client — handles authentication, signing, and HTTP requests."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from execution.adapters.bybit.config import BybitConfig

logger = logging.getLogger(__name__)


class BybitRestClient:
    """Low-level Bybit V5 REST API client with HMAC-SHA256 signing."""

    def __init__(self, config: BybitConfig) -> None:
        self._config = config

    def _sign(self, ts: str, params: str) -> str:
        """Compute HMAC-SHA256 signature per Bybit V5 spec."""
        msg = ts + self._config.api_key + self._config.recv_window + params
        return hmac.new(
            self._config.api_secret.encode(),
            msg.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, ts: str, signature: str) -> dict[str, str]:
        return {
            "X-BAPI-API-KEY": self._config.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": self._config.recv_window,
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: dict[str, str] | None = None) -> dict:
        """Authenticated GET request."""
        ts = str(int(time.time() * 1000))
        query = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        signature = self._sign(ts, query)
        url = f"{self._config.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        req = Request(url, headers=self._headers(ts, signature))
        return self._send(req)

    def post(self, path: str, body: dict[str, Any]) -> dict:
        """Authenticated POST request."""
        ts = str(int(time.time() * 1000))
        body_str = json.dumps(body)
        signature = self._sign(ts, body_str)
        url = f"{self._config.base_url}{path}"
        req = Request(
            url, data=body_str.encode(),
            headers=self._headers(ts, signature), method="POST",
        )
        return self._send(req)

    def _send(self, req: Request) -> dict:
        """Execute HTTP request and parse JSON response."""
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("retCode", -1) != 0:
                    logger.warning(
                        "Bybit API error: retCode=%s msg=%s",
                        data.get("retCode"), data.get("retMsg"),
                    )
                return data
        except HTTPError as e:
            body = e.read().decode(errors="ignore")
            retryable = e.code in (429, 500, 502, 503, 504)
            logger.error("Bybit HTTP %d (retryable=%s): %s", e.code, retryable, body[:200])
            return {"retCode": e.code, "retMsg": body[:200], "retryable": retryable}
        except Exception as e:
            # Timeouts and network errors are retryable
            logger.error("Bybit request failed: %s", e)
            return {"retCode": -1, "retMsg": str(e), "retryable": True}
