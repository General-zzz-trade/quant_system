"""Polymarket CLOB REST client using stdlib urllib."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from execution.adapters.polymarket.auth import PolymarketAuth


class PolymarketRestError(RuntimeError):
    """Base error for Polymarket REST API calls."""
    pass


class PolymarketRetryableError(PolymarketRestError):
    """Network / rate-limit / 5xx errors that may be retried."""
    pass


class PolymarketNonRetryableError(PolymarketRestError):
    """Client errors (4xx) that should not be retried."""
    pass


class PolymarketRestClient:
    """Polymarket CLOB API REST client.

    Follows the same pattern as BinanceRestClient: urllib.request, JSON parsing,
    signed headers via PolymarketAuth.
    """

    def __init__(
        self,
        auth: PolymarketAuth,
        base_url: str = "https://clob.polymarket.com",
        timeout_s: float = 10.0,
    ) -> None:
        self._auth = auth
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_markets(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        active: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch available markets."""
        params: Dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        if active is not None:
            params["active"] = "true" if active else "false"
        return self._request_public(method="GET", path="/markets", params=params)

    def get_orderbook(self, *, token_id: str) -> Dict[str, Any]:
        """Fetch orderbook for a token."""
        return self._request_public(
            method="GET", path="/book", params={"token_id": token_id}
        )

    def get_trades(
        self,
        *,
        token_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent trades for a token."""
        params: Dict[str, Any] = {"token_id": token_id}
        if limit is not None:
            params["limit"] = limit
        return self._request_public(method="GET", path="/trades", params=params)

    # ------------------------------------------------------------------ #
    # Authenticated API
    # ------------------------------------------------------------------ #

    def create_order(
        self,
        *,
        token_id: str,
        side: str,
        price: str,
        size: str,
        order_type: str = "GTC",
    ) -> Dict[str, Any]:
        """Place a new order."""
        body = json.dumps({
            "tokenID": token_id,
            "side": side.upper(),
            "price": price,
            "size": size,
            "type": order_type,
        })
        return self._request_signed(method="POST", path="/order", body=body)

    def cancel_order(self, *, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        body = json.dumps({"orderID": order_id})
        return self._request_signed(method="DELETE", path="/order", body=body)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Fetch current positions."""
        return self._request_signed(method="GET", path="/positions")

    def get_balances(self) -> Dict[str, Any]:
        """Fetch account balances."""
        return self._request_signed(method="GET", path="/balances")

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _request_public(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        m = method.upper()
        qs = _encode_params(params) if params else ""
        if m == "GET" and qs:
            url = f"{url}?{qs}"
        return self._send(method=m, url=url, body=None, headers={})

    def _request_signed(
        self,
        *,
        method: str,
        path: str,
        body: str = "",
    ) -> Any:
        headers = self._auth.sign_request(method, path, body)
        headers["Content-Type"] = "application/json"
        url = f"{self._base_url}{path}"
        data = body.encode("utf-8") if body else None
        return self._send(method=method.upper(), url=url, body=data, headers=headers)

    def _send(
        self,
        *,
        method: str,
        url: str,
        body: Optional[bytes],
        headers: Dict[str, str],
    ) -> Any:
        req = Request(url=url, data=body, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read().decode("utf-8").strip()
                if not raw:
                    return {}
                return json.loads(raw)
        except json.JSONDecodeError as e:
            raise PolymarketRetryableError(f"Invalid JSON response: {e}") from e
        except HTTPError as e:
            raw_body = e.read().decode("utf-8", errors="replace")
            if e.code == 429 or 500 <= e.code <= 599:
                raise PolymarketRetryableError(
                    f"HTTP {e.code}: {raw_body}"
                ) from e
            raise PolymarketNonRetryableError(
                f"HTTP {e.code}: {raw_body}"
            ) from e
        except URLError as e:
            raise PolymarketRetryableError(f"Network error: {e}") from e


def _encode_params(params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return ""
    normalized: Dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, bool):
            normalized[k] = "true" if v else "false"
        else:
            normalized[k] = v
    return urlencode(normalized, doseq=True)
