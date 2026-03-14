"""Polymarket CLOB API authentication."""
from __future__ import annotations
import hashlib
import hmac
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PolymarketAuth:
    api_key: str
    api_secret: str

    def sign_request(self, method: str, path: str, body: str = "") -> dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "POLY_ADDRESS": self.api_key,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": timestamp,
        }
