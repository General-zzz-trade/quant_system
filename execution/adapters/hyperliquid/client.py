# execution/adapters/hyperliquid/client.py
"""Hyperliquid REST client — handles HTTP requests and EIP-712 signing."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from execution.adapters.hyperliquid.config import HyperliquidConfig

logger = logging.getLogger(__name__)

# Try importing eth_account for EIP-712 signing (optional dependency)
try:
    from eth_account import Account
    from eth_account.messages import encode_structured_data
    _HAS_ETH_ACCOUNT = True
except ImportError:
    _HAS_ETH_ACCOUNT = False


# EIP-712 domain and types for Hyperliquid exchange actions
_EIP712_DOMAIN = {
    "name": "Exchange",
    "version": "1",
    "chainId": 1337,
    "verifyingContract": "0x0000000000000000000000000000000000000000",
}

_EIP712_TYPES_ACTION = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Agent": [
        {"name": "source", "type": "string"},
        {"name": "connectionId", "type": "bytes32"},
    ],
}


class HyperliquidRestClient:
    """Low-level Hyperliquid REST API client.

    Info requests (POST /info) are public and require no authentication.
    Exchange requests (POST /exchange) require EIP-712 signing via eth_account.
    """

    def __init__(self, config: HyperliquidConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public info requests (no auth required)
    # ------------------------------------------------------------------

    def info_request(self, payload: dict[str, Any]) -> Any:
        """Send a POST request to /info endpoint (public, no auth).

        Args:
            payload: JSON body, e.g. {"type": "meta"} or {"type": "l2Book", "coin": "BTC"}.

        Returns:
            Parsed JSON response (list or dict).
        """
        url = f"{self._config.base_url}/info"
        body_str = json.dumps(payload)
        req = Request(
            url, data=body_str.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(req)

    # ------------------------------------------------------------------
    # Authenticated exchange requests
    # ------------------------------------------------------------------

    def exchange_request(
        self, action: dict[str, Any], *, nonce: Optional[int] = None,
    ) -> Any:
        """Send a signed POST request to /exchange endpoint.

        Args:
            action: The action payload (e.g. order, cancel, updateLeverage).
            nonce: Request nonce (defaults to current time in ms).

        Returns:
            Parsed JSON response.

        Raises:
            NotImplementedError: If eth_account is not installed.
            ValueError: If no private key is configured.
        """
        if not _HAS_ETH_ACCOUNT:
            raise NotImplementedError(
                "eth_account library is required for Hyperliquid exchange requests. "
                "Install it with: pip install eth-account"
            )
        if not self._config.private_key:
            raise ValueError("private_key is required for exchange requests")

        if nonce is None:
            nonce = int(time.time() * 1000)

        signature, vault_address = self._sign_action(action, nonce)

        body = {
            "action": action,
            "nonce": nonce,
            "signature": signature,
        }
        if vault_address:
            body["vaultAddress"] = vault_address

        url = f"{self._config.base_url}/exchange"
        body_str = json.dumps(body)
        req = Request(
            url, data=body_str.encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(req)

    # ------------------------------------------------------------------
    # EIP-712 signing
    # ------------------------------------------------------------------

    def _sign_action(
        self, action: dict[str, Any], nonce: int,
    ) -> tuple[dict[str, Any], Optional[str]]:
        """Sign an action using EIP-712 structured data.

        Returns:
            (signature_dict, vault_address_or_None)
        """
        if not _HAS_ETH_ACCOUNT:
            raise NotImplementedError(
                "eth_account library is required for signing. "
                "Install it with: pip install eth-account"
            )

        # Hyperliquid uses a phantom agent for signing
        # The connection_id is a hash of the action + nonce
        connection_id = _action_hash(action, nonce, is_mainnet=not self._config.is_testnet)

        typed_data = {
            "types": _EIP712_TYPES_ACTION,
            "domain": _EIP712_DOMAIN,
            "primaryType": "Agent",
            "message": {
                "source": "a" if not self._config.is_testnet else "b",
                "connectionId": connection_id,
            },
        }

        key = self._config.private_key
        if not key.startswith("0x"):
            key = f"0x{key}"

        signable = encode_structured_data(typed_data)
        signed = Account.sign_message(signable, key)

        sig = {
            "r": hex(signed.r),
            "s": hex(signed.s),
            "v": signed.v,
        }
        return sig, None

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _send(self, req: Request) -> Any:
        """Execute HTTP request and parse JSON response."""
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                return data
        except HTTPError as e:
            body = e.read().decode(errors="ignore")
            retryable = e.code in (429, 500, 502, 503, 504)
            logger.error(
                "Hyperliquid HTTP %d (retryable=%s): %s",
                e.code, retryable, body[:200],
            )
            return {"status": "error", "code": e.code,
                    "msg": body[:200], "retryable": retryable}
        except Exception as e:
            logger.error("Hyperliquid request failed: %s", e)
            return {"status": "error", "code": -1,
                    "msg": str(e), "retryable": True}


def _action_hash(
    action: dict[str, Any], nonce: int, *, is_mainnet: bool = True,
) -> bytes:
    """Compute the connection_id (action hash) for EIP-712 signing.

    Hyperliquid hashes the action + nonce + vault_address to create the
    phantom agent's connectionId.
    """
    import hashlib

    # Serialize action deterministically
    action_str = json.dumps(action, sort_keys=True, separators=(",", ":"))
    # Hash: action_str + nonce + is_mainnet flag
    msg = f"{action_str}{nonce}{is_mainnet}"
    h = hashlib.sha256(msg.encode()).digest()
    return h
