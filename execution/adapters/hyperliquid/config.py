# execution/adapters/hyperliquid/config.py
"""Hyperliquid adapter configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HyperliquidConfig:
    """Hyperliquid API connection configuration.

    Attributes:
        private_key: Ethereum private key for EIP-712 signing (hex, with or without 0x prefix).
        base_url: API endpoint — mainnet or testnet.
        wallet_address: Ethereum wallet address (derived from private_key if not provided).
    """
    private_key: str = ""
    base_url: str = "https://api.hyperliquid.xyz"
    wallet_address: str = ""

    def __post_init__(self) -> None:
        # Derive wallet_address from private_key if not explicitly set
        if self.private_key and not self.wallet_address:
            addr = _derive_address(self.private_key)
            # frozen dataclass — use object.__setattr__
            object.__setattr__(self, "wallet_address", addr)

    @property
    def is_testnet(self) -> bool:
        return "testnet" in self.base_url

    @property
    def is_mainnet(self) -> bool:
        return not self.is_testnet

    @property
    def has_private_key(self) -> bool:
        return bool(self.private_key)

    @classmethod
    def mainnet(cls, *, private_key: str = "", **kw: Any) -> "HyperliquidConfig":
        return cls(private_key=private_key,
                   base_url="https://api.hyperliquid.xyz", **kw)

    @classmethod
    def testnet(cls, *, private_key: str = "", **kw: Any) -> "HyperliquidConfig":
        return cls(private_key=private_key,
                   base_url="https://api.hyperliquid-testnet.xyz", **kw)


def _derive_address(private_key: str) -> str:
    """Derive Ethereum address from private key.

    Falls back to empty string if eth_account is not installed.
    """
    try:
        from eth_account import Account
        key = private_key if private_key.startswith("0x") else f"0x{private_key}"
        return Account.from_key(key).address
    except ImportError:
        return ""
    except Exception:
        return ""
