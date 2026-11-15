"""Composition root: builds the venue adapters. Nothing here decides anything about trading."""
from __future__ import annotations

from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.infrastructure.target import Account


def build_adapters(*, execution: bool, testnet: bool, symbols: list[str],
                   credentials: BinanceConfig | None = None) -> tuple[BinanceAdapter, BinanceAdapter | None]:
    """(public market-data adapter, account adapter or None). Connecting is read-only: leverage and margin mode
    are set by a person and checked at start-up (ADR-0010). Call this after every local start-up check."""
    market = BinanceAdapter(BinanceConfig(testnet=False))  # public mainnet data only (no keys)
    account = None
    if execution:
        if credentials is None:
            raise RuntimeError("execution needs the Binance credentials of the configuration (ADR-0017)")
        account = BinanceAdapter(credentials)
        if not account.connect(symbols=tuple(symbols)):
            raise RuntimeError("Binance account connection failed")
        blockers = account.execution_blockers()
        if blockers:
            raise RuntimeError("Execution blocked: " + "; ".join(blockers))
    return market, account


def as_account(adapter: BinanceAdapter) -> Account:
    """The adapter as the typed account the sessions are wired with. The type checker proves here that the
    Binance adapter implements every exchange port with the declared result types."""
    return adapter


__all__ = ["as_account", "build_adapters"]
