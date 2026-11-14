"""Composition root: builds the venue adapters. Nothing here decides anything about trading."""
from __future__ import annotations

from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.infrastructure.settings import binance_config_from_env, exchange_settings
from trading.infrastructure.target import Account


def build_adapters(*, execution: bool, testnet: bool,
                   symbols: list[str]) -> tuple[BinanceAdapter, BinanceAdapter | None]:
    """(public market-data adapter, account adapter or None). Connecting the account sets leverage and
    margin type on the exchange, so call this only after every local start-up check has passed."""
    market = BinanceAdapter(BinanceConfig(testnet=False))  # public mainnet data only (no keys)
    account = None
    if execution:
        leverage, margin = exchange_settings()
        account = BinanceAdapter(binance_config_from_env(testnet=testnet))
        if not account.connect(symbols=tuple(symbols), configure_leverage=True, leverage=leverage,
                               margin_type=margin):
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
