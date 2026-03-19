"""Helpers for mixed legacy and canonical balance snapshots on ops paths."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_asset_balance(balances: Any, asset: str) -> Any | None:
    """Return one asset balance from either a mapping or BalanceSnapshot-like object."""
    if balances is None:
        return None
    getter = getattr(balances, "get", None)
    if callable(getter):
        return getter(asset)
    if isinstance(balances, Mapping):
        return balances.get(asset)
    return None


def read_balance_value(balance: Any, *field_names: str) -> float | None:
    """Read the first numeric field present on a legacy or canonical balance object."""
    if balance is None:
        return None

    if isinstance(balance, Mapping):
        for field_name in field_names:
            if field_name in balance:
                try:
                    return float(balance[field_name])
                except (TypeError, ValueError):
                    continue

    for field_name in field_names:
        if hasattr(balance, field_name):
            try:
                return float(getattr(balance, field_name))
            except (TypeError, ValueError):
                continue
    return None


def get_total_and_free_balance(balances: Any, asset: str = "USDT") -> tuple[float | None, float | None]:
    """Return (total, free) balance for a given asset across supported schemas."""
    asset_balance = get_asset_balance(balances, asset)
    total = read_balance_value(asset_balance, "total", "balance", "walletBalance")
    free = read_balance_value(asset_balance, "free", "available", "availableBalance")
    return total, free
