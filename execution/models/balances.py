# execution/models/balances.py
"""Canonical balance snapshot from venue."""
from __future__ import annotations

from trading.domain.venue import BalanceSnapshot, CanonicalBalance  # noqa: F401  re-exported


# CanonicalBalance: moved to trading.domain.venue (migration step 8)


# BalanceSnapshot: moved to trading.domain.venue (migration step 8)


# 兼容别名 — sim_venue.py 等模块使用
AssetBalance = CanonicalBalance
