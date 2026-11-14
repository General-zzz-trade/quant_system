"""Compatibility shell: the drawdown latch is trading.adapters.sqlite.drawdown_store (migration step 4)."""
from __future__ import annotations

from trading.adapters.sqlite.drawdown_store import SqliteDrawdownGuard as DrawdownGuard

__all__ = ["DrawdownGuard"]
