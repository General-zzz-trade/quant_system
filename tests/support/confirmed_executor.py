"""Compatibility shell: the order lifecycle is trading.application.execution.lifecycle (migration step 9).

Legacy callers (the Binance, Bybit and OKX execution adapters and their tests) keep constructing
``ConfirmedExecutor(adapter, journal_path, scope, ...)``. This class supplies what the application layer
receives by injection: the SQLite journal at ``journal_path``, ``time.time``, ``time.sleep`` and a
re-entrant thread lock.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import TERMINAL, TRANSITION_VIOLATIONS, OrderLifecycle, PendingOrder

if TYPE_CHECKING:
    from trading.application.ports import Clock, OrderGateway


class ConfirmedExecutor(OrderLifecycle):
    def __init__(self, adapter: OrderGateway, journal_path: Path, scope: str, *,
                 wait_seconds: float = 3.0, owner: str = "", entry_order_type: str = "limit",
                 not_found_grace_seconds: float | None = None, clock: Clock = time.time,
                 entry_ttl_seconds: float | None = None) -> None:
        # Validate before the journal file is created, as before the move.
        self.check_settings(scope, wait_seconds=wait_seconds, entry_order_type=entry_order_type,
                            not_found_grace_seconds=not_found_grace_seconds, entry_ttl_seconds=entry_ttl_seconds)
        self.path = Path(journal_path)
        super().__init__(adapter, SqliteOrderJournal(self.path, scope), scope, clock=clock,
                         sleeper=lambda seconds: time.sleep(seconds),   # looked up per call, as before
                         lock=threading.RLock(), wait_seconds=wait_seconds, owner=owner,
                         entry_order_type=entry_order_type, not_found_grace_seconds=not_found_grace_seconds,
                         entry_ttl_seconds=entry_ttl_seconds)

    def _connect(self):  # raw access kept for the frozen OKX adapter only
        return self.journal.connect()


__all__ = ["ConfirmedExecutor", "PendingOrder", "TERMINAL", "TRANSITION_VIOLATIONS"]
