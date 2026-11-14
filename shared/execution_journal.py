"""Compatibility shell: the order journal is trading.adapters.sqlite.order_journal (migration step 6)."""
from __future__ import annotations

from trading.adapters.sqlite.order_journal import SqliteOrderJournal as ExecutionJournal

__all__ = ["ExecutionJournal"]
