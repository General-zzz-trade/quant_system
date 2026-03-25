"""Storage — backend protocol definitions and implementations.

Defines BarStore/TickStore protocols and the Bar/Tick data types re-exported
from data.store.  Storage backends implement these protocols for different
persistence targets (parquet, SQLite, JSONL).
"""
from __future__ import annotations

from data.backends.base import Bar, BarStore, Tick, TickStore

__all__ = ["Bar", "BarStore", "Tick", "TickStore"]
