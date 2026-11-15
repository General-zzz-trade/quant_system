"""Compatibility shell: the single-writer lease is trading.adapters.files.lease (migration step 12b)."""
from __future__ import annotations

from trading.adapters.files.lease import ExecutionLease

__all__ = ["ExecutionLease"]
