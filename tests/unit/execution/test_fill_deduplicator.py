from __future__ import annotations

import pytest

from execution.ingress.router import FillDeduplicator


def test_fill_deduplicator_accepts_new_and_drops_duplicate() -> None:
    dedup = FillDeduplicator()
    key = ("binance", "BTCUSDT", "fill-1")

    assert dedup.accept_or_raise(key=key, digest="digest-1") is True
    assert dedup.accept_or_raise(key=key, digest="digest-1") is False


def test_fill_deduplicator_fails_fast_on_payload_mismatch() -> None:
    dedup = FillDeduplicator()
    key = ("binance", "BTCUSDT", "fill-1")

    assert dedup.accept_or_raise(key=key, digest="digest-1") is True
    with pytest.raises(ValueError, match="payload mismatch"):
        dedup.accept_or_raise(key=key, digest="digest-2")
