"""Replay event log smoke tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from event.bootstrap import bootstrap_event_layer
from event.codec import encode_event_json
from event.header import EventHeader
from event.types import EventType, MarketEvent
from runner.replay_runner import run_replay


def test_event_ordering_replay_smoke(tmp_path: Path) -> None:
    bootstrap_event_layer()

    events = [
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("10"),
        ),
        MarketEvent(
            header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="test"),
            ts=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("99"),
            close=Decimal("101"),
            volume=Decimal("12"),
        ),
    ]
    event_log = tmp_path / "events.jsonl"
    event_log.write_text("\n".join(encode_event_json(e) for e in events) + "\n", encoding="utf-8")

    processed = run_replay(event_log_path=event_log, out_dir=tmp_path / "out")

    assert processed == 2
    summary = json.loads((tmp_path / "out" / "replay_summary.json").read_text(encoding="utf-8"))
    assert summary["events_processed"] == 2
