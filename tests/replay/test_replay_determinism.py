"""Replay determinism tests."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from event.bootstrap import bootstrap_event_layer
from event.codec import encode_event_json
from event.header import EventHeader
from event.types import EventType, MarketEvent
from runner.replay_runner import run_replay


def _write_market_log(path: Path) -> None:
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
            low=Decimal("100"),
            close=Decimal("101"),
            volume=Decimal("11"),
        ),
    ]
    path.write_text("\n".join(encode_event_json(e) for e in events) + "\n", encoding="utf-8")


def test_replay_determinism(tmp_path: Path) -> None:
    bootstrap_event_layer()
    event_log = tmp_path / "events.jsonl"
    _write_market_log(event_log)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"

    r1 = run_replay(event_log_path=event_log, out_dir=out1)
    r2 = run_replay(event_log_path=event_log, out_dir=out2)

    assert r1.events_processed == 2
    assert r2.events_processed == 2
    assert (out1 / "replay_summary.json").read_text() == (out2 / "replay_summary.json").read_text()
