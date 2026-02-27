from __future__ import annotations

import json
from pathlib import Path

from runner.replay_runner import JsonlEventSource, run_replay


def _write_jsonl(path: Path, events: list) -> None:
    with path.open("w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_jsonl_event_source_len(tmp_path):
    p = tmp_path / "events.jsonl"
    _write_jsonl(p, [{"event_type": "market", "ts": 1}, {"event_type": "market", "ts": 2}])
    source = JsonlEventSource(p)
    assert len(source) == 2


def test_jsonl_event_source_iter(tmp_path):
    p = tmp_path / "events.jsonl"
    events = [{"event_type": "market", "ts": i} for i in range(5)]
    _write_jsonl(p, events)
    source = JsonlEventSource(p)
    result = list(source)
    assert len(result) == 5


def test_jsonl_event_source_skips_blank_lines(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"event_type": "market"}\n\n{"event_type": "fill"}\n\n')
    source = JsonlEventSource(p)
    assert len(source) == 2
    assert len(list(source)) == 2


def test_run_replay_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    count = run_replay(event_log_path=p, symbol="BTCUSDT", out_dir=tmp_path / "out")
    assert count == 0
    summary = json.loads((tmp_path / "out" / "replay_summary.json").read_text())
    assert summary["events_processed"] == 0
