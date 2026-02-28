"""Tests for runner/replay_runner.py — JsonlEventSource."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from runner.replay_runner import JsonlEventSource


class TestJsonlEventSource:
    def _write_jsonl(self, tmp_path: Path, lines: list[str]) -> Path:
        p = tmp_path / "events.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def test_parse_valid_lines(self, tmp_path: Path):
        lines = [
            json.dumps({"event_type": "market", "symbol": "BTCUSDT", "close": 40000}),
            json.dumps({"event_type": "market", "symbol": "ETHUSDT", "close": 2500}),
        ]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        events = list(source)
        assert len(events) == 2
        assert events[0]["event_type"] == "market"
        assert events[1]["symbol"] == "ETHUSDT"

    def test_len_counts_non_empty_lines(self, tmp_path: Path):
        lines = [
            json.dumps({"event_type": "a"}),
            "",
            json.dumps({"event_type": "b"}),
            "",
        ]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        assert len(source) == 2

    def test_skip_empty_lines(self, tmp_path: Path):
        lines = [
            "",
            json.dumps({"event_type": "market"}),
            "   ",
            json.dumps({"event_type": "fill"}),
        ]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        events = list(source)
        assert len(events) == 2

    def test_empty_file(self, tmp_path: Path):
        path = self._write_jsonl(tmp_path, [])
        source = JsonlEventSource(path)
        assert list(source) == []
        assert len(source) == 0

    def test_single_event(self, tmp_path: Path):
        lines = [json.dumps({"event_type": "order", "symbol": "BTCUSDT"})]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        events = list(source)
        assert len(events) == 1
        assert events[0]["event_type"] == "order"

    def test_len_cached(self, tmp_path: Path):
        lines = [json.dumps({"event_type": "x"}) for _ in range(5)]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        assert source._count is None
        n = len(source)
        assert n == 5
        assert source._count == 5
        # Second call uses cache
        assert len(source) == 5

    def test_malformed_json_falls_back(self, tmp_path: Path):
        """Malformed JSON raises on json.loads — verify we get an error."""
        lines = [
            json.dumps({"event_type": "good"}),
            "not-valid-json{{{",
        ]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        events = []
        with pytest.raises(json.JSONDecodeError):
            for ev in source:
                events.append(ev)
        # First event was parsed before error
        assert len(events) == 1

    def test_iterable_multiple_times(self, tmp_path: Path):
        lines = [json.dumps({"event_type": "a"}), json.dumps({"event_type": "b"})]
        path = self._write_jsonl(tmp_path, lines)
        source = JsonlEventSource(path)
        first = list(source)
        second = list(source)
        assert len(first) == 2
        assert len(second) == 2
