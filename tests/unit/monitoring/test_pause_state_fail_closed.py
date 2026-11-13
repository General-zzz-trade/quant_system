"""The operator pause file must fail closed and never be silently erased."""
from __future__ import annotations

import json

import pytest

import monitoring.live_ic_killswitch as ks


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    path = tmp_path / "symbol_pause_state.json"
    monkeypatch.setattr(ks, "STATE_PATH", path)
    return path


def test_missing_file_means_not_paused(state_path):
    assert ks.is_runner_paused("BTCUSDT") is False


def test_explicit_pause_and_resume(state_path):
    ks._save_state({"BTCUSDT": {"paused": True}, "ETHUSDT": {"paused": False}})
    assert ks.is_runner_paused("BTCUSDT") is True
    assert ks.is_runner_paused("ETHUSDT") is False
    assert ks.is_runner_paused("SOLUSDT") is False


@pytest.mark.parametrize("content", ["{not json", "[]", '{"BTCUSDT": "yes"}'])
def test_unreadable_or_malformed_state_blocks_entries(state_path, content):
    state_path.write_text(content)
    assert ks.is_runner_paused("BTCUSDT") is True


def test_corrupt_state_is_not_overwritten_by_load(state_path):
    state_path.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        ks._load_state()
    assert state_path.read_text() == "{not json"


def test_save_is_atomic_and_leaves_no_temp_files(state_path):
    ks._save_state({"BTCUSDT": {"paused": True}})
    assert json.loads(state_path.read_text()) == {"BTCUSDT": {"paused": True}}
    assert [p.name for p in state_path.parent.iterdir()] == [state_path.name]
