"""Tests for CheckpointManager."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    return tmp_path / "checkpoints"


@pytest.fixture
def mgr(tmp_ckpt_dir):
    from scripts.ops.checkpoint_manager import CheckpointManager
    return CheckpointManager(checkpoint_dir=tmp_ckpt_dir)


class TestCheckpointManager:

    def test_save_and_restore(self, mgr):
        mgr.save("BTC_4h", '{"engine": "data"}', {"inference": "state"},
                  extra={"bars_processed": 100})
        data = mgr.restore("BTC_4h")
        assert data is not None
        assert data["engine"] == '{"engine": "data"}'
        assert data["inference"] == {"inference": "state"}
        assert data["bars_processed"] == 100

    def test_restore_nonexistent(self, mgr):
        assert mgr.restore("NONEXISTENT") is None

    def test_exists(self, mgr):
        assert not mgr.exists("BTC_4h")
        mgr.save("BTC_4h", "{}", {})
        assert mgr.exists("BTC_4h")

    def test_delete(self, mgr):
        mgr.save("BTC_4h", "{}", {})
        assert mgr.delete("BTC_4h") is True
        assert not mgr.exists("BTC_4h")
        assert mgr.delete("BTC_4h") is False

    def test_creates_directory(self, tmp_ckpt_dir):
        from scripts.ops.checkpoint_manager import CheckpointManager
        mgr = CheckpointManager(checkpoint_dir=tmp_ckpt_dir / "deep" / "nested")
        mgr.save("test", "{}", {})
        assert mgr.exists("test")

    def test_nan_handling(self, mgr):
        """NaN values in extra dict should be sanitized."""
        mgr.save("BTC_4h", "{}", {}, extra={"val": float("nan")})
        data = mgr.restore("BTC_4h")
        assert data is not None
        # NaN becomes null (None in JSON)
        assert data["val"] is None

    def test_infinity_handling(self, mgr):
        mgr.save("BTC_4h", "{}", {}, extra={"val": float("inf")})
        data = mgr.restore("BTC_4h")
        assert data is not None
        assert data["val"] is None

    def test_inference_dict_serialized(self, mgr):
        """Inference checkpoint dict should be JSON-serialized on save."""
        mgr.save("BTC_4h", "engine_str", {"key": [1, 2, 3]})
        data = mgr.restore("BTC_4h")
        assert data["inference"] == {"key": [1, 2, 3]}

    def test_inference_string_passthrough(self, mgr):
        """Inference checkpoint already a string should be deserialized on restore."""
        mgr.save("BTC_4h", "engine_str", '{"key": "val"}')
        data = mgr.restore("BTC_4h")
        # String gets double-serialized then deserialized back
        assert data["inference"] == {"key": "val"}

    def test_overwrite(self, mgr):
        mgr.save("BTC_4h", "{}", {}, extra={"bars": 50})
        mgr.save("BTC_4h", "{}", {}, extra={"bars": 100})
        data = mgr.restore("BTC_4h")
        assert data["bars"] == 100

    def test_multiple_runners(self, mgr):
        mgr.save("BTC_4h", "a", {})
        mgr.save("ETH_4h", "b", {})
        assert mgr.restore("BTC_4h")["engine"] == "a"
        assert mgr.restore("ETH_4h")["engine"] == "b"

    def test_corrupt_file_returns_none(self, mgr, tmp_ckpt_dir):
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
        (tmp_ckpt_dir / "BAD.json").write_text("not json{{{")
        assert mgr.restore("BAD") is None
