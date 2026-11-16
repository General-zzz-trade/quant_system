"""The publish tool signs a spec the runtime will accept, schema 1 or 2 (R2), and refuses one it would not."""
from __future__ import annotations

import json
import sys

import pytest

from scripts import publish_strategy_spec as publish
from trading.domain.strategy.overlay import StrategySpec

V2 = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": StrategySpec().record()},
      "execution": {"band": 0.1, "execution": "market", "ttl_bars": 1}, "symbols": ["ETHUSDT"], "allow_short": False,
      "passed": False}


def _publish(tmp_path, monkeypatch, document):
    study = tmp_path / "study"
    study.mkdir()
    (study / "final_spec.json").write_text(json.dumps(document))
    monkeypatch.setenv("QUANT_MODEL_SIGN_KEY", "test-signing-key-0123456789")
    monkeypatch.setattr(sys, "argv", ["publish", "--study", str(study), "--out", str(tmp_path / "out")])
    publish.main()
    return tmp_path / "out" / "final_spec.json"


def test_a_schema_2_spec_is_published_signed(tmp_path, monkeypatch):
    path = _publish(tmp_path, monkeypatch, V2)
    assert json.loads(path.read_text())["schema"] == 2
    assert list((tmp_path / "out").iterdir()) != [path]                        # the signature file beside it


def test_a_spec_the_runtime_would_refuse_is_not_published(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="symbol"):
        _publish(tmp_path, monkeypatch, {**V2, "symbols": []})
    assert not (tmp_path / "out").exists()
