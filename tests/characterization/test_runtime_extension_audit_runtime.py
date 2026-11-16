"""The runtime-level checks of the fixes for the R1–R4 audit (see tests/unit/trading/test_runtime_extension_audit.py
for the list): a paper record does not resume under another strategy, the start-up record names the strategy, and an
unknown funding interval still lets a position be reduced."""
from __future__ import annotations

import json

import pytest

from tests.unit.trading.test_runtime_extension_audit import OVERLAY, _params, _v2, _world


def test_a_paper_record_does_not_resume_under_other_strategy_parameters(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    first = _v2()
    world = _world(tmp_path, first)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    assert runtime.paper_document()["spec"] == OVERLAY.record()       # the overlay as traded, as before R2
    world["spec_path"].write_text(json.dumps(_v2(strategy=_params(target_vol=0.5))))     # another parameter
    again = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                          market=world["adapter"], clock=lambda: world["clock"]["now"])
    with pytest.raises(RuntimeError, match="another spec"):
        again.warmup()


def test_the_start_up_record_names_the_strategy_symbols_and_shorts(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    from tests.support.state_db import state_dir_db
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    world = _world(tmp_path, _v2(allow_short=True))
    state_dir_db(tmp_path / "exec")
    TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                  market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                  testnet=True)
    events = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    (config,) = [e for e in events if e["kind"] == "config"]
    assert config["strategy"]["strategy"] == "overlay" and config["strategy"]["allow_short"] is True
    assert config["strategy"]["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_an_unknown_funding_interval_still_lets_a_position_be_reduced(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    from tests.support.state_db import state_dir_db
    from tests.unit.lab.test_target_runtime import _advance
    from trading.domain.strategy.base import PerSymbol
    import numpy as np
    import pandas as pd
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)

    class Fixed(PerSymbol):
        name, weight = "fixed", -0.5

        def window_hours(self):
            return OVERLAY.window_hours()

        def funding_window(self):
            return OVERLAY.funding_window()

        def symbol_history(self, window):
            times = window.hourly["available_at"].to_numpy(dtype=np.int64)
            return pd.DataFrame({"available_at": times, "target": self.weight, "ready": True})

    world = _world(tmp_path, _v2(allow_short=True))
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True)
    runtime.sessions["BTCUSDT"].gate.inner._limiter._monotonic = lambda: world["clock"]["now"]
    strategy = Fixed()
    for session in runtime.sessions.values():
        session.strategy = strategy
    runtime.warmup()
    i = world["warm"]

    def close(k):
        _advance(world, runtime, k)
        for _ in range(2):                           # the next 5-second cycles send what the per-second limit held
            world["clock"]["now"] += 5
            runtime.cycle()
    close(i)
    eth = world["fakes"]["ETHUSDT"]
    assert eth.positions["ETHUSDT"] < 0
    for fake in world["fakes"].values():
        fake.funding_hours = {"BTCUSDT": 8}                          # ETHUSDT is no longer listed
    strategy.weight = 0.0
    close(i + 1)                                                      # the next hour reads the intervals again
    session = runtime.sessions["ETHUSDT"]
    assert session.funding_interval is None and session.entry_reason == "funding interval not known"
    assert eth.positions["ETHUSDT"] == 0 and eth.trades[-1]["side"] == "BUY"
