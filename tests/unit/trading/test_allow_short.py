"""R6: a session trades shorts only when its spec allows them (``allow_short`` of the signed spec; without one, the
strategy spec's own ``long_only``). A negative target is traded as 0 otherwise - nothing new is opened and a short
already held is closed - in both modes."""
from __future__ import annotations

import json

import pytest

from trading.application.session import TargetSession
from trading.domain.strategy.overlay import StrategySpec

BOTH = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
LONG = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market",
                    long_only=True)


class Parts:
    executor = consumer = stops = gate = flattener = equity = None


def _session(spec, allow_short=None):
    return TargetSession("BTCUSDT", spec, journal=None, rules=None, make_order=None, execution=Parts(),
                         allow_short=allow_short)


class Bar:
    open_time, available_at = 0, 3_600_000


def test_without_a_spec_value_the_strategy_spec_decides():
    assert _session(BOTH).allow_short is True and _session(LONG).allow_short is False
    assert _session(BOTH, allow_short=False).allow_short is False
    assert _session(LONG, allow_short=True).allow_short is True


@pytest.mark.parametrize("allow,target,traded", [(False, -0.3, 0.0), (True, -0.3, -0.3), (False, 0.4, 0.4),
                                                 (False, 0.0, 0.0)])
def test_a_negative_target_is_traded_as_zero_unless_shorts_are_allowed(allow, target, traded):
    session = _session(BOTH, allow_short=allow)
    session.decide(Bar(), now_ms=Bar.available_at, target=(target, False))
    assert session.last_target == (traded, False)


def test_paper_sessions_follow_the_spec_too(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    from tests.unit.lab.test_target_runtime import _world
    world = _world(tmp_path, BOTH)
    doc = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": BOTH.record()},
           "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1}, "symbols": ["BTCUSDT", "ETHUSDT"],
           "allow_short": False, "passed": False}
    world["spec_path"].write_text(json.dumps(doc))
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    assert all(s.allow_short is False for s in runtime.sessions.values())
