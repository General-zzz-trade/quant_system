"""R2 in the running runtime: a schema-2 spec that names the overlay on BTC/ETH trades exactly like the schema-1 spec
it restates (same orders and fills on the same bars), and the runtime builds its sessions from the spec's symbols
and strategy."""
from __future__ import annotations

import json
from decimal import Decimal as D

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _run(tmp_path, document):
    tmp_path.mkdir(parents=True, exist_ok=True)
    world = _world(tmp_path, SPEC)
    if document is not None:
        world["spec_path"].write_text(json.dumps(document))
    state_dir_db(tmp_path / "exec")
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"],
                            clock=lambda: world["clock"]["now"], testnet=True)
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + 40):
        _advance(world, runtime, i)
    trades = {s: [(t["side"], t["qty"]) for t in f.trades] for s, f in world["fakes"].items()}
    return runtime, trades


def test_the_same_strategy_in_either_schema_trades_the_same(tmp_path):
    _, one = _run(tmp_path / "one", None)
    two_doc = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": SPEC.record()},
               "execution": {"band": SPEC.band, "execution": SPEC.execution, "ttl_bars": SPEC.ttl_bars},
               "symbols": ["BTCUSDT", "ETHUSDT"], "allow_short": not SPEC.long_only, "passed": False}
    runtime, two = _run(tmp_path / "two", two_doc)
    assert any(one.values()) and one == two
    assert set(runtime.sessions) == {"BTCUSDT", "ETHUSDT"}
    assert all(s.strategy.name == "overlay" and s.strategy.spec == SPEC for s in runtime.sessions.values())
    assert all(s.allocation == D(1) / D(2) for s in runtime.sessions.values())
