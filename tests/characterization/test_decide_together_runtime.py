"""R3 in the running runtime: a strategy that decides every symbol together with one call trades exactly like the same
targets decided symbol by symbol, in execution and in paper, when every symbol's bar arrives at each close.

``Together`` gives the overlay's targets but says it is cross-sectional and records each call: one call per close,
with every symbol's window ending at that close."""
from __future__ import annotations

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.support.state_db import state_dir_db
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.domain.strategy.base import OverlayStrategy

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")
CLOSES = 40


class Together(OverlayStrategy):
    cross_sectional = True

    def __init__(self, spec):
        super().__init__(spec)
        self.calls = []

    def latest(self, windows):
        self.calls.append(sorted((s, int(w.hourly["open_time"].iloc[-1])) for s, w in windows.items()))
        return super().latest(windows)


def _run(tmp_path, mode, together):
    tmp_path.mkdir(parents=True, exist_ok=True)
    world = _world(tmp_path, SPEC)
    if mode == "execution":
        state_dir_db(tmp_path / "exec")
        runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                                market=world["adapter"], account=world["adapter"],
                                clock=lambda: world["clock"]["now"], testnet=True)
    else:
        runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                                market=world["adapter"], clock=lambda: world["clock"]["now"])
    strategy = Together(SPEC) if together else None
    if strategy is not None:
        for session in runtime.sessions.values():
            session.strategy = strategy
    runtime.warmup()
    for i in range(world["warm"], world["warm"] + CLOSES):
        _advance(world, runtime, i)
    if mode == "execution":
        trades = {s: [(t["side"], t["qty"], t["price"]) for t in f.trades] for s, f in world["fakes"].items()}
    else:
        trades = {s: [(f["at"], f["side"], f["qty"], f["price"]) for f in session.sleeve.broker.fills]
                  for s, session in runtime.sessions.items()}
    return trades, strategy, world


def test_execution_decided_together_trades_like_symbol_by_symbol(tmp_path):
    alone, _, _ = _run(tmp_path / "alone", "execution", together=False)
    together, strategy, world = _run(tmp_path / "together", "execution", together=True)
    assert any(alone.values()) and together == alone
    assert len(strategy.calls) == CLOSES
    for call in strategy.calls:
        assert [s for s, _ in call] == ["BTCUSDT", "ETHUSDT"] and call[0][1] == call[1][1]


def test_paper_decided_together_trades_like_symbol_by_symbol(tmp_path):
    alone, _, _ = _run(tmp_path / "alone", "paper", together=False)
    together, strategy, _ = _run(tmp_path / "together", "paper", together=True)
    assert any(alone.values()) and together == alone
    assert len(strategy.calls) == CLOSES and all(c[0][1] == c[1][1] for c in strategy.calls)
