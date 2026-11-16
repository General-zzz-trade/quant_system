"""``STRATEGY_LEVERAGE`` in the running runtime (owner, 2026-11-16 evening): with a strategy capital of 300 and a
leverage of 2, each symbol sizes from twice its share of the capital. The fake account holds 10,000 USDT."""
from __future__ import annotations

from decimal import Decimal as D

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_strategy_capital_sizing import _execution, _first_entry_notional


def test_each_symbol_sizes_from_twice_its_share_at_a_leverage_of_2(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    monkeypatch.delenv("STRATEGY_LEVERAGE", raising=False)
    world, single = _execution(tmp_path / "one")
    _tick(world, single)
    monkeypatch.setenv("STRATEGY_LEVERAGE", "2")
    world, double = _execution(tmp_path / "two")
    _tick(world, double)
    assert double.config.strategy_leverage == D(2)
    for symbol in ("BTCUSDT", "ETHUSDT"):                            # 300 split in two, then times the leverage
        assert single.sessions[symbol].sleeve_equity() == D(150)
        assert double.sessions[symbol].sleeve_equity() == D(300)


def test_a_leverage_of_3_triples_each_symbol_s_share(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    monkeypatch.setenv("STRATEGY_LEVERAGE", "3")
    monkeypatch.setenv("BINANCE_EXCHANGE_LEVERAGE", "10")
    world, runtime = _execution(tmp_path)
    _tick(world, runtime)
    assert runtime.config.strategy_leverage == D(3)
    assert all(runtime.sessions[s].sleeve_equity() == D(450) for s in ("BTCUSDT", "ETHUSDT"))


def test_a_leveraged_entry_is_larger_than_the_symbol_s_share_and_within_the_multiple(tmp_path, monkeypatch):
    monkeypatch.setenv("STRATEGY_CAPITAL_USDT", "300")
    monkeypatch.setenv("STRATEGY_LEVERAGE", "2")
    notional, _ = _first_entry_notional(tmp_path)
    assert D(150) < notional <= D(300) * D("1.01")                  # more than 1x the share, at most 2x


def test_without_a_strategy_capital_the_leverage_stays_1(tmp_path, monkeypatch):
    monkeypatch.delenv("STRATEGY_CAPITAL_USDT", raising=False)
    monkeypatch.delenv("STRATEGY_LEVERAGE", raising=False)
    _, runtime = _first_entry_notional(tmp_path)
    assert runtime.config.strategy_leverage == D(1)
