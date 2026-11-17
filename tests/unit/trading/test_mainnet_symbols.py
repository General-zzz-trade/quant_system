"""Mainnet trades only BTCUSDT and ETHUSDT.

Owner 2026-11-17: "目前如何去改成做BTC和ETH，不再全市场这样做", and "加硬限制".

The start check refuses a mainnet execution whose signed spec lists any other symbol, before any account is contacted.
The testnet and paper keep the runtime general (R1-R7: up to 10 symbols), for tests. The pilot's start check
(``account_check --pilot``) names it too.
"""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from lab.strategy import StrategySpec
from trading.domain.strategy.spec import MAINNET_SYMBOLS
from trading.infrastructure.cli import target as cli

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _doc(symbols, passed=True):
    return {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": SPEC.record()},
            "execution": {"band": SPEC.band, "execution": SPEC.execution, "ttl_bars": SPEC.ttl_bars},
            "symbols": list(symbols), "allow_short": False, "passed": passed}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name, value in (("RISK_SCOPE", "binance-main-ut"), ("BINANCE_API_KEY", "k"), ("BINANCE_API_SECRET", "s")):
        monkeypatch.setenv(name, value)
    for name in ("MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "BINANCE_EXCHANGE_LEVERAGE",
                 "BINANCE_MARGIN_TYPE"):
        monkeypatch.delenv(name, raising=False)


def _preflight(symbols, *, testnet, execution=True, tmp_path=None, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("BINANCE_TESTNET", "1" if testnet else "0")
        monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut" if testnet else "binance-main-ut")
    return cli.preflight("unused.json", execution=execution, testnet=testnet, stop_pct=D("0.08"),
                         load_spec=lambda path: _doc(symbols))


def test_the_mainnet_symbols_are_btc_and_eth():
    assert MAINNET_SYMBOLS == frozenset({"BTCUSDT", "ETHUSDT"})


@pytest.mark.parametrize("symbols", [("BTCUSDT", "SOLUSDT"), ("XRPUSDT",), ("ETHUSDT", "BTCUSDT", "DOGEUSDT"),
                                     ("SOLUSDT", "BTCUSDT")])
def test_mainnet_refuses_a_spec_with_any_other_symbol(symbols, monkeypatch):
    with pytest.raises(RuntimeError, match="BTCUSDT and ETHUSDT"):
        _preflight(symbols, testnet=False, monkeypatch=monkeypatch)


@pytest.mark.parametrize("symbols", [("BTCUSDT", "ETHUSDT"), ("BTCUSDT",), ("ETHUSDT",)])
def test_mainnet_takes_btc_and_eth(symbols, monkeypatch):
    assert _preflight(symbols, testnet=False, monkeypatch=monkeypatch)["symbols"] == list(symbols)


def test_the_testnet_and_paper_stay_general(monkeypatch):
    general = _preflight(("SOLUSDT", "XRPUSDT"), testnet=True, monkeypatch=monkeypatch)
    assert general["symbols"] == ["SOLUSDT", "XRPUSDT"]
    assert _preflight(("SOLUSDT",), testnet=False, execution=False, monkeypatch=monkeypatch)["symbols"] == ["SOLUSDT"]


def _doc_v1(symbols):
    return {"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": list(symbols), "spec": SPEC.record(),
            "passed": True}


def test_a_schema_1_spec_is_refused_too(monkeypatch):
    """Audit of PR #153, M1: the first schema (the overlay) is held to the same list."""
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    with pytest.raises(RuntimeError, match="SOLUSDT"):
        cli.preflight("unused.json", execution=True, testnet=False, stop_pct=D("0.08"),
                      load_spec=lambda path: _doc_v1(("SOLUSDT", "ETHUSDT")))


def test_the_runtime_refuses_a_schema_1_spec_and_follows_the_config_not_its_testnet_argument(tmp_path, monkeypatch):
    """Audit of PR #153, M1: ``TargetRuntime(testnet=...)`` defaults to True; the configuration decides."""
    from trading.infrastructure.config import load_config
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    config, secrets = load_config(execution=True, testnet=False, stop_pct=D("0.08"))
    with pytest.raises(RuntimeError, match="SOLUSDT"):
        cli.TargetRuntime(spec_path=tmp_path / "unused.json", state_dir=tmp_path / "state", mode="execution",
                          market=None, config=config, secrets=secrets, testnet=True,
                          load_spec=lambda path: _doc_v1(("SOLUSDT", "ETHUSDT")))
    assert not (tmp_path / "state").exists()


def test_preflight_follows_the_config_not_its_testnet_argument(monkeypatch):
    from trading.infrastructure.config import load_config
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    config, _ = load_config(execution=True, testnet=False, stop_pct=D("0.08"))
    with pytest.raises(RuntimeError, match="SOLUSDT"):
        cli.preflight("unused.json", execution=True, testnet=True, stop_pct=D("0.08"), config=config,
                      load_spec=lambda path: _doc(("SOLUSDT",)))


def test_the_runtime_itself_refuses_too(tmp_path, monkeypatch):
    """A second check where the runtime is composed, for a caller that skips the start check (tests, tools)."""
    from trading.infrastructure.config import load_config
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    config, secrets = load_config(execution=True, testnet=False, stop_pct=D("0.08"))
    with pytest.raises(RuntimeError, match="BTCUSDT and ETHUSDT"):
        cli.TargetRuntime(spec_path=tmp_path / "unused.json", state_dir=tmp_path / "state", mode="execution",
                          market=None, config=config, secrets=secrets, testnet=False,
                          load_spec=lambda path: _doc(("BTCUSDT", "SOLUSDT")))
    assert not (tmp_path / "state").exists()


def test_the_mainnet_entry_point_refuses_before_any_account_setting_is_written(tmp_path, monkeypatch):
    """As the research-gate refusal (tests/characterization/test_risk_state.py): local checks before the account."""
    import json

    from infra.model_signing import sign_file
    from runner.binance_target import main
    from tests.unit.binance.fake_venue import FakeBinance, make_bars
    from trading.adapters.binance import adapter as binance
    from trading.infrastructure import bootstrap

    fake = FakeBinance(make_bars(10), clock=lambda: 1_700_000_000.0)

    class Adapter(binance.BinanceAdapter):
        def __init__(self, config, client=None):
            super().__init__(config, client=fake)

    monkeypatch.setattr(bootstrap, "BinanceAdapter", Adapter)
    monkeypatch.delenv("QUANT_ALLOW_UNSIGNED_MODELS", raising=False)
    alerts = tmp_path / "alerts"
    alerts.mkdir()
    # Every other mainnet setting present (audit of PR #153, M2): the symbol refusal is what stops the start.
    for name, value in (("BINANCE_TESTNET", "0"), ("QUANT_MODEL_SIGN_KEY", "test-key"),
                        ("STRATEGY_CAPITAL_USDT", "150"), ("ALERT_FILE", str(alerts / "mainnet.jsonl")),
                        ("HEARTBEAT_URL", "https://hc.example.invalid/ping")):
        monkeypatch.setenv(name, value)
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps(_doc(("SOLUSDT", "ETHUSDT"))))
    sign_file(spec_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="SOLUSDT"):
        main(["--spec", str(spec_path), "--enable-execution", "--state-dir", str(tmp_path / "state")])
    assert fake.leverage == {} and fake.margin_type == {} and fake.posts == []
    assert not (tmp_path / "state").exists()
    assert not (alerts / "mainnet.jsonl").exists()                          # not even "execution starting"
