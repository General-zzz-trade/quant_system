"""ADR-0017 (S4): the configuration is read from the environment once, at start-up, into one frozen RuntimeConfig.
It is recorded without any secret; the same environment gives the same digest; changing the environment after the
start changes nothing; the account identity (ADR-0016 point 5) is derived here once."""
from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal as D

import pytest

from trading.infrastructure.config import RuntimeConfig, load_config

SCOPE = "binance-testnet-ut"
SECRETS = {"BINANCE_TESTNET_API_KEY": "key-0123456789abcdef", "BINANCE_TESTNET_API_SECRET": "secret-fedcba9876543210",
           "TELEGRAM_BOT_TOKEN": "123456:telegram-token-value", "TELEGRAM_CHAT_ID": "-1001234567890"}


def _env(**over):
    env = {"RISK_SCOPE": SCOPE, "BINANCE_TESTNET": "1", "MAX_DAILY_DRAWDOWN_PCT": "4", **SECRETS}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def test_execution_config_has_every_value_and_the_account_identity():
    config, secrets = load_config(_env(), execution=True, stop_pct=D("0.08"), require_credentials=True)
    assert isinstance(config, RuntimeConfig)
    assert (config.testnet, config.execution, config.scope, config.stop_pct) == (True, True, SCOPE, D("0.08"))
    assert config.limits.daily_drawdown_pct == 4.0 and config.limits.weekly_drawdown_pct == 12.0
    assert (config.leverage, config.margin) == (3, None) and config.alerts == "telegram"
    assert config.identity == hashlib.sha256(f"testnet:{SECRETS['BINANCE_TESTNET_API_KEY']}".encode()).hexdigest()
    assert secrets.binance is not None and secrets.binance.api_key == SECRETS["BINANCE_TESTNET_API_KEY"]
    with pytest.raises(Exception):
        config.scope = "other"                                         # frozen


def test_paper_config_reads_no_account_setting():
    config, secrets = load_config({}, execution=False, stop_pct=D("0.08"))
    assert (config.execution, config.scope, config.identity, config.alerts) == (False, "paper-target", None, None)
    assert secrets.binance is None and secrets.telegram is None


def test_the_record_and_its_digest_hold_no_secret():
    config, secrets = load_config(_env(), execution=True, stop_pct=D("0.08"), require_credentials=True)
    text = json.dumps(config.record()) + config.digest() + repr(config) + repr(secrets)
    for value in SECRETS.values():
        assert value not in text


def test_the_same_environment_gives_the_same_digest_and_a_changed_limit_another():
    first, _ = load_config(_env(), execution=True, stop_pct=D("0.08"), require_credentials=True)
    again, _ = load_config(_env(), execution=True, stop_pct=D("0.08"), require_credentials=True)
    other_secret, _ = load_config(_env(BINANCE_TESTNET_API_SECRET="another-secret"), execution=True,
                                  stop_pct=D("0.08"), require_credentials=True)
    other_limit, _ = load_config(_env(MAX_DAILY_DRAWDOWN_PCT="3"), execution=True, stop_pct=D("0.08"),
                                 require_credentials=True)
    assert first.digest() == again.digest() == other_secret.digest()
    assert other_limit.digest() != first.digest()


@pytest.mark.parametrize("over,named", [
    ({"MAX_DAILY_DRAWDOWN_PCT": "50"}, "daily drawdown"),
    ({"RISK_SCOPE": None}, "RISK_SCOPE"),
    ({"TELEGRAM_CHAT_ID": None}, "TELEGRAM"),
    ({"BINANCE_EXCHANGE_LEVERAGE": "99"}, "BINANCE_EXCHANGE_LEVERAGE"),
    ({"BINANCE_TESTNET_API_KEY": None}, "API key"),
])
def test_a_missing_or_invalid_setting_refuses_and_names_it(over, named):
    with pytest.raises(RuntimeError, match=named):
        load_config(_env(**over), execution=True, stop_pct=D("0.08"), require_credentials=True)


# -- in the runtime -----------------------------------------------------------------------------------------------
class _NoEnvironment(dict):
    """os.environ after the start: any read fails the test."""

    def __getitem__(self, key):
        raise AssertionError(f"the environment was read after the start: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"the environment was read after the start: {key}")

    def __contains__(self, key):
        raise AssertionError(f"the environment was read after the start: {key}")


def test_nothing_reads_the_environment_after_the_start_and_the_config_is_recorded(tmp_path, monkeypatch):
    from tests.characterization.test_exit_paths import _execution, _tick
    monkeypatch.setenv("RISK_SCOPE", SCOPE)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    world, runtime = _execution(tmp_path, execution="market")
    monkeypatch.setattr(os, "environ", _NoEnvironment())
    for _ in range(3):
        _tick(world, runtime)
    rows = [json.loads(line) for line in (tmp_path / "exec/target_events.jsonl").read_text().splitlines()]
    (config,) = [r for r in rows if r["kind"] == "config"]
    assert config["digest"] == runtime.config.digest() and config["scope"] == SCOPE
    status = json.loads((tmp_path / "exec/target_status.json").read_text())
    assert status["config_digest"] == runtime.config.digest()


def test_the_credentials_come_last_and_bring_the_identity():
    from trading.infrastructure.config import with_credentials
    config, secrets = load_config(_env(), execution=True, stop_pct=D("0.08"))
    assert config.identity is None and secrets.binance is None          # the local checks need no credentials
    config, secrets = with_credentials(config, secrets, _env())
    assert config.identity == hashlib.sha256(f"testnet:{SECRETS['BINANCE_TESTNET_API_KEY']}".encode()).hexdigest()
    assert secrets.binance is not None and secrets.binance.api_secret == SECRETS["BINANCE_TESTNET_API_SECRET"]
    with pytest.raises(RuntimeError, match="API key"):
        with_credentials(config, secrets, _env(BINANCE_TESTNET_API_KEY=None))


def test_paper_config_carries_no_limits_and_the_start_up_check_uses_the_configured_settings(tmp_path, monkeypatch):
    config, _ = load_config({}, execution=False, stop_pct=D("0.08"))
    assert (config.limits, config.leverage, config.margin) == (None, None, None)
    from trading.infrastructure import target
    seen = {}

    def foreign_holdings(account, journal, *, symbols, leverage, margin):
        seen.update(leverage=leverage, margin=margin)
        return []
    monkeypatch.setattr(target, "foreign_holdings", foreign_holdings)
    target.refuse_foreign_holdings(object(), state_dir=tmp_path, scope=SCOPE, symbols=["BTCUSDT"], leverage=5,
                                   margin="ISOLATED")
    assert seen == {"leverage": 5, "margin": "isolated"}


def test_the_binance_config_never_shows_its_key_or_secret():
    # Found while checking S4's mutants: its generated repr printed both in plain text.
    from trading.adapters.binance.config import BinanceConfig
    text = repr(BinanceConfig(api_key="key-0123456789abcdef", api_secret="secret-fedcba9876543210", testnet=True))
    assert "key-0123456789abcdef" not in text and "secret-fedcba9876543210" not in text and "testnet=True" in text
