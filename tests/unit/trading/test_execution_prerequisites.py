"""ADR-0016 points 1 and 3 (S7), the configuration part: execution mode (testnet and mainnet) needs Telegram and a
heartbeat address, or the start is refused before anything else happens; paper needs neither. The heartbeat
address is a secret: in the Secrets, never in the configuration record, its digest or any printed form.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from trading.adapters.notify.heartbeat import HttpHeartbeat
from trading.infrastructure.config import load_config, make_heartbeat, require_operator_channels

HEARTBEAT = "https://hc.example.invalid/ping/SECRET-PING-ID"
ENV = {"RISK_SCOPE": "binance-testnet-ut", "BINANCE_TESTNET": "1", "TELEGRAM_BOT_TOKEN": "123:tok",
       "TELEGRAM_CHAT_ID": "42", "HEARTBEAT_URL": HEARTBEAT}


def _env(**over):
    env = dict(ENV, **over)
    return {k: v for k, v in env.items() if v is not None}


def _load(env, execution=True):
    return load_config(env, execution=execution, stop_pct=D("0.08"))


def test_execution_with_telegram_and_a_heartbeat_may_start():
    config, secrets = _load(_env())
    require_operator_channels(config, secrets)
    assert isinstance(make_heartbeat(config, secrets), HttpHeartbeat)


@pytest.mark.parametrize("missing,named", [
    ({"TELEGRAM_BOT_TOKEN": None, "TELEGRAM_CHAT_ID": None}, "TELEGRAM_BOT_TOKEN"),
    ({"HEARTBEAT_URL": None}, "HEARTBEAT_URL"),
    ({"HEARTBEAT_URL": "  "}, "HEARTBEAT_URL"),
])
def test_execution_without_an_alert_channel_or_a_heartbeat_is_refused(missing, named):
    config, secrets = _load(_env(**missing))
    with pytest.raises(RuntimeError, match=named):
        require_operator_channels(config, secrets)


@pytest.mark.parametrize("address", ["http://hc.example.invalid/ping/x", "hc.example.invalid/ping/x", "ftp://x/y"])
def test_a_heartbeat_address_that_is_not_https_is_refused_and_not_echoed(address):
    with pytest.raises(RuntimeError, match="HEARTBEAT_URL") as refused:
        _load(_env(HEARTBEAT_URL=address))
    assert address not in str(refused.value)


def test_paper_needs_neither_and_has_no_heartbeat():
    config, secrets = _load({}, execution=False)
    require_operator_channels(config, secrets)
    assert make_heartbeat(config, secrets) is None


def test_the_heartbeat_address_is_never_recorded_or_printed():
    config, secrets = _load(_env())
    assert "SECRET-PING-ID" not in json.dumps(config.record()) + config.digest() + repr(config) + repr(secrets)
    assert "SECRET-PING-ID" not in repr(make_heartbeat(config, secrets))
