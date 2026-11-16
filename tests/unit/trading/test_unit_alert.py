"""ADR-0016 point 4 (S7/S8): the OnFailure= alert of the systemd units. ``quant-alert@<unit>.service`` runs
``python -m trading.infrastructure.unit_alert <unit>``: one Telegram message, confirmed or the command fails."""
from __future__ import annotations

import io
import json

from trading.adapters.notify.telegram import TelegramNotifier
from trading.infrastructure import unit_alert

ENV = {"TELEGRAM_BOT_TOKEN": "123:SECRET-BOT-TOKEN", "TELEGRAM_CHAT_ID": "42"}


def _telegram(ok, sent):
    def request(url, body, timeout):
        sent.append((json.loads(body), timeout))
        return json.dumps({"ok": ok}).encode()

    def make(env):
        assert env == ENV                                    # built from the environment it was given
        return TelegramNotifier(env["TELEGRAM_BOT_TOKEN"], env["TELEGRAM_CHAT_ID"], request=request)
    return make


def _run(argv, env, **kwargs):
    out = io.StringIO()
    return unit_alert.main(argv, env=env, stdout=out, **kwargs), out.getvalue()


def test_a_failed_unit_is_named_in_a_confirmed_message():
    sent = []
    code, out = _run(["binance-target-testnet.service"], ENV, make=_telegram(True, sent))
    assert code == 0
    ((body, timeout),) = sent
    assert "binance-target-testnet.service" in body["text"] and body["chat_id"] == "42" and timeout > 0
    assert "SECRET-BOT-TOKEN" not in out


def test_an_unconfirmed_message_fails_the_alert_unit():
    sent = []
    code, out = _run(["state-backup@testnet.service"], ENV, make=_telegram(False, sent))
    assert code == 1 and "not confirmed" in out and len(sent) == 1


def test_without_telegram_the_alert_unit_fails_and_says_why():
    code, out = _run(["binance-target-testnet.service"], {})
    assert code == 2 and "TELEGRAM_BOT_TOKEN" in out


def test_half_a_telegram_configuration_fails_too():
    code, out = _run(["binance-target-testnet.service"], {"TELEGRAM_BOT_TOKEN": "123:SECRET-BOT-TOKEN"})
    assert code == 2 and "TELEGRAM_CHAT_ID" in out and "SECRET-BOT-TOKEN" not in out
