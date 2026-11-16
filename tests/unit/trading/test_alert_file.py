"""Alerts to a file instead of Telegram (owner, 2026-11-16): the owner does not use Telegram; a Claude Code session
reads the file about once an hour, and the external heartbeat stays mandatory.

``ALERT_FILE`` (an absolute path) makes the runtime append one JSON line per alert. Execution may start with it in
place of the Telegram pair; "the start alert was confirmed" then means the line was written. A write that fails is
never raised into the trading loop. Telegram stays available; setting both is refused.
"""
from __future__ import annotations

import io
import json
import os
import stat
from decimal import Decimal as D

import pytest

from trading.adapters.notify.file import FileNotifier
from trading.adapters.notify.log import LogNotifier
from trading.adapters.notify.telegram import TelegramNotifier
from trading.infrastructure import unit_alert
from trading.infrastructure.config import (load_config, make_notifier, notifier_from_env,
                                           require_operator_channels)

HEARTBEAT = "https://hc.example.invalid/ping/SECRET-PING"


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_alert_is_one_json_line_appended_and_synced(tmp_path):
    path = tmp_path / "alerts.jsonl"
    notifier = FileNotifier(path, clock=lambda: 1_800_000_000.25)
    notifier.alert("critical", "[mainnet] BTCUSDT", "entries paused: 止损触发")
    notifier.alert("warning", "second")
    first, second = _lines(path)
    assert first == {"at": 1_800_000_000.25, "utc": "2027-01-15T08:00:00Z", "level": "critical",
                     "title": "[mainnet] BTCUSDT", "detail": "entries paused: 止损触发"}
    assert (second["level"], second["title"], second["detail"]) == ("warning", "second", "")
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert notifier.channel == "file" and notifier.background is False
    notifier.flush(1.0)                                              # nothing is queued: writes are immediate


def test_a_write_that_fails_never_raises_and_a_delivery_is_then_not_confirmed(tmp_path, caplog):
    notifier = FileNotifier(tmp_path / "no-such-directory" / "alerts.jsonl")
    notifier.alert("critical", "position without a stop")          # the trading loop carries on
    assert notifier.deliver("MAINNET execution starting", "scope x", timeout_s=1.0) is False
    assert "position without a stop" in caplog.text                 # and it is at least in the process log


def test_a_delivery_is_confirmed_when_the_line_is_in_the_file(tmp_path):
    path = tmp_path / "alerts.jsonl"
    assert FileNotifier(path).deliver("MAINNET execution starting", "scope x", timeout_s=1.0) is True
    (line,) = _lines(path)
    assert line["title"] == "MAINNET execution starting" and line["detail"] == "scope x" and line["level"] == "info"


ENV = {"RISK_SCOPE": "binance-ut", "BINANCE_TESTNET": "0", "HEARTBEAT_URL": HEARTBEAT,
       "STRATEGY_CAPITAL_USDT": "150"}


def _load(**over):
    env = {k: v for k, v in dict(ENV, **over).items() if v is not None}
    return load_config(env, execution=True, stop_pct=D("0.08"))


def test_execution_may_start_with_an_alert_file_and_a_heartbeat(tmp_path):
    config, secrets = _load(ALERT_FILE=str(tmp_path / "alerts.jsonl"))
    require_operator_channels(config, secrets)
    assert config.alerts == "file" and json.loads(json.dumps(config.record()))["alerts"] == "file"
    notifier = make_notifier(secrets)
    assert isinstance(notifier, FileNotifier) and notifier.path == tmp_path / "alerts.jsonl"


def test_without_any_alert_channel_execution_is_refused_and_the_message_names_both():
    config, secrets = _load()
    with pytest.raises(RuntimeError, match="ALERT_FILE.*TELEGRAM_BOT_TOKEN"):
        require_operator_channels(config, secrets)


def test_the_heartbeat_stays_mandatory_with_an_alert_file(tmp_path):
    config, secrets = _load(ALERT_FILE=str(tmp_path / "a.jsonl"), HEARTBEAT_URL=None)
    with pytest.raises(RuntimeError, match="HEARTBEAT_URL"):
        require_operator_channels(config, secrets)


def test_an_alert_file_and_telegram_together_are_refused(tmp_path):
    with pytest.raises(RuntimeError, match="ALERT_FILE"):
        _load(ALERT_FILE=str(tmp_path / "a.jsonl"), TELEGRAM_BOT_TOKEN="1:t", TELEGRAM_CHAT_ID="1")
    with pytest.raises(RuntimeError, match="ALERT_FILE"):
        notifier_from_env({"ALERT_FILE": str(tmp_path / "a.jsonl"), "TELEGRAM_BOT_TOKEN": "1:t",
                           "TELEGRAM_CHAT_ID": "1"})


@pytest.mark.parametrize("raw", ["alerts.jsonl", "./alerts.jsonl", "~/alerts.jsonl"])
def test_a_relative_alert_file_is_refused(raw):
    with pytest.raises(RuntimeError, match="ALERT_FILE"):
        _load(ALERT_FILE=raw)
    with pytest.raises(RuntimeError, match="ALERT_FILE"):
        notifier_from_env({"ALERT_FILE": raw})


def test_the_notifier_from_the_environment_is_the_file_telegram_or_the_log(tmp_path):
    assert isinstance(notifier_from_env({"ALERT_FILE": str(tmp_path / "a.jsonl")}), FileNotifier)
    assert isinstance(notifier_from_env({"TELEGRAM_BOT_TOKEN": "1:t", "TELEGRAM_CHAT_ID": "1"}), TelegramNotifier)
    assert isinstance(notifier_from_env({}), LogNotifier)
    assert isinstance(notifier_from_env({"ALERT_FILE": "  "}), LogNotifier)


def test_paper_reads_no_alert_file(tmp_path):
    config, secrets = load_config({"ALERT_FILE": str(tmp_path / "a.jsonl")}, execution=False, stop_pct=D("0.08"))
    assert config.alerts is None and secrets.alert_file is None


def test_the_failure_alert_of_a_unit_is_written_to_the_file(tmp_path):
    path, out = tmp_path / "alerts.jsonl", io.StringIO()
    code = unit_alert.main(["binance-target-mainnet.service"], env={"ALERT_FILE": str(path)}, stdout=out)
    (line,) = _lines(path)
    assert code == 0 and "binance-target-mainnet.service" in line["title"] and "alert sent" in out.getvalue()


def test_the_failure_alert_fails_when_the_file_cannot_be_written(tmp_path):
    out = io.StringIO()
    code = unit_alert.main(["x.service"], env={"ALERT_FILE": str(tmp_path / "missing" / "a.jsonl")}, stdout=out)
    assert code == 1 and "not confirmed" in out.getvalue()


def test_a_read_only_file_is_not_confirmed(tmp_path):
    path = tmp_path / "alerts.jsonl"
    path.write_text("")
    os.chmod(tmp_path, 0o500)
    os.chmod(path, 0o400)
    try:
        if os.geteuid() != 0:                                        # root writes through the mode bits
            assert FileNotifier(path).deliver("t", timeout_s=1.0) is False
    finally:
        os.chmod(tmp_path, 0o700)
