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
    assert line["title"] == "MAINNET execution starting" and line["detail"] == "scope x" and line["level"] == "notice"


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


# -- hardening after the independent audit of 2026-11-16 (findings M1, L2, L5, L6, L8, L9) ---------------------------
def _short_write(monkeypatch, keep):
    """os.write that takes only ``keep`` bytes, as a full disk or a size limit does, without an error."""
    import trading.adapters.notify.file as module
    real = os.write
    monkeypatch.setattr(module.os, "write", lambda fd, data: real(fd, data[:keep]))


def test_a_short_write_is_not_a_confirmed_delivery(tmp_path, monkeypatch):
    path = tmp_path / "alerts.jsonl"
    _short_write(monkeypatch, 40)
    assert FileNotifier(path).deliver("MAINNET execution starting", "scope x", timeout_s=1.0) is False


def test_the_alert_after_a_cut_off_line_starts_on_its_own_line(tmp_path, monkeypatch):
    path = tmp_path / "alerts.jsonl"
    notifier = FileNotifier(path)
    with monkeypatch.context() as patched:
        _short_write(patched, 40)
        notifier.alert("warning", "cut off")                          # 40 bytes, no newline, and no exception
    notifier.alert("critical", "[mainnet] BTCUSDT", "position without a stop")
    last = json.loads(path.read_text().splitlines()[-1])
    assert (last["level"], last["detail"]) == ("critical", "position without a stop")


def test_each_line_is_synced_and_the_file_is_closed(tmp_path, monkeypatch):
    import trading.adapters.notify.file as module
    synced, closed = [], []
    real_sync, real_close = os.fsync, os.close
    monkeypatch.setattr(module.os, "fsync", lambda fd: (synced.append(fd), real_sync(fd))[1])
    monkeypatch.setattr(module.os, "close", lambda fd: (closed.append(fd), real_close(fd))[1])
    FileNotifier(tmp_path / "alerts.jsonl").alert("info", "one")
    assert len(synced) == 1 and closed == synced


@pytest.mark.parametrize("make", [lambda p: p.mkdir(), lambda p: os.symlink(p.parent / "elsewhere.jsonl", p),
                                  lambda p: os.mkfifo(p)])
def test_a_path_that_is_not_a_plain_file_is_never_written_and_never_blocks(tmp_path, make):
    path = tmp_path / "alerts.jsonl"
    (tmp_path / "elsewhere.jsonl").write_text("")
    make(path)
    notifier = FileNotifier(path)
    notifier.alert("critical", "x")                                   # logged, not raised (a directory, a link, a pipe)
    assert notifier.deliver("t", timeout_s=1.0) is False
    assert (tmp_path / "elsewhere.jsonl").read_text() == ""          # a symlink is not followed


@pytest.mark.parametrize("title,detail", [("lone surrogate \udcff", "x"), ("t", "\udcff"), (object(), 7)])
def test_text_that_cannot_be_encoded_is_still_written_and_nothing_raises(tmp_path, title, detail):
    path = tmp_path / "alerts.jsonl"
    notifier = FileNotifier(path)
    notifier.alert("critical", title, detail)
    assert notifier.deliver(title, detail, timeout_s=1.0) is True
    assert len(_lines(path)) == 2


def test_a_broken_clock_never_raises(tmp_path):
    def clock():
        raise ValueError("no time")
    notifier = FileNotifier(tmp_path / "alerts.jsonl", clock=clock)
    notifier.alert("critical", "x")
    assert notifier.deliver("t", timeout_s=1.0) is False


def test_a_very_long_detail_is_cut(tmp_path):
    path = tmp_path / "alerts.jsonl"
    FileNotifier(path).alert("warning", "t" * 5000, "HTTP 500: " + "x" * 100_000)
    (line,) = _lines(path)
    assert len(line["title"]) <= 300 and len(line["detail"]) <= 4000 and line["detail"].endswith("…")


def test_a_confirmed_delivery_is_marked_so_that_a_reader_never_ranks_it_low(tmp_path):
    path = tmp_path / "alerts.jsonl"
    FileNotifier(path).deliver("binance-target-mainnet.service failed on host", timeout_s=1.0)
    assert _lines(path)[0]["level"] == "notice"                       # start, refused start, failed unit: always read


def test_the_host_installation_makes_the_alert_directory_for_the_service_user():
    from pathlib import Path
    script = (Path(__file__).resolve().parents[3] / "infra" / "install_host.sh").read_text()
    code = script.split("cat <<EOF", 1)[0]
    assert 'install -d -m 0750 -o "$SERVICE_USER" -g "$SERVICE_USER" "$STATE_ROOT/alerts"' in code
    assert "ALERT_FILE" in script.split("cat <<EOF", 1)[1]            # and the closing instructions name the setting


def test_the_health_watchdog_s_alerts_reach_the_alert_file(tmp_path, monkeypatch):
    from monitoring import notify
    path = tmp_path / "alerts.jsonl"
    monkeypatch.setenv("ALERT_FILE", str(path))
    monkeypatch.setattr(notify, "send_telegram", lambda *a, **k: False)
    monkeypatch.setattr(notify, "_history_path", lambda: str(tmp_path / "history.jsonl"))
    assert notify.send_alert(notify.AlertLevel.CRITICAL, "disk 96 % full", {"mount": "/"}) is True
    (line,) = _lines(path)
    assert line["level"] == "critical" and "disk 96 % full" in line["title"] and "mount" in line["detail"]


def test_without_an_alert_file_the_watchdog_writes_none(tmp_path, monkeypatch):
    from monitoring import notify
    monkeypatch.delenv("ALERT_FILE", raising=False)
    monkeypatch.setattr(notify, "send_telegram", lambda *a, **k: False)
    monkeypatch.setattr(notify, "_history_path", lambda: str(tmp_path / "history.jsonl"))
    assert notify.send_alert(notify.AlertLevel.WARNING, "x") is False


def test_tests_never_inherit_an_alert_file_from_the_environment():
    assert "ALERT_FILE" not in os.environ                             # tests/conftest.py removes it for every test


def test_an_exported_alert_file_is_removed_for_every_test(tmp_path):
    """The check above in a child process that does export one (the audit's finding L7, reproduced)."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                          f"{Path(__file__).relative_to(root)}::test_tests_never_inherit_an_alert_file_from_the_environment"],
                         cwd=root, env={**os.environ, "ALERT_FILE": str(tmp_path / "host.jsonl")},
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout[-500:]
    assert not (tmp_path / "host.jsonl").exists()
