"""The entry point with an alert file in place of Telegram (owner, 2026-11-16): "execution starting" is written to
the file before any account request, and a file that cannot be written refuses the start with no account request.
Driven with the fake venue of the start-up tests; never contacts Telegram or an exchange."""
from __future__ import annotations

import json

import pytest

from tests.characterization.test_startup_account_check import Started, world  # noqa: F401  (fixture)

KEY, SECRET = "KEY-0123456789", "SECRET-9876543210"


@pytest.fixture
def file_channel(world, monkeypatch, tmp_path):  # noqa: F811
    path = tmp_path / "alerts" / "testnet.jsonl"
    path.parent.mkdir()
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("ALERT_FILE", str(path)), ("HEARTBEAT_URL", "https://hc.example.invalid/ping/SECRET-PING"),
                        ("BINANCE_TESTNET_API_KEY", KEY), ("BINANCE_TESTNET_API_SECRET", SECRET)):
        monkeypatch.setenv(name, value)
    from trading.infrastructure import config
    from trading.infrastructure.cli import target as cli
    monkeypatch.setattr(cli, "make_notifier", config.make_notifier)      # the real one, not the fixture's stand-in
    return path


def test_the_start_is_written_to_the_alert_file_before_any_account_request(world, file_channel):  # noqa: F811
    seen = {}
    real = world["fake"].request_signed if hasattr(world["fake"], "request_signed") else None
    assert real is not None

    def spy(*args, **kwargs):
        seen.setdefault("lines_at_first_request", len(file_channel.read_text().splitlines()))
        return real(*args, **kwargs)
    world["fake"].request_signed = spy
    with pytest.raises(Started):
        world["main"](world["argv"])
    lines = [json.loads(line) for line in file_channel.read_text().splitlines()]
    assert "execution starting" in lines[0]["title"] and "binance-testnet-ut" in lines[0]["detail"]
    assert seen["lines_at_first_request"] >= 1                       # the line was there before the account was asked
    text = file_channel.read_text()
    assert KEY not in text and SECRET not in text and "SECRET-PING" not in text


def test_an_alert_file_that_cannot_be_written_refuses_the_start_with_no_account_request(world, file_channel,  # noqa: F811
                                                                                        monkeypatch, tmp_path):
    monkeypatch.setenv("ALERT_FILE", str(tmp_path / "no-such-directory" / "alerts.jsonl"))
    with pytest.raises(RuntimeError, match="not confirmed"):
        world["main"](world["argv"])
    assert world["fake"].requests == []
