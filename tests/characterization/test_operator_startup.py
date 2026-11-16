"""ADR-0016 points 1-3 (S7), at the entry point: execution mode needs Telegram and a heartbeat address; after every
local check and before any account request it sends "execution mode starting" and starts only when Telegram
confirmed it; a refusal by the account check is sent too (best effort); the heartbeat beats while waiting for
market data and between cycles, and its failures never stop the loop. Paper needs none of it.
Driven through the entry point with the fake venue (ADR-0010 harness); never contacts Telegram.
"""
from __future__ import annotations

import pytest

from tests.characterization.test_startup_account_check import Started, world  # noqa: F401  (fixture)
from trading.infrastructure.cli import target as cli
from trading.infrastructure.config import account_identity

KEY, SECRET = "KEY-0123456789", "SECRET-9876543210"


class Notifier:
    channel = "telegram"

    def __init__(self, fake, confirm=True):
        self.fake, self.confirm = fake, confirm
        self.delivered: list[tuple[str, str, float, int]] = []
        self.alerts: list[tuple[str, str, str]] = []

    def deliver(self, title, detail="", *, timeout_s):
        self.delivered.append((title, detail, timeout_s, len(self.fake.requests)))
        return self.confirm

    def alert(self, level, title, detail=""):
        self.alerts.append((level, title, detail))

    def flush(self, timeout):
        pass


@pytest.fixture
def channels(world, monkeypatch):  # noqa: F811
    for name, value in (("TELEGRAM_BOT_TOKEN", "123:tok"), ("TELEGRAM_CHAT_ID", "42"),
                        ("HEARTBEAT_URL", "https://hc.example.invalid/ping/SECRET-PING"),
                        ("BINANCE_TESTNET_API_KEY", KEY), ("BINANCE_TESTNET_API_SECRET", SECRET)):
        monkeypatch.setenv(name, value)
    notifier = Notifier(world["fake"])
    monkeypatch.setattr(cli, "make_notifier", lambda secrets: notifier)
    return notifier


@pytest.mark.parametrize("unset,named", [(("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"), "TELEGRAM_BOT_TOKEN"),
                                         (("HEARTBEAT_URL",), "HEARTBEAT_URL")])
def test_execution_without_alerts_or_a_heartbeat_is_refused_before_any_account_request(world, channels,  # noqa: F811
                                                                                     monkeypatch, unset, named):
    for name in unset:
        monkeypatch.delenv(name)
    with pytest.raises(RuntimeError, match=named):
        world["main"](world["argv"])
    assert world["fake"].requests == [] and channels.delivered == []


def test_the_start_is_announced_and_confirmed_before_any_account_request(world, channels):  # noqa: F811
    with pytest.raises(Started):
        world["main"](world["argv"])
    ((title, detail, timeout, requests_before),) = channels.delivered
    assert requests_before == 0 and timeout > 0
    assert "starting" in title and "testnet" in (title + detail).lower()
    assert account_identity(testnet=True, api_key=KEY)[:12] in detail and "binance-testnet-ut" in detail
    assert KEY not in title + detail and SECRET not in title + detail and "SECRET-PING" not in title + detail


def test_an_unconfirmed_start_alert_refuses_the_start_with_no_account_request(world, channels):  # noqa: F811
    channels.confirm = False
    with pytest.raises(RuntimeError, match="not confirmed"):
        world["main"](world["argv"])
    assert world["fake"].requests == []


def test_a_refusal_by_the_account_check_is_sent_too(world, channels):  # noqa: F811
    from decimal import Decimal as D
    world["fake"].positions["ETHUSDT"] = D("-1")
    with pytest.raises(RuntimeError, match="not in the strategy ledger"):
        world["main"](world["argv"])
    assert len(channels.delivered) == 2
    title, detail = channels.delivered[1][:2]
    assert "refused" in title and "not in the strategy ledger" in detail


def test_the_runtime_alerts_through_the_same_notifier(world, channels, monkeypatch):  # noqa: F811
    import runner.binance_target as entry
    seen = {}

    def stand_in(**kwargs):
        seen.update(kwargs)
        raise Started()
    monkeypatch.setattr(entry, "TargetRuntime", stand_in)
    with pytest.raises(Started):
        world["main"](world["argv"])
    assert seen["notifier"] is channels


def test_paper_needs_no_alerts_or_heartbeat_and_announces_nothing(world, monkeypatch):  # noqa: F811
    for name in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "HEARTBEAT_URL"):
        monkeypatch.delenv(name, raising=False)
    delivered = []
    monkeypatch.setattr(cli, "make_notifier", lambda secrets: delivered.append(secrets) or None)
    paper = [a for a in world["argv"] if a != "--enable-execution"]
    paper[paper.index("--state-dir") + 1] = str(world["state"].parent / "paper")   # paper refuses an execution dir
    with pytest.raises(Started):
        world["main"](paper)
    assert delivered == []


# -- the heartbeat in the loop ---------------------------------------------------------------------------------
class Runtime:
    def __init__(self, order):
        self.order, self.holds = order, 2

    def warmup_or_hold(self):
        self.order.append("warmup")
        self.holds -= 1
        return self.holds < 0

    def cycle(self):
        self.order.append("cycle")


class Heartbeat:
    def __init__(self, order, fails=False):
        self.order, self.fails = order, fails

    def beat(self):
        self.order.append("beat")
        if self.fails:
            raise ConnectionError("monitor unreachable")


@pytest.mark.parametrize("fails", [False, True])
def test_the_heartbeat_beats_while_waiting_and_between_cycles_and_never_stops_the_loop(world, channels,  # noqa: F811
                                                                                      monkeypatch, fails):
    import runner.binance_target as entry
    order: list[str] = []
    monkeypatch.setattr(entry, "TargetRuntime", lambda **kwargs: Runtime(order))
    monkeypatch.setattr(cli, "make_heartbeat", lambda config, secrets: Heartbeat(order, fails))
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    world["main"](world["argv"] + ["--max-cycles", "3"])
    assert order.count("cycle") == 3                                   # a failing heartbeat stops nothing
    first_ready = [i for i, step in enumerate(order) if step == "warmup"][2]
    assert "beat" in order[:first_ready]                               # beating while market data is down
    cycles = [i for i, step in enumerate(order) if step == "cycle"]
    assert all("beat" in order[a:b] for a, b in zip(cycles, cycles[1:]))


def test_the_start_waits_at_most_thirty_seconds_for_telegram(world, channels):  # noqa: F811
    # S6-S8 audit, mutant C6: a start alert that may wait for minutes is no check at all.
    with pytest.raises(Started):
        world["main"](world["argv"])
    assert channels.delivered[0][2] == cli.START_ALERT_SECONDS and 0 < cli.START_ALERT_SECONDS <= 30
