"""Alerts reach a person (migration step 12, part 3): Telegram or the process log, never breaking the loop."""
from __future__ import annotations

import json
import logging

import pytest

from tests.characterization.test_exit_paths import _tick
from tests.characterization.test_runtime_decisions import MARKET, _until_position
from tests.unit.lab.test_target_runtime import _world
from trading.adapters.notify.log import LogNotifier
from trading.adapters.notify.telegram import TelegramNotifier
from trading.application.alerts import ALERT_AFTER_SECONDS, REPEAT_SECONDS, IssueAlerts, NotifyingEventLog
from trading.application.issues import Issue
from trading.domain.events import EventKind
from trading.domain.issues import IssueKind
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure.config import notifier_from_env
from tests.support.state_db import state_dir_db

TOKEN = "123456:secret-token"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "BINANCE_EXCHANGE_LEVERAGE", "BINANCE_MARGIN_TYPE",
                 "MAX_DAILY_DRAWDOWN_PCT", "MAX_WEEKLY_DRAWDOWN_PCT", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.setenv("RISK_SCOPE", "binance-testnet-ut")


class Recorder:
    channel = "recorder"

    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def alert(self, level, title, detail=""):
        if self.fail:
            raise OSError("network down")
        self.sent.append((level, title, detail))


class Rows:
    def __init__(self):
        self.rows = []

    def write(self, kind, **fields):
        self.rows.append((kind.value, fields))            # the word, as the JSONL log writes it (S3b)


# -- the Telegram adapter ---------------------------------------------------------------------------------------
def test_telegram_posts_one_message_to_the_configured_chat():
    posts = []
    TelegramNotifier(TOKEN, "42", post=lambda url, body: posts.append((url, json.loads(body)))).alert(
        "critical", "BTCUSDT: alert", "entries paused")
    ((url, body),) = posts
    assert url == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert body == {"chat_id": "42", "text": "[CRITICAL] BTCUSDT: alert\nentries paused",
                    "disable_web_page_preview": True}


def test_a_failed_delivery_is_logged_without_the_token_and_never_raises(caplog):
    def down(url, body):
        raise OSError(f"cannot reach {url}")

    with caplog.at_level(logging.WARNING):
        TelegramNotifier(TOKEN, "42", post=down).alert("warning", "title", "detail")
    assert "not delivered (OSError)" in caplog.text and "secret-token" not in caplog.text


def test_a_long_detail_is_cut_to_what_telegram_accepts():
    posts = []
    TelegramNotifier(TOKEN, "42", post=lambda url, body: posts.append(json.loads(body))).alert(
        "info", "t", "x" * 10_000)
    assert len(posts[0]["text"]) <= 4096 and posts[0]["text"].endswith("…")


def test_the_log_notifier_writes_to_the_process_log(caplog):
    with caplog.at_level(logging.INFO):
        LogNotifier().alert("critical", "BTCUSDT: alert", "entries paused")
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [(logging.CRITICAL,
                                                                     "BTCUSDT: alert: entries paused")]


# -- configuration ------------------------------------------------------------------------------------------
def test_both_settings_choose_telegram_neither_the_log_and_one_alone_is_an_error():
    chosen = notifier_from_env({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": "42"})
    assert isinstance(chosen, TelegramNotifier) and chosen.channel == "telegram"
    assert notifier_from_env({}).channel == "log only"
    for env in ({"TELEGRAM_BOT_TOKEN": TOKEN}, {"TELEGRAM_CHAT_ID": "42"},
                {"TELEGRAM_BOT_TOKEN": " ", "TELEGRAM_CHAT_ID": "42"}):
        with pytest.raises(RuntimeError, match="both") as raised:
            notifier_from_env(env)
        assert "secret" not in str(raised.value)


# -- alert rows go out at once ----------------------------------------------------------------------------------
def test_alert_rows_are_sent_at_once_and_other_rows_are_not():
    inner, notifier = Rows(), Recorder()
    log = NotifyingEventLog(inner, notifier, prefix="binance-testnet-ut")
    log.write(EventKind.ISSUE, symbol="BTCUSDT", issue="x")
    log.write(EventKind.ALERT, symbol="BTCUSDT", reason="entries paused: protective stop triggered")
    assert [k for k, _ in inner.rows] == ["issue", "alert"]
    assert notifier.sent == [("critical", "[binance-testnet-ut] BTCUSDT",
                              "entries paused: protective stop triggered")]


def test_a_notifier_that_fails_never_stops_the_event_log():
    inner = Rows()
    NotifyingEventLog(inner, Recorder(fail=True), prefix="p").write(EventKind.ALERT, symbol="BTCUSDT", reason="r")
    assert inner.rows == [("alert", {"symbol": "BTCUSDT", "reason": "r"})]


# -- lasting issues -----------------------------------------------------------------------------------------
def test_an_issue_is_sent_once_it_lasts_then_repeated_rarely_and_its_end_is_reported():
    clock, notifier = {"t": 0.0}, Recorder()
    alerts = IssueAlerts(notifier, lambda: clock["t"], prefix="p")
    issue = Issue(IssueKind.LEDGER_MISMATCH, "BTCUSDT", "BTCUSDT: venue qty=0.1, ledger qty=0")
    for t in (0, 60, ALERT_AFTER_SECONDS - 1):
        clock["t"] = t
        alerts.observe("BTCUSDT", [issue])
    assert notifier.sent == []                                                # a passing issue is not sent
    clock["t"] = ALERT_AFTER_SECONDS
    alerts.observe("BTCUSDT", [issue])
    assert notifier.sent == [("warning", "[p] BTCUSDT", f"{issue} (for 2 min)")]
    clock["t"] = ALERT_AFTER_SECONDS + REPEAT_SECONDS - 1
    alerts.observe("BTCUSDT", [issue])
    assert len(notifier.sent) == 1
    clock["t"] = ALERT_AFTER_SECONDS + REPEAT_SECONDS
    alerts.observe("BTCUSDT", [issue])
    assert len(notifier.sent) == 2
    alerts.observe("ETHUSDT", [])                                             # another symbol changes nothing
    alerts.observe("BTCUSDT", [])
    assert notifier.sent[-1] == ("info", "[p] BTCUSDT", f"resolved: {issue}")
    alerts.observe("BTCUSDT", [])
    assert len(notifier.sent) == 3


def test_an_issue_that_ends_before_it_was_sent_ends_silently_and_starts_afresh():
    clock, notifier = {"t": 0.0}, Recorder()
    alerts = IssueAlerts(notifier, lambda: clock["t"], prefix="p")
    alerts.observe("BTCUSDT", [Issue(IssueKind.UNDELIVERED_FILL, "BTCUSDT", "BTCUSDT: undelivered fills")])
    clock["t"] = 100
    alerts.observe("BTCUSDT", [])
    clock["t"] = 200
    alerts.observe("BTCUSDT", [Issue(IssueKind.UNDELIVERED_FILL, "BTCUSDT", "BTCUSDT: undelivered fills")])
    clock["t"] = 300
    alerts.observe("BTCUSDT", [Issue(IssueKind.UNDELIVERED_FILL, "BTCUSDT", "BTCUSDT: undelivered fills")])
    assert notifier.sent == []


def test_some_issues_may_wait_longer_before_they_are_sent():
    clock, notifier = {"t": 0.0}, Recorder()
    alerts = IssueAlerts(notifier, lambda: clock["t"], prefix="p",
                         wait=lambda issue: 4000.0 if "unresolved" in issue else ALERT_AFTER_SECONDS)
    for t in (0, 3999):
        clock["t"] = t
        alerts.observe("BTCUSDT", [Issue(IssueKind.UNRESOLVED_ORDER, "BTCUSDT",
                                         "BTCUSDT: unresolved orders or legacy journal records")])
    assert notifier.sent == []
    clock["t"] = 4000
    alerts.observe("BTCUSDT", [Issue(IssueKind.UNRESOLVED_ORDER, "BTCUSDT",
                                         "BTCUSDT: unresolved orders or legacy journal records")])
    assert len(notifier.sent) == 1


def test_a_notifier_that_fails_never_stops_the_issue_tracking():
    clock = {"t": 0.0}
    alerts = IssueAlerts(Recorder(fail=True), lambda: clock["t"], prefix="p")
    alerts.observe("BTCUSDT", [Issue(IssueKind.RECONCILE_FAILED, "BTCUSDT", "x")])
    clock["t"] = ALERT_AFTER_SECONDS
    alerts.observe("BTCUSDT", [Issue(IssueKind.RECONCILE_FAILED, "BTCUSDT", "x")])
    alerts.observe("BTCUSDT", [])


# -- in the runtime -----------------------------------------------------------------------------------------
def _runtime(tmp_path, notifier, **spec):
    from runner.binance_target import TargetRuntime
    world = _world(tmp_path, StrategySpec(**MARKET, **spec))
    state_dir_db(tmp_path / "exec")   # a state database as init creates it (ADR-0009)
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "exec", mode="execution",
                            market=world["adapter"], account=world["adapter"], clock=lambda: world["clock"]["now"],
                            testnet=True, notifier=notifier)
    runtime.warmup()
    return world, runtime


def test_the_runtime_sends_a_lasting_missing_stop_and_its_end(tmp_path):
    notifier = Recorder()
    world, runtime = _runtime(tmp_path, notifier, execution="market")
    _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    original = btc.request_signed

    def refuse(*, method, path, params=None):
        if path == "/fapi/v1/algoOrder" and method == "POST":
            btc._error(-4120, "not supported")
        return original(method=method, path=path, params=params)

    btc.request_signed = refuse
    for stop in btc.algo.values():
        stop["algoStatus"] = "CANCELED"
    for _ in range(25):                                   # 150 s without a stop
        _tick(world, runtime)
    sent = [d for level, title, d in notifier.sent if level == "warning" and "BTCUSDT" in title]
    assert any("position without an active protective stop" in d for d in sent)
    btc.request_signed = original
    _tick(world, runtime)
    assert any(level == "info" and "resolved" in d and "protective stop" in d for level, _, d in notifier.sent)
    assert runtime.status()["alerts"] == "recorder"


def test_the_runtime_sends_an_entry_pause_at_once(tmp_path):
    notifier = Recorder()
    world, runtime = _runtime(tmp_path, notifier, execution="market")
    _until_position(world, runtime)
    btc = world["fakes"]["BTCUSDT"]
    (client_id,) = [k for k, v in btc.algo.items() if v["algoStatus"] == "NEW"]
    btc.trigger_stop(client_id, btc.mark("BTCUSDT"))
    _tick(world, runtime)
    critical = [(t, d) for level, t, d in notifier.sent if level == "critical"]
    assert [t for t, _ in critical] == ["[binance-testnet-ut] BTCUSDT"]
    assert critical[0][1].startswith("entries paused: protective stop triggered")


def test_without_telegram_settings_execution_alerts_go_to_the_log(tmp_path):
    world, runtime = _runtime(tmp_path, None, execution="market")
    assert runtime.status()["alerts"] == "log only"


@pytest.mark.parametrize("execution,ttl,wait", [("maker", 3, 3 * 3600.0 + 600.0), ("market", 1, ALERT_AFTER_SECONDS)])
def test_an_unresolved_order_is_news_only_after_a_maker_entry_s_lifetime(tmp_path, execution, ttl, wait):
    _, runtime = _runtime(tmp_path, Recorder(), execution=execution, ttl_bars=ttl)
    unresolved = Issue(IssueKind.UNRESOLVED_ORDER, "BTCUSDT", "BTCUSDT: unresolved orders or legacy journal records")
    mismatch = Issue(IssueKind.LEDGER_MISMATCH, "BTCUSDT", "BTCUSDT: venue qty=0.1, ledger qty=0")
    assert runtime._alert_wait(unresolved) == wait
    assert runtime._alert_wait(mismatch) == ALERT_AFTER_SECONDS
