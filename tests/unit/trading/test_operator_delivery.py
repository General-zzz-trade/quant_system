"""ADR-0016 points 2 and 3 (S7), the adapter parts.

- ``deliver``: a synchronous message that returns True only when Telegram confirmed it (its answer says
  ``"ok": true``) within the timeout. Anything else is False and is logged without the token. The process log
  never counts as delivered.
- ``HttpHeartbeat``: at most one ping a minute, sent in the background with a timeout. A failure is logged
  without the address and never reaches the caller: the heartbeat is a bystander.
"""
from __future__ import annotations

import json
import logging
import threading
import time

from trading.adapters.notify.heartbeat import HttpHeartbeat
from trading.adapters.notify.log import LogNotifier
from trading.adapters.notify.telegram import TelegramNotifier

TOKEN = "123456:SECRET-BOT-TOKEN"
URL = "https://hc.example.invalid/ping/SECRET-PING-ID"


def _answer(ok=True):
    return json.dumps({"ok": ok, "result": {}}).encode()


def test_deliver_is_true_only_when_telegram_confirms():
    seen = []

    def request(url, body, timeout):
        seen.append((json.loads(body), timeout))
        return _answer(True)
    notifier = TelegramNotifier(TOKEN, "42", request=request, background=True)
    assert notifier.deliver("execution mode starting", "testnet, identity 1a2b3c", timeout_s=7.5) is True
    ((body, timeout),) = seen
    assert body["chat_id"] == "42" and "execution mode starting" in body["text"] and "1a2b3c" in body["text"]
    assert timeout == 7.5


def test_an_answer_that_is_not_ok_is_not_delivered():
    notifier = TelegramNotifier(TOKEN, "42", request=lambda url, body, timeout: _answer(False))
    assert notifier.deliver("t", "d", timeout_s=5) is False


def test_only_a_true_ok_counts_as_delivered():
    # S6-S8 audit, mutant C5: a truthy "ok" that is not true (a string, a number) is no confirmation.
    for ok in ("false", "true", 1, [1]):
        answer = json.dumps({"ok": ok}).encode()
        assert TelegramNotifier(TOKEN, "42", request=lambda url, body, timeout, a=answer: a).deliver(
            "t", "d", timeout_s=5) is False


def test_an_unreadable_answer_is_not_delivered():
    notifier = TelegramNotifier(TOKEN, "42", request=lambda url, body, timeout: b"<html>bad gateway</html>")
    assert notifier.deliver("t", "d", timeout_s=5) is False


def test_an_error_or_timeout_is_not_delivered_and_the_token_is_never_logged(caplog):
    def fails(url, body, timeout):
        raise TimeoutError(f"timed out talking to {url}")
    notifier = TelegramNotifier(TOKEN, "42", request=fails)
    with caplog.at_level(logging.DEBUG):
        assert notifier.deliver("t", "d", timeout_s=5) is False
    assert caplog.records and TOKEN not in caplog.text and "TimeoutError" in caplog.text


def test_deliver_goes_out_now_even_with_alerts_queued_behind_a_slow_sender():
    release = threading.Event()

    def slow(url, body):
        release.wait(5)
    notifier = TelegramNotifier(TOKEN, "42", post=slow, request=lambda url, body, timeout: _answer(True),
                                background=True)
    notifier.alert("critical", "queued")
    started = time.monotonic()
    assert notifier.deliver("now", "", timeout_s=5) is True
    assert time.monotonic() - started < 1.0
    release.set()


def test_the_process_log_never_counts_as_delivered():
    assert LogNotifier().deliver("t", "d", timeout_s=5) is False


# -- heartbeat -----------------------------------------------------------------------------------------------
class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _heartbeat(get, clock):
    return HttpHeartbeat(URL, get=get, clock=clock)


def test_at_most_one_ping_a_minute():
    sent, clock = [], _Clock()
    beat = _heartbeat(lambda url, timeout: sent.append((url, timeout)), clock)
    for step in (0, 10, 59.9):
        clock.now = 1000.0 + step
        beat.beat()
    beat.flush(5)
    assert len(sent) == 1
    clock.now = 1060.0
    beat.beat()
    beat.flush(5)
    assert len(sent) == 2 and sent[0] == (URL, 5.0)


def test_a_slow_ping_never_holds_up_the_caller():
    release, clock = threading.Event(), _Clock()
    beat = _heartbeat(lambda url, timeout: release.wait(5), clock)
    started = time.monotonic()
    beat.beat()
    assert time.monotonic() - started < 0.5
    release.set()
    beat.flush(5)


def test_a_failing_ping_is_logged_without_the_address_and_never_raises(caplog):
    clock = _Clock()

    def fails(url, timeout):
        raise ConnectionError(f"cannot reach {url}")
    beat = _heartbeat(fails, clock)
    with caplog.at_level(logging.DEBUG):
        beat.beat()
        beat.flush(5)
    assert "ConnectionError" in caplog.text and "SECRET-PING-ID" not in caplog.text and URL not in caplog.text


def test_a_ping_still_in_flight_is_not_doubled():
    release, sent, clock = threading.Event(), [], _Clock()

    def slow(url, timeout):
        sent.append(url)
        release.wait(5)
    beat = _heartbeat(slow, clock)
    beat.beat()
    clock.now += 61
    beat.beat()                               # the first is still waiting: no second thread
    time.sleep(0.05)
    assert len(sent) == 1
    release.set()
    beat.flush(5)


def test_the_address_is_not_in_the_heartbeat_s_text():
    beat = HttpHeartbeat(URL)
    assert "SECRET-PING-ID" not in repr(beat) and "SECRET-PING-ID" not in str(beat)
