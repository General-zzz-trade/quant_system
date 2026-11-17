"""Which Binance answers to an order placement mean "rejected" and which mean "outcome unknown" (task 1, 2026-11-17).

Only an answer whose Binance error code refuses the request itself (parameters, permissions, account or symbol
rules) is ``rejected``. Everything else is ``error``: the order may exist, so the lifecycle keeps looking it up for
the whole not-found grace and keeps the symbol's entries locked meanwhile. Sources (checked 2026-11-17):
https://developers.binance.com/docs/derivatives/usds-margined-futures/error-code and .../general-info
(-1006/-1007 and the HTTP 503 "Unknown error" are documented as "execution status unknown"; HTTP 408 is a timeout
waiting for the backend; for HTTP 429/418 the documentation does not say whether the request was processed).
"""
from __future__ import annotations

import threading
from decimal import Decimal as D

import pytest

from tests.support.binance_errors import network_error, rest_error
from tests.support.state_db import state_db
from tests.unit.binance.fake_venue import FakeBinance, make_bars
from trading.adapters.binance.adapter import BinanceAdapter
from trading.adapters.binance.config import BinanceConfig
from trading.adapters.binance.rest import BinanceRetryableError
from trading.adapters.sqlite.order_journal import SqliteOrderJournal
from trading.application.execution.lifecycle import OrderLifecycle
from trading.domain.orders import ENTRY_UNRESOLVED

NOW = 1_800_000_000.0
SCOPE = "binance-testnet-ut"
GRACE = 90.0                                   # trading.infrastructure.target.NOT_FOUND_GRACE_SECONDS

# (HTTP status, Binance code): the request itself was refused, nothing was placed.
REFUSED = [(400, -2019), (400, -2010), (400, -2018), (400, -2022), (400, -4164), (400, -4131), (400, -5022),
           (400, -1121), (400, -1102), (400, -1111), (400, -1021), (400, -5028), (401, -2015), (400, -4116),
           (400, -4120), (400, -2021)]
# Everything else: the outcome is unknown and must be looked up.
UNKNOWN = [(400, -1000), (400, -1001), (400, -1006), (400, -1007), (408, -1007), (408, None), (400, -1008),
           (400, -1003), (400, -1015), (400, -1016), (400, -4117), (400, None), (400, -9999), (403, None),
           (429, -1003), (429, None), (418, None), (500, -1000), (502, None), (503, None)]


def _adapter(fake: FakeBinance) -> BinanceAdapter:
    fake.price["BTCUSDT"] = D("100")
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    assert adapter.connect(symbols=("BTCUSDT",))
    return adapter


def _failing_posts(fake: FakeBinance, error: Exception, *, times: int = 1_000) -> list[int]:
    """Every order and algo-order POST raises ``error`` before reaching the venue, ``times`` times."""
    original, attempts = fake.request_signed, [0]

    def failing(*, method, path, params=None):
        if method == "POST" and path in ("/fapi/v1/order", "/fapi/v1/algoOrder"):
            attempts[0] += 1
            if attempts[0] <= times:
                raise error
        return original(method=method, path=path, params=params)

    fake.request_signed = failing
    return attempts


def _place(adapter: BinanceAdapter, kind: str) -> dict:
    if kind == "market":
        return dict(adapter.send_market_order("BTCUSDT", "buy", 0.1, client_order_id="c1"))
    if kind == "limit":
        return dict(adapter.send_limit_order("BTCUSDT", "buy", 0.1, 99.0, client_order_id="c1"))
    return dict(adapter.place_stop_market("BTCUSDT", "sell", D("96"), client_algo_id="s1"))


KINDS = ["market", "limit", "stop"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status,code", REFUSED)
def test_a_refusal_of_the_request_itself_is_rejected(kind, status, code):
    fake = FakeBinance(make_bars(10), clock=lambda: NOW)
    adapter = _adapter(fake)
    _failing_posts(fake, rest_error(status, code))
    placed = _place(adapter, kind)
    assert (placed["status"], placed["code"]) == ("rejected", code)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status,code", UNKNOWN)
def test_any_other_answer_is_an_unknown_outcome(kind, status, code):
    fake = FakeBinance(make_bars(10), clock=lambda: NOW)
    adapter = _adapter(fake)
    _failing_posts(fake, rest_error(status, code))
    placed = _place(adapter, kind)
    assert placed["status"] == "error" and not placed.get("breached", False)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("error", [network_error(), BinanceRetryableError("Rate limited (local): /fapi/v1/order"),
                                   TimeoutError("read timed out"), OSError("connection reset")])
def test_no_http_answer_is_an_unknown_outcome(kind, error):
    fake = FakeBinance(make_bars(10), clock=lambda: NOW)
    adapter = _adapter(fake)
    _failing_posts(fake, error)
    assert _place(adapter, kind)["status"] == "error"


@pytest.mark.parametrize("status,code,breached", [(400, -2021, True), (400, -4142, True), (400, -2019, False),
                                                  (400, -4120, False)])
def test_only_an_immediate_trigger_refusal_reports_a_breached_stop(status, code, breached):
    """-2021 "Order would immediately trigger." and -4142 "REJECT: take profit or stop order will be triggered
    immediately" both mean the price is already beyond the trigger (tech lead 2026-11-17: -4142 counts as breached;
    which of the two the algo-order endpoint returns is not verified)."""
    fake = FakeBinance(make_bars(10), clock=lambda: NOW)
    adapter = _adapter(fake)
    _failing_posts(fake, rest_error(status, code))
    placed = _place(adapter, "stop")
    assert (placed["status"], placed["breached"]) == ("rejected", breached)


@pytest.mark.parametrize("http_status,expected", [(400, -2019), (401, -2019), (499, -2019), (403, None), (408, None),
                                                  (418, None), (429, None), (399, None), (500, None), (503, None)])
def test_a_listed_code_counts_only_with_an_http_status_that_can_carry_a_refusal(http_status, expected):
    """Defence in depth: the REST client maps 418/429 and 5xx to other classes, but a refusal code arriving with a
    timeout, a WAF block, a rate limit or a server error status is still an unknown outcome."""
    from trading.adapters.binance.adapter import _refused_code
    from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRateLimitError
    assert _refused_code(BinanceNonRetryableError("x", http_status=http_status, code=-2019)) == expected
    assert _refused_code(BinanceNonRetryableError("x", http_status=400, code=None)) is None
    assert _refused_code(BinanceRateLimitError("HTTP 429")) is None


# -- the lifecycle on top of the real adapter -----------------------------------------------------------------------
def _lifecycle(tmp_path, fake: FakeBinance, clock: dict) -> OrderLifecycle:
    journal = SqliteOrderJournal(state_db(tmp_path / "orders.sqlite"), SCOPE)
    return OrderLifecycle(_adapter(fake), journal, SCOPE, clock=lambda: clock["t"], sleeper=lambda s: None,
                          lock=threading.RLock(), wait_seconds=0, owner="o", entry_order_type="market",
                          not_found_grace_seconds=GRACE)


def _row(lifecycle: OrderLifecycle, intent_id: str) -> tuple[str, int]:
    row = next(r for r in lifecycle.journal.orders("o") if r["intent_id"] == intent_id)
    return row["status"], row["settled"]


@pytest.mark.parametrize("status,code", [(408, -1007), (408, None), (429, -1003), (418, None), (400, -1000),
                                         (400, -1001), (400, -1006), (400, -1007), (503, None)])
def test_an_unknown_outcome_keeps_the_entry_lock_for_the_whole_grace(tmp_path, status, code):
    clock = {"t": NOW}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["t"])
    lifecycle = _lifecycle(tmp_path, fake, clock)
    attempts = _failing_posts(fake, rest_error(status, code), times=1)
    assert lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("0.1")) == ()
    assert _row(lifecycle, "i1") == ("unknown", 0) and "i1" in lifecycle.pending
    # A second entry on the symbol is refused while the first one's outcome is unknown.
    assert lifecycle.submit(intent_id="i2", symbol="BTCUSDT", side="buy", qty=D("0.1")) == ()
    assert lifecycle.refusals["i2"] == ENTRY_UNRESOLVED and attempts[0] == 1
    clock["t"] = NOW + GRACE - 0.1
    lifecycle.poll()
    assert _row(lifecycle, "i1") == ("unknown", 0) and "i1" in lifecycle.pending
    clock["t"] = NOW + GRACE
    lifecycle.poll()
    assert _row(lifecycle, "i1") == ("not_found", 1) and lifecycle.pending == {}
    facts = lifecycle.submit(intent_id="i3", symbol="BTCUSDT", side="buy", qty=D("0.1"))
    assert attempts[0] == 2 and [f.qty for f in facts] == [D("0.1")]


def test_an_order_that_appears_after_an_unknown_answer_is_followed_and_never_sent_again(tmp_path):
    """HTTP 408: the venue took the order but answered with a timeout, and its order lookup lags behind."""
    clock = {"t": NOW}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["t"])
    lifecycle = _lifecycle(tmp_path, fake, clock)
    original, hidden, posts = fake.request_signed, {"on": True}, [0]

    def accepted_then_timeout(*, method, path, params=None):
        if method == "POST" and path == "/fapi/v1/order":
            posts[0] += 1
            original(method=method, path=path, params=params)          # the venue places and fills it ...
            raise rest_error(408, -1007)                                 # ... but the answer is a timeout
        if method == "GET" and path == "/fapi/v1/order" and hidden["on"]:
            fake._error(-2013, "Order does not exist.")                  # not visible to lookups yet
        return original(method=method, path=path, params=params)

    fake.request_signed = accepted_then_timeout
    assert lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("0.1")) == ()
    assert _row(lifecycle, "i1") == ("unknown", 0)
    assert lifecycle.submit(intent_id="i2", symbol="BTCUSDT", side="buy", qty=D("0.1")) == ()
    assert lifecycle.refusals["i2"] == ENTRY_UNRESOLVED
    hidden["on"] = False
    clock["t"] = NOW + 5
    booked = lifecycle.poll()
    assert [(intent, fill.qty) for intent, fill in booked] == [("i1", D("0.1"))]
    assert _row(lifecycle, "i1") == ("filled", 1) and posts[0] == 1
    assert fake.positions["BTCUSDT"] == D("0.1")                         # one order, booked once


def test_a_refused_request_settles_at_once_and_frees_the_symbol(tmp_path):
    clock = {"t": NOW}
    fake = FakeBinance(make_bars(10), clock=lambda: clock["t"])
    lifecycle = _lifecycle(tmp_path, fake, clock)
    attempts = _failing_posts(fake, rest_error(400, -2019), times=1)
    assert lifecycle.submit(intent_id="i1", symbol="BTCUSDT", side="buy", qty=D("0.1")) == ()
    assert _row(lifecycle, "i1") == ("rejected", 1)
    facts = lifecycle.submit(intent_id="i2", symbol="BTCUSDT", side="buy", qty=D("0.1"))
    assert attempts[0] == 2 and [f.qty for f in facts] == [D("0.1")]
