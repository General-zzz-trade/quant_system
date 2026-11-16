"""ADR-0012 (S5b): the protective stop and the breached-stop flatten follow the venue position; a stop fill beyond
the ledger is booked as a foreign fill. Every case is from the ADR's acceptance list."""
from __future__ import annotations

from decimal import Decimal as D

from tests.characterization.test_stops_and_fills import _drain, _with_position


def _active(fake):
    return {k: v for k, v in fake.algo.items() if v["algoStatus"] == "NEW"}


def _venue(fake, qty, entry):
    fake.positions["BTCUSDT"], fake.entry["BTCUSDT"] = D(qty), D(entry)


def test_ledger_0_1_venue_0_15_the_trigger_follows_the_venue_entry_price(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)          # ledger 0.1 long at 100
    _venue(fake, "0.15", "102")                                          # a manual 0.05 on top
    stops.sync()
    (stop,) = _active(fake).values()
    # 97.92 from the venue's entry price, which the venue rounds to its tick; from the ledger it would be 96.
    assert abs(D(str(stop["triggerPrice"])) - D("102") * (1 - D("0.04"))) < D("0.1")


def test_ledger_0_venue_0_1_no_strategy_stop(tmp_path):
    from tests.characterization.test_stops_and_fills import _order
    fake, executor, consumer, stops = _with_position(tmp_path)
    for event in executor.send_order(_order(2, "sell", "0.1", reduce_only=True)):
        consumer.consume(event)                                          # the strategy owns nothing ...
    assert consumer.position.qty == 0
    _venue(fake, "0.1", "100")                                           # ... the venue holds a manual 0.1
    stops.sync()
    assert _active(fake) == {}


def _placed(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)
    stops.sync()
    assert len(_active(fake)) == 1
    return fake, executor, consumer, stops


def test_a_flat_venue_keeps_the_stop_and_blocks_entries(tmp_path):
    fake, executor, consumer, stops = _placed(tmp_path)
    before = set(_active(fake))
    _venue(fake, "0", "0")
    issues = stops.sync()
    assert set(_active(fake)) == before and issues and stops.unprotected() is False


def test_an_opposite_venue_position_keeps_the_stop(tmp_path):
    fake, executor, consumer, stops = _placed(tmp_path)
    before = set(_active(fake))
    _venue(fake, "-0.1", "100")
    assert stops.sync() and set(_active(fake)) == before


def test_an_unreadable_venue_keeps_the_stop(tmp_path, monkeypatch):
    fake, executor, consumer, stops = _placed(tmp_path)
    before = set(_active(fake))

    def unreadable(*args, **kwargs):
        raise ConnectionError("positions unavailable")
    monkeypatch.setattr(stops.adapter, "get_positions", unreadable)
    monkeypatch.setattr(stops.adapter, "get_position", unreadable, raising=False)
    assert stops.sync() and set(_active(fake)) == before


def test_the_same_venue_position_in_another_text_form_keeps_the_same_stop(tmp_path):
    fake, executor, consumer, stops = _placed(tmp_path)
    before = set(_active(fake))
    _venue(fake, "0.100", "100.00")
    stops.sync()
    assert set(_active(fake)) == before and len(fake.algo) == 1           # nothing cancelled, nothing re-placed


def test_a_stop_fill_beyond_the_ledger_books_the_excess_as_a_foreign_fill(tmp_path):
    fake, executor, consumer, stops = _with_position(tmp_path)          # ledger 0.1
    _venue(fake, "0.15", "100")
    stops.sync()
    (client_id,) = _active(fake)
    fake.trigger_stop(client_id, D("95.9"))                               # closes the whole venue position
    stops.sync()
    _drain(executor, consumer)
    assert consumer.position.qty == 0
    foreign = [r for r in executor.journal.fills("foreign:BTCUSDT")]
    assert len(foreign) == 1 and D(foreign[0]["qty"]) == D("0.05")


def test_a_breached_stop_flattens_the_venue_position(tmp_path):
    from tests.characterization.test_exit_paths import _execution, _until
    from tests.unit.lab.test_target_runtime import _advance
    from tests.unit.trading.test_entry_pause import _breach
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    ledger = runtime.sessions["BTCUSDT"].position_qty()
    btc.positions["BTCUSDT"] = ledger * D("1.5")                         # a manual addition in the same direction
    _breach(world, runtime, i + 1)
    assert btc.positions["BTCUSDT"] == 0                                  # the whole venue position is closed


def test_a_breach_with_an_unreadable_venue_sends_nothing(tmp_path, monkeypatch):
    from tests.characterization.test_exit_paths import _execution, _until
    from tests.unit.lab.test_target_runtime import _advance
    world, runtime = _execution(tmp_path, execution="market")
    btc = world["fakes"]["BTCUSDT"]
    i = _until(world, runtime, lambda: btc.positions.get("BTCUSDT", D(0)) != 0)
    _advance(world, runtime, i)
    session = runtime.sessions["BTCUSDT"]
    posts = len(btc.posts)

    def unreadable(*args, **kwargs):
        raise ConnectionError("positions unavailable")
    monkeypatch.setattr(session.flattener.account, "get_positions", unreadable)
    session.flatten("stop_breached", now_ms=int(world["clock"]["now"] * 1000))
    assert len(btc.posts) == posts


def test_an_answer_without_the_symbol_s_row_is_unknown_and_never_flat(tmp_path):
    # ADR-0012 point 3 (audit M2): the ledger went to 0, but the venue's answer has no row for the symbol. That is
    # unknown, not "both flat", so the stop in place is not cancelled.
    from tests.characterization.test_stops_and_fills import _order
    fake, executor, consumer, stops = _placed(tmp_path)
    before = set(_active(fake))
    for event in executor.send_order(_order(2, "sell", "0.1", reduce_only=True)):
        consumer.consume(event)
    assert consumer.position.qty == 0
    original = fake.request_signed

    def no_row(*, method, path, params=None):
        if path == "/fapi/v2/positionRisk":
            return []
        return original(method=method, path=path, params=params)
    fake.request_signed = no_row
    stops.sync()
    assert set(_active(fake)) == before
