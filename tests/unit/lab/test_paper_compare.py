"""S11, design G6: a paper record compared fill by fill with research on the same bars (``lab/paper_compare.py``).

Acceptance (design 2026-11 M5, section 2, G6): a paper run and research on the same bars compare equal ("一致"); an
injected difference is reported at the first differing fill; missing or unusable data is "无法比较", never "一致".
An intended divergence (a ``paper_divergence`` event) is listed apart and is no inconsistency, but only with its
event: the same record without its divergence events is "不一致".
"""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest

from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.binance.fake_venue import HOUR
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.adapters.codec import decode, encode

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def _snapshot(world, out: Path) -> None:
    """The bars and funding paper was served, as a binance_public snapshot directory."""
    out.mkdir(parents=True)
    for symbol in SYMBOLS:
        rows = world["rows"][symbol]
        with (out / f"{symbol}_1h.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["open_time", "open", "high", "low", "close", "volume"])
            for r in rows:
                writer.writerow([r["start"], *(str(r[k]) for k in ("open", "high", "low", "close", "volume"))])
        opens = {r["start"]: r["open"] for r in rows}
        with (out / f"{symbol}_funding.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["funding_time", "rate", "mark_price"])
            for at, rate in world["funding"]:
                if at in opens:
                    writer.writerow([at, str(rate), str(opens[at])])


def _run(tmp: Path, script, *, execution="market", ttl=1) -> dict:
    with pytest.MonkeyPatch.context() as mp:
        for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
            mp.delenv(name, raising=False)
        mp.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
        spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                            ttl_bars=ttl)
        world = _world(tmp, spec)
        runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp / "paper", mode="paper",
                                market=world["adapter"], clock=lambda: world["clock"]["now"])
        runtime.warmup()
        script(world, runtime)
    _snapshot(world, tmp / "data")
    return {"paper": tmp / "paper", "data": tmp / "data"}


def _bars(world, runtime, start, count):
    for i in range(start, start + count):
        _advance(world, runtime, i)
    return start + count


def _until(world, runtime, done, limit=300):
    i = world["warm"]
    while not done():
        assert i < world["warm"] + limit, "the scenario never happened"
        _advance(world, runtime, i)
        i += 1
    return i


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("plain"), lambda w, r: _bars(w, r, w["warm"], 150))


@pytest.fixture(scope="module")
def kill_flatten(tmp_path_factory):
    def script(world, runtime):
        i = _until(world, runtime, lambda: any(s.position_qty() != 0 for s in runtime.sessions.values()))
        i = _bars(world, runtime, i, 3)
        runtime.kill.write(active=True, reason="test", flatten=True)
        _bars(world, runtime, i, 12)
    return _run(tmp_path_factory.mktemp("kill"), script)


@pytest.fixture(scope="module")
def entries_refused(tmp_path_factory):
    def script(world, runtime):
        runtime.kill.write(active=True, reason="test", flatten=False)
        _bars(world, runtime, world["warm"], 60)
    return _run(tmp_path_factory.mktemp("refused"), script)


@pytest.fixture(scope="module")
def kill_cancel(tmp_path_factory):
    """The kill switch (no flatten) comes on while a maker entry rests: the exit side cancels it (only limit entries
    are cancelled), and research fills it at a later bar."""
    def script(world, runtime):
        venue = runtime.account
        i = _until(world, runtime, lambda: any(not row["reduce_only"] for s in SYMBOLS
                                               for row in venue.broker(s).pending))
        runtime.kill.write(active=True, reason="test", flatten=False)
        world["clock"]["now"] += 1800                          # half an hour on: no new bar, only the exit side acts
        runtime.cycle()
        assert not any(not row["reduce_only"] for s in SYMBOLS for row in venue.broker(s).pending)
        _bars(world, runtime, i, 6)
    return _run(tmp_path_factory.mktemp("cancel"), script, execution="maker", ttl=4)


@pytest.fixture(scope="module")
def stop_crash(tmp_path_factory):
    def script(world, runtime):
        eth = runtime.sessions["ETHUSDT"]
        i = _until(world, runtime, lambda: eth.position_qty() != 0 and eth.stops.has_active_stop())
        (stop,) = [s for s in runtime.account.get_open_algo_orders() if s.symbol == "ETHUSDT"]
        row = world["rows"]["ETHUSDT"][i]                     # the next bar crashes through the stop
        if eth.position_qty() > 0:
            row["low"] = float(stop.trigger_price * D("0.97"))
        else:
            row["high"] = float(stop.trigger_price * D("1.03"))
        _bars(world, runtime, i, 8)
    return _run(tmp_path_factory.mktemp("stop"), script)


# -- helpers ----------------------------------------------------------------------------------------------------
def _copy(run, tmp_path):
    paper, data = tmp_path / "paper", tmp_path / "data"
    shutil.copytree(run["paper"], paper)
    shutil.copytree(run["data"], data)
    return paper, data


def _compare(paper, data, capsys):
    from lab.paper_compare import main       # imported here: before the tool existed each test failed by itself
    code = main(["--paper-dir", str(paper), "--data-dir", str(data)])
    return code, capsys.readouterr().out


def _when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")


def _document(paper):
    with sqlite3.connect(paper / "state.sqlite") as db:
        return json.loads(db.execute("SELECT document FROM paper_state").fetchone()[0], object_hook=decode)


def _save_document(paper, document):
    with sqlite3.connect(paper / "state.sqlite") as db:
        db.execute("UPDATE paper_state SET document=?", (json.dumps(document, default=encode),))


def _venue_fills(document, symbol):
    return document["venue"]["brokers"][symbol]["broker"]["fills"]


def _journal_fill(paper, symbol, trade_id, change):
    """Rewrite (``change`` a dict) or delete (``change`` None) the order journal's record of a venue fill."""
    with sqlite3.connect(paper / "state.sqlite") as db:
        rows = [(fill_id, json.loads(payload)) for fill_id, payload in
                db.execute("SELECT fill_id, payload FROM execution_fills")]
        (fill_id, payload), = [(f, p) for f, p in rows if p["symbol"] == symbol and p["trade_id"] == trade_id]
        if change is None:
            db.execute("DELETE FROM execution_fills WHERE fill_id=?", (fill_id,))
        else:
            db.execute("UPDATE execution_fills SET payload=? WHERE fill_id=?",
                       (json.dumps({**payload, **change}), fill_id))


def _events(paper):
    return [json.loads(line) for line in (paper / "target_events.jsonl").read_text().splitlines()]


def _keep_events(paper, keep):
    rows = [e for e in _events(paper) if keep(e)]
    (paper / "target_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in rows))


def _csv(path, edit):
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([rows[0], *edit(rows[1:])])


def _files(paper):
    """The record's files. SQLite's own -wal and -shm files are not: any reader of a WAL database may create them."""
    return {p.name: p.read_bytes() for p in sorted(paper.iterdir()) if not p.name.endswith(("-shm", "-wal"))}


# -- acceptance 1: equal ----------------------------------------------------------------------------------------
def test_a_paper_run_and_research_on_the_same_bars_compare_equal(plain, tmp_path, capsys):
    paper, data = _copy(plain, tmp_path)
    before = _files(paper)
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (0, "结论：一致"), out
    document = _document(paper)
    for symbol in SYMBOLS:
        traded = len(_venue_fills(document, symbol))
        assert traded > 0 and f"{symbol}：逐笔一致 {traded} 笔" in out
    assert _files(paper) == before                               # the tool only reads the paper directory


# -- acceptance 2: an injected difference ----------------------------------------------------------------------
@pytest.mark.parametrize("change", ["price", "missing"])
def test_an_injected_difference_is_reported_at_the_first_differing_fill(plain, tmp_path, capsys, change):
    paper, data = _copy(plain, tmp_path)
    document = _document(paper)
    fills = _venue_fills(document, "BTCUSDT")
    second = fills[1]
    if change == "price":
        second["price"] = str(D(second["price"]) + 1)
        _journal_fill(paper, "BTCUSDT", second["id"], {"price": second["price"]})
    else:
        fills.remove(second)
        _journal_fill(paper, "BTCUSDT", second["id"], None)
    _save_document(paper, document)
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (1, "结论：不一致"), out
    assert "BTCUSDT：前 1 笔一致，第 2 笔不同" in out and _when(second["at"]) in out
    assert "ETHUSDT：逐笔一致" in out                              # the other symbol is still compared


# -- acceptance 3: missing data is never "equal" ---------------------------------------------------------------
def _first_btc_fill(paper):
    return _venue_fills(_document(paper), "BTCUSDT")[0]


def _gap(paper, data):
    at = _first_btc_fill(paper)["at"]
    _csv(data / "BTCUSDT_1h.csv", lambda rows: [r for r in rows if int(r[0]) != at])


def _short(paper, data):
    cursor = _document(paper)["symbols"]["BTCUSDT"]["cursor"]
    _csv(data / "BTCUSDT_1h.csv", lambda rows: [r for r in rows if int(r[0]) <= cursor - 10 * HOUR])


def _no_funding(paper, data):
    at = _first_btc_fill(paper)["at"] // (8 * HOUR) * (8 * HOUR)
    _csv(data / "BTCUSDT_funding.csv", lambda rows: [r for r in rows if int(r[0]) != at])


def _other_close(paper, data):
    order = _document(paper)["venue"]["brokers"]["BTCUSDT"]["broker"]["orders"][0]
    decided = order["submitted_at"] - HOUR                          # the bar whose close the order was sent at

    def edit(rows):
        for r in rows:
            if int(r[0]) == decided:
                low, high = D(r[3]), D(r[2])
                r[4] = str((low + (high - low) / 3).quantize(D("0.01")))
                assert D(r[4]) != D(order["reference"])
        return rows
    _csv(data / "BTCUSDT_1h.csv", edit)


def _no_log(paper, data):
    (paper / "target_events.jsonl").unlink()


def _no_warmup(paper, data):
    _keep_events(paper, lambda e: e["kind"] != "warmup")


def _no_db(paper, data):
    for path in paper.glob("state.sqlite*"):
        path.unlink()


def _no_document(paper, data):
    with sqlite3.connect(paper / "state.sqlite") as db:
        db.execute("DELETE FROM paper_state")


def _unjournaled(paper, data):
    _journal_fill(paper, "BTCUSDT", _first_btc_fill(paper)["id"], None)


def _undecided(paper, data):
    document = _document(paper)                         # a crash between a bar's matching and its decision
    document["symbols"]["BTCUSDT"]["cursor"] -= HOUR
    _save_document(paper, document)


MISSING = {"a gap in the bars": (_gap, "K 线快照"),
           "bars that end before the paper record": (_short, "K 线快照"),
           "a missing funding settlement": (_no_funding, "资金费"),
           "bars other than those paper saw": (_other_close, "与模拟盘看到的不同"),
           "no event log": (_no_log, "事件日志"),
           "the record's start missing from the event log": (_no_warmup, "事件日志"),
           "no state database": (_no_db, "state.sqlite"),
           "no saved paper state": (_no_document, "模拟盘状态"),
           "a venue fill missing from the order journal": (_unjournaled, "订单库"),
           "a bar matched but not decided yet": (_undecided, "还没有决策")}


@pytest.mark.parametrize("case", list(MISSING))
def test_missing_or_unusable_data_cannot_be_compared_and_is_never_equal(plain, tmp_path, capsys, case):
    paper, data = _copy(plain, tmp_path)
    damage, reason = MISSING[case]
    damage(paper, data)
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (2, "结论：无法比较"), out
    assert reason in out


def test_the_record_starts_at_its_own_first_warm_up_not_at_an_older_record_s(plain, tmp_path, capsys):
    paper, data = _copy(plain, tmp_path)
    events = _events(paper)
    older = [{**e, "last_open": e["last_open"] - 100 * HOUR} for e in events if e["kind"] == "warmup"]
    (paper / "target_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in [*older, *events]))
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (0, "结论：一致"), out


# -- intended divergences --------------------------------------------------------------------------------------
DIVERGED = [("kill_flatten", "flatten"), ("entries_refused", "entry_refused"), ("kill_cancel", "cancel_entries"),
            ("stop_crash", "stop_triggered")]


@pytest.mark.parametrize("run,cause", DIVERGED)
def test_an_intended_divergence_is_listed_apart_and_is_no_inconsistency(request, tmp_path, capsys, run, cause):
    paper, data = _copy(request.getfixturevalue(run), tmp_path)
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (0, "结论：一致"), out
    assert f"起与研究分叉（{cause}" in out                         # the fills differed from there on, for this cause
    listed = out.split("已解释的分叉", 1)[1]
    assert cause in listed


@pytest.mark.parametrize("run,cause", DIVERGED)
def test_without_its_divergence_events_the_same_record_is_inconsistent(request, tmp_path, capsys, run, cause):
    paper, data = _copy(request.getfixturevalue(run), tmp_path)
    _keep_events(paper, lambda e: e["kind"] != "paper_divergence")
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (1, "结论：不一致"), out


def test_an_intended_divergence_does_not_hide_an_earlier_difference(kill_flatten, tmp_path, capsys):
    paper, data = _copy(kill_flatten, tmp_path)
    document = _document(paper)
    symbol = next(s for s in SYMBOLS if _venue_fills(document, s))
    first = _venue_fills(document, symbol)[0]                     # traded before the kill switch
    first["price"] = str(D(first["price"]) + 1)
    _journal_fill(paper, symbol, first["id"], {"price": first["price"]})
    _save_document(paper, document)
    code, out = _compare(paper, data, capsys)
    assert (code, out.splitlines()[0]) == (1, "结论：不一致"), out
    assert f"{symbol}：前 0 笔一致，第 1 笔不同" in out
