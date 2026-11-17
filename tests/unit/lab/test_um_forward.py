"""lab/um_forward.py: the weekly forward looks of a frozen lab champion (tech lead 2026-11-16 evening; criterion
2026-11-17). Synthetic market (the judge tests'), a frozen file and a look registry in a temporary directory.

- A look that is not registered for this frozen file, registered on or after its end, or not fully passed is refused,
  for the download and for the look; a look is evaluated once.
- The look replays the frozen configuration from the frozen start to the look's end; the criterion is the lowest
  hourly portfolio equity against 92.5 % of the capital.
- The download asks the downloader for the frozen symbols and BTCUSDT only, up to the look's end.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from lab import forward
from lab import um_forward as fw
from lab import um_lab as lab
from tests.unit.lab.test_um_judge_runs import market  # noqa: F401  (a fixture)

DAY = lab.DAY
START = lab.START + 100 * DAY            # 2023-02-21, inside the synthetic market
END = lab.START + 130 * DAY


@pytest.fixture(autouse=True)
def _synthetic_forward_start(monkeypatch):
    """The synthetic market lies in 2023: its forward window starts at the market's start (test setup only)."""
    monkeypatch.setattr(forward, "FORWARD_START", lab.START)


def _day(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat()


def _setup(tmp_path, *, symbols=("UPUSDT", "CRASHUSDT"), registered=None, end=END, stop=None, params=None):
    frozen = {"kind": "lab_champion_forward", "passed": False, "registered": True, "start": _day(START),
              "spec": {"strategy": {"name": "A1_tsmom_long", "params": params or {"n": 20}}, "symbols": list(symbols)},
              "stop": stop or {"stop_pct": "0.08", "pause_hours": None, "reenter": "signal_reset"},
              "capital_usdt": "300", "slots": 10}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    looks = tmp_path / "looks.jsonl"
    looks.write_text(json.dumps({"study": "s", "spec": str(path), "spec_sha256": forward.spec_digest(path),
                                 "end": _day(end), "registered": registered or _day(START - DAY)}) + "\n")
    return path, looks


def _look(path, looks, market, out, *, end=END, now=None):  # noqa: F811
    return fw.look(path, study="s", end=_day(end), data=market, out=out, looks=looks,
                   now_ms=now if now is not None else end + 2 * DAY)


def test_a_look_replays_the_frozen_configuration_and_judges_the_lowest_hourly_equity(market, tmp_path):  # noqa: F811
    path, looks = _setup(tmp_path)
    result = _look(path, looks, market, tmp_path / "out")
    assert result["holds"] == (result["lowest_equity_share"] > 0.925)
    assert result["lowest_equity_share"] <= min(1.0, 1.0 + result["total_return"]) + 1e-12   # the low is a low
    on_disk = json.loads((tmp_path / "out" / "result.json").read_text())
    assert on_disk == json.loads(json.dumps(result, sort_keys=True))
    assert "前向查看" in (tmp_path / "out" / "README.md").read_text()
    assert result["round_trips"] >= 0 and result["btc_hold_return"] is not None


def test_a_crash_below_the_line_does_not_hold(tmp_path):
    """A rising contract (the momentum idea is long) whose low drops 30 % in one hour inside the window, below its
    stop 15 % under the entry: with every slot in it, the equity falls below 92.5 % although the stop caps the
    loss."""
    import math
    from tests.unit.lab.test_um_judge import finish, write_contract
    data = tmp_path / "dip"
    data.mkdir()
    write_contract(data, "BTCUSDT", price=lambda d: 100.0)
    write_contract(data, "DIPUSDT", price=lambda d: 100 * math.exp(0.01 * d), lows={110 * 24 + 5: 0.70})
    finish(data, ["BTCUSDT", "DIPUSDT"])
    path, looks = _setup(tmp_path, symbols=("DIPUSDT",), stop={"stop_pct": "0.15", "pause_hours": 4,
                                                                "reenter": "pause"})
    frozen = json.loads(path.read_text())
    frozen["slots"] = 1
    path.write_text(json.dumps(frozen, indent=1, sort_keys=True) + "\n")
    looks.write_text(json.dumps({"study": "s", "spec": str(path), "spec_sha256": forward.spec_digest(path),
                                 "end": _day(END), "registered": _day(START - DAY)}) + "\n")
    result = _look(path, looks, data, tmp_path / "out")
    assert result["exchange_stops"] >= 1
    assert result["lowest_equity_share"] < 0.925 and result["holds"] is False


@pytest.mark.parametrize("case", ["unregistered", "edited", "registered_late", "not_passed"])
def test_a_look_that_is_not_registered_or_not_due_is_refused(market, tmp_path, case):  # noqa: F811
    path, looks = _setup(tmp_path, registered=_day(END) if case == "registered_late" else None)
    if case == "unregistered":
        looks.write_text("")
    if case == "edited":
        path.write_text(path.read_text().replace('"300"', '"3000"'))
    now = END + DAY // 2 if case == "not_passed" else None
    with pytest.raises(PermissionError):
        _look(path, looks, market, tmp_path / "out", now=now)
    with pytest.raises(PermissionError):
        fw.download(path, study="s", end=_day(END), out=tmp_path / "data", looks=looks,
                    now_ms=now if now is not None else END + 2 * DAY, fetch=lambda url: b"")
    assert not (tmp_path / "out").exists() and not (tmp_path / "data").exists()


def test_a_look_is_evaluated_once(market, tmp_path):  # noqa: F811
    path, looks = _setup(tmp_path)
    _look(path, looks, market, tmp_path / "out")
    with pytest.raises(FileExistsError):
        _look(path, looks, market, tmp_path / "out")


def test_the_download_asks_for_the_frozen_symbols_and_btc_up_to_the_end(tmp_path, monkeypatch):
    from data.downloads import binance_um_all as um
    path, looks = _setup(tmp_path)
    asked = {}

    def fake(out, **kwargs):
        asked.update(kwargs, out=out)
        return {"symbols_with_bars": 3, "failed": []}
    monkeypatch.setattr(um, "download", fake)
    fw.download(path, study="s", end=_day(END), out=tmp_path / "data", looks=looks, now_ms=END + 2 * DAY)
    assert asked["symbols"] == ["BTCUSDT", "CRASHUSDT", "UPUSDT"] and asked["forward_look"] is True
    assert asked["end"].isoformat() == _day(END) and asked["start"] == date(2026, 9, 1)


def test_the_downloader_takes_a_forward_look_only_for_named_symbols_and_a_passed_end(tmp_path):
    from data.downloads import binance_um_all as um
    with pytest.raises(ValueError, match="forward look"):
        um.download(tmp_path / "a", start=date(2026, 9, 1), end=date(2026, 11, 25), symbols=[], forward_look=True,
                    today=date(2026, 11, 26), fetch=lambda url: b"")
    with pytest.raises(ValueError, match="forward look"):
        um.download(tmp_path / "b", start=date(2026, 9, 1), end=date(2026, 11, 25), symbols=["ETHUSDT"],
                    forward_look=True, today=date(2026, 11, 24), fetch=lambda url: b"")
    with pytest.raises(ValueError, match="forward"):           # without the flag the guard is unchanged
        um.download(tmp_path / "c", start=date(2026, 9, 1), end=date(2026, 11, 25), symbols=["ETHUSDT"],
                    today=date(2026, 11, 26), fetch=lambda url: b"")
