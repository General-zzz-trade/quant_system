"""The prediction feed (data/downloads/perps_feed.py; owner 2026-11-18 "先做BTC和ETH，直接就在此对话推送"): a rolling
store seeded from a snapshot's recent rows, then extended from Binance's public endpoints (no key). Fake HTTP only."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from data.downloads import perps_feed as pf

H = 3_600_000
T0 = 1_762_992_000_000                       # 2025-11-13 00:00 UTC


def _write(path: Path, header: str, rows: list[str]) -> None:
    path.write_bytes(gzip.compress((header + "\n" + "".join(r + "\n" for r in rows)).encode()))


def _bar(t: int, close: float = 100.0) -> str:
    return f"{t},{close},{close + 1},{close - 1},{close},10,1000,5,4,400"


def _snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir(parents=True)
    for s in ("BTCUSDT", "XAUUSDT", "OLDUSDT"):
        _write(snap / f"{s}_1h.csv.gz", pf.BAR_HEADER, [_bar(T0 + k * H) for k in range(48)])
        _write(snap / f"{s}_funding.csv.gz", pf.FUNDING_HEADER, [f"{T0 + k * 8 * H},0.0001" for k in range(6)])
    return snap


def _rows(path: Path) -> list[str]:
    return gzip.decompress(path.read_bytes()).decode().splitlines()[1:]


def test_seeding_copies_the_recent_rows_leaves_tradfi_out_and_never_overwrites(tmp_path):
    feed = tmp_path / "feed"
    out = pf.seed(_snapshot(tmp_path), feed, start_ms=T0 + 24 * H, exclude={"XAUUSDT"})
    assert sorted(out["symbols"]) == ["BTCUSDT", "OLDUSDT"]
    assert _rows(feed / "BTCUSDT_1h.csv.gz")[0].startswith(str(T0 + 24 * H))
    assert len(_rows(feed / "BTCUSDT_1h.csv.gz")) == 24
    assert not (feed / "XAUUSDT_1h.csv.gz").exists()
    assert json.loads((feed / "state.json").read_text())["seeded_from"] == "snap"
    with pytest.raises(ValueError):
        pf.seed(_snapshot(tmp_path / "again"), feed, start_ms=T0, exclude=set())


def _info(*entries):
    return json.dumps({"symbols": [
        {"symbol": s, "contractType": kind, "quoteAsset": quote, "status": status, "onboardDate": onboard,
         "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.001"},
                     {"filterType": "MIN_NOTIONAL", "notional": "100"}]}
        for s, kind, quote, status, onboard in entries]}).encode()


class FakeBinance:
    """The public endpoints the feed uses, serving bars up to ``now`` (the one still open included, as Binance
    does) and funding every 8 hours; records every URL."""

    def __init__(self, now: int, listed: dict[str, int], fail: frozenset[str] = frozenset()):
        self.now, self.listed, self.fail, self.urls = now, listed, fail, []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        assert url.startswith("https://fapi.binance.com/fapi/v1/")
        assert "signature" not in url and "timestamp" not in url
        if url.endswith("exchangeInfo"):
            return _info(*[(s, "PERPETUAL", "USDT", "TRADING", t) for s, t in self.listed.items()],
                         ("XAUUSDT", "TRADIFI_PERPETUAL", "USDT", "TRADING", T0),
                         ("BTCUSDC", "PERPETUAL", "USDC", "TRADING", T0),
                         ("GONEUSDT", "PERPETUAL", "USDT", "SETTLING", T0))
        q = dict(p.split("=") for p in url.split("?")[1].split("&"))
        symbol, start, end = q["symbol"], int(q["startTime"]), int(q["endTime"])
        if symbol in self.fail:
            raise OSError("network")
        lo, hi = max(start, self.listed[symbol]), min(end, self.now)
        if "klines" in url:
            times = list(range(-(-lo // H) * H, hi + 1, H))
            return json.dumps([[t, "1", "2", "0.5", "1.5", "10", t + H - 1, "1000", 5, "4", "400", "0"]
                               for t in times[:1500]]).encode()
        times = list(range(T0 + -(-(lo - T0) // (8 * H)) * 8 * H, hi + 1, 8 * H))
        return json.dumps([{"symbol": symbol, "fundingTime": t, "fundingRate": "0.0002"} for t in times]).encode()


def _seeded(tmp_path):
    feed = tmp_path / "feed"
    pf.seed(_snapshot(tmp_path), feed, start_ms=T0, exclude={"XAUUSDT"})
    return feed


def test_an_update_appends_closed_bars_and_funding_after_the_last_and_adds_new_listings(tmp_path):
    feed = _seeded(tmp_path)
    now = T0 + 60 * H + 30 * 60_000                                       # half way through hour 60
    fake = FakeBinance(now, {"BTCUSDT": T0, "NEWUSDT": T0 + 50 * H})
    out = pf.update(feed, now_ms=now, fetch=fake, wait=lambda: None)
    bars = _rows(feed / "BTCUSDT_1h.csv.gz")
    times = [int(r.split(",")[0]) for r in bars]
    assert times == [T0 + k * H for k in range(60)]                       # hour 60 has not closed
    funding = [int(r.split(",")[0]) for r in _rows(feed / "BTCUSDT_funding.csv.gz")]
    assert funding == sorted(set(funding)) and funding[-1] == T0 + 56 * H
    assert [int(r.split(",")[0]) for r in _rows(feed / "NEWUSDT_1h.csv.gz")][0] == T0 + 50 * H
    assert out["new"] == ["NEWUSDT"] and out["failed"] == []
    assert "XAUUSDT" in out["tradfi"] and not (feed / "XAUUSDT_1h.csv.gz").exists()
    assert out["rules"]["BTCUSDT"] == {"step": "0.001", "min_notional": "100"}
    assert len(bars[-1].split(",")) == len(pf.BAR_HEADER.split(","))


def test_a_second_update_at_the_same_time_changes_nothing(tmp_path):
    feed = _seeded(tmp_path)
    now = T0 + 60 * H
    fake = FakeBinance(now, {"BTCUSDT": T0})
    pf.update(feed, now_ms=now, fetch=fake, wait=lambda: None)
    before = (feed / "BTCUSDT_1h.csv.gz").read_bytes(), (feed / "BTCUSDT_funding.csv.gz").read_bytes()
    pf.update(feed, now_ms=now, fetch=fake, wait=lambda: None)
    assert ((feed / "BTCUSDT_1h.csv.gz").read_bytes(), (feed / "BTCUSDT_funding.csv.gz").read_bytes()) == before


def test_a_failing_contract_is_recorded_and_the_others_still_update(tmp_path):
    feed = _seeded(tmp_path)
    now = T0 + 60 * H
    fake = FakeBinance(now, {"BTCUSDT": T0, "OLDUSDT": T0}, fail=frozenset({"OLDUSDT"}))
    out = pf.update(feed, now_ms=now, fetch=fake, wait=lambda: None)
    assert out["failed"] == ["OLDUSDT"] and out["updated"] == 1
    assert len(_rows(feed / "OLDUSDT_1h.csv.gz")) == 48                   # untouched
    state = json.loads((feed / "state.json").read_text())
    assert state["last_update"]["failed"] == ["OLDUSDT"] and state["last_update"]["now_ms"] == now


def test_every_funding_request_waits_for_its_slot(tmp_path):
    feed = _seeded(tmp_path)
    calls = []
    pf.update(feed, now_ms=T0 + 60 * H, fetch=FakeBinance(T0 + 60 * H, {"BTCUSDT": T0, "OLDUSDT": T0}),
              wait=lambda: calls.append(1))
    assert len(calls) == 2
