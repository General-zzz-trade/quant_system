"""Two event sources for the alpha search (owner 2026-11-18: "执行" on the plan that downloads them beside the
overnight study): Nasdaq's earnings calendar (data/downloads/us_earnings.py) and DefiLlama's protocol fees and revenue
(data/downloads/defillama_fees.py). Parsers on canned payloads, the forward cut, the snapshot rules. No network."""
from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from data.downloads import defillama_fees as dl
from data.downloads import us_earnings as ue


class Fetch:
    def __init__(self, pages):
        self.pages, self.seen = pages, []

    def __call__(self, url):
        self.seen.append(url)
        for key, body in self.pages.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                return body
        raise AssertionError(f"unexpected URL {url}")


def _calendar(rows):
    return json.dumps({"data": {"asOf": "x", "headers": {}, "rows": rows}}).encode()


ROW = {"symbol": "MSFT", "name": "Microsoft", "time": "time-not-supplied", "eps": "$4.74", "epsForecast": "$4.21",
       "surprise": "12.59", "fiscalQuarterEnding": "Jun/2026", "marketCap": "$4,000,000,000", "noOfEsts": "12"}


# -- the earnings calendar -------------------------------------------------------------------------------------------
def test_a_calendar_day_becomes_rows_with_its_date():
    rows = ue.parse(_calendar([ROW]), date(2026, 7, 29))
    assert rows == [{"date": "2026-07-29", **{k: ROW[k] for k in ue.FIELDS}}]
    assert ue.parse(json.dumps({"data": None}).encode(), date(2026, 8, 1)) == []           # a weekend: no rows


def test_the_earnings_snapshot_asks_each_day_once_and_records_it(tmp_path):
    fetch = Fetch({"date=2026-07-29": _calendar([ROW]), "date=2026-07-30": _calendar([]),
                   "date=2026-07-31": _calendar([{**ROW, "symbol": "AAPL"}])})
    m = ue.snapshot(tmp_path / "e", start=date(2026, 7, 29), end=date(2026, 8, 1), fetch=fetch, pause=0,
                    now=lambda: "2026-11-18T00:00:00Z")
    assert len(fetch.seen) == 3 and m["days"] == 3 and m["days_with_rows"] == 2 and m["rows"] == 2
    with (tmp_path / "e" / "earnings.csv").open() as handle:
        assert [r["symbol"] for r in csv.DictReader(handle)] == ["MSFT", "AAPL"]
    assert json.loads((tmp_path / "e" / "manifest.json").read_text())["end_exclusive"] == "2026-08-01"


def test_the_earnings_snapshot_respects_the_forward_guard_and_is_written_once(tmp_path):
    with pytest.raises(ValueError, match="2026-11-13"):
        ue.snapshot(tmp_path / "e", start=date(2026, 11, 1), end=date(2026, 11, 20), fetch=Fetch({}), pause=0)
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "x").write_text("old")
    with pytest.raises(FileExistsError):
        ue.snapshot(tmp_path / "f", start=date(2026, 7, 29), end=date(2026, 7, 30), fetch=Fetch({}), pause=0)


# -- DefiLlama ---------------------------------------------------------------------------------------------------------
PROTOCOLS = [{"name": "Uniswap V3", "slug": "uniswap-v3", "symbol": "UNI", "gecko_id": "uniswap", "category": "Dexs",
              "mcap": 5e9},
             {"name": "Aave V3", "slug": "aave-v3", "symbol": "AAVE", "gecko_id": "aave", "category": "Lending"},
             {"name": "No token", "slug": "none", "symbol": "-", "gecko_id": None, "category": "Dexs"},
             {"name": "Pepe thing", "slug": "pepe", "symbol": "PEPE", "gecko_id": None, "category": "Meme"}]


def _chart(points):
    return json.dumps({"name": "x", "totalDataChart": points}).encode()


def test_protocols_are_matched_to_perpetuals_by_token_symbol():
    bases = {"UNI", "AAVE", "PEPE"}
    assert [p["slug"] for p in dl.select(PROTOCOLS, bases)] == ["aave-v3", "pepe", "uniswap-v3"]
    assert dl.bases(["UNIUSDT", "1000PEPEUSDT", "BTCUSDT"]) == {"UNI", "PEPE", "BTC"}


def test_a_chart_is_cut_before_the_end():
    end = date(2026, 11, 13)
    rows, dropped = dl.parse_chart(_chart([[1794441600, 10], [1794528000, 20], [1794960000, 30]]), end)
    assert rows == [{"date": "2026-11-12", "value": "10"}] and dropped == 2       # 11-13 and 11-17 are forward data


def test_the_fees_snapshot_writes_each_series_and_records_what_failed(tmp_path):
    import urllib.error
    pages = {"/protocols": json.dumps(PROTOCOLS).encode(),
             "uniswap-v3?dataType=dailyFees": _chart([[1794441600, 10]]),
             "uniswap-v3?dataType=dailyRevenue": _chart([[1794441600, 4]]),
             "aave-v3?dataType=dailyFees": urllib.error.HTTPError("u", 404, "no adapter", {}, None),
             "aave-v3?dataType=dailyRevenue": urllib.error.HTTPError("u", 404, "no adapter", {}, None)}
    m = dl.snapshot(tmp_path / "d", bases={"UNI", "AAVE"}, end=date(2026, 11, 13), fetch=Fetch(pages), pause=0,
                    now=lambda: "2026-11-18T00:00:00Z")
    names = sorted(p.name for p in (tmp_path / "d").iterdir())
    assert names == ["manifest.json", "protocols.json", "uniswap-v3_dailyFees.csv", "uniswap-v3_dailyRevenue.csv"]
    assert m["missing"] == {"aave-v3_dailyFees": "HTTPError", "aave-v3_dailyRevenue": "HTTPError"}
    kept = json.loads((tmp_path / "d" / "protocols.json").read_text())
    assert [p["slug"] for p in kept] == ["aave-v3", "uniswap-v3"] and "mcap" not in kept[0]   # today's value: dropped
