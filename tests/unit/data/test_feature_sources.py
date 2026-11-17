"""The feature-pool study's extra sources (data/downloads/feature_sources.py).

Owner 2026-11-17: "来使用加入特征池的方式去做，参考更多的数据去做", "执行". Each parser on canned payloads, the cut at
the snapshot's end, the checksum, the pagination, and the snapshot rules (immutable directory, manifest). No network."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import date

import pytest

from data.downloads import feature_sources as fs

H = 3_600_000
END = fs.ms(date(2024, 1, 2))                       # the cut used below: 2024-01-02 00:00 UTC


def _zip(name, text):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr(name, text)
    return buffer.getvalue()


class Fetch:
    """URL -> bytes; a missing URL is a 404, as data.binance.vision answers."""

    def __init__(self, pages=None):
        self.pages, self.seen = dict(pages or {}), []

    def add_zip(self, url, name, text, *, checksum=None):
        raw = _zip(name, text)
        self.pages[url] = raw
        self.pages[url + ".CHECKSUM"] = f"{checksum or hashlib.sha256(raw).hexdigest()}  {name}.zip".encode()

    def __call__(self, url):
        self.seen.append(url)
        if url not in self.pages:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return self.pages[url]


def test_spot_klines_take_microseconds_and_stop_at_the_end():
    fetch = Fetch()
    base = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-"
    ms_row = f"{END - 2 * H},1,2,0.5,1.5,10,{END - H - 1},100,7,4,40,0"
    us_row = f"{(END - H) * 1000},1.5,2,1,1.8,11,{END * 1000 - 1},110,8,5,50,0"           # 2025+: microseconds
    late = f"{END * 1000},2,3,1,2,1,{(END + H) * 1000 - 1},5,1,1,1,0"                     # at the end: dropped
    fetch.add_zip(base + "2024-01.zip", "x.csv", "\n".join([ms_row, us_row, late]))
    rows = fs.spot_klines("BTCUSDT", ["2024-01"], END, fetch=fetch)
    assert [r[0] for r in rows] == [END - 2 * H, END - H]
    assert rows[1][1:5] == ("1.5", "2", "1", "1.8") and rows[1][-1] == "50"            # quote taker-buy volume


def test_a_checksum_mismatch_is_refused():
    fetch = Fetch()
    url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    fetch.add_zip(url, "x.csv", f"{END - H},1,1,1,1,1,{END - 1},1,1,1,1,0", checksum="0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        fs.spot_klines("BTCUSDT", ["2024-01"], END, fetch=fetch)


def test_a_missing_month_is_skipped():
    assert fs.spot_klines("BTCUSDT", ["2024-01"], END, fetch=Fetch()) == []


def test_book_depth_is_averaged_per_hour_and_level():
    fetch = Fetch()
    text = "\n".join(["timestamp,percentage,depth,notional",
                      "2024-01-01 00:00:10,-1,1,100", "2024-01-01 00:30:10,-1,1,300",
                      "2024-01-01 00:00:10,1,1,50", "2024-01-01 01:00:10,-1,1,700"])
    fetch.add_zip(fs.depth_url("BTCUSDT", date(2024, 1, 1)), "d.csv", text)
    rows = fs.book_depth("BTCUSDT", [date(2024, 1, 1)], END, fetch=fetch)
    start = fs.ms(date(2024, 1, 1))
    assert rows == [(start, -1, 200.0, 300.0), (start, 1, 50.0, 50.0), (start + H, -1, 700.0, 700.0)]


def test_bvol_keeps_each_hour_s_last_value():
    fetch = Fetch()
    start = fs.ms(date(2024, 1, 1))
    text = "\n".join(["calc_time,symbol,base_asset,quote_asset,index_value",
                      f"{start},BTCBVOLUSDT,BTCBVOL,USDT,60", f"{start + H - 1000},BTCBVOLUSDT,BTCBVOL,USDT,61",
                      f"{start + H},BTCBVOLUSDT,BTCBVOL,USDT,62"])
    fetch.add_zip(fs.bvol_url("BTCBVOLUSDT", date(2024, 1, 1)), "b.csv", text)
    assert fs.bvol("BTCBVOLUSDT", [date(2024, 1, 1)], END, fetch=fetch) == [(start, 61.0), (start + H, 62.0)]


def test_the_first_day_of_a_daily_archive_is_found_by_bisection():
    fetch = Fetch({fs.depth_url("BTCUSDT", d) + ".CHECKSUM": b"x" for d in fs.days(date(2023, 3, 7), date(2023, 4, 1))})
    assert fs.first_day(lambda d: fs.depth_url("BTCUSDT", d), date(2022, 1, 1), date(2023, 3, 31),
                        fetch=fetch) == date(2023, 3, 7)


def test_dvol_is_read_in_windows_sorted_and_cut():
    start = fs.ms(date(2024, 1, 1)) - 40 * 24 * H
    pages = {}

    class Deribit(Fetch):
        def __call__(self, url):
            self.seen.append(url)
            q = dict(p.split("=") for p in url.split("?")[1].split("&"))
            a, b = int(q["start_timestamp"]), int(q["end_timestamp"])
            data = [[t, 1, 2, 0.5, 50 + (t - start) / H] for t in range(a - a % H, b, H) if start <= t]
            return json.dumps({"result": {"data": list(reversed(data)), "continuation": None}}).encode()
    fetch = Deribit(pages)
    rows = fs.dvol("BTC", start, END, fetch=fetch)
    assert len(fetch.seen) > 1                                                      # more than one window
    assert [r[0] for r in rows] == list(range(start, END, H))                       # each hour once, in order
    assert rows[-1][0] == END - H


def test_fred_drops_missing_values_and_rows_at_or_after_the_end():
    text = "observation_date,VIXCLS\n2023-12-29,12.4\n2024-01-01,.\n2024-01-02,13.1\n"
    fetch = Fetch({fs.fred_url("VIXCLS"): text.encode()})
    assert fs.fred("VIXCLS", END, fetch=fetch) == [("2023-12-29", 12.4)]


def test_stablecoins_take_the_usd_total_before_the_end():
    payload = [{"date": str(END // 1000 - 86400), "totalCirculatingUSD": {"peggedUSD": 5.0, "peggedEUR": 1.0}},
               {"date": str(END // 1000), "totalCirculatingUSD": {"peggedUSD": 6.0}}]
    fetch = Fetch({fs.STABLECOINS: json.dumps(payload).encode()})
    assert fs.stablecoins(END, fetch=fetch) == [("2024-01-01", 5.0)]


def test_coinmetrics_follows_the_next_page_and_cuts():
    first = fs.coinmetrics_url("btc", ("AdrActCnt", "TxCnt"), date(2023, 12, 30), date(2024, 1, 2))
    second = "https://community-api.coinmetrics.io/v4/next?page=2"
    pages = {first: json.dumps({"data": [{"asset": "btc", "time": "2023-12-30T00:00:00.000000000Z",
                                          "AdrActCnt": "10", "TxCnt": "5"}], "next_page_url": second}).encode(),
             second: json.dumps({"data": [{"asset": "btc", "time": "2024-01-01T00:00:00.000000000Z",
                                           "AdrActCnt": "11"},
                                          {"asset": "btc", "time": "2024-01-02T00:00:00.000000000Z",
                                           "AdrActCnt": "12", "TxCnt": "6"}]}).encode()}
    rows = fs.coinmetrics("btc", ("AdrActCnt", "TxCnt"), date(2023, 12, 30), END, fetch=Fetch(pages))
    assert rows == [("2023-12-30", "10", "5"), ("2024-01-01", "11", "")]


def test_a_snapshot_directory_must_be_new_and_its_manifest_lists_every_file(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "x").write_text("x")
    with pytest.raises(ValueError, match="immutable"):
        fs.write_snapshot(tmp_path / "old", {}, END, sources={})
    files = {"a.csv.gz": (("t", "v"), [(1, 2), (3, 4)])}
    manifest = fs.write_snapshot(tmp_path / "new", files, END, sources={"a.csv.gz": {"known": "at t"}})
    raw = (tmp_path / "new" / "a.csv.gz").read_bytes()
    assert manifest["files"]["a.csv.gz"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["files"]["a.csv.gz"]["rows"] == 2 and manifest["end_exclusive"] == "2024-01-02T00:00:00+00:00"
    assert gzip.decompress(raw).decode().splitlines() == ["t,v", "1,2", "3,4"]
    assert json.loads((tmp_path / "new" / "manifest.json").read_text()) == manifest


def test_a_cut_off_read_is_retried_and_a_404_is_not():
    import http.client
    calls = []

    def flaky(url):
        calls.append(url)
        if len(calls) < 3:
            raise http.client.IncompleteRead(b"x", 10)
        return b"ok"
    assert fs.retrying(flaky, attempts=4, pause=0)("u") == b"ok" and len(calls) == 3
    with pytest.raises(urllib.error.HTTPError):
        fs.retrying(Fetch(), attempts=4, pause=0)("missing")
    gone = []

    def always_cut(url):
        gone.append(url)
        raise http.client.IncompleteRead(b"", 5)
    with pytest.raises(http.client.IncompleteRead):
        fs.retrying(always_cut, attempts=2, pause=0)("u")
    assert len(gone) == 2


def test_a_finished_source_is_kept_for_the_next_attempt(tmp_path):
    made = []

    def rows():
        made.append(1)
        return [(1, 2.5), (3, 4.25)]
    first = fs.staged(tmp_path, "a.csv.gz", ("t", "v"), rows)
    second = fs.staged(tmp_path, "a.csv.gz", ("t", "v"), lambda: pytest.fail("made again"))
    assert len(made) == 1 and [tuple(map(str, r)) for r in first] == [tuple(r) for r in second]


def test_a_finished_depth_day_is_kept_and_not_fetched_again(tmp_path):
    fetch = Fetch()
    text = "timestamp,percentage,depth,notional\n2024-01-01 00:00:10,-1,1,100\n2024-01-01 00:30:10,-1,1,300\n"
    fetch.add_zip(fs.depth_url("BTCUSDT", date(2024, 1, 1)), "d.csv", text)
    first = fs.book_depth("BTCUSDT", [date(2024, 1, 1)], END, fetch=fetch, cache=tmp_path)
    again = fs.book_depth("BTCUSDT", [date(2024, 1, 1)], END, fetch=Fetch(), cache=tmp_path)   # would be a 404
    assert first == again == [(fs.ms(date(2024, 1, 1)), -1, 200.0, 300.0)]


def test_the_download_reads_long_files_with_a_long_timeout(monkeypatch):
    seen = {}
    monkeypatch.setattr(fs, "http_get", lambda url, **kw: seen.update(kw) or b"x")
    fs.slow_fetch("u")
    assert seen["timeout"] >= 120


def test_only_the_chosen_sources_are_downloaded_and_the_rest_are_named(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "FRED_LAG_DAYS", {"VIXCLS": 2})
    monkeypatch.setattr(fs, "FRED_V2_LAG_DAYS", {})
    fetch = Fetch({fs.fred_url("VIXCLS"): b"observation_date,VIXCLS\n2023-12-29,12.4\n"})
    manifest = fs.download(tmp_path / "snap", date(2024, 1, 2), fetch=fetch, sources=("fred",))
    assert list(manifest["files"]) == ["fred_VIXCLS.csv.gz"]
    assert set(manifest["not_downloaded"]) >= {"spot", "premium", "depth", "dvol", "stable", "coinmetrics", "BVOL"}
    assert all("fred" in url for url in fetch.seen) and not (tmp_path / "snap.partial").exists()
    with pytest.raises(ValueError, match="unknown source"):
        fs.download(tmp_path / "other", date(2024, 1, 2), fetch=fetch, sources=("fred", "tea"))


# -- the second version's sources (owner 2026-11-17: "还有哪些特征是需要用到", then "执行") -------------------------
def test_quarterly_expiries_are_the_last_friday_of_each_quarter_at_eight():
    assert fs.quarterly_expiry(2024, 3) == fs.ms(date(2024, 3, 29)) + 8 * H
    assert fs.quarterly_expiry(2025, 12) == fs.ms(date(2025, 12, 26)) + 8 * H
    assert fs.quarterly_symbol("BTCUSDT", 2024, 3) == "BTCUSDT_240329"


def test_quarterly_klines_carry_their_contract_and_expiry():
    fetch = Fetch()
    symbol = "BTCUSDT_240329"
    url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-2024-01.zip"
    rows = [f"{END - 2 * H},1,2,0.5,1.5,1,{END - H - 1},1,1,1,1,0", f"{END},1,1,1,9,1,{END + H},1,1,1,1,0"]
    fetch.add_zip(url, "q.csv", "\n".join(rows))
    rows = fs.quarterly_klines("BTCUSDT", [(2024, 3)], END, fetch=fetch, months_back=0, first_month="2024-01")
    assert rows == [(END - 2 * H, symbol, fs.quarterly_expiry(2024, 3), "1.5")]


def test_coinbase_candles_are_paged_backwards_sorted_and_cut():
    start = END - 700 * H

    class Coinbase(Fetch):
        def __call__(self, url):
            self.seen.append(url)
            q = dict(p.split("=") for p in url.split("?")[1].split("&"))
            a, b = fs.iso_ms(q["start"]), fs.iso_ms(q["end"])
            data = [[t // 1000, 1, 2, 1, 100 + (t - start) / H, 5] for t in range(b - H, a - 1, -H)]  # newest first
            return json.dumps(data).encode()
    fetch = Coinbase()
    rows = fs.coinbase("BTC-USD", start, END, fetch=fetch)
    assert len(fetch.seen) == 3                                                   # 300 candles a request
    assert [r[0] for r in rows] == list(range(start, END, H)) and rows[0][1] == 100.0


def test_upbit_candles_walk_back_from_the_end():
    start = END - 450 * H

    class Upbit(Fetch):
        def __call__(self, url):
            self.seen.append(url)
            q = dict(p.split("=") for p in url.split("?")[1].split("&"))
            to = fs.iso_ms(q["to"])
            first = max(start, to - int(q["count"]) * H)
            return json.dumps([{"candle_date_time_utc": fs.ms_iso(t)[:19], "trade_price": float(t // H % 1000)}
                               for t in range(to - H, first - 1, -H)]).encode()
    rows = fs.upbit("KRW-BTC", start, END, fetch=Upbit())
    assert [r[0] for r in rows] == list(range(start, END, H))


def test_fear_and_greed_keeps_days_before_the_end():
    payload = {"data": [{"value": "40", "timestamp": str(END // 1000 - 86400)},
                        {"value": "55", "timestamp": str(END // 1000)}]}
    assert fs.fear_greed(END, fetch=Fetch({fs.FEAR_GREED: json.dumps(payload).encode()})) == [("2024-01-01", 40.0)]


FOMC_PAGE = """<h4><a id="1"></a>2023 FOMC Meetings</h4>
<div class="fomc-meeting__month col-xs-5"><strong>Jan/Feb</strong></div><div class="fomc-meeting__date">31-1</div>
<div class="fomc-meeting__month col-xs-5"><strong>March</strong></div><div class="fomc-meeting__date">21-22*</div>
<div class="fomc-meeting__month col-xs-5"><strong>December</strong></div><div class="fomc-meeting__date">12-13*</div>
<h4>2024 FOMC Meetings</h4>
<div class="fomc-meeting__month col-xs-5"><strong>Jan</strong></div><div class="fomc-meeting__date">30-31</div>"""


def test_fomc_statement_days_are_the_last_day_of_each_meeting():
    assert fs.fomc_days(FOMC_PAGE) == ["2023-02-01", "2023-03-22", "2023-12-13", "2024-01-31"]
