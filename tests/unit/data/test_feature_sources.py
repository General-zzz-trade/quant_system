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
