"""Archive downloader against a fake HTTP source: universe ranking, checksums, cut-off, immutability."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import date

import pytest

from data.downloads import binance_archive as ba

BAR = 300_000
OCT = 1_664_582_400_000          # 2022-10-01 00:00 UTC


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", text)
    return buffer.getvalue()


class Source:
    """Serves the listing, daily klines for ranking, 5m klines, funding, checksums and exchangeInfo."""

    def __init__(self, daily: dict, *, bars_per_month: int = 4, corrupt: str = ""):
        self.daily, self.bars_per_month, self.corrupt, self.requests = daily, bars_per_month, corrupt, []

    def _body(self, url: str) -> bytes:
        name = url.rsplit("/", 1)[1]
        symbol, label, month = name[:-4].split("-", 2)
        if label == "1d":
            volumes = self.daily[symbol].get(month)
            if volumes is None:
                raise urllib.error.HTTPError(url, 404, "missing", None, None)
            header = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tbv,tbqv,ignore\n"
            return _zip(header + "".join(f"{k},1,1,1,1,1,0,{v},1,0,0,0\n" for k, v in enumerate(volumes)))
        year, mon = (int(x) for x in month.split("-"))
        start = OCT + ((year - 2022) * 12 + mon - 10) * 31 * 86_400_000
        if label == "5m":
            if symbol == "GONEUSDT" and month >= "2022-12":
                raise urllib.error.HTTPError(url, 404, "delisted", None, None)
            rows = "".join(f"{start + k * BAR},10,11,9,10.5,7,0,{100 + k},3,2,{40 + k},0\n"
                           for k in range(self.bars_per_month))
            return _zip(rows)                                   # an older archive: no header line
        return _zip("calc_time,funding_interval_hours,last_funding_rate\n" + f"{start},8,0.0001\n")

    def __call__(self, url: str) -> bytes:
        self.requests.append(url)
        if "s3-ap-northeast-1" in url:
            names = "".join(f"<Prefix>data/futures/um/monthly/klines/{s}/</Prefix>" for s in sorted(self.daily))
            return f"<IsTruncated>false</IsTruncated>{names}".encode()
        if url.endswith("exchangeInfo"):
            filters = [{"filterType": "LOT_SIZE", "stepSize": "0.1"}, {"filterType": "MIN_NOTIONAL", "notional": "5"}]
            return json.dumps({"symbols": [{"symbol": "AAAUSDT", "filters": filters}]}).encode()
        if url.endswith(".CHECKSUM"):
            body = self._body(url[:-9])
            digest = "0" * 64 if self.corrupt and self.corrupt in url else hashlib.sha256(body).hexdigest()
            return f"{digest}  file.zip\n".encode()
        return self._body(url)


FULL = {"2022-10": [5, 5], "2022-11": [5], "2022-12": [5]}
DAILY = {"AAAUSDT": {m: [v * 10 for v in vs] for m, vs in FULL.items()}, "BBBUSDT": FULL,
         "GONEUSDT": {m: [v * 4 for v in vs] for m, vs in FULL.items()},       # delisted later, still ranked
         "NEWUSDT": {"2022-12": [10_000]},                                      # listed too late: not a candidate
         "USDCUSDT": {m: [9_999] for m in FULL}, "BTCUSDT_230331": {m: [9_999] for m in FULL},
         "BTCDOMUSDT": {m: [9_999] for m in FULL}, "ETHBUSD": {m: [9_999] for m in FULL}}


def test_the_universe_is_ranked_on_2022_q4_volume_and_keeps_delisted_symbols():
    universe = ba.rank_universe(ba.list_symbols(fetch=Source(DAILY)), 3, fetch=Source(DAILY), workers=2)
    assert [(u["symbol"], u["rank"], u["ranking_quote_volume"]) for u in universe] == [
        ("AAAUSDT", 1, 200.0), ("GONEUSDT", 2, 80.0), ("BBBUSDT", 3, 20.0)]
    assert not any(ba.eligible(s) for s in ("USDCUSDT", "BTCUSDT_230331", "BTCDOMUSDT", "ETHBUSD"))


def test_a_snapshot_holds_bars_before_the_end_only_and_records_what_is_missing(tmp_path):
    out = tmp_path / "snap"
    end = date(2022, 12, 1)                                     # exclusive: October and November archives only
    manifest = ba.download(out, end=end, top=3, fetch=Source(DAILY), workers=2)
    assert [u["symbol"] for u in manifest["universe"]] == ["AAAUSDT", "GONEUSDT", "BBBUSDT"]
    with gzip.open(out / "AAAUSDT_5m.csv.gz", "rt") as stream:
        lines = stream.read().splitlines()
    assert lines[0] == ",".join(ba.KLINE_COLUMNS) and len(lines) == 1 + 8
    assert lines[1] == f"{OCT},10,11,9,10.5,100,40"
    assert manifest["files"]["AAAUSDT_5m.csv.gz"] == {"rows": 8, "sha256": hashlib.sha256(
        (out / "AAAUSDT_5m.csv.gz").read_bytes()).hexdigest()}
    assert json.loads((out / "exchange_info.json").read_text()) == {"AAAUSDT": {"step": "0.1", "min_notional": "5"}}
    later = ba.download(tmp_path / "later", end=date(2023, 1, 1), top=3, fetch=Source(DAILY), workers=2)
    assert later["missing_months"] == {"GONEUSDT": ["2022-12"]}
    with pytest.raises(ValueError, match="immutable"):
        ba.download(out, end=end, top=3, fetch=Source(DAILY))


def test_the_same_inputs_give_byte_identical_files(tmp_path):
    for name in ("a", "b"):
        ba.download(tmp_path / name, end=date(2022, 12, 1), top=2, fetch=Source(DAILY), workers=2)
    assert (tmp_path / "a/AAAUSDT_5m.csv.gz").read_bytes() == (tmp_path / "b/AAAUSDT_5m.csv.gz").read_bytes()


def test_a_bar_that_closes_after_the_end_is_dropped(tmp_path):
    source = Source(DAILY, bars_per_month=3)
    rows = ba.download_symbol("AAAUSDT", tmp_path, ["2022-10"], OCT + 2 * BAR, fetch=source)
    assert rows["files"]["AAAUSDT_5m.csv.gz"]["rows"] == 2      # the third bar closes at OCT + 3 bars
    assert rows["files"]["AAAUSDT_funding.csv.gz"]["rows"] == 1


def test_a_checksum_mismatch_stops_the_download(tmp_path):
    with pytest.raises(ValueError, match="checksum mismatch"):
        ba.download(tmp_path / "bad", end=date(2022, 12, 1), top=2, fetch=Source(DAILY, corrupt="AAAUSDT-5m"))


def test_too_few_qualifying_symbols_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="only 3 symbols"):
        ba.download(tmp_path / "few", end=date(2022, 12, 1), top=4, fetch=Source(DAILY))
    assert ba.months("2022-10", date(2023, 2, 1)) == ["2022-10", "2022-11", "2022-12", "2023-01"]
