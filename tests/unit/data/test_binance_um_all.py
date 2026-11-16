"""The all-USD(S)-M downloader against a fake archive: no network."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import urllib.error
import zipfile
from datetime import date

import pytest

from data.downloads import binance_um_all as um

H = 3_600_000
NOV_1 = 1_793_491_200_000          # 2026-11-01 00:00 UTC


def _zip(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("x.csv", text)
    return buffer.getvalue()


def _kline(t: int) -> str:
    return f"{t},1,2,0.5,1.5,10,{t + H - 1},15,3,4,6,0"


class Archive:
    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.requests: list[str] = []

    def add(self, url: str, body: bytes):
        self.files[url] = body
        self.files[url + ".CHECKSUM"] = f"{hashlib.sha256(body).hexdigest()}  x.zip".encode()

    def __call__(self, url: str) -> bytes:
        self.requests.append(url)
        if url.startswith(um.FUNDING_API):
            return json.dumps([{"fundingTime": NOV_1 + 8 * H, "fundingRate": "0.0001"}]).encode()
        if url not in self.files:
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        return self.files[url]


def test_whole_months_and_the_days_of_the_open_month_with_funding(tmp_path):
    fake = Archive()
    oct_1 = NOV_1 - 31 * 24 * H
    fake.add("https://data.binance.vision/data/futures/um/monthly/klines/ABCUSDT/1h/ABCUSDT-1h-2026-10.zip",
             _zip("open_time,o,h,l,c,v,ct,qv,n,tbv,tbqv,i\n" + "\n".join(_kline(oct_1 + k * H) for k in range(3))))
    fake.add("https://data.binance.vision/data/futures/um/monthly/fundingRate/ABCUSDT/ABCUSDT-fundingRate-2026-10.zip",
             _zip(f"calc_time,interval,rate\n{oct_1},8,0.0002"))
    fake.add("https://data.binance.vision/data/futures/um/daily/klines/ABCUSDT/1h/ABCUSDT-1h-2026-11-01.zip",
             _zip("\n".join(_kline(NOV_1 + k * H) for k in range(24))))
    entry = um.download_symbol("ABCUSDT", tmp_path, date(2026, 10, 1), date(2026, 11, 3), fetch=fake)
    assert entry["type"] == "perpetual" and entry["margin"] == "USDT"
    assert entry["missing"] == ["2026-11-02"]                     # the daily archive for the 2nd does not exist
    bars = gzip.decompress((tmp_path / "ABCUSDT_1h.csv.gz").read_bytes()).decode().splitlines()
    assert bars[0] == ",".join(um.COLUMNS) and len(bars) == 1 + 3 + 24
    assert bars[1] == f"{oct_1},1,2,0.5,1.5,10,15,3,4,6"
    funding = gzip.decompress((tmp_path / "ABCUSDT_funding.csv.gz").read_bytes()).decode().splitlines()
    assert funding[1:] == [f"{oct_1},0.0002", f"{NOV_1 + 8 * H},0.0001"]


def test_a_tampered_archive_is_refused(tmp_path):
    fake = Archive()
    url = "https://data.binance.vision/data/futures/um/daily/klines/ABCUSDT/1h/ABCUSDT-1h-2026-11-01.zip"
    fake.add(url, _zip(_kline(NOV_1)))
    fake.files[url] = _zip(_kline(NOV_1 + H))
    with pytest.raises(ValueError, match="checksum"):
        um.download_symbol("ABCUSDT", tmp_path, date(2026, 11, 1), date(2026, 11, 2), fetch=fake)


def test_contract_types_and_margins():
    assert um.contract("BTCUSDT_250328") == {"type": "delivery", "margin": "USDT"}
    assert um.contract("BTCUSDC") == {"type": "perpetual", "margin": "USDC"}


def test_a_delivery_contract_has_no_funding_request(tmp_path):
    fake = Archive()
    um.fetch_recent_funding("BTCUSDT_261225", NOV_1, NOV_1 + 24 * H, fetch=fake)
    assert not fake.requests


def test_the_forward_period_and_an_existing_snapshot_are_refused(tmp_path):
    with pytest.raises(ValueError, match="forward"):
        um.download(tmp_path / "a", start=date(2026, 1, 1), end=date(2026, 11, 14), symbols=[], fetch=Archive())
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        um.download(tmp_path / "b", start=date(2026, 1, 1), symbols=[], fetch=Archive())
