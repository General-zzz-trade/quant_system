"""Daily history of the US stocks and ETFs behind Binance's US-equity perpetuals (data/downloads/us_equity_daily.py).

Owner 2026-11-18: "你来去收集数据" (after asking how to develop strategies on the US-stock perpetuals, whose own history
is under a year). Each parser on canned payloads, the forward cut, the snapshot rules (immutable directory, manifest
with source, time, checksums), and a symbol a source does not know. No network."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import date
from xml.sax.saxutils import escape

import pytest

from data.downloads import us_equity_daily as us


def _nasdaq(rows, total=None):
    return json.dumps({"data": {"symbol": "X", "totalRecords": total if total is not None else len(rows),
                                "tradesTable": {"rows": rows}}, "message": None, "status": {"rCode": 200}}).encode()


UNKNOWN = json.dumps({"data": None, "status": {"rCode": 400, "bCodeMessage": [
    {"code": 1001, "errorMessage": "Symbol not exists."}]}}).encode()


def _row(day, close, *, dollar=True, volume="1,000"):
    d = "$" if dollar else ""
    return {"date": day, "close": f"{d}{close}", "volume": volume, "open": f"{d}{close}", "high": f"{d}{close}",
            "low": f"{d}{close}"}


def _xlsx(rows):
    """A workbook as SPDR publishes it: inline strings, one sheet."""
    cells = []
    for i, row in enumerate(rows, start=1):
        cs = "".join(f'<c r="{chr(65 + j)}{i}" t="inlineStr"><is><t>{escape(v)}</t></is></c>'
                     for j, v in enumerate(row))
        cells.append(f'<row r="{i}">{cs}</row>')
    sheet = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
             + "".join(cells) + "</sheetData></worksheet>")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


NAV = [["Fund Name:", "SPDR S&P 500 ETF Trust"], ["Ticker Symbol:", "SPY"],
       ["Date", "NAV", "Shares Outstanding", "Total Net Assets"],
       ["16-Nov-2026", "739.7", "1", "2"], ["12-Nov-2026", "757.0", "1", "2"], ["31-Jan-2003", "85.5", "1", "2"],
       [], ["Past performance is not a reliable indicator of future performance. Visit www.ssga.com", "", ""]]


class Fetch:
    def __init__(self, pages):
        self.pages, self.seen = pages, []

    def __call__(self, url):
        self.seen.append(url)
        for key, body in self.pages.items():
            if key in url:
                return body
        raise AssertionError(f"unexpected URL {url}")


END = date(2026, 11, 13)


def test_nasdaq_rows_are_parsed_sorted_and_cut_before_the_end():
    payload = _nasdaq([_row("11/13/2026", "170.00"), _row("11/12/2026", "166.19", volume="747,707,600"),
                       _row("11/17/2016", "2.3098"), _row("11/18/2016", "2.40", dollar=False, volume="N/A")])
    rows, dropped = us.parse_nasdaq(payload, END)
    assert [r["date"] for r in rows] == ["2016-11-17", "2016-11-18", "2026-11-12"]
    assert rows[0]["close"] == "2.3098" and rows[2]["volume"] == "747707600" and rows[1]["volume"] == ""
    assert dropped == 1                                                   # 2026-11-13: forward data, never written


def test_a_symbol_nasdaq_does_not_know_is_none():
    assert us.parse_nasdaq(UNKNOWN, END) is None
    assert us.parse_nasdaq(_nasdaq([]), END) is None


def test_the_spdr_nav_sheet_is_parsed_and_cut():
    rows, dropped = us.parse_nav(_xlsx(NAV), END)
    assert rows == [{"date": "2003-01-31", "nav": "85.5"}, {"date": "2026-11-12", "nav": "757.0"}]
    assert dropped == 1


def test_a_snapshot_writes_files_and_a_manifest_with_checksums(tmp_path):
    fetch = Fetch({"/quote/NVDA/historical?assetclass=stocks": _nasdaq([_row("11/12/2026", "166.19"),
                                                                           _row("11/17/2016", "2.3")]),
                   "/quote/QQQ/historical?assetclass=stocks": UNKNOWN,
                   "/quote/QQQ/historical?assetclass=etf": _nasdaq([_row("11/12/2026", "600", dollar=False)]),
                   "/quote/ZZZQ/historical": UNKNOWN,
                   "navhist-us-en-spy.xlsx": _xlsx(NAV)})
    out = tmp_path / "snap"
    manifest = us.snapshot(out, ["NVDA", "QQQ", "ZZZQ"], nav=["SPY"], end=END, fetch=fetch, pause=0,
                           now=lambda: "2026-11-18T00:00:00Z")
    assert sorted(p.name for p in out.iterdir()) == ["NVDA_1d.csv", "QQQ_1d.csv", "SPY_nav.csv", "manifest.json"]
    with (out / "NVDA_1d.csv").open() as handle:
        assert [r["date"] for r in csv.DictReader(handle)] == ["2016-11-17", "2026-11-12"]
    saved = json.loads((out / "manifest.json").read_text())
    assert saved == manifest and saved["end_exclusive"] == "2026-11-13"
    assert saved["fetched_at"] == "2026-11-18T00:00:00Z"
    files = saved["files"]
    assert files["QQQ_1d.csv"]["asset_class"] == "etf" and files["NVDA_1d.csv"]["rows"] == 2
    assert files["NVDA_1d.csv"]["sha256"] == hashlib.sha256((out / "NVDA_1d.csv").read_bytes()).hexdigest()
    assert files["SPY_nav.csv"]["first"] == "2003-01-31" and files["SPY_nav.csv"]["dropped_on_or_after_end"] == 1
    assert saved["missing"] == {"ZZZQ": "not known to Nasdaq as a stock or an ETF"}
    assert all("nasdaq.com" in s or "ssga.com" in s for s in saved["sources"].values())


def test_a_snapshot_directory_is_written_once(tmp_path):
    out = tmp_path / "snap"
    out.mkdir()
    (out / "x").write_text("old")
    with pytest.raises(FileExistsError):
        us.snapshot(out, ["NVDA"], nav=[], end=END, fetch=Fetch({}), pause=0)


def test_an_end_after_the_forward_guard_is_refused(tmp_path):
    with pytest.raises(ValueError, match="2026-11-13"):
        us.snapshot(tmp_path / "snap", ["NVDA"], nav=[], end=date(2026, 11, 20), fetch=Fetch({}), pause=0)


def test_the_tickers_come_from_the_recorded_tradfi_list(tmp_path):
    path = tmp_path / "tradfi.json"
    path.write_text(json.dumps({"underlying": {"NVDAUSDT": "EQUITY", "XAUUSDT": "COMMODITY", "HK0700USDT": "HK_EQUITY",
                                               "OPENAIUSDT": "PREMARKET", "SPYUSDT": "EQUITY"}}))
    assert us.tickers(path) == ["NVDA", "SPY"]
