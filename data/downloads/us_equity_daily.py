"""Daily price history of the US stocks and ETFs behind Binance's US-equity perpetuals.

Owner 2026-11-18: "你来去收集数据".

The perpetuals listed between 2025-12 and 2026-07, so their own history is too short to develop a strategy on; the
research uses the underlyings' history for the signals and the perpetuals' own data for the costs (funding, basis,
weekend gaps).

    python -m data.downloads.us_equity_daily --out data_files/us_equity_daily_2026-11-12 \\
        --tradfi research/lab/runs/2026-11-17b/tradfi_contracts.json --nav SPY,GLD

Sources (public, no account, no key; one request a second):
- Nasdaq's historical quotes (``api.nasdaq.com``, the data behind nasdaq.com's "historical data" page): the last 10
  years of daily open, high, low, close and volume, adjusted by Nasdaq for splits (checked 2026-11-18: NVDA is
  continuous across its 10-for-1 split of 2024-06-10), not for dividends. A ticker is asked as a stock, then as an ETF.
- State Street's SPDR NAV history (``ssga.com``): SPY's and GLD's daily net asset value since 2003 and 2004; NAV only,
  no distributions, no high or low.
- FRED (``fred.stlouisfed.org``, ``--fred``): index levels at the close, e.g. the Nasdaq Composite since 1971-02-05
  (added 2026-11-18 for the US trend study's long check).

The tickers are the EQUITY underlyings of the TradFi list recorded on 2026-11-17 (``tradfi_contracts.json``): the
Binance symbol without "USDT". A ticker Nasdaq does not know (a Binance name that is not the US ticker) is listed under
``missing`` in the manifest, never guessed.

``end`` is exclusive and at most 2026-11-13: data from 2026-11-13 on is read only at registered forward looks
(``lab/forward.py``). Rows on or after it are dropped before anything is written; the manifest counts them. Snapshots
are immutable: an existing non-empty output directory is refused. ``manifest.json`` records the sources, the time of
the download, and every file's rows, first and last date and SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from data.downloads.binance_public import http_get

FORWARD_GUARD = date(2026, 11, 13)
NASDAQ = ("https://api.nasdaq.com/api/quote/{ticker}/historical?assetclass={kind}&fromdate={start}&todate={last}"
          "&limit=9999")
SPDR_NAV = "https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/navhist-us-en-{fund}.xlsx"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
ASSET_CLASSES = ("stocks", "etf")
NASDAQ_YEARS = 10
_SHEET = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _number(text: str) -> str:
    """'$1,234.50' -> '1234.50'; 'N/A' or '' -> ''."""
    cleaned = text.replace("$", "").replace(",", "").strip()
    return cleaned if re.fullmatch(r"-?\d+(\.\d+)?", cleaned) else ""


def parse_nasdaq(payload: bytes, end: date) -> tuple[list[dict], int] | None:
    """(rows before ``end``, oldest first; how many on or after it), or None when Nasdaq has no rows for the ticker."""
    data = json.loads(payload).get("data")
    table = (data or {}).get("tradesTable") or {}
    raw = table.get("rows") or []
    if not raw:
        return None
    rows, dropped = [], 0
    for r in raw:
        day = datetime.strptime(r["date"], "%m/%d/%Y").date()
        if day >= end:
            dropped += 1
            continue
        rows.append({"date": day.isoformat(), **{k: _number(r[k]) for k in ("open", "high", "low", "close")},
                     "volume": _number(r.get("volume") or "")})
    rows.sort(key=lambda r: r["date"])
    return rows, dropped


def _xlsx_rows(payload: bytes) -> list[list[str]]:
    """The first sheet's rows as text (stdlib only: shared or inline strings and plain values)."""
    z = zipfile.ZipFile(io.BytesIO(payload))
    shared: list[str] = []
    if "xl/sharedStrings.xml" in z.namelist():
        for item in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{_SHEET}si"):
            shared.append("".join(t.text or "" for t in item.iter(f"{_SHEET}t")))
    sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
    out = []
    for row in ElementTree.fromstring(z.read(sheet)).iter(f"{_SHEET}row"):
        cells = []
        for c in row.iter(f"{_SHEET}c"):
            value = c.find(f"{_SHEET}v")
            if c.get("t") == "s" and value is not None:
                cells.append(shared[int(value.text or 0)])
            elif c.get("t") == "inlineStr":
                cells.append("".join(t.text or "" for t in c.iter(f"{_SHEET}t")))
            else:
                cells.append(value.text or "" if value is not None else "")
        out.append(cells)
    return out


def parse_nav(payload: bytes, end: date) -> tuple[list[dict], int]:
    """SPDR's NAV history: (rows before ``end``, oldest first; how many on or after it)."""
    rows, dropped, started = [], 0, False
    for cells in _xlsx_rows(payload):
        if cells[:2] == ["Date", "NAV"]:
            started = True
            continue
        if not started or len(cells) < 2 or not re.fullmatch(r"\d{2}-[A-Z][a-z]{2}-\d{4}", cells[0]):
            continue                          # the sheet ends with disclaimer text below the rows
        day = datetime.strptime(cells[0], "%d-%b-%Y").date()
        if day >= end:
            dropped += 1
            continue
        rows.append({"date": day.isoformat(), "nav": _number(cells[1])})
    rows.sort(key=lambda r: r["date"])
    return rows, dropped


def parse_fred(payload: bytes, end: date) -> tuple[list[dict], int]:
    """A FRED series' CSV (date, value; "." or empty on a day without a value): (rows before ``end``, oldest first;
    how many on or after it)."""
    rows, dropped = [], 0
    for line in payload.decode().splitlines()[1:]:
        day_text, _, value = line.partition(",")
        if not _number(value):
            continue
        day = date.fromisoformat(day_text)
        if day >= end:
            dropped += 1
            continue
        rows.append({"date": day.isoformat(), "close": value.strip()})
    rows.sort(key=lambda r: r["date"])
    return rows, dropped


def tickers(tradfi: Path) -> list[str]:
    """The US-equity underlyings of the recorded TradFi list (the Binance symbol without "USDT")."""
    underlying = json.loads(Path(tradfi).read_text())["underlying"]
    return sorted(s[:-4] for s, kind in underlying.items() if kind == "EQUITY" and s.endswith("USDT"))


def _write(path: Path, rows: list[dict]) -> dict:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    path.write_bytes(data)
    return {"rows": len(rows), "first": rows[0]["date"], "last": rows[-1]["date"],
            "sha256": hashlib.sha256(data).hexdigest()}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def snapshot(out: Path, symbols: Sequence[str], *, nav: Sequence[str], fred: Sequence[str] = (),
             end: date = FORWARD_GUARD,
             fetch: Callable[[str], bytes] | None = None, pause: float = 1.0,
             now: Callable[[], str] = _now) -> dict[str, Any]:
    """Download into the new directory ``out`` and write its manifest; returns the manifest."""
    if end > FORWARD_GUARD:
        raise ValueError(f"end {end} is after the forward guard 2026-11-13: forward data is read only at registered "
                         "looks")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: snapshots are immutable")
    get = fetch or (lambda url: http_get(url, retries=3, timeout=60))
    out.mkdir(parents=True, exist_ok=True)
    last = end - timedelta(days=1)
    start = date(last.year - NASDAQ_YEARS, 1, 1)
    files: dict[str, dict] = {}
    missing: dict[str, str] = {}
    for ticker in symbols:
        for kind in ASSET_CLASSES:
            url = NASDAQ.format(ticker=ticker, kind=kind, start=start.isoformat(), last=last.isoformat())
            parsed = parse_nasdaq(get(url), end)
            time.sleep(pause)
            if parsed is not None and parsed[0]:
                rows, dropped = parsed
                files[f"{ticker}_1d.csv"] = {**_write(out / f"{ticker}_1d.csv", rows), "source": "nasdaq",
                                             "asset_class": "etf" if kind == "etf" else "stock",
                                             "dropped_on_or_after_end": dropped}
                break
        else:
            missing[ticker] = "not known to Nasdaq as a stock or an ETF"
    for fund in nav:
        rows, dropped = parse_nav(get(SPDR_NAV.format(fund=fund.lower())), end)
        time.sleep(pause)
        files[f"{fund.upper()}_nav.csv"] = {**_write(out / f"{fund.upper()}_nav.csv", rows), "source": "spdr_nav",
                                            "dropped_on_or_after_end": dropped}
    for series in fred:
        rows, dropped = parse_fred(get(FRED.format(series=series)), end)
        time.sleep(pause)
        files[f"{series}_index.csv"] = {**_write(out / f"{series}_index.csv", rows), "source": "fred",
                                        "dropped_on_or_after_end": dropped}
    manifest = {"what": "daily history of the US stocks and ETFs behind Binance's US-equity perpetuals",
                "fetched_at": now(), "end_exclusive": end.isoformat(),
                "sources": {k: v for k, v in (("nasdaq", NASDAQ.split("?")[0]), ("spdr_nav", SPDR_NAV), ("fred", FRED))
                            if any(f["source"] == k for f in files.values())},
                "notes": {"nasdaq": f"last {NASDAQ_YEARS} years; split-adjusted by Nasdaq, not dividend-adjusted",
                          "spdr_nav": "net asset value only; no distributions, no open/high/low",
                          "fred": "index level at the close only (e.g. NASDAQCOM, a price index without dividends)"},
                "files": files, "missing": missing}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tradfi", type=Path, help="tradfi_contracts.json: its EQUITY underlyings")
    parser.add_argument("--symbols", default="", help="comma-separated US tickers, besides --tradfi")
    parser.add_argument("--nav", default="SPY,GLD", help="SPDR funds whose NAV history to add")
    parser.add_argument("--fred", default="", help="FRED series (index levels) to add, e.g. NASDAQCOM")
    parser.add_argument("--end", type=date.fromisoformat, default=FORWARD_GUARD)
    args = parser.parse_args(argv)
    symbols = sorted(set((tickers(args.tradfi) if args.tradfi else [])
                         + [s.strip().upper() for s in args.symbols.split(",") if s.strip()]))
    manifest = snapshot(args.out, symbols, nav=[f for f in args.nav.split(",") if f],
                        fred=[f for f in args.fred.split(",") if f], end=args.end)
    print(f"{len(manifest['files'])} files written to {args.out}; missing: {len(manifest['missing'])}")


if __name__ == "__main__":
    main()


__all__ = ["FORWARD_GUARD", "parse_fred", "parse_nasdaq", "parse_nav", "snapshot", "tickers"]
