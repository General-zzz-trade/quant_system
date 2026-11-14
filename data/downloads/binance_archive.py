"""Download a mid-frequency research snapshot from Binance's public archive (no account access).

    python -m data.downloads.binance_archive --out data_files/binance_midfreq_2025-11-16

Protocol: research/protocols/2026-11-14-midfreq-feasibility.md. Source: data.binance.vision monthly
archives of USD-M perpetuals; every zip is checked against its published SHA-256.

1. Universe: every symbol in the archive listing that ends in USDT (delisted ones included), has daily
   klines for each ranking month and is not a delivery contract, a stablecoin pair or an index, ranked by
   quote volume over the ranking months. The top ``--top`` symbols are kept for the whole study.
2. For each kept symbol: 5-minute klines (open time, OHLC, quote volume, taker-buy quote volume) and
   funding settlements from the first ranking month up to ``--end`` (exclusive).
3. ``exchange_info.json``: lot step and minimum notional per symbol as published today.

Snapshots are immutable: an existing non-empty output directory is refused. ``manifest.json`` records the
universe, the ranking volumes and a SHA-256 of every file written.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import urllib.error
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from data.downloads.binance_public import http_get

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly"
LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/futures/um/monthly/klines/"
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
RANKING_MONTHS = ("2022-10", "2022-11", "2022-12")
EXCLUDED = ("USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "USDPUSDT", "DAIUSDT", "BTCDOMUSDT", "DEFIUSDT",
            "BLUEBIRDUSDT", "FOOTBALLUSDT")
KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume")
Fetch = Callable[[str], bytes]


def eligible(symbol: str) -> bool:
    """A USDT-margined perpetual on a single asset: no delivery contract, stablecoin pair or index."""
    return symbol.endswith("USDT") and "_" not in symbol and symbol not in EXCLUDED


def list_symbols(*, fetch: Fetch = http_get) -> list[str]:
    """Every symbol directory in the monthly kline archive, delisted symbols included."""
    names: list[str] = []
    marker = ""
    while True:
        body = fetch(LISTING + (f"&marker={urllib.parse.quote(marker)}" if marker else "")).decode()
        found = re.findall(r"<Prefix>data/futures/um/monthly/klines/([^/<]+)/</Prefix>", body)
        names += found
        if "<IsTruncated>true</IsTruncated>" not in body or not found:
            return sorted(set(names))
        marker = f"data/futures/um/monthly/klines/{found[-1]}/"


def fetch_archive(kind: str, symbol: str, month: str, *, fetch: Fetch = http_get) -> list[list[str]] | None:
    """Rows of one monthly archive (header dropped), or None when the archive does not exist.

    ``kind`` is ``klines/5m``, ``klines/1d`` or ``fundingRate``. The zip must match its published checksum.
    """
    label, name = kind.split("/")[-1], urllib.parse.quote(symbol)      # some listed symbols are not ASCII
    url = f"{ARCHIVE}/{kind.split('/')[0]}/{name}/" + (f"{label}/" if "/" in kind else "") + \
        f"{name}-{label}-{month}.zip"
    try:
        raw = fetch(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    expected = fetch(url + ".CHECKSUM").decode().split()[0]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        text = archive.read(archive.namelist()[0]).decode()
    rows = [line.split(",") for line in text.splitlines() if line]
    return rows[1:] if rows and not rows[0][0].lstrip("-").isdigit() else rows


def months(first: str, end: date) -> list[str]:
    """Calendar months from ``first`` (YYYY-MM) through the month that contains the day before ``end``."""
    year, month = (int(x) for x in first.split("-"))
    out = []
    while date(year, month, 1) < end:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def rank_universe(symbols: Iterable[str], top: int, *, fetch: Fetch = http_get, workers: int = 8) -> list[dict]:
    """The ``top`` eligible symbols by quote volume over RANKING_MONTHS; all three months must exist."""
    def volume(symbol: str) -> tuple[str, float] | None:
        total = 0.0
        for month in RANKING_MONTHS:
            rows = fetch_archive("klines/1d", symbol, month, fetch=fetch)
            if not rows:
                return None
            total += sum(float(r[7]) for r in rows)
        return symbol, total

    candidates = sorted(s for s in symbols if eligible(s))
    with ThreadPoolExecutor(workers) as pool:
        ranked = sorted((v for v in pool.map(volume, candidates) if v), key=lambda v: (-v[1], v[0]))
    return [{"symbol": s, "ranking_quote_volume": q, "rank": k + 1} for k, (s, q) in enumerate(ranked[:top])]


def _write_gz(path: Path, header: tuple, rows: Iterable[tuple]) -> tuple[str, int]:
    """Deterministic gzip CSV (no timestamp in the header); returns its SHA-256 and row count."""
    buffer, count = io.StringIO(), 0
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
        count += 1
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
        stream.write(buffer.getvalue().encode())
    return hashlib.sha256(path.read_bytes()).hexdigest(), count


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def download_symbol(symbol: str, out: Path, month_list: list[str], end_ms: int, *, fetch: Fetch = http_get) -> dict:
    """Write ``<symbol>_5m.csv.gz`` and ``<symbol>_funding.csv.gz``; bars and settlements before ``end_ms`` only."""
    bars, funding, missing = [], [], []
    for month in month_list:
        rows = fetch_archive("klines/5m", symbol, month, fetch=fetch)
        if rows is None:
            missing.append(month)
            continue
        bars += [(int(r[0]), r[1], r[2], r[3], r[4], r[7], r[10]) for r in rows if int(r[0]) + 300_000 <= end_ms]
        for r in fetch_archive("fundingRate", symbol, month, fetch=fetch) or []:
            if int(r[0]) < end_ms:
                funding.append((int(r[0]), r[2]))
    bars = sorted(set(bars))
    if len({b[0] for b in bars}) != len(bars):
        raise ValueError(f"{symbol}: two different bars share one open time")
    digest_b, n_b = _write_gz(out / f"{symbol}_5m.csv.gz", KLINE_COLUMNS, bars)
    digest_f, n_f = _write_gz(out / f"{symbol}_funding.csv.gz", ("funding_time", "rate"), sorted(set(funding)))
    return {"symbol": symbol, "missing_months": missing,
            "files": {f"{symbol}_5m.csv.gz": {"rows": n_b, "sha256": digest_b},
                      f"{symbol}_funding.csv.gz": {"rows": n_f, "sha256": digest_f}}}


def exchange_rules(symbols: Iterable[str], *, fetch: Fetch = http_get) -> dict:
    """Lot step and minimum notional per symbol as published now; a symbol no longer listed is left out."""
    wanted, out = set(symbols), {}
    for entry in json.loads(fetch(EXCHANGE_INFO))["symbols"]:
        if entry["symbol"] in wanted:
            filters = {f["filterType"]: f for f in entry["filters"]}
            out[entry["symbol"]] = {"step": filters["LOT_SIZE"]["stepSize"],
                                    "min_notional": filters["MIN_NOTIONAL"]["notional"]}
    return out


def download(out: Path, *, end: date, top: int = 15, fetch: Fetch = http_get, workers: int = 8) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("snapshots are immutable: use a new output directory")
    universe = rank_universe(list_symbols(fetch=fetch), top, fetch=fetch, workers=workers)
    if len(universe) < top:
        raise ValueError(f"only {len(universe)} symbols qualify for a universe of {top}")
    out.mkdir(parents=True, exist_ok=True)
    month_list = months(RANKING_MONTHS[0], end)
    with ThreadPoolExecutor(min(workers, 4)) as pool:
        parts = list(pool.map(lambda u: download_symbol(u["symbol"], out, month_list, _ms(end), fetch=fetch),
                              universe))
    rules = exchange_rules([u["symbol"] for u in universe], fetch=fetch)
    (out / "exchange_info.json").write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n")
    files = {name: info for part in parts for name, info in part["files"].items()}
    files["exchange_info.json"] = {"sha256": hashlib.sha256((out / "exchange_info.json").read_bytes()).hexdigest()}
    manifest = {"source": "data.binance.vision USD-M monthly archives", "ranking_months": list(RANKING_MONTHS),
                "start": f"{RANKING_MONTHS[0]}-01", "end": end.isoformat(), "bar": "5m", "universe": universe,
                "missing_months": {p["symbol"]: p["missing_months"] for p in parts if p["missing_months"]},
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 11, 16), help="exclusive")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args(argv)
    manifest = download(args.out, end=args.end, top=args.top)
    print(json.dumps({"universe": [u["symbol"] for u in manifest["universe"]],
                      "missing_months": manifest["missing_months"]}, indent=2))


if __name__ == "__main__":
    main()
