"""Download every Binance USD(S)-M contract's hourly bars and funding from the public archive (no account access).

    python -m data.downloads.binance_um_all --start 2022-11-13 --end 2026-11-13 \\
        --out data_files/binance_um_all_2026-11-12

The research lab of 2026-11-16 (tech lead: research and trading over all USD(S)-M contracts). Source:
data.binance.vision; every zip is checked against its published SHA-256.

- Universe: every symbol directory in the monthly kline archive, delisted symbols included, perpetual and
  delivery contracts (``BTCUSDT_250328``), USDT- and USDC-margined. Nothing is ranked or filtered here: the
  research decides what to use, and the delisted ones are kept against survivorship bias.
- Per symbol: 1-hour klines and funding settlements in [start, end). Whole months come from the monthly archive;
  the days of a month that has not closed yet come from the daily kline archive and the public funding endpoint.
- ``end`` is exclusive and defaults to 2026-11-13: data from 2026-11-13 on is read only at registered forward
  looks (``lab/forward.py``), so this snapshot never contains it.

Snapshots are immutable: an existing non-empty output directory is refused. ``manifest.json`` records the source,
the window, every file's SHA-256 and row count, the months that were missing and the contract type.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.error
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from data.downloads.binance_archive import Fetch, _write_gz, exchange_rules, fetch_archive, list_symbols, months
from data.downloads.binance_public import http_get

DAILY = "https://data.binance.vision/data/futures/um/daily"
FUNDING_API = "https://fapi.binance.com/fapi/v1/fundingRate"
COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume",
           "taker_buy_quote_volume")
HOUR_MS = 3_600_000
DEFAULT_END = date(2026, 11, 13)


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def contract(symbol: str) -> dict:
    """Perpetual or delivery, and the margin asset, from the symbol's name."""
    base = symbol.split("_")[0]
    margin = "USDC" if base.endswith("USDC") else "USDT" if base.endswith("USDT") else "other"
    return {"type": "delivery" if "_" in symbol else "perpetual", "margin": margin}


def fetch_daily(symbol: str, day: date, *, fetch: Fetch = http_get) -> list[list[str]] | None:
    """Rows of one daily 1h kline archive (header dropped), or None when it does not exist."""
    name, stamp = urllib.parse.quote(symbol), day.isoformat()
    url = f"{DAILY}/klines/{name}/1h/{name}-1h-{stamp}.zip"
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


def fetch_recent_funding(symbol: str, start_ms: int, end_ms: int, *, fetch: Fetch = http_get) -> list[tuple]:
    """Funding settlements in [start_ms, end_ms) from the public endpoint (no key); [] for a delivery contract."""
    if "_" in symbol:
        return []
    out: list[tuple] = []
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "startTime": cursor, "endTime": end_ms - 1, "limit": 1000})
        try:
            rows = json.loads(fetch(f"{FUNDING_API}?{query}"))
        except urllib.error.HTTPError as exc:
            if exc.code == 400:                       # the symbol is no longer listed
                return out
            raise
        rows = [r for r in rows if start_ms <= int(r["fundingTime"]) < end_ms]
        if not rows:
            return out
        out += [(int(r["fundingTime"]), r["fundingRate"]) for r in rows]
        cursor = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < 1000:
            return out
    return out


def download_symbol(symbol: str, out: Path, start: date, end: date, *, fetch: Fetch = http_get) -> dict:
    """Write ``<symbol>_1h.csv.gz`` and ``<symbol>_funding.csv.gz`` with bars and settlements in [start, end)."""
    start_ms, end_ms = _ms(start), _ms(end)
    closed_until = date(end.year, end.month, 1)               # months before this one are closed and archived
    bars, funding, missing = [], [], []
    for month in months(f"{start.year:04d}-{start.month:02d}", closed_until):
        rows = fetch_archive("klines/1h", symbol, month, fetch=fetch)
        if rows is None:
            missing.append(month)
            continue
        bars += [tuple([int(r[0])] + r[1:6] + [r[7], r[8], r[9], r[10]]) for r in rows
                 if start_ms <= int(r[0]) and int(r[0]) + HOUR_MS <= end_ms]
        funding += [(int(r[0]), r[2]) for r in fetch_archive("fundingRate", symbol, month, fetch=fetch) or []
                    if start_ms <= int(r[0]) < end_ms]
    day = max(closed_until, start)
    while day < end:
        rows = fetch_daily(symbol, day, fetch=fetch)
        if rows is None:
            missing.append(day.isoformat())
        else:
            bars += [tuple([int(r[0])] + r[1:6] + [r[7], r[8], r[9], r[10]]) for r in rows
                     if start_ms <= int(r[0]) and int(r[0]) + HOUR_MS <= end_ms]
        day += timedelta(days=1)
    if closed_until < end:
        funding += fetch_recent_funding(symbol, max(_ms(closed_until), start_ms), end_ms, fetch=fetch)
    bars = sorted(set(bars))
    if len({b[0] for b in bars}) != len(bars):
        raise ValueError(f"{symbol}: two different bars share one open time")
    entry = {"symbol": symbol, **contract(symbol), "missing": missing, "files": {}}
    if not bars:
        return entry
    digest_b, n_b = _write_gz(out / f"{symbol}_1h.csv.gz", COLUMNS, bars)
    digest_f, n_f = _write_gz(out / f"{symbol}_funding.csv.gz", ("funding_time", "rate"), sorted(set(funding)))
    entry["files"] = {f"{symbol}_1h.csv.gz": {"rows": n_b, "sha256": digest_b},
                      f"{symbol}_funding.csv.gz": {"rows": n_f, "sha256": digest_f}}
    return entry


def download(out: Path, *, start: date, end: date = DEFAULT_END, fetch: Fetch = http_get, workers: int = 16,
             symbols: list[str] | None = None) -> dict:
    if end > DEFAULT_END:
        raise ValueError("data from 2026-11-13 on is read only at registered forward looks (lab/forward.py)")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; snapshots are immutable")
    out.mkdir(parents=True, exist_ok=True)
    universe = symbols if symbols is not None else list_symbols(fetch=fetch)
    with ThreadPoolExecutor(workers) as pool:
        entries = list(pool.map(lambda s: download_symbol(s, out, start, end, fetch=fetch), universe))
    kept = [e for e in entries if e["files"]]
    try:
        rules = exchange_rules([e["symbol"] for e in kept], fetch=fetch)
    except Exception as exc:                                  # the rules are a convenience, not the data
        rules = {"error": f"{type(exc).__name__}"}
    (out / "exchange_info.json").write_text(json.dumps(rules, indent=1, sort_keys=True) + "\n")
    manifest = {"source": "data.binance.vision USD(S)-M archives (monthly and daily 1h klines, monthly funding); "
                          "fapi.binance.com public funding for the unclosed month",
                "window": [start.isoformat(), end.isoformat()], "end_exclusive": True,
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbols_listed": len(universe), "symbols_with_bars": len(kept),
                "contracts": sorted(entries, key=lambda e: e["symbol"])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 11, 13))
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    manifest = download(args.out, start=args.start, end=args.end, workers=args.workers)
    print(f"{manifest['symbols_with_bars']} of {manifest['symbols_listed']} symbols written to {args.out}")


if __name__ == "__main__":
    main()


__all__ = ["COLUMNS", "contract", "download", "download_symbol", "fetch_daily", "fetch_recent_funding", "main"]
