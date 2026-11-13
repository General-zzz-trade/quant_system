"""Download Binance USD-M perpetual public research data (no account access).

Sources (public, unauthenticated):
- 1h klines:   GET https://fapi.binance.com/fapi/v1/klines
- funding:     GET https://fapi.binance.com/fapi/v1/fundingRate
- OI / ratios: https://data.binance.vision/data/futures/um/daily/metrics/<SYM>/<SYM>-metrics-<day>.zip
               (5-minute snapshots, verified against the published .CHECKSUM)

Only CLOSED bars are written. Funding rows whose markPrice is empty (older
history) use the open of the 1h bar that starts at the settlement time as the
mark proxy, flagged in ``mark_source``. Every output file gets a SHA-256 in
``manifest.json`` so a study can prove which inputs it used.

Usage::

    python -m data.downloads.binance_public --symbols BTCUSDT ETHUSDT \
        --start 2020-01-01 --out data_files/binance_public
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision"
HOUR_MS = 3_600_000
_USER_AGENT = "quant-research-downloader/1.0"

KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "quote_volume",
                 "trades", "taker_buy_volume", "taker_buy_quote_volume")
FUNDING_COLUMNS = ("funding_time", "rate", "mark_price", "mark_source", "observed_funding_time")
METRIC_COLUMNS = ("timestamp", "sum_open_interest", "sum_open_interest_value",
                  "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
                  "count_long_short_ratio", "sum_taker_long_short_vol_ratio")

Fetch = Callable[[str], bytes]


def http_get(url: str, *, retries: int = 4, timeout: float = 20.0) -> bytes:
    """GET with bounded retries; 404 is returned to the caller as an error."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
            if exc.code in (418, 429):
                time.sleep(30 * (attempt + 1))
                continue
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int, *, fetch: Fetch = http_get,
                 now_ms: int | None = None) -> list[tuple]:
    """Closed 1h klines with open_time in [start_ms, end_ms)."""
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    rows: dict[int, tuple] = {}
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "interval": "1h", "limit": 1500,
                                        "startTime": cursor, "endTime": end_ms - 1})
        batch = json.loads(fetch(f"{FAPI}/fapi/v1/klines?{query}"))
        if not batch:
            break
        for k in batch:
            open_time, close_time = int(k[0]), int(k[6])
            if open_time % HOUR_MS or close_time != open_time + HOUR_MS - 1:
                raise ValueError(f"unexpected kline boundaries at {open_time}")
            if close_time >= now_ms or not start_ms <= open_time < end_ms:
                continue  # never store a bar that has not closed
            rows[open_time] = (open_time, k[1], k[2], k[3], k[4], k[5], k[7], int(k[8]), k[9], k[10])
        last_open = int(batch[-1][0])
        if last_open + HOUR_MS <= cursor:
            raise ValueError("kline pagination did not advance")
        cursor = last_open + HOUR_MS
    return [rows[t] for t in sorted(rows)]


def fetch_funding(symbol: str, start_ms: int, end_ms: int, *, fetch: Fetch = http_get) -> list[dict]:
    """Raw funding events with fundingTime in [start_ms, end_ms)."""
    events: dict[int, dict] = {}
    cursor = start_ms
    while cursor < end_ms:
        query = urllib.parse.urlencode({"symbol": symbol, "limit": 1000,
                                        "startTime": cursor, "endTime": end_ms - 1})
        batch = json.loads(fetch(f"{FAPI}/fapi/v1/fundingRate?{query}"))
        if not batch:
            break
        for event in batch:
            observed = int(event["fundingTime"])
            if start_ms <= observed < end_ms:
                events[observed] = event
        last = int(batch[-1]["fundingTime"])
        if last < cursor:
            raise ValueError("funding pagination did not advance")
        cursor = last + 1
        if len(batch) < 1000:
            break
    return [events[t] for t in sorted(events)]


def normalize_funding(events: Iterable[dict], opens: dict[int, str]) -> list[tuple]:
    """Align to nominal hour boundaries (<=1s jitter) and attach a mark price."""
    out = []
    for event in events:
        observed = int(event["fundingTime"])
        nominal = round(observed / HOUR_MS) * HOUR_MS
        if abs(observed - nominal) > 1000:
            raise ValueError(f"funding time {observed} is not within 1s of an hour boundary")
        mark = str(event.get("markPrice") or "")
        source = "exchange"
        if not mark or float(mark) <= 0:
            if nominal not in opens:
                raise ValueError(f"no mark price and no kline open for funding at {nominal}")
            mark, source = opens[nominal], "kline_open_proxy"
        out.append((nominal, event["fundingRate"], mark, source, observed))
    return out


def fetch_metrics_day(symbol: str, day: date, *, fetch: Fetch = http_get) -> list[tuple]:
    """One day of 5-minute OI/ratio snapshots, checksum-verified."""
    name = f"{symbol}-metrics-{day.isoformat()}.zip"
    url = f"{VISION}/data/futures/um/daily/metrics/{symbol}/{name}"
    payload = fetch(url)
    expected = fetch(url + ".CHECKSUM").decode().split()[0].strip().lower()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {name}")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.namelist()
        if len(members) != 1:
            raise ValueError(f"unexpected archive layout in {name}")
        text = archive.read(members[0]).decode()
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        stamp = datetime.strptime(row["create_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        rows.append((int(stamp.timestamp() * 1000), *(row.get(c, "") for c in METRIC_COLUMNS[1:])))
    return rows


def _write_csv(path: Path, header: tuple, rows: Iterable[tuple]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(symbols: list[str], start: date, end: date, out: Path, *, metrics: bool = True,
             fetch: Fetch = http_get) -> dict:
    start_ms, end_ms = _ms(start), _ms(end)
    manifest: dict = {"source": "binance-usdm-public", "start": start.isoformat(), "end": end.isoformat(),
                      "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "files": {}}
    for symbol in symbols:
        logger.info("%s: klines", symbol)
        klines = fetch_klines(symbol, start_ms, end_ms, fetch=fetch)
        manifest["files"][f"{symbol}_1h.csv"] = {
            "rows": len(klines), "sha256": _write_csv(out / f"{symbol}_1h.csv", KLINE_COLUMNS, klines)}
        logger.info("%s: funding", symbol)
        opens = {row[0]: row[1] for row in klines}
        funding = normalize_funding(fetch_funding(symbol, start_ms, end_ms, fetch=fetch), opens)
        manifest["files"][f"{symbol}_funding.csv"] = {
            "rows": len(funding), "sha256": _write_csv(out / f"{symbol}_funding.csv", FUNDING_COLUMNS, funding),
            "mark_proxy_rows": sum(1 for row in funding if row[3] != "exchange")}
        if not metrics:
            continue
        rows, missing, day = [], [], start
        while day < end:
            try:
                rows.extend(fetch_metrics_day(symbol, day, fetch=fetch))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                missing.append(day.isoformat())  # metrics archive begins around 2021-12
            day += timedelta(days=1)
        rows.sort(key=lambda r: r[0])
        manifest["files"][f"{symbol}_metrics.csv"] = {
            "rows": len(rows), "missing_days": len(missing),
            "first_missing": missing[:3], "last_missing": missing[-3:],
            "sha256": _write_csv(out / f"{symbol}_metrics.csv", METRIC_COLUMNS, rows)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat,
                        default=datetime.now(timezone.utc).date())
    parser.add_argument("--out", type=Path, default=Path("data_files/binance_public"))
    parser.add_argument("--no-metrics", action="store_true", help="Skip OI/long-short metrics archives")
    args = parser.parse_args(argv)
    if args.start >= args.end:
        parser.error("--start must be before --end")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    manifest = download(args.symbols, args.start, args.end, args.out, metrics=not args.no_metrics)
    print(json.dumps(manifest["files"], indent=2))


if __name__ == "__main__":
    main()
