"""Download the extra BTC/ETH sources of the feature-pool study into one immutable dated snapshot.

    python -m data.downloads.feature_sources --out data_files/btc_eth_features_2026-11-12 --end 2026-11-13

Owner 2026-11-17: "来使用加入特征池的方式去做，参考更多的数据去做", then "执行". Every source is public (no account,
no key). Nothing at or after ``--end`` is written: data from 2026-11-13 on is read only at registered forward looks
(lab/forward.py). Sources, each checked reachable on 2026-11-17:

- Binance spot 1h klines (data.binance.vision, checksum-checked; microsecond timestamps from 2025 become ms);
- Binance USD-M premium index 1h klines (data.binance.vision);
- Binance USD-M order book depth (daily archives from the first day there is one), the hourly mean and last notional
  of each percentage level;
- Binance options BVOL index: ``bvol`` reads its daily archives, but ``download`` leaves it out (``BVOL_LEFT_OUT``);
- Deribit DVOL index, hourly candles (public API);
- FRED daily series: broad dollar index, 10-year Treasury yield, S&P 500, Nasdaq composite, VIX;
- DefiLlama total stablecoin supply in USD, daily;
- Coin Metrics community asset metrics, daily, for btc and eth.

``manifest.json`` records, per file, its SHA-256, its row count and when each row is known (the feature pool must
use a value only from then on), and every source's caveat. Coin Metrics, FRED's dollar index and DefiLlama revise
values after first publication, so their history is the revised value, not what was known at the time.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.client
import io
import json
import shutil
import time
import urllib.error
import zipfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data.downloads.binance_public import http_get

Fetch = Callable[[str], bytes]
H = 3_600_000
DAY_MS = 24 * H
VISION = "https://data.binance.vision/data"
DERIBIT = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
STABLECOINS = "https://stablecoins.llama.fi/stablecoincharts/all"
COINMETRICS = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
SYMBOLS = ("BTCUSDT", "ETHUSDT")
BVOL = ("BTCBVOLUSDT", "ETHBVOLUSDT")
BVOL_LEFT_OUT = ("not downloaded (2026-11-17): it duplicates Deribit's DVOL, also a 30-day implied volatility index, "
                 "from 2021 and read from an API; its per-second daily archives start in 2023 and are the slowest "
                 "source on this host (about 1 GB at 150-250 KB/s in all)")
DVOL = ("BTC", "ETH")
FRED_LAG_DAYS = {"DTWEXBGS": 9, "DGS10": 2, "SP500": 2, "NASDAQCOM": 2, "VIXCLS": 2}   # release schedules unverified
CM_METRICS = ("FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "AdrActCnt", "TxCnt", "CapMrktCurUSD")
CM_ASSETS = ("btc", "eth")
SPOT_FIRST, PREMIUM_FIRST = "2017-08", "2020-01"
DVOL_FIRST, CM_FIRST = date(2021, 1, 1), date(2017, 1, 1)


def slow_fetch(url: str) -> bytes:
    """http_get with a timeout long enough for this host's link to data.binance.vision (about 150-250 KB/s in all,
    measured 2026-11-17: a 0.5 MB daily archive takes 3 s alone and much longer beside others)."""
    return http_get(url, timeout=120.0, retries=6)


def retrying(fetch: Fetch, *, attempts: int = 5, pause: float = 5.0) -> Fetch:
    """``fetch`` again when a read is cut off or the connection drops (http_get retries HTTP errors itself); an HTTP
    error, a 404 included, goes to the caller at once."""
    def get(url: str) -> bytes:
        for attempt in range(attempts):
            try:
                return fetch(url)
            except urllib.error.HTTPError:
                raise
            except (http.client.HTTPException, OSError):
                if attempt == attempts - 1:
                    raise
                time.sleep(pause * (attempt + 1))
        raise AssertionError("unreachable")
    return get


def staged(stage: Path, name: str, header: tuple, make: Callable[[], list]) -> list:
    """The rows of one source: read back from ``stage`` when an earlier attempt finished it, else made and kept
    there, so a failure later in a long download loses nothing already done."""
    path = Path(stage) / name
    if path.exists():
        with gzip.open(path, "rt", newline="") as handle:
            return [tuple(r) for r in list(csv.reader(handle))[1:]]
    rows = make()
    Path(stage).mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    tmp.replace(path)
    return rows


def ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp() * 1000)


def days(first: date, end: date) -> list[date]:
    """Every day from ``first`` up to ``end`` (exclusive)."""
    return [first + timedelta(k) for k in range((end - first).days)]


def months(first: str, end: date) -> list[str]:
    """Calendar months from ``first`` (YYYY-MM) through the month holding the day before ``end``."""
    year, month = (int(x) for x in first.split("-"))
    out = []
    while date(year, month, 1) < end:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def _day(t_ms: int) -> str:
    return datetime.fromtimestamp(t_ms / 1000, timezone.utc).date().isoformat()


# -- data.binance.vision ----------------------------------------------------------------------------------------
def vision_rows(url: str, *, fetch: Fetch = http_get) -> list[list[str]] | None:
    """Rows of one archive, the header dropped; None when the archive does not exist. The zip must match its
    published checksum."""
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


def _to_ms(t: str) -> int:
    value = int(t)
    return value // 1000 if value > 10**14 else value          # spot archives use microseconds from 2025


def spot_klines(symbol: str, month_list: Sequence[str], end_ms: int, *, fetch: Fetch = http_get) -> list[tuple]:
    """(open time ms, open, high, low, close, base volume, quote volume, trades, taker-buy quote volume)."""
    out = []
    for month in month_list:
        url = f"{VISION}/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
        for r in vision_rows(url, fetch=fetch) or []:
            t = _to_ms(r[0])
            if t + H <= end_ms:
                out.append((t, r[1], r[2], r[3], r[4], r[5], r[7], r[8], r[10]))
    return sorted(set(out))


def premium_klines(symbol: str, month_list: Sequence[str], end_ms: int, *, fetch: Fetch = http_get) -> list[tuple]:
    """(open time ms, open, high, low, close) of the premium index."""
    out = []
    for month in month_list:
        url = f"{VISION}/futures/um/monthly/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{month}.zip"
        for r in vision_rows(url, fetch=fetch) or []:
            t = _to_ms(r[0])
            if t + H <= end_ms:
                out.append((t, r[1], r[2], r[3], r[4]))
    return sorted(set(out))


def depth_url(symbol: str, day: date) -> str:
    return f"{VISION}/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{day.isoformat()}.zip"


def bvol_url(symbol: str, day: date) -> str:
    return f"{VISION}/option/daily/BVOLIndex/{symbol}/{symbol}-BVOLIndex-{day.isoformat()}.zip"


def _parallel(fn: Callable[[date], list], day_list: Sequence[date], workers: int) -> list:
    if workers <= 1:
        return [row for d in day_list for row in fn(d)]
    with ThreadPoolExecutor(workers) as pool:
        return [row for part in pool.map(fn, day_list) for row in part]


def _depth_day(rows: list[list[str]], end_ms: int) -> list[tuple]:
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=["timestamp", "percentage", "depth", "notional"])
    t = pd.to_datetime(frame["timestamp"], format="%Y-%m-%d %H:%M:%S", utc=True).astype("int64") // 10**6
    frame = frame.assign(hour=t - t % H, percentage=frame["percentage"].astype(float).astype(int),
                         notional=frame["notional"].astype(float))
    frame = frame[frame["hour"] + H <= end_ms]
    g = frame.groupby(["hour", "percentage"], sort=True)["notional"]
    out = pd.DataFrame({"mean": g.mean(), "last": g.last()}).reset_index()
    return [(int(h), int(p), float(m), float(v)) for h, p, m, v in out.itertuples(index=False)]


def book_depth(symbol: str, day_list: Sequence[date], end_ms: int, *, fetch: Fetch = http_get,
               workers: int = 1, cache: Path | None = None) -> list[tuple]:
    """(hour open ms, percentage, mean notional, last notional) of the snapshots taken in that hour; a negative
    percentage is the bid side. With ``cache``, every finished day is kept there and not fetched again."""
    def one(day: date) -> list[tuple]:
        if cache is not None:
            return [tuple(r) for r in staged(Path(cache), f"{symbol}_{day.isoformat()}.csv.gz",
                                             ("hour", "percentage", "mean", "last"),
                                             lambda: _depth_day(vision_rows(depth_url(symbol, day), fetch=fetch)
                                                                or [], end_ms))]
        return _depth_day(vision_rows(depth_url(symbol, day), fetch=fetch) or [], end_ms)
    rows = _parallel(one, day_list, workers)
    return sorted((int(h), int(p), float(m), float(v)) for h, p, m, v in rows)


def bvol(symbol: str, day_list: Sequence[date], end_ms: int, *, fetch: Fetch = http_get,
         workers: int = 1) -> list[tuple]:
    """(hour open ms, last index value in that hour)."""
    def one(day: date) -> list[tuple]:
        last: dict[int, tuple[int, float]] = {}
        for r in vision_rows(bvol_url(symbol, day), fetch=fetch) or []:
            t = int(r[0])
            hour = t - t % H
            if hour + H <= end_ms and (hour not in last or t >= last[hour][0]):
                last[hour] = (t, float(r[4]))
        return [(hour, value) for hour, (_, value) in last.items()]
    return sorted(_parallel(one, day_list, workers))


def first_day(url_for: Callable[[date], str], lo: date, hi: date, *, fetch: Fetch = http_get) -> date | None:
    """The first day in [lo, hi] whose daily archive exists (its checksum file answers), assuming every later day
    has one; None when ``hi`` has none."""
    def exists(day: date) -> bool:
        try:
            fetch(url_for(day) + ".CHECKSUM")
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise
    if not exists(hi):
        return None
    while lo < hi:
        mid = lo + timedelta((hi - lo).days // 2)
        if exists(mid):
            hi = mid
        else:
            lo = mid + timedelta(1)
    return lo


# -- other public sources ---------------------------------------------------------------------------------------
def dvol(currency: str, start_ms: int, end_ms: int, *, fetch: Fetch = http_get, window_days: int = 30) -> list[tuple]:
    """(open time ms, open, high, low, close) of Deribit's hourly DVOL candles in [start_ms, end_ms)."""
    found: dict[int, tuple] = {}
    a = start_ms
    while a < end_ms:
        b = min(a + window_days * DAY_MS, end_ms)
        stop = b
        while True:
            url = f"{DERIBIT}?currency={currency}&start_timestamp={a}&end_timestamp={stop}&resolution=3600"
            result = json.loads(fetch(url))["result"]
            for t, o, h, low, c in result["data"]:
                if start_ms <= int(t) and int(t) + H <= end_ms:
                    found[int(t)] = (int(t), o, h, low, c)
            if not result.get("continuation") or int(result["continuation"]) <= a:
                break
            stop = int(result["continuation"])
        a = b
    return [found[t] for t in sorted(found)]


def fred_url(series: str) -> str:
    return f"{FRED}?id={series}"


def fred(series: str, end_ms: int, *, fetch: Fetch = http_get) -> list[tuple]:
    """(date, value) of a FRED daily series, dates before the end's day; missing values ('.') left out."""
    end_day = _day(end_ms)
    rows = list(csv.reader(io.StringIO(fetch(fred_url(series)).decode())))[1:]
    return [(d, float(v)) for d, v in rows if v not in (".", "") and d < end_day]


def stablecoins(end_ms: int, *, fetch: Fetch = http_get) -> list[tuple]:
    """(date, total stablecoin supply pegged to USD) before the end."""
    out = []
    for row in json.loads(fetch(STABLECOINS)):
        t = int(row["date"]) * 1000
        usd = row.get("totalCirculatingUSD", {}).get("peggedUSD")
        if t < end_ms and usd is not None:
            out.append((_day(t), float(usd)))
    return out


def coinmetrics_url(asset: str, metrics: Sequence[str], first: date, end: date) -> str:
    return (f"{COINMETRICS}?assets={asset}&metrics={','.join(metrics)}&frequency=1d&start_time={first.isoformat()}"
            f"&end_time={end.isoformat()}&end_inclusive=false&page_size=10000")


def coinmetrics(asset: str, metrics: Sequence[str], first: date, end_ms: int, *,
                fetch: Fetch = http_get) -> list[tuple]:
    """(date, value of each metric, '' when missing) of Coin Metrics' community API, days before the end."""
    end_day = _day(end_ms)
    url: str | None = coinmetrics_url(asset, metrics, first, date.fromisoformat(end_day))
    out = []
    while url:
        page = json.loads(fetch(url))
        for row in page.get("data", []):
            day = row["time"][:10]
            if day < end_day:
                out.append((day, *(row.get(m, "") for m in metrics)))
        url = page.get("next_page_url")
    return out


# -- the snapshot ---------------------------------------------------------------------------------------------
def write_snapshot(out: Path, files: dict[str, tuple[tuple, Iterable[tuple]]], end_ms: int, *,
                   sources: dict[str, dict], extra: dict | None = None) -> dict:
    """Write each file as a deterministic gzip CSV and ``manifest.json``; a non-empty directory is refused."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"snapshots are immutable: {out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    listed = {}
    for name, (header, rows) in files.items():
        buffer, count = io.StringIO(), 0
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            count += 1
        path = out / name
        with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
            stream.write(buffer.getvalue().encode())
        listed[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": count,
                        **sources.get(name, {})}
    manifest = {"end_exclusive": datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(), "files": listed,
                **(extra or {})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return json.loads((out / "manifest.json").read_text())


def download(out: Path, end: date, *, fetch: Fetch = slow_fetch, workers: int = 4) -> dict:
    """Every source, each kept in ``<out>.partial`` as soon as it is done (a rerun after a failure resumes there),
    then the snapshot and its manifest; the staging directory is removed at the end."""
    end_ms = ms(end)
    fetch = retrying(fetch)
    stage = Path(str(out) + ".partial")
    files: dict[str, tuple[tuple, Any]] = {}
    sources: dict[str, dict] = {}
    started = datetime.now(timezone.utc).isoformat()

    def add(name: str, header: tuple, make: Callable[[], list], meta: dict) -> None:
        files[name] = (header, staged(stage, name, header, make))
        sources[name] = meta
        print(name, len(files[name][1]), "rows", flush=True)
    last = end - timedelta(1)
    for symbol in SYMBOLS:
        add(f"{symbol}_spot_1h.csv.gz", ("open_time", "open", "high", "low", "close", "volume", "quote_volume",
                                         "trades", "taker_buy_quote_volume"),
            lambda s=symbol: spot_klines(s, months(SPOT_FIRST, end), end_ms, fetch=fetch),
            {"source": f"{VISION}/spot/monthly/klines/{symbol}/1h", "known": "open_time + 1 h"})
        add(f"{symbol}_premium_1h.csv.gz", ("open_time", "open", "high", "low", "close"),
            lambda s=symbol: premium_klines(s, months(PREMIUM_FIRST, end), end_ms, fetch=fetch),
            {"source": f"{VISION}/futures/um/monthly/premiumIndexKlines/{symbol}/1h", "known": "open_time + 1 h"})
        first = first_day(lambda d, s=symbol: depth_url(s, d), date(2020, 1, 1), last, fetch=fetch)
        add(f"{symbol}_depth_1h.csv.gz", ("hour", "percentage", "mean_notional", "last_notional"),
            lambda s=symbol, f=first: book_depth(s, days(f, end), end_ms, fetch=fetch, workers=workers,
                                                 cache=stage / "depth_days") if f else [],
            {"source": f"{VISION}/futures/um/daily/bookDepth/{symbol}",
             "first_day": first.isoformat() if first else None,
             "known": "hour + 1 h (snapshots taken in [hour, hour + 1 h))"})
    for currency in DVOL:
        add(f"{currency}_dvol_1h.csv.gz", ("open_time", "open", "high", "low", "close"),
            lambda c=currency: dvol(c, ms(DVOL_FIRST), end_ms, fetch=fetch),
            {"source": f"{DERIBIT} (resolution 3600)", "known": "open_time + 1 h"})
    for series, lag in FRED_LAG_DAYS.items():
        add(f"fred_{series}.csv.gz", ("date", "value"), lambda x=series: fred(x, end_ms, fetch=fetch),
            {"source": fred_url(series), "known": f"date + {lag} days 00:00 UTC (release schedule unverified)"})
    add("stablecoins_usd.csv.gz", ("date", "total_usd"), lambda: stablecoins(end_ms, fetch=fetch),
        {"source": STABLECOINS, "known": "date + 1 day 00:00 UTC", "caveat": "revised after first publication"})
    for asset in CM_ASSETS:
        add(f"coinmetrics_{asset}.csv.gz", ("date", *CM_METRICS),
            lambda a=asset: coinmetrics(a, CM_METRICS, CM_FIRST, end_ms, fetch=fetch),
            {"source": COINMETRICS, "known": "date + 2 days 00:00 UTC",
             "caveat": "flash values are revised: history is the revised value"})
    extra = {"downloaded_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
             "purpose": "feature-pool study (owner 2026-11-17)", "terms": "each source's terms of use unverified",
             "not_downloaded": {"BVOL": BVOL_LEFT_OUT}}
    manifest = write_snapshot(out, files, end_ms, sources=sources, extra=extra)
    shutil.rmtree(stage)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="first day NOT written (UTC)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    manifest = download(args.out, args.end, workers=args.workers)
    print(json.dumps({k: {"rows": v["rows"]} for k, v in manifest["files"].items()}, indent=1))


if __name__ == "__main__":
    main()
