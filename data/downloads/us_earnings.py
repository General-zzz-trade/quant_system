"""Nasdaq's earnings calendar for the alpha search (owner 2026-11-18: "执行" on the plan that downloads it).

    python -m data.downloads.us_earnings --out data_files/us_earnings_2026-11-12 --start 2025-12-01

One public request a day of the window (``api.nasdaq.com/api/calendar/earnings?date=``, the data behind nasdaq.com's
earnings calendar; no account, no key), one a second. Every row is kept as served: symbol, name, the reported and the
forecast EPS, the surprise, the fiscal quarter, the market cap and the number of estimates. Facts of the source,
checked 2026-11-18:
- ``time`` (before the open or after the close) reads "time-not-supplied" for past days, so the report's hour is not
  known from this source;
- the values are as served on the fetch date: a later revision of a reported EPS would not show.

``end`` is exclusive and at most 2026-11-13 (the forward guard). Snapshots are immutable; ``manifest.json`` records
the source, the time of the download, the days asked, the rows and the file's SHA-256.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data.downloads.binance_public import http_get
from data.downloads.us_equity_daily import FORWARD_GUARD, _now, _write

CALENDAR = "https://api.nasdaq.com/api/calendar/earnings?date={day}"
FIELDS = ("symbol", "name", "time", "eps", "epsForecast", "surprise", "fiscalQuarterEnding", "marketCap", "noOfEsts")


def parse(payload: bytes, day: date) -> list[dict]:
    """One day's rows, each with its date; a day without reports (a weekend, a holiday) has none."""
    data = json.loads(payload).get("data") or {}
    rows = data.get("rows") or []
    return [{"date": day.isoformat(), **{k: str(r.get(k) or "").strip() for k in FIELDS}} for r in rows]


def snapshot(out: Path, *, start: date, end: date = FORWARD_GUARD, fetch: Callable[[str], bytes] | None = None,
             pause: float = 1.0, now: Callable[[], str] = _now) -> dict[str, Any]:
    if end > FORWARD_GUARD:
        raise ValueError(f"end {end} is after the forward guard 2026-11-13: forward data is read only at registered "
                         "looks")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: snapshots are immutable")
    get = fetch or (lambda url: http_get(url, retries=3, timeout=60))
    out.mkdir(parents=True, exist_ok=True)
    rows, days, with_rows = [], 0, 0
    day = start
    while day < end:
        found = parse(get(CALENDAR.format(day=day.isoformat())), day)
        rows += found
        days += 1
        with_rows += bool(found)
        time.sleep(pause)
        day += timedelta(days=1)
    entry = _write(out / "earnings.csv", rows) if rows else {"rows": 0}
    manifest = {"what": "Nasdaq earnings calendar, one request a day", "source": CALENDAR, "fetched_at": now(),
                "start": start.isoformat(), "end_exclusive": end.isoformat(), "days": days, "days_with_rows": with_rows,
                "rows": len(rows), "file": entry,
                "notes": ["time is 'time-not-supplied' for past days: the report's hour is not in this source",
                          "values as served on the fetch date; later revisions would not show"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=FORWARD_GUARD)
    args = parser.parse_args(argv)
    m = snapshot(args.out, start=args.start, end=args.end)
    print(f"{m['rows']} rows over {m['days']} days written to {args.out}")


if __name__ == "__main__":
    main()


__all__ = ["CALENDAR", "FIELDS", "parse", "snapshot"]
