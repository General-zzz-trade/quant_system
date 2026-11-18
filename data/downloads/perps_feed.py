"""The daily prediction report's data (owner 2026-11-18: "先做BTC和ETH，直接就在此对话推送"): a rolling store of every
crypto USDT perpetual's hourly bars and funding, laid out like a snapshot (``<symbol>_1h.csv.gz``,
``<symbol>_funding.csv.gz``) so the frozen panel code reads it unchanged.

    python -m data.downloads.perps_feed seed --snapshot data_files/binance_usdt_perps_2020_2026-11-12 --feed <dir>
    python -m data.downloads.perps_feed update --feed <dir>

- ``seed`` copies a snapshot's rows from ``start`` on (the features need about 400 days of history) into a new
  directory; TradFi and stablecoin contracts are left out.
- ``update`` appends, from Binance's public endpoints (no key, nothing signed), the bars that have closed and the
  funding settled since each contract's last row; contracts listed since are added from their first bar.
  Contracts: ``status`` TRADING, ``contractType`` PERPETUAL, quote USDT (TradFi contracts are TRADIFI_PERPETUAL).
  A contract that fails is recorded and left as it was; the report decides whether the day is usable.
- This store is not a research snapshot: it changes every day and is never used to select anything. The data it
  holds from 2026-11-13 on is read for the report's predictions only (no outcome, no performance).
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from data.downloads.binance_public import FAPI, fetch_funding, fetch_klines, http_get
from data.downloads.binance_um_all import COLUMNS, _wait_for_funding_slot

H = 3_600_000
BAR_HEADER = ",".join(COLUMNS)
FUNDING_HEADER = "funding_time,rate"
EXCHANGE_INFO = f"{FAPI}/fapi/v1/exchangeInfo"
STATE = "state.json"
Fetch = Callable[[str], bytes]


def _read(path: Path) -> list[str]:
    return gzip.decompress(path.read_bytes()).decode().splitlines()[1:] if path.exists() else []


def _write(path: Path, header: str, rows: Iterable[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(gzip.compress((header + "\n" + "".join(r + "\n" for r in rows)).encode()))
    tmp.replace(path)


def _last_time(rows: list[str]) -> int | None:
    return int(rows[-1].split(",", 1)[0]) if rows else None


def seed(snapshot: Path, feed: Path, *, start_ms: int, exclude: Iterable[str]) -> dict:
    """A new feed holding ``snapshot``'s rows from ``start_ms`` on; refuses a directory that is not empty."""
    snapshot, feed = Path(snapshot), Path(feed)
    if feed.exists() and any(feed.iterdir()):
        raise ValueError(f"{feed} is not empty: a feed is seeded once")
    feed.mkdir(parents=True, exist_ok=True)
    skip, symbols = set(exclude), []
    for path in sorted(snapshot.glob("*_1h.csv.gz")):
        symbol = path.name[:-len("_1h.csv.gz")]
        bars = [r for r in _read(path) if int(r.split(",", 1)[0]) >= start_ms]
        if symbol in skip or not bars:
            continue
        _write(feed / path.name, BAR_HEADER, bars)
        funding = [r for r in _read(snapshot / f"{symbol}_funding.csv.gz") if int(r.split(",", 1)[0]) >= start_ms]
        _write(feed / f"{symbol}_funding.csv.gz", FUNDING_HEADER, funding)
        symbols.append(symbol)
    state = {"seeded_from": snapshot.name, "start_ms": start_ms, "symbols_seeded": len(symbols)}
    (feed / STATE).write_text(json.dumps(state, indent=1) + "\n")
    return {"symbols": symbols, **state}


def contracts(info: dict) -> tuple[dict[str, dict], list[str]]:
    """The crypto USDT perpetuals trading now ({symbol: entry}) and the TradFi ones (names)."""
    crypto = {e["symbol"]: e for e in info["symbols"] if e.get("status") == "TRADING"
              and e.get("contractType") == "PERPETUAL" and e.get("quoteAsset") == "USDT"}
    tradfi = sorted(e["symbol"] for e in info["symbols"] if e.get("contractType") == "TRADIFI_PERPETUAL")
    return crypto, tradfi


def _rules(entry: dict) -> dict:
    filters = {f["filterType"]: f for f in entry.get("filters", [])}
    return {"step": filters.get("LOT_SIZE", {}).get("stepSize"),
            "min_notional": filters.get("MIN_NOTIONAL", {}).get("notional")}


def update(feed: Path, *, now_ms: int | None = None, fetch: Fetch = http_get,
           wait: Callable[[], None] = _wait_for_funding_slot) -> dict:
    """Append every trading crypto perpetual's closed bars and settled funding since its last row."""
    feed = Path(feed)
    state = json.loads((feed / STATE).read_text())
    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    crypto, tradfi = contracts(json.loads(fetch(EXCHANGE_INFO)))
    failed, new, updated = [], [], 0
    for symbol in sorted(crypto):
        bars_path, funding_path = feed / f"{symbol}_1h.csv.gz", feed / f"{symbol}_funding.csv.gz"
        try:
            bars, funding = _read(bars_path), _read(funding_path)
            last = _last_time(bars)
            if last is None:
                onboard = int(crypto[symbol].get("onboardDate") or state["start_ms"])
                start = max(int(state["start_ms"]), onboard // H * H)
            else:
                start = last + H
            added = [",".join(str(v) for v in row) for row in fetch_klines(symbol, start, now_ms, fetch=fetch,
                                                                           now_ms=now_ms)]
            last_funding = _last_time(funding)
            wait()
            events = fetch_funding(symbol, (start - 1 if last_funding is None else last_funding) + 1, now_ms,
                                   fetch=fetch)
        except Exception:                                            # noqa: BLE001 - recorded, the rest go on
            failed.append(symbol)
            continue
        if last is None:
            new.append(symbol)
        if added or last is None:
            _write(bars_path, BAR_HEADER, bars + added)
        if events or last_funding is None:
            _write(funding_path, FUNDING_HEADER, funding + [f"{e['fundingTime']},{e['fundingRate']}" for e in events])
        updated += 1
    out = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "now_ms": now_ms, "updated": updated,
           "failed": failed, "new": new, "tradfi": tradfi, "trading": sorted(crypto),
           "rules": {s: _rules(e) for s, e in crypto.items()}}
    state["last_update"] = out
    (feed / STATE).write_text(json.dumps(state, indent=1) + "\n")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--snapshot", type=Path, required=True)
    s.add_argument("--feed", type=Path, required=True)
    s.add_argument("--start", default="2024-11-01", help="first day kept (UTC)")
    u = sub.add_parser("update")
    u.add_argument("--feed", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "seed":
        from lab import um_judge
        tradfi = json.loads(Path("research/lab/runs/2026-11-17b/tradfi_contracts.json").read_text())["symbols"]
        start = int(datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc).timestamp() * 1000)
        out = seed(args.snapshot, args.feed, start_ms=start, exclude=set(tradfi) | set(um_judge.STABLE))
        print(json.dumps({k: v for k, v in out.items() if k != "symbols"}))
    else:
        out = update(args.feed)
        print(json.dumps({k: out[k] for k in ("at", "updated", "failed", "new")}))


if __name__ == "__main__":
    main()
