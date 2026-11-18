"""DefiLlama's daily protocol fees and revenue for the alpha search (owner 2026-11-18: "执行" on the plan that downloads
them; a crypto "price to revenue" value factor, monthly, cross-sectional).

    python -m data.downloads.defillama_fees --out data_files/defillama_fees_2026-11-12 \\
        --perps data_files/binance_usdt_perps_2020_2026-11-12

Public, no key (``api.llama.fi``), one request a second:
- ``/protocols``: every protocol with its token symbol; a token is the base asset of a USDT perpetual in the
  perpetuals' snapshot ("1000PEPE" counts as PEPE). A child protocol is asked through its parent (whose series sums
  the versions); a symbol whose protocols carry different CoinGecko ids is a different coin each time and is left out
  (``ambiguous``). The matched protocols are saved as ``protocols.json`` with name, slug, symbol, CoinGecko id,
  category and parent; today's market cap and TVL are dropped (they are not as of past dates).
- ``/summary/fees/<slug>?dataType=dailyFees`` and ``dailyRevenue``: the daily series, ``<TOKEN>__<slug>_<kind>.csv``.
  A slug without a series (no adapter: 404 or another error) is listed under ``missing``, never guessed.

Rows on or after ``end`` (at most 2026-11-13, the forward guard) are dropped before anything is written. DefiLlama
recomputes its series; the values are as served on the fetch date. Snapshots are immutable; ``manifest.json`` records
the source, the time of the download, every file's rows, first and last date and SHA-256.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from data.downloads.binance_public import http_get
from data.downloads.us_equity_daily import FORWARD_GUARD, _now, _write

PROTOCOLS = "https://api.llama.fi/protocols"
SUMMARY = "https://api.llama.fi/summary/fees/{slug}?dataType={kind}"
KINDS = ("dailyFees", "dailyRevenue")
KEEP = ("name", "slug", "symbol", "gecko_id", "category", "parentProtocol")


def bases(perpetuals: Iterable[str]) -> set[str]:
    """The base assets of USDT perpetuals ("1000PEPEUSDT" -> "PEPE")."""
    return {re.sub(r"^1000+", "", s[:-4]) for s in perpetuals if s.endswith("USDT")}


def select(protocols: list[dict], wanted: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(token -> the slugs to ask, token -> the slugs of an ambiguous token). A token is a symbol in ``wanted`` (any
    case). A child protocol is asked through its parent, whose series sums its versions. A symbol whose protocols
    carry two or more CoinGecko ids belongs to different coins: it is left out, since the perpetual's coin is not
    known."""
    by: dict[str, list[dict]] = {}
    for p in protocols:
        symbol = str(p.get("symbol") or "").upper()
        if symbol in wanted and p.get("slug"):
            by.setdefault(symbol, []).append(p)
    targets, ambiguous = {}, {}
    for symbol, group in sorted(by.items()):
        coins = {p["gecko_id"] for p in group if p.get("gecko_id")}
        if len(coins) > 1:
            ambiguous[symbol] = sorted(p["slug"] for p in group)
            continue
        targets[symbol] = sorted({str(p["parentProtocol"]).removeprefix("parent#") if p.get("parentProtocol")
                                  else p["slug"] for p in group})
    return targets, ambiguous


def parse_chart(payload: bytes, end: date) -> tuple[list[dict], int]:
    """A summary's ``totalDataChart`` ([seconds, value], ...): (rows before ``end``, oldest first; how many after)."""
    rows, dropped = [], 0
    for ts, value in json.loads(payload).get("totalDataChart") or []:
        day = datetime.fromtimestamp(int(ts), timezone.utc).date()
        if day >= end:
            dropped += 1
            continue
        rows.append({"date": day.isoformat(), "value": str(value)})
    rows.sort(key=lambda r: r["date"])
    return rows, dropped


def snapshot(out: Path, *, bases: set[str], end: date = FORWARD_GUARD, fetch: Callable[[str], bytes] | None = None,
             pause: float = 1.0, now: Callable[[], str] = _now) -> dict[str, Any]:
    if end > FORWARD_GUARD:
        raise ValueError(f"end {end} is after the forward guard 2026-11-13: forward data is read only at registered "
                         "looks")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty: snapshots are immutable")
    get = fetch or (lambda url: http_get(url, retries=2, timeout=60))
    out.mkdir(parents=True, exist_ok=True)
    listed = json.loads(get(PROTOCOLS))
    targets, ambiguous = select(listed, bases)
    kept = [{k: p.get(k) for k in KEEP} for p in listed
            if str(p.get("symbol") or "").upper() in targets and p.get("slug")]
    (out / "protocols.json").write_text(json.dumps(sorted(kept, key=lambda p: p["slug"]), indent=1, sort_keys=True)
                                        + "\n")
    time.sleep(pause)
    files: dict[str, dict] = {}
    missing: dict[str, str] = {}
    for symbol, slugs in targets.items():
        for slug in slugs:
            for kind in KINDS:
                name = f"{symbol}__{slug}_{kind}"
                try:
                    rows, dropped = parse_chart(get(SUMMARY.format(slug=slug, kind=kind)), end)
                except Exception as exc:             # no adapter for this protocol: recorded, not guessed
                    missing[name] = type(exc).__name__
                    rows, dropped = [], 0
                time.sleep(pause)
                if rows:
                    files[f"{name}.csv"] = {**_write(out / f"{name}.csv", rows), "dropped_on_or_after_end": dropped}
                elif name not in missing:
                    missing[name] = "empty series"
    manifest = {"what": "DefiLlama daily fees and revenue of protocols whose token has a USDT perpetual",
                "sources": {"protocols": PROTOCOLS, "summary": SUMMARY}, "fetched_at": now(),
                "end_exclusive": end.isoformat(), "tokens": len(targets), "files": files, "missing": missing,
                "ambiguous": ambiguous,
                "notes": ["values as served on the fetch date: DefiLlama recomputes its series",
                          "today's market cap and TVL are not kept: they are not as of past dates"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perps", type=Path, required=True, help="the perpetuals' snapshot (its manifest.json)")
    parser.add_argument("--end", type=date.fromisoformat, default=FORWARD_GUARD)
    args = parser.parse_args(argv)
    contracts = json.loads((args.perps / "manifest.json").read_text())["contracts"]
    wanted = bases(c["symbol"] for c in contracts if c.get("type") == "perpetual")
    m = snapshot(args.out, bases=wanted, end=args.end)
    print(f"{m['tokens']} tokens, {len(m['files'])} series written to {args.out}; missing {len(m['missing'])}, "
          f"ambiguous {len(m['ambiguous'])}")


if __name__ == "__main__":
    main()


__all__ = ["KINDS", "bases", "parse_chart", "select", "snapshot"]
