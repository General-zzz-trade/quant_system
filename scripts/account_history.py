"""Read-only export of the account's own records, for checking the backtest against reality (tech lead 2026-11-16).

Run through the account check, the one command that reads the credentials file:

    python3 -m scripts.account_check --history <private directory> [--days 89]

It asks three GET endpoints, all on the account check's allowlist: the income history (funding, commissions,
realized PnL, transfers), and, for every symbol that history names, the account's trades and its orders. Requests
cover at most a week each and are followed page by page. The raw records go to the directory (mode 700, files 600)
and nowhere else; the screen shows counts only. Nothing is placed, cancelled or changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

FAPI = "https://fapi.binance.com"
ENDPOINTS = ("/fapi/v1/income", "/fapi/v1/userTrades", "/fapi/v1/allOrders")
DAY = 86_400_000
WINDOW = 7 * DAY
PAGE = 1000
MAX_DAYS = 89          # Binance keeps three months of income history; older rows would be missing without a word


def _paged(client: Any, path: str, lo: int, hi: int, key: Callable[[dict], Any], **params: Any) -> tuple[list, bool]:
    """Every row of ``path`` with lo <= time <= hi, in windows of a week, each followed until a page is short.
    Returns the rows (once each, by ``key``) and whether every page came back in ascending time."""
    rows: dict[Any, dict] = {}
    ascending = True
    start = lo
    while start <= hi:
        end = min(start + WINDOW, hi)
        cursor = start
        while True:
            page = client.get(FAPI, path, startTime=cursor, endTime=end, limit=PAGE, **params)
            if not isinstance(page, list):
                raise RuntimeError(f"{path}: not a list")
            times = [int(r["time"]) for r in page]
            ascending = ascending and times == sorted(times)
            for r in page:
                rows[key(r)] = r
            if len(page) < PAGE:
                break
            if max(times) <= cursor:
                raise RuntimeError(f"{path}: a full page does not advance past {cursor}")
            cursor = max(times)
        start = end + 1
    return sorted(rows.values(), key=lambda r: int(r["time"])), ascending


def _write(directory: Path, name: str, value: Any) -> None:
    fd = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(value, handle)


def export(client: Any, directory: Path, *, days: int, now_ms: int) -> dict[str, Any]:
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_DAYS} (what Binance keeps)")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    lo, hi = now_ms - days * DAY, now_ms
    income, ascending = _paged(client, ENDPOINTS[0], lo, hi,
                               lambda r: (r.get("tranId"), r.get("incomeType"), r.get("asset"), r.get("symbol")))
    _write(directory, "income.json", income)
    symbols = sorted({str(r.get("symbol")) for r in income if r.get("symbol")})
    trades = orders = 0
    for symbol in symbols:
        rows, _ = _paged(client, ENDPOINTS[1], lo, hi, lambda r: r["id"], symbol=symbol)
        _write(directory, f"trades_{symbol}.json", rows)
        trades += len(rows)
        rows, _ = _paged(client, ENDPOINTS[2], lo, hi, lambda r: r["orderId"], symbol=symbol)
        _write(directory, f"orders_{symbol}.json", rows)
        orders += len(rows)
    kinds: dict[str, int] = {}
    for r in income:
        kinds[str(r.get("incomeType"))] = kinds.get(str(r.get("incomeType")), 0) + 1
    summary = {"from_ms": lo, "to_ms": hi, "days": days, "income_rows": len(income), "income_kinds": kinds,
               "income_ascending": ascending, "symbols": symbols, "trades": trades, "orders": orders}
    _write(directory, "summary.json", summary)
    return summary


def lines(summary: dict[str, Any]) -> list[str]:
    return [f"days: {summary['days']}", f"income rows: {summary['income_rows']}",
            "income kinds: " + ", ".join(f"{k} {v}" for k, v in sorted(summary["income_kinds"].items())),
            f"income pages in ascending time: {summary['income_ascending']}",
            "symbols: " + ", ".join(summary["symbols"]), f"trades: {summary['trades']}", f"orders: {summary['orders']}"]


__all__ = ["ENDPOINTS", "MAX_DAYS", "export", "lines"]
