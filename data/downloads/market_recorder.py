"""Records the public USD-M data the archive lacks (owner 2026-11-18, "执行"): liquidations (the ``forceOrder`` stream;
data.binance.vision keeps none) and best bid/ask snapshots (its bookTicker archive stopped on 2024-03-30). Depth,
open interest, long/short ratios and trades are archived and not recorded here. Public endpoints only: no key, no
account.

    python -m data.downloads.market_recorder --root /root/lab_cache/market_recorder

- Files: ``<root>/liquidations/<day>.csv`` and ``<root>/book_ticker/<day>.csv``, one per UTC day of receipt,
  appended row by row. When a later day starts, an earlier day's file is closed: compressed to ``.csv.gz``, its
  SHA-256 and row count appended to ``<root>/manifest.jsonl``, and never written again (a row arriving for a closed
  day goes to a new part file).
- ``<root>/gaps.csv``: every span the liquidation stream was not listening (not connected, connected but silent for
  ``SILENT_S``, the recorder not running) and every ticker snapshot that failed or was skipped. Research treats those
  spans as missing, not as quiet. The stream's address is ``/market/ws/``: the old ``/ws/`` route still accepts a
  connection but sends nothing (checked 2026-11-18), which only the silence rule would catch.
- Writing pauses while the disk has less than ``MIN_FREE_GB`` free (the trading runtime shares the disk); the pause
  is a gap. ``<root>/status.json`` shows the last message and snapshot times.
- Data from 2026-11-13 on is forward data: research reads it only in a study registered before it looks
  (lab/forward.py, research/protocols/).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import signal
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import websocket

from data.downloads.binance_um_all import usdt_perpetual

LIQUIDATIONS_URL = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
LIQUIDATION_COLUMNS = ("event_time", "trade_time", "symbol", "side", "order_type", "time_in_force", "qty", "price",
                       "avg_price", "status", "last_filled_qty", "filled_qty")
TICKER_COLUMNS = ("snapshot_time", "symbol", "bid", "bid_qty", "ask", "ask_qty", "exchange_time")
GAP_COLUMNS = ("stream", "start_ms", "end_ms", "reason")
EVERY_S = 300                 # one best bid/ask snapshot of every USDT perpetual each 5 minutes
SILENT_S = 900                # a connected stream with no liquidation for 15 minutes is reconnected and counted a gap
MIN_FREE_GB = 5.0
DAY = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


def _day(t: int) -> str:
    return datetime.fromtimestamp(t / 1000, timezone.utc).date().isoformat()


def liquidation_rows(message: str) -> list[tuple]:
    """The liquidations of one stream message (one event, or a list of them)."""
    data = json.loads(message)
    rows = []
    for e in data if isinstance(data, list) else [data]:
        if not isinstance(e, dict) or e.get("e") != "forceOrder":
            continue
        o = e["o"]
        rows.append((int(e["E"]), int(o["T"]), o["s"], o["S"], o["o"], o["f"], o["q"], o["p"], o["ap"], o["X"],
                     o["l"], o["z"]))
    return rows


def ticker_rows(payload: Sequence[dict], snapshot: int) -> list[tuple]:
    return [(snapshot, r["symbol"], r["bidPrice"], r["bidQty"], r["askPrice"], r["askQty"], int(r["time"]))
            for r in payload if usdt_perpetual(r["symbol"])]


class DayFiles:
    """One CSV per UTC day of receipt; earlier days are closed (compressed, hashed, listed) and never reopened."""

    def __init__(self, root: Path, kind: str, columns: Sequence[str], *,
                 free_gb: Callable[[], float] | None = None) -> None:
        self.root, self.dir, self.columns = Path(root), Path(root) / kind, tuple(columns)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.free_gb = free_gb or (lambda: shutil.disk_usage(self.root).free / 1e9)
        self.lock = threading.Lock()

    def append(self, rows: Sequence[tuple], at: int) -> bool:
        """False when the disk is too full to write (nothing written)."""
        if self.free_gb() < MIN_FREE_GB:
            return False
        day = _day(at)
        with self.lock:
            self.close_before(day)
            path = self.dir / f"{day}.csv"
            if (self.dir / f"{day}.csv.gz").exists() and not path.exists():
                path = self._part(day)
            new = not path.exists()
            with path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(self.columns)
                writer.writerows(rows)
        return True

    def _part(self, day: str) -> Path:
        k = 2
        while (self.dir / f"{day}.part{k}.csv.gz").exists() or (self.dir / f"{day}.part{k}.csv").exists():
            k += 1
        return self.dir / f"{day}.part{k}.csv"

    def close_before(self, day: str) -> None:
        for path in sorted(self.dir.glob("*.csv")):
            if path.name[:10] >= day:
                continue
            target = path.with_name(path.name + ".gz")
            if target.exists():
                raise FileExistsError(f"{target} exists: a closed day is never rewritten")
            raw = path.read_bytes()
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(gzip.compress(raw, mtime=0))
            tmp.rename(target)
            record = {"file": str(target.relative_to(self.root)), "sha256": hashlib.sha256(target.read_bytes())
                      .hexdigest(), "rows": max(raw.count(b"\n") - 1, 0), "closed_at": now_ms()}
            with (self.root / "manifest.jsonl").open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            path.unlink()


class Gaps:
    def __init__(self, root: Path) -> None:
        self.path, self.lock = Path(root) / "gaps.csv", threading.Lock()

    def add(self, stream: str, start: int, end: int, reason: str) -> None:
        with self.lock:
            new = not self.path.exists()
            with self.path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(GAP_COLUMNS)
                writer.writerow((stream, int(start), int(end), reason[:200]))


class Status:
    def __init__(self, root: Path, clock: Callable[[], int] = now_ms) -> None:
        self.path, self.clock, self.lock = Path(root) / "status.json", clock, threading.Lock()
        self.data: dict[str, Any] = {}

    def previous_alive(self) -> int | None:
        try:
            return int(json.loads(self.path.read_text())["last_alive"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value

    def write(self) -> None:
        with self.lock:
            record = {**self.data, "last_alive": self.clock()}
        tmp = self.path.with_name("status.json.tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        tmp.rename(self.path)


def record_liquidations(files: DayFiles, gaps: Gaps, status: Status, stop: threading.Event, *,
                        connect: Callable[[str], Any] | None = None, clock: Callable[[], int] = now_ms,
                        sleep: Callable[[float], None] = time.sleep) -> None:
    """Listen until ``stop``; every span not listening becomes a gap (from the last message or the disconnect)."""
    connect = connect or (lambda url: websocket.create_connection(url, timeout=30))
    down_since, reason, backoff = clock(), "starting", 1.0
    while not stop.is_set():
        try:
            ws = connect(LIQUIDATIONS_URL)
        except Exception as exc:                                    # noqa: BLE001 - retried, recorded as a gap
            reason = f"connect failed: {type(exc).__name__}: {exc}"
            sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        connected = clock()
        if connected - down_since > 1000:
            gaps.add("liquidations", down_since, connected, reason)
        backoff, last = 1.0, connected
        try:
            ws.settimeout(30)
            while not stop.is_set():
                try:
                    message = ws.recv()
                except websocket.WebSocketTimeoutException:
                    if clock() - last >= SILENT_S * 1000:
                        down_since, reason = last, f"no liquidation for {SILENT_S} s: reconnected"
                        break
                    continue
                at = clock()
                rows = liquidation_rows(message)
                if rows and not files.append(rows, at):
                    gaps.add("liquidations", at, at, f"disk below {MIN_FREE_GB} GB: {len(rows)} rows not written")
                last = at
                status.set("liquidations_last_message", at)
            else:
                down_since, reason = clock(), "stopped"
        except Exception as exc:                                    # noqa: BLE001 - reconnected, recorded as a gap
            down_since, reason = clock(), f"disconnected: {type(exc).__name__}: {exc}"
        finally:
            try:
                ws.close()
            except Exception:                                       # noqa: BLE001
                pass


def record_tickers(files: DayFiles, gaps: Gaps, status: Status, stop: threading.Event, *,
                   fetch: Callable[[str], bytes] | None = None, clock: Callable[[], int] = now_ms) -> None:
    """A snapshot at every ``EVERY_S`` boundary of the clock until ``stop``; a failed one is a gap."""
    fetch = fetch or (lambda url: urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "research-recorder"}), timeout=30).read())
    while not stop.is_set():
        due = (clock() // (EVERY_S * 1000) + 1) * EVERY_S * 1000
        if stop.wait(max(0.0, (due - clock()) / 1000)):
            return
        try:
            written = files.append(ticker_rows(json.loads(fetch(TICKER_URL)), due), due)
        except Exception as exc:                                    # noqa: BLE001 - recorded as a gap
            gaps.add("book_ticker", due, due, f"snapshot failed: {type(exc).__name__}: {exc}")
            continue
        if not written:
            gaps.add("book_ticker", due, due, f"disk below {MIN_FREE_GB} GB")
            continue
        status.set("book_ticker_last_snapshot", due)


def run(root: Path, stop: threading.Event, *, clock: Callable[[], int] = now_ms) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    gaps, status = Gaps(root), Status(root, clock)
    before = status.previous_alive()
    if before is not None:
        gaps.add("all", before, clock(), "recorder not running")
    status.set("started_at", clock())
    liquidations = DayFiles(root, "liquidations", LIQUIDATION_COLUMNS)
    tickers = DayFiles(root, "book_ticker", TICKER_COLUMNS)
    threads = [threading.Thread(target=record_liquidations, args=(liquidations, gaps, status, stop), daemon=True),
               threading.Thread(target=record_tickers, args=(tickers, gaps, status, stop), daemon=True)]
    for t in threads:
        t.start()
    while not stop.is_set():
        status.set("disk_free_gb", round(shutil.disk_usage(root).free / 1e9, 1))
        status.write()
        stop.wait(60)
    status.write()
    for t in threads:
        t.join(timeout=40)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    run(args.root, stop)


if __name__ == "__main__":
    main()


__all__ = ["DayFiles", "Gaps", "Status", "liquidation_rows", "record_liquidations", "record_tickers", "run",
           "ticker_rows"]
