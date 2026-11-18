"""data/downloads/market_recorder.py: liquidations and best bid/ask snapshots, the public data the archive lacks
(owner 2026-11-18, "执行"). No network: fake connections, fetches and clocks.

- Rows: a stream message's liquidations; a snapshot's USDT perpetuals.
- Files: one per UTC day of receipt; an earlier day is closed (compressed, hashed, listed) and never rewritten.
- Gaps: every span the stream was not listening (disconnected, silent, recorder down) and every failed snapshot; the
  disk guard writes nothing and records a gap.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import threading

import pytest
import websocket

from data.downloads import market_recorder as mr

DAY = mr.DAY
T0 = 1_795_000_000_000 - 1_795_000_000_000 % DAY        # a UTC midnight
SAMPLE = ('{"e":"forceOrder","E":1795021277177,"o":{"s":"ZECUSDT","S":"BUY","o":"LIMIT","f":"IOC","q":"1.791",'
          '"p":"938.81","ap":"929.55","X":"FILLED","l":"0.467","z":"1.791","T":1795021276173,"ps":"ZECUSDT","st":1}}')


def _csv(path):
    with path.open() as handle:
        return list(csv.reader(handle))


# -- rows -------------------------------------------------------------------------------------------------------
def test_a_stream_message_becomes_its_liquidation_rows():
    row = mr.liquidation_rows(SAMPLE)
    assert row == [(1795021277177, 1795021276173, "ZECUSDT", "BUY", "LIMIT", "IOC", "1.791", "938.81", "929.55",
                    "FILLED", "0.467", "1.791")]
    assert len(row[0]) == len(mr.LIQUIDATION_COLUMNS)
    assert mr.liquidation_rows(f"[{SAMPLE},{SAMPLE}]") == row * 2
    assert mr.liquidation_rows('{"e":"other"}') == [] and mr.liquidation_rows("[1]") == []


def test_a_snapshot_keeps_the_usdt_perpetuals():
    payload = [{"symbol": s, "bidPrice": "1", "bidQty": "2", "askPrice": "3", "askQty": "4", "time": 5}
               for s in ("BTCUSDT", "BTCUSDT_261225", "ETHUSDC", "ZECUSDT")]
    assert mr.ticker_rows(payload, 7) == [(7, "BTCUSDT", "1", "2", "3", "4", 5), (7, "ZECUSDT", "1", "2", "3", "4", 5)]


# -- files ------------------------------------------------------------------------------------------------------
def test_a_day_is_closed_when_the_next_begins_and_never_rewritten(tmp_path):
    files = mr.DayFiles(tmp_path, "liquidations", ("a", "b"), free_gb=lambda: 100.0)
    assert files.append([(1, "x"), (2, "y")], T0 + 5)
    assert files.append([(3, "z")], T0 + DAY + 5)                             # day two closes day one
    folder = tmp_path / "liquidations"
    one, two = folder / f"{mr._day(T0)}.csv.gz", folder / f"{mr._day(T0 + DAY)}.csv"
    assert not (tmp_path / "liquidations" / f"{mr._day(T0)}.csv").exists()
    assert gzip.decompress(one.read_bytes()).decode().splitlines() == ["a,b", "1,x", "2,y"]
    assert _csv(two) == [["a", "b"], ["3", "z"]]
    listed = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert listed == [{"file": f"liquidations/{one.name}", "sha256": hashlib.sha256(one.read_bytes()).hexdigest(),
                       "rows": 2, "closed_at": listed[0]["closed_at"]}]
    before = one.read_bytes()
    assert files.append([(4, "late")], T0 + 7)                                # a row for the closed day
    assert one.read_bytes() == before
    assert _csv(tmp_path / "liquidations" / f"{mr._day(T0)}.part2.csv") == [["a", "b"], ["4", "late"]]


def test_nothing_is_written_while_the_disk_is_short(tmp_path):
    files = mr.DayFiles(tmp_path, "book_ticker", ("a",), free_gb=lambda: mr.MIN_FREE_GB - 0.1)
    assert files.append([(1,)], T0) is False
    assert list((tmp_path / "book_ticker").iterdir()) == []


# -- the liquidation stream -------------------------------------------------------------------------------------
class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


class Connection:
    """Plays a script: a message (str), a timeout (None, the clock moves ``step`` ms) or an exception."""

    def __init__(self, script, clock, stop, step):
        self.script, self.clock, self.stop, self.step, self.closed = list(script), clock, stop, step, False

    def settimeout(self, s):
        pass

    def recv(self):
        if not self.script:
            self.stop.set()
            raise websocket.WebSocketTimeoutException("end of script")
        item = self.script.pop(0)
        if item is None:
            self.clock.t += self.step
            raise websocket.WebSocketTimeoutException("timeout")
        if isinstance(item, Exception):
            raise item
        self.clock.t += 1000
        return item

    def close(self):
        self.closed = True


def _stream(tmp_path, scripts, *, step=30_000, fail_first=0):
    clock, stop = Clock(T0), threading.Event()
    files = mr.DayFiles(tmp_path, "liquidations", mr.LIQUIDATION_COLUMNS, free_gb=lambda: 100.0)
    gaps, status = mr.Gaps(tmp_path), mr.Status(tmp_path, clock)
    connections, attempts, urls = [], [0], []

    def connect(url):
        urls.append(url)
        attempts[0] += 1
        clock.t += 2000
        if url != mr.LIQUIDATIONS_URL or len(connections) >= len(scripts) or attempts[0] > 20:
            stop.set()                                         # a wrong address or an unplanned reconnect ends it
            raise OSError("not in the script")
        if attempts[0] <= fail_first:
            raise OSError("refused")
        c = Connection(scripts[len(connections)], clock, stop, step)
        connections.append(c)
        return c
    mr.record_liquidations(files, gaps, status, stop, connect=connect, clock=clock, sleep=lambda s: None)
    rows = (_csv(tmp_path / "gaps.csv")[1:] if (tmp_path / "gaps.csv").exists() else [])
    assert urls[0] == "wss://fstream.binance.com/market/ws/!forceOrder@arr"     # /ws/ connects but sends nothing
    return files, rows, connections, status


def test_a_disconnect_is_a_gap_from_the_disconnect_to_the_reconnect(tmp_path):
    files, gaps, connections, status = _stream(tmp_path, [[SAMPLE, ConnectionResetError("reset")], [SAMPLE]])
    written = _csv(tmp_path / "liquidations" / f"{mr._day(T0)}.csv")
    assert len(written) == 3 and written[1][2] == "ZECUSDT"
    starting, reset = gaps
    assert starting[0] == "liquidations" and starting[3] == "starting"              # start to the first connect
    assert reset[3].startswith("disconnected: ConnectionResetError")
    assert int(reset[2]) - int(reset[1]) == 2000                                     # disconnect to reconnect
    assert all(c.closed for c in connections) and status.data["liquidations_last_message"] > T0


def test_a_silent_stream_is_reconnected_and_the_silence_is_a_gap(tmp_path):
    quiet = [SAMPLE] + [None] * (mr.SILENT_S // 30)                                  # 30 s timeouts until 15 min
    _, gaps, connections, _ = _stream(tmp_path, [quiet, [SAMPLE]])
    silent = gaps[-1]
    assert silent[3].startswith("no liquidation for 900 s") and len(connections) == 2
    assert int(silent[2]) - int(silent[1]) >= mr.SILENT_S * 1000                     # from the last message


def test_a_failed_connect_is_retried_and_counted_in_the_gap(tmp_path):
    _, gaps, connections, _ = _stream(tmp_path, [[SAMPLE]], fail_first=2)
    assert len(connections) == 1 and gaps[0][3] == "connect failed: OSError: refused"
    assert int(gaps[0][2]) - int(gaps[0][1]) == 6000                                 # three attempts of 2 s each


# -- snapshots and restarts -------------------------------------------------------------------------------------
def test_a_snapshot_is_written_at_each_boundary_and_a_failed_one_is_a_gap(tmp_path):
    clock, stop = Clock(T0 + 10), threading.Event()
    files = mr.DayFiles(tmp_path, "book_ticker", mr.TICKER_COLUMNS, free_gb=lambda: 100.0)
    gaps, status = mr.Gaps(tmp_path), mr.Status(tmp_path, clock)
    calls = []

    def fetch(url):
        calls.append(clock.t)
        if len(calls) == 2:
            raise TimeoutError("slow")
        if len(calls) == 3:
            stop.set()
        return json.dumps([{"symbol": "BTCUSDT", "bidPrice": "1", "bidQty": "2", "askPrice": "3", "askQty": "4",
                            "time": 5}]).encode()

    real_wait = stop.wait

    def wait(seconds):
        clock.t += int(seconds * 1000)
        return real_wait(0)
    stop.wait = wait
    mr.record_tickers(files, gaps, status, stop, fetch=fetch, clock=clock)
    step = mr.EVERY_S * 1000
    assert calls == [T0 + step, T0 + 2 * step, T0 + 3 * step]                        # on the 5-minute boundaries
    rows = _csv(tmp_path / "book_ticker" / f"{mr._day(T0)}.csv")
    assert [r[0] for r in rows[1:]] == [str(T0 + step), str(T0 + 3 * step)]
    assert _csv(tmp_path / "gaps.csv")[1] == ["book_ticker", str(T0 + 2 * step), str(T0 + 2 * step),
                                              "snapshot failed: TimeoutError: slow"]
    assert status.data["book_ticker_last_snapshot"] == T0 + 3 * step


def test_the_time_the_recorder_was_down_is_a_gap_at_the_next_start(tmp_path):
    (tmp_path / "status.json").write_text(json.dumps({"last_alive": T0}))
    stop = threading.Event()
    stop.set()
    mr.run(tmp_path, stop, clock=lambda: T0 + 3 * 3_600_000)
    assert _csv(tmp_path / "gaps.csv")[1] == ["all", str(T0), str(T0 + 3 * 3_600_000), "recorder not running"]
    assert json.loads((tmp_path / "status.json").read_text())["last_alive"] == T0 + 3 * 3_600_000


@pytest.mark.parametrize("bad", ["", "{}", "null"])
def test_an_unreadable_status_starts_without_a_gap(tmp_path, bad):
    (tmp_path / "status.json").write_text(bad)
    assert mr.Status(tmp_path).previous_alive() is None
