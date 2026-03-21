from __future__ import annotations

from execution.store.event_log import SQLiteEventLog


def test_sqlite_event_log_append_and_iter(tmp_path) -> None:
    db = tmp_path / "ev.sqlite"
    log = SQLiteEventLog(path=str(db))

    id1 = log.append(event_type="A", payload={"x": 1}, correlation_id="c")
    id2 = log.append(event_type="B", payload={"x": 2}, correlation_id="c")

    rows = list(log.iter(after_id=0))
    assert [r["id"] for r in rows] == [id1, id2]
    assert rows[0]["event_type"] == "A"
    assert rows[1]["payload"]["x"] == 2

    rows2 = list(log.iter(after_id=id1))
    assert len(rows2) == 1
    assert rows2[0]["id"] == id2


def test_sqlite_event_log_context_manager_closes_connection(tmp_path) -> None:
    db = tmp_path / "ev.sqlite"

    with SQLiteEventLog(path=str(db)) as log:
        log.append(event_type="A", payload={"x": 1})
        assert log._closed is False

    assert log._closed is True
