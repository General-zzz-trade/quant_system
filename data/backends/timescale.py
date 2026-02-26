"""TimescaleDB storage backend implementing BarStore and TickStore."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from data.backends.base import Tick
from data.store import Bar

logger = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None  # type: ignore[assignment]
    execute_values = None  # type: ignore[assignment]

_CREATE_BARS = """
CREATE TABLE IF NOT EXISTS bars (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    open        NUMERIC     NOT NULL,
    high        NUMERIC     NOT NULL,
    low         NUMERIC     NOT NULL,
    close       NUMERIC     NOT NULL,
    volume      NUMERIC,
    exchange    TEXT        NOT NULL DEFAULT ''
);
"""

_CREATE_TICKS = """
CREATE TABLE IF NOT EXISTS ticks (
    ts          TIMESTAMPTZ NOT NULL,
    symbol      TEXT        NOT NULL,
    price       NUMERIC     NOT NULL,
    qty         NUMERIC     NOT NULL,
    side        TEXT        NOT NULL,
    trade_id    TEXT        NOT NULL DEFAULT ''
);
"""

_HYPERTABLE_BARS = (
    "SELECT create_hypertable('bars', 'ts', if_not_exists => TRUE);"
)

_HYPERTABLE_TICKS = (
    "SELECT create_hypertable('ticks', 'ts', if_not_exists => TRUE);"
)


def _ensure_psycopg2() -> None:
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for TimescaleDB backend. "
            "Install it with: pip install psycopg2-binary"
        )


class TimescaleBarStore:
    """BarStore backed by TimescaleDB."""

    def __init__(self, dsn: str, *, create_tables: bool = True) -> None:
        _ensure_psycopg2()
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        if create_tables:
            self._init_tables()

    def _init_tables(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_BARS)
            try:
                cur.execute(_HYPERTABLE_BARS)
            except Exception:
                logger.debug("hypertable creation skipped (may already exist)")

    def write_bars(self, symbol: str, bars: Sequence[Bar]) -> None:
        if not bars:
            return
        sql = (
            "INSERT INTO bars (ts, symbol, open, high, low, close, volume, exchange) "
            "VALUES %s"
        )
        rows = [
            (
                b.ts,
                b.symbol,
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                float(b.volume) if b.volume is not None else None,
                b.exchange,
            )
            for b in bars
        ]
        with self._conn.cursor() as cur:
            execute_values(cur, sql, rows)
        logger.info("Inserted %d bars for %s", len(rows), symbol)

    def read_bars(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Bar]:
        clauses = ["symbol = %s"]
        params: list = [symbol]
        if start is not None:
            clauses.append("ts >= %s")
            params.append(start)
        if end is not None:
            clauses.append("ts <= %s")
            params.append(end)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT ts, symbol, open, high, low, close, volume, exchange "
            f"FROM bars WHERE {where} ORDER BY ts"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            Bar(
                ts=row[0] if row[0].tzinfo else row[0].replace(tzinfo=timezone.utc),
                symbol=row[1],
                open=Decimal(str(row[2])),
                high=Decimal(str(row[3])),
                low=Decimal(str(row[4])),
                close=Decimal(str(row[5])),
                volume=Decimal(str(row[6])) if row[6] is not None else None,
                exchange=row[7],
            )
            for row in rows
        ]

    def symbols(self) -> List[str]:
        with self._conn.cursor() as cur:
            cur.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol")
            return [row[0] for row in cur.fetchall()]

    def date_range(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(ts), MAX(ts) FROM bars WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return (row[0], row[1])

    def close(self) -> None:
        self._conn.close()


class TimescaleTickStore:
    """TickStore backed by TimescaleDB."""

    def __init__(self, dsn: str, *, create_tables: bool = True) -> None:
        _ensure_psycopg2()
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        if create_tables:
            self._init_tables()

    def _init_tables(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TICKS)
            try:
                cur.execute(_HYPERTABLE_TICKS)
            except Exception:
                logger.debug("hypertable creation skipped (may already exist)")

    def write_ticks(self, symbol: str, ticks: Sequence[Tick]) -> None:
        if not ticks:
            return
        sql = (
            "INSERT INTO ticks (ts, symbol, price, qty, side, trade_id) "
            "VALUES %s"
        )
        rows = [
            (
                t.ts,
                t.symbol,
                float(t.price),
                float(t.qty),
                t.side,
                t.trade_id,
            )
            for t in ticks
        ]
        with self._conn.cursor() as cur:
            execute_values(cur, sql, rows)
        logger.info("Inserted %d ticks for %s", len(rows), symbol)

    def read_ticks(
        self,
        symbol: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Tick]:
        clauses = ["symbol = %s"]
        params: list = [symbol]
        if start is not None:
            clauses.append("ts >= %s")
            params.append(start)
        if end is not None:
            clauses.append("ts <= %s")
            params.append(end)
        where = " AND ".join(clauses)
        sql = (
            f"SELECT ts, symbol, price, qty, side, trade_id "
            f"FROM ticks WHERE {where} ORDER BY ts"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            Tick(
                ts=row[0] if row[0].tzinfo else row[0].replace(tzinfo=timezone.utc),
                symbol=row[1],
                price=Decimal(str(row[2])),
                qty=Decimal(str(row[3])),
                side=row[4],
                trade_id=row[5],
            )
            for row in rows
        ]

    def count(self, symbol: str) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM ticks WHERE symbol = %s",
                (symbol,),
            )
            row = cur.fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()
