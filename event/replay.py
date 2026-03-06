# event/replay.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, List, Sequence

from event.clock import EventClock
from event.errors import EventReplayError
from event.factory.market import MarketEventFactory
from event.runtime import EventRuntime
from event.security import RunMode, make_actor


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    ts: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    symbol: str
    source: str = "system.replay"
    strict_ordering: bool = True
    emit_bar_events: bool = True


class EventReplay:
    def __init__(
        self,
        *,
        runtime: EventRuntime,
        clock: EventClock,
        cfg: ReplayConfig,
    ) -> None:
        self._runtime = runtime
        self._clock = clock
        self._cfg = cfg
        self._actor = make_actor(
            module="event.replay",
            roles={"replay"},
            mode=RunMode.REPLAY,
            source=cfg.source,
        )

    def replay_bars(self, bars: Iterable[HistoricalBar]) -> None:
        last_ts: datetime | None = None

        for bar in bars:
            if bar.ts.tzinfo is None:
                raise EventReplayError("bar.ts must be tz-aware datetime")

            ts = bar.ts.astimezone(timezone.utc)

            if self._cfg.strict_ordering and last_ts is not None and ts <= last_ts:
                raise EventReplayError(f"bar out of order: {ts} <= {last_ts}")
            last_ts = ts

            self._clock.update_from_event_time(event_time=ts, bar_index=self._clock.bar_index + 1)

            if self._cfg.emit_bar_events:
                ev = MarketEventFactory.bar(
                    ts=ts,
                    symbol=self._cfg.symbol,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    source=self._cfg.source,
                )
                self._runtime.emit(ev, actor=self._actor)


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------
# Older modules may import EventReplayer. Keep an alias to avoid breaking
# existing code while standardizing on the shorter name.
EventReplayer = EventReplay



def bars_from_ohlcv(
    rows: Sequence[Sequence[object]],
    *,
    ts_index: int = 0,
    open_index: int = 1,
    high_index: int = 2,
    low_index: int = 3,
    close_index: int = 4,
    volume_index: int = 5,
) -> List[HistoricalBar]:
    bars: List[HistoricalBar] = []
    for r in rows:
        ts = r[ts_index]
        if not isinstance(ts, datetime):
            raise ValueError("rows[*][ts_index] must be datetime")
        if ts.tzinfo is None:
            raise ValueError("ts must be tz-aware datetime")

        bars.append(
            HistoricalBar(
                ts=ts.astimezone(timezone.utc),
                open=Decimal(str(r[open_index])),
                high=Decimal(str(r[high_index])),
                low=Decimal(str(r[low_index])),
                close=Decimal(str(r[close_index])),
                volume=Decimal(str(r[volume_index])),
            )
        )
    return bars
