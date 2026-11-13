"""Warmup logic for alpha_main: fetch historical bars and push through coordinator."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from data.quality.validators import BarValidator
from data.quality.gaps import GapDetector
from engine.coordinator import EngineCoordinator
from event.header import EventHeader
from event.types import EventType, MarketEvent

logger = logging.getLogger(__name__)

HOUR_MS = 3_600_000


def bar_open_ms(bar: dict) -> int:
    """Open time in ms. Adapters put ms in ``start`` and SECONDS in ``time``.

    A ``time`` above 1e11 is already milliseconds (1e11 s is year 5138; 1e11 ms
    is 1973), so both conventions are recognized without ambiguity.
    """
    if bar.get("start"):
        return int(bar["start"])
    if bar.get("time"):
        value = int(bar["time"])
        return value if value > 100_000_000_000 else value * 1000
    raise ValueError("bar has no open time")


def closed_chronological(bars: list[dict], interval: str, *, now_ms: int | None = None) -> list[dict]:
    """Oldest-first, de-duplicated, closed bars only — regardless of adapter ordering."""
    import time as _time
    step = int(interval) * 60_000 if str(interval).isdigit() else HOUR_MS
    now_ms = int(_time.time() * 1000) if now_ms is None else now_ms
    unique = {}
    for bar in bars:
        start = bar_open_ms(bar)
        if bar.get("confirm") is False or start + step > now_ms:
            continue  # an unclosed bar must never enter warmup
        unique[start] = bar
    return [unique[t] for t in sorted(unique)]


def validate_warmup_bars(
    bars: list[dict],
    symbol: str,
    interval: str,
) -> None:
    """Run data quality checks on warmup bars and log results.

    Converts adapter dicts to Bar objects for the validators, then logs
    any errors/warnings. Never blocks warmup -- quality issues are warnings only.
    """
    from data.store import Bar as StoreBar

    if not bars:
        return

    interval_seconds = 14400 if interval == "240" else 3600

    # Convert adapter dicts to Bar objects
    store_bars: list[StoreBar] = []
    for raw in bars:
        ts = datetime.fromtimestamp(bar_open_ms(raw) / 1000, tz=timezone.utc)
        try:
            store_bars.append(StoreBar(
                ts=ts,
                open=Decimal(str(raw["open"])),
                high=Decimal(str(raw["high"])),
                low=Decimal(str(raw["low"])),
                close=Decimal(str(raw["close"])),
                volume=Decimal(str(raw["volume"])),
                symbol=symbol,
            ))
        except (KeyError, ValueError) as exc:
            logger.warning("Warmup bar conversion failed at ts=%s: %s", ts, exc)

    if not store_bars:
        return

    # Bar validation (OHLC consistency, time continuity, anomaly detection)
    validator = BarValidator(
        zscore_threshold=5.0,
        max_gap_seconds=interval_seconds * 2,
    )
    result = validator.validate(store_bars)

    if not result.valid:
        logger.warning(
            "Warmup data quality ERRORS for %s (interval=%s): %d errors -- %s",
            symbol, interval, len(result.errors),
            "; ".join(result.errors[:5]),
        )
    if result.warnings:
        logger.info(
            "Warmup data quality warnings for %s (interval=%s): %d warnings -- %s",
            symbol, interval, len(result.warnings),
            "; ".join(result.warnings[:5]),
        )

    # Gap detection
    gap_detector = GapDetector(interval_seconds=interval_seconds)
    gap_report = gap_detector.detect(
        store_bars,
        start=store_bars[0].ts,
        end=store_bars[-1].ts,
    )
    if gap_report.gaps:
        logger.warning(
            "Warmup gap report for %s (interval=%s): %d gaps, %.1f%% complete",
            symbol, interval, len(gap_report.gaps), gap_report.completeness_pct,
        )
    else:
        logger.info(
            "Warmup quality OK for %s (interval=%s): %d bars, 0 gaps, %d anomalies",
            symbol, interval, result.stats["total_bars"], result.stats["anomalies"],
        )


def warmup(
    coordinator: EngineCoordinator,
    adapter: Any,
    symbol: str,
    interval: str,
    limit: int,
) -> int:
    """Fetch historical bars and push through the coordinator.

    Returns the number of bars processed.
    """
    logger.info("Warmup: fetching %d bars for %s interval=%s", limit, symbol, interval)
    # One extra row: some venues include the still-open bar, which is dropped.
    bars = adapter.get_klines(symbol, interval=interval, limit=limit + 1)

    # Adapters differ (Bybit newest-first, OKX/Binance oldest-first): normalize.
    bars = closed_chronological(bars, interval)[-limit:]

    # Data quality validation (log-only, never blocks warmup)
    try:
        validate_warmup_bars(bars, symbol, interval)
    except Exception:
        logger.exception("Warmup data quality check failed (non-fatal)")

    # Detach execution bridge during warmup to prevent real orders from
    # historical bar replay. Reattach after warmup completes.
    saved_exec_bridge = getattr(coordinator, "_execution_bridge", None)
    if saved_exec_bridge is not None:
        coordinator._execution_bridge = None

    count = 0
    for bar in bars:
        ts = datetime.fromtimestamp(bar_open_ms(bar) / 1000, tz=timezone.utc)

        header = EventHeader.new_root(
            event_type=EventType.MARKET,
            version=1,
            source="warmup",
        )
        event = MarketEvent(
            header=header,
            ts=ts,
            symbol=symbol,
            open=Decimal(str(bar["open"])),
            high=Decimal(str(bar["high"])),
            low=Decimal(str(bar["low"])),
            close=Decimal(str(bar["close"])),
            volume=Decimal(str(bar["volume"])),
        )
        coordinator.emit(event, actor="warmup")
        count += 1

    # Reattach execution bridge for live trading
    if saved_exec_bridge is not None:
        coordinator._execution_bridge = saved_exec_bridge

    logger.info("Warmup complete: %s %d bars", symbol, count)
    return count
