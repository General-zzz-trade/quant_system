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
        ts_ms = raw.get("time") or raw.get("start") or 0
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
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
            logger.warning("Warmup bar conversion failed at ts=%s: %s", ts_ms, exc)

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
    bars = adapter.get_klines(symbol, interval=interval, limit=limit)

    # Bybit returns newest first -- reverse for chronological order
    bars = list(reversed(bars))

    # Data quality validation (log-only, never blocks warmup)
    try:
        validate_warmup_bars(bars, symbol, interval)
    except Exception:
        logger.exception("Warmup data quality check failed (non-fatal)")

    count = 0
    for bar in bars:
        ts_ms = bar.get("time") or bar.get("start") or 0
        ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)

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

    logger.info("Warmup complete: %s %d bars", symbol, count)
    return count
