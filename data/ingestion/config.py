"""Ingestion configuration — per-symbol and global settings."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IngestionSymbolConfig:
    """Per-symbol collection configuration."""

    symbol: str
    collect_bars: bool = True
    collect_ticks: bool = False
    collect_funding: bool = True
    bar_interval: str = "1h"


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Global ingestion configuration."""

    symbols: tuple[IngestionSymbolConfig, ...]
    backfill_on_start: bool = True
    backfill_days: int = 30
    health_check_interval_sec: float = 60.0
