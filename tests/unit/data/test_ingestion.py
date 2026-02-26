"""Tests for IngestionOrchestrator — lifecycle, backfill, and health reporting."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock

import pytest

from data.ingestion.config import IngestionConfig, IngestionSymbolConfig
from data.ingestion.orchestrator import IngestionOrchestrator


class FakeCollector:
    """Minimal collector for testing the orchestrator."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._running = False
        self._last_active: Optional[datetime] = None

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self._running = True
        self._last_active = datetime.now(timezone.utc)

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def last_active_ts(self) -> datetime | None:
        return self._last_active


def _make_config(
    *, backfill_on_start: bool = False, symbols: tuple[str, ...] = ("BTCUSDT",)
) -> IngestionConfig:
    return IngestionConfig(
        symbols=tuple(IngestionSymbolConfig(symbol=s) for s in symbols),
        backfill_on_start=backfill_on_start,
        backfill_days=7,
        health_check_interval_sec=600.0,
    )


class TestIngestionOrchestrator:
    def test_start_stop_lifecycle(self) -> None:
        c1 = FakeCollector("ws-btc")
        c2 = FakeCollector("ws-eth")
        config = _make_config()
        orch = IngestionOrchestrator(
            config=config, bar_store=MagicMock(), collectors=[c1, c2]
        )
        assert orch.running is False

        orch.start()
        assert orch.running is True
        assert c1.is_running is True
        assert c2.is_running is True

        orch.stop()
        assert orch.running is False
        assert c1.is_running is False
        assert c2.is_running is False

    def test_double_start_noop(self) -> None:
        config = _make_config()
        orch = IngestionOrchestrator(
            config=config, bar_store=MagicMock(), collectors=[]
        )
        orch.start()
        orch.start()  # should not raise
        assert orch.running is True
        orch.stop()

    def test_health_report_structure(self) -> None:
        c1 = FakeCollector("ws-btc")
        c1.start()  # mark as running
        config = _make_config()
        orch = IngestionOrchestrator(
            config=config, bar_store=MagicMock(), collectors=[c1]
        )
        orch.start()

        report = orch.health_report()
        assert report["orchestrator_running"] is True
        assert report["collector_count"] == 1
        assert len(report["collectors"]) == 1

        cr = report["collectors"][0]
        assert cr["name"] == "ws-btc"
        assert cr["is_running"] is True
        assert cr["last_active_ts"] is not None
        assert isinstance(cr["lag_seconds"], float)

        orch.stop()

    def test_health_report_stopped_collector(self) -> None:
        c1 = FakeCollector("ws-btc")
        config = _make_config()
        orch = IngestionOrchestrator(
            config=config, bar_store=MagicMock(), collectors=[c1]
        )
        report = orch.health_report()
        cr = report["collectors"][0]
        assert cr["is_running"] is False
        assert cr["last_active_ts"] is None
        assert cr["lag_seconds"] is None

    def test_backfill_on_start(self) -> None:
        backfiller = MagicMock()
        backfiller.backfill.return_value = 100
        config = _make_config(backfill_on_start=True, symbols=("BTCUSDT", "ETHUSDT"))
        orch = IngestionOrchestrator(
            config=config,
            bar_store=MagicMock(),
            backfiller=backfiller,
            collectors=[],
        )
        orch.start()
        assert backfiller.backfill.call_count == 2
        called_symbols = {
            call.args[0] for call in backfiller.backfill.call_args_list
        }
        assert called_symbols == {"BTCUSDT", "ETHUSDT"}
        orch.stop()

    def test_no_backfill_when_disabled(self) -> None:
        backfiller = MagicMock()
        config = _make_config(backfill_on_start=False)
        orch = IngestionOrchestrator(
            config=config,
            bar_store=MagicMock(),
            backfiller=backfiller,
            collectors=[],
        )
        orch.start()
        backfiller.backfill.assert_not_called()
        orch.stop()
