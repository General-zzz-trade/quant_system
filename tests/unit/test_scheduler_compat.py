from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if "data" in sys.modules and not hasattr(sys.modules["data"], "__path__"):
    sys.modules.pop("data", None)


def _load_scheduler_modules():
    data_scheduler = importlib.import_module("data.scheduler.data_scheduler")
    freshness_monitor = importlib.import_module("data.scheduler.freshness_monitor")
    return data_scheduler, freshness_monitor


def test_data_scheduler_builds_expected_job_set():
    data_scheduler, _ = _load_scheduler_modules()
    DataScheduler = data_scheduler.DataScheduler
    DataSchedulerConfig = data_scheduler.DataSchedulerConfig
    scheduler = DataScheduler(DataSchedulerConfig(symbols=("BTCUSDT",)))
    assert len(scheduler._jobs) == 7
    assert [job.name for job in scheduler._jobs] == [
        "bars",
        "funding",
        "ticks",
        "orderbook",
        "open_interest",
        "ls_ratio",
        "taker_flow",
    ]


def test_data_scheduler_start_stop_is_idempotent():
    data_scheduler, _ = _load_scheduler_modules()
    DataScheduler = data_scheduler.DataScheduler
    DataSchedulerConfig = data_scheduler.DataSchedulerConfig
    scheduler = DataScheduler(DataSchedulerConfig(symbols=("BTCUSDT",), interval_sec=0.01))
    scheduler.start()
    scheduler.start()
    assert scheduler.running is True
    scheduler.stop()
    scheduler.stop()
    assert scheduler.running is False


def test_freshness_monitor_start_stop_is_idempotent(tmp_path):
    _, freshness_monitor = _load_scheduler_modules()
    FreshnessConfig = freshness_monitor.FreshnessConfig
    FreshnessMonitor = freshness_monitor.FreshnessMonitor
    monitor = FreshnessMonitor(
        FreshnessConfig(
            data_dir=str(tmp_path),
            symbols=("BTCUSDT",),
            check_interval_sec=0.01,
        )
    )
    monitor.start()
    monitor.start()
    assert monitor.running is True
    monitor.stop()
    monitor.stop()
    assert monitor.running is False
