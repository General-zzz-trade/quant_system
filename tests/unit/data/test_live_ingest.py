"""Tests for LiveDataIngestor.

Uses importlib to avoid conflict between tests/unit/data/ and data/ packages.
"""
import sys
import time
from pathlib import Path

# Register the data.ingest module properly
_root = str(Path(__file__).resolve().parent.parent.parent.parent)
_ingest_path = Path(_root) / "data" / "ingest" / "live_ingest.py"

import importlib.util
# Must register the module in sys.modules before exec_module
_spec = importlib.util.spec_from_file_location(
    "data_ingest_live_ingest", str(_ingest_path)
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["data_ingest_live_ingest"] = _mod
_spec.loader.exec_module(_mod)

LiveDataIngestor = _mod.LiveDataIngestor
CachedValue = _mod.CachedValue
DataSourceConfig = _mod.DataSourceConfig


class TestCachedValue:
    def test_fresh_value(self):
        cv = CachedValue(value=50.0, fetched_at=time.time(), ttl_sec=3600)
        assert not cv.is_stale
        assert cv.age_sec < 1.0

    def test_stale_value(self):
        cv = CachedValue(value=50.0, fetched_at=time.time() - 7200, ttl_sec=3600)
        assert cv.is_stale


class TestLiveDataIngestor:
    def test_get_fgi_empty(self):
        ingestor = LiveDataIngestor()
        assert ingestor.get_fgi() is None

    def test_set_and_get_value(self):
        ingestor = LiveDataIngestor()
        ingestor._set_value("fgi", 55.0, ttl_sec=3600)
        assert ingestor.get_fgi() == 55.0

    def test_stale_returns_none(self):
        ingestor = LiveDataIngestor()
        ingestor._set_value("fgi", 55.0, ttl_sec=0.0)
        assert ingestor.get_fgi() is None

    def test_funding_rate_callable(self):
        ingestor = LiveDataIngestor(symbols=("BTCUSDT",))
        ingestor._set_value("funding:BTCUSDT", 0.0001, ttl_sec=3600)
        getter = ingestor.get_funding_rate("BTCUSDT")
        assert callable(getter)
        assert getter() == 0.0001

    def test_basis_callable(self):
        ingestor = LiveDataIngestor(symbols=("BTCUSDT",))
        ingestor._set_value("basis:BTCUSDT", 0.005, ttl_sec=3600)
        getter = ingestor.get_basis("BTCUSDT")
        assert getter() == 0.005

    def test_get_all_status(self):
        ingestor = LiveDataIngestor()
        ingestor._set_value("fgi", 42.0, ttl_sec=3600)
        status = ingestor.get_all_status()
        assert "fgi" in status["sources"]
        assert status["sources"]["fgi"]["value"] == 42.0
        assert not status["sources"]["fgi"]["stale"]

    def test_prometheus_metrics(self):
        calls = []

        class MockProm:
            def set_gauge(self, name, value, labels=None):
                calls.append((name, value, labels))

        ingestor = LiveDataIngestor(prometheus=MockProm())
        ingestor._set_value("fgi", 55.0, ttl_sec=3600)
        assert any(c[0] == "external_data_age_seconds" for c in calls)
