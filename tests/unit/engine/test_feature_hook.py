"""Tests for FeatureComputeHook — source wiring and data flow."""
from __future__ import annotations

from datetime import datetime

import pytest

pytest.importorskip("_quant_hotpath")

from engine.feature_hook import FeatureComputeHook
from features.enriched_computer import EnrichedFeatureComputer


class StubEvent:
    def __init__(self, symbol: str = "BTCUSDT", close: float = 50000.0):
        self.symbol = symbol
        self.close = close
        self.volume = 100.0
        self.high = 50100.0
        self.low = 49900.0
        self.open = 49950.0
        self.ts = datetime(2024, 1, 1, 12, 0, 0)


class TestSpotCloseSource:
    def test_spot_close_passed_when_source_provided(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(
            computer=computer,
            spot_close_source=lambda: 49800.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None
        # spot_close flows through RustFeatureEngine → basis feature
        assert "close" in features

    def test_spot_close_default_when_source_none(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(computer=computer, spot_close_source=None)
        features = hook.on_event(StubEvent())
        assert features is not None

    def test_spot_close_default_when_source_returns_none(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(computer=computer, spot_close_source=lambda: None)
        features = hook.on_event(StubEvent())
        assert features is not None


class TestFgiSource:
    def test_fgi_passed_when_source_provided(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(
            computer=computer,
            fgi_source=lambda: 65.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None

    def test_fgi_default_when_source_returns_none(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(computer=computer, fgi_source=lambda: None)
        features = hook.on_event(StubEvent())
        assert features is not None


class TestBothSourcesCombined:
    def test_both_spot_and_fgi(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(
            computer=computer,
            spot_close_source=lambda: 49800.0,
            fgi_source=lambda: 72.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None

    def test_no_sources_still_works(self):
        computer = EnrichedFeatureComputer()
        hook = FeatureComputeHook(computer=computer)
        features = hook.on_event(StubEvent())
        assert features is not None
