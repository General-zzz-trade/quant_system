"""Tests for FeatureComputeHook — spot_close, fear_greed, and data source wiring."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from engine.feature_hook import FeatureComputeHook


class StubComputer:
    """Minimal feature computer that records on_bar calls."""

    def __init__(self, accept_spot: bool = False, accept_fgi: bool = False):
        self.calls: list[dict] = []
        self._accept_spot = accept_spot
        self._accept_fgi = accept_fgi
        # Build the on_bar method dynamically so inspect.signature picks up the params
        params = ["self", "symbol", "close", "volume", "high", "low"]
        optional_params = []
        if accept_spot:
            optional_params.append("spot_close=None")
        if accept_fgi:
            optional_params.append("fear_greed=None")
        all_params = params + optional_params
        param_str = ", ".join(all_params)
        # Collect all non-self param names for the dict
        param_names = [p for p in params if p != "self"] + [
            p.split("=")[0] for p in optional_params
        ]
        assign_str = "kwargs = {" + ", ".join(
            f"'{p}': {p}" for p in param_names
        ) + "}"
        code = f"def on_bar({param_str}):\n    {assign_str}\n    self.calls.append(kwargs)"
        ns: Dict[str, Any] = {"self": self}
        exec(code, ns)  # noqa: S102
        import types
        self.on_bar = types.MethodType(ns["on_bar"], self)

    def get_features_dict(self, symbol: str) -> Dict[str, Any]:
        return {"feat_a": 1.0}


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
        computer = StubComputer(accept_spot=True)
        hook = FeatureComputeHook(
            computer=computer,
            spot_close_source=lambda: 49800.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None
        # Verify spot_close was passed to on_bar
        assert len(computer.calls) == 1
        assert computer.calls[0]["spot_close"] == 49800.0

    def test_spot_close_default_when_source_none(self):
        computer = StubComputer(accept_spot=True)
        hook = FeatureComputeHook(computer=computer, spot_close_source=None)
        features = hook.on_event(StubEvent())
        assert features is not None
        # spot_close not injected — computer gets its default (None)
        assert computer.calls[0]["spot_close"] is None

    def test_spot_close_default_when_source_returns_none(self):
        computer = StubComputer(accept_spot=True)
        hook = FeatureComputeHook(computer=computer, spot_close_source=lambda: None)
        features = hook.on_event(StubEvent())
        assert features is not None
        # Source returned None → not injected → computer gets its default
        assert computer.calls[0]["spot_close"] is None


class TestFgiSource:
    def test_fgi_passed_when_source_provided(self):
        computer = StubComputer(accept_fgi=True)
        hook = FeatureComputeHook(
            computer=computer,
            fgi_source=lambda: 65.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None
        assert len(computer.calls) == 1
        assert computer.calls[0]["fear_greed"] == 65.0

    def test_fgi_default_when_source_returns_none(self):
        computer = StubComputer(accept_fgi=True)
        hook = FeatureComputeHook(computer=computer, fgi_source=lambda: None)
        features = hook.on_event(StubEvent())
        assert features is not None
        assert computer.calls[0]["fear_greed"] is None


class TestBothSourcesCombined:
    def test_both_spot_and_fgi(self):
        computer = StubComputer(accept_spot=True, accept_fgi=True)
        hook = FeatureComputeHook(
            computer=computer,
            spot_close_source=lambda: 49800.0,
            fgi_source=lambda: 72.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None
        call = computer.calls[0]
        assert call["spot_close"] == 49800.0
        assert call["fear_greed"] == 72.0

    def test_no_sources_still_works(self):
        """Computer that doesn't accept spot_close/fgi should still work."""
        computer = StubComputer(accept_spot=False, accept_fgi=False)
        hook = FeatureComputeHook(
            computer=computer,
            spot_close_source=lambda: 49800.0,
            fgi_source=lambda: 72.0,
        )
        features = hook.on_event(StubEvent())
        assert features is not None
        # Computer doesn't accept these params → not in the call
        call = computer.calls[0]
        assert "spot_close" not in call
        assert "fear_greed" not in call
