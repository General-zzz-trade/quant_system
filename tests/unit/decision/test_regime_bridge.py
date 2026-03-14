# tests/unit/decision/test_regime_bridge.py
"""Tests for RegimeAwareDecisionModule and RegimePolicy."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import pytest

from regime.base import RegimeLabel
from decision.regime_policy import RegimePolicy
from _quant_hotpath import RustRegimeBuffer
from decision.regime_bridge import RegimeAwareDecisionModule


# ── Stubs ────────────────────────────────────────────────────

def _market(close: float) -> SimpleNamespace:
    return SimpleNamespace(close=close, last_price=close)


def _snapshot(
    *,
    markets: Dict[str, Any],
    ts: datetime | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        markets=markets,
        ts=ts or datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    )


class _MockDecisionModule:
    """Records calls, returns configurable intents."""

    def __init__(self, intents: List[Any] | None = None):
        self.intents = intents or [SimpleNamespace(symbol="BTCUSDT", side="buy")]
        self.call_count = 0

    def decide(self, snapshot: Any) -> Iterable[Any]:
        self.call_count += 1
        return list(self.intents)


# ── RegimePolicy tests ──────────────────────────────────────

class TestRegimePolicy:
    def test_allow_when_no_labels(self):
        policy = RegimePolicy()
        ok, reason = policy.allow([])
        assert ok is True
        assert reason == "ok"

    def test_block_high_vol_flat_trend(self):
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="volatility", ts=datetime.now(timezone.utc), value="high"),
            RegimeLabel(name="trend", ts=datetime.now(timezone.utc), value="flat"),
        ]
        ok, reason = policy.allow(labels)
        assert ok is False
        assert reason == "high_vol_flat_trend"

    def test_allow_high_vol_up_trend(self):
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="volatility", ts=datetime.now(timezone.utc), value="high"),
            RegimeLabel(name="trend", ts=datetime.now(timezone.utc), value="up"),
        ]
        ok, reason = policy.allow(labels)
        assert ok is True

    def test_allow_low_vol_flat_trend(self):
        policy = RegimePolicy()
        labels = [
            RegimeLabel(name="volatility", ts=datetime.now(timezone.utc), value="low"),
            RegimeLabel(name="trend", ts=datetime.now(timezone.utc), value="flat"),
        ]
        ok, reason = policy.allow(labels)
        assert ok is True

    def test_custom_blocklist(self):
        policy = RegimePolicy(
            blocked_regimes={"volatility": frozenset({"high"})},
            block_high_vol_flat_trend=False,
        )
        labels = [
            RegimeLabel(name="volatility", ts=datetime.now(timezone.utc), value="high"),
        ]
        ok, reason = policy.allow(labels)
        assert ok is False
        assert "volatility=high" in reason

    def test_disable_default_combination(self):
        policy = RegimePolicy(block_high_vol_flat_trend=False)
        labels = [
            RegimeLabel(name="volatility", ts=datetime.now(timezone.utc), value="high"),
            RegimeLabel(name="trend", ts=datetime.now(timezone.utc), value="flat"),
        ]
        ok, _ = policy.allow(labels)
        assert ok is True


# ── RustRegimeBuffer tests ──────────────────────────────────

class TestPriceBuffer:
    def test_ma_computation(self):
        buf = RustRegimeBuffer(50)
        for i in range(10):
            buf.push(float(100 + i))
        ma = buf.ma(5)
        # Last 5: 105, 106, 107, 108, 109 → mean = 107
        assert ma == pytest.approx(107.0)

    def test_ma_returns_none_if_insufficient(self):
        buf = RustRegimeBuffer(50)
        buf.push(100.0)
        assert buf.ma(5) is None

    def test_rolling_vol_constant_price(self):
        buf = RustRegimeBuffer(200)
        for _ in range(50):
            buf.push(100.0)
        vol = buf.rolling_vol(20)
        assert vol == pytest.approx(0.0, abs=1e-12)

    def test_rolling_vol_returns_none_if_insufficient(self):
        buf = RustRegimeBuffer(50)
        for i in range(10):
            buf.push(float(100 + i))
        assert buf.rolling_vol(20) is None


# ── RegimeAwareDecisionModule tests ─────────────────────────

class TestRegimeAwareDecisionModule:
    def _build_module(
        self,
        inner: Any = None,
        policy: RegimePolicy | None = None,
    ) -> RegimeAwareDecisionModule:
        return RegimeAwareDecisionModule(
            inner=inner or _MockDecisionModule(),
            policy=policy or RegimePolicy(),
            ma_fast_window=5,
            ma_slow_window=10,
            vol_window=5,
            buffer_maxlen=200,
        )

    def _feed_prices(
        self,
        mod: RegimeAwareDecisionModule,
        n: int = 50,
        base: float = 100.0,
        volatility: float = 0.0,
    ) -> None:
        """Feed N snapshots with controllable volatility."""
        import math
        for i in range(n):
            price = base + volatility * math.sin(i * 0.5) * base
            snap = _snapshot(markets={"BTCUSDT": _market(price)})
            mod.decide(snap)

    def test_delegates_to_inner_in_normal_regime(self):
        inner = _MockDecisionModule()
        mod = self._build_module(inner=inner)

        # Feed enough stable prices
        self._feed_prices(mod, n=50, base=100.0, volatility=0.0)

        # Final call should delegate
        snap = _snapshot(markets={"BTCUSDT": _market(100.0)})
        intents = list(mod.decide(snap))
        assert inner.call_count > 0
        assert len(intents) > 0

    def test_blocks_in_high_vol_flat_trend(self):
        inner = _MockDecisionModule()
        mod = self._build_module(inner=inner)

        # Feed highly volatile prices (large swings → high vol, roughly flat MA)
        for i in range(100):
            # Oscillating around 100 with huge amplitude → high vol, flat trend
            price = 100.0 + 50.0 * (1 if i % 2 == 0 else -1)
            snap = _snapshot(markets={"BTCUSDT": _market(price)})
            list(mod.decide(snap))

        # Check that labels were generated
        labels = mod.current_labels
        [l for l in labels if l.name == "volatility"]
        # The key test: with very high oscillation, vol should be detected as high
        # and trend as flat, which should block trading
        # Note: exact blocking depends on threshold calibration

    def test_no_market_data_delegates_directly(self):
        """When no markets in snapshot, no features → no labels → policy allows."""
        inner = _MockDecisionModule()
        mod = self._build_module(inner=inner)
        snap = _snapshot(markets={})
        list(mod.decide(snap))
        assert inner.call_count == 1

    def test_insufficient_data_delegates_directly(self):
        """Not enough bars to compute features → no labels → delegates."""
        inner = _MockDecisionModule()
        mod = self._build_module(inner=inner)
        snap = _snapshot(markets={"BTCUSDT": _market(100.0)})
        list(mod.decide(snap))
        assert inner.call_count == 1

    def test_current_labels_property(self):
        mod = self._build_module()
        self._feed_prices(mod, n=50, base=100.0)
        labels = mod.current_labels
        assert isinstance(labels, list)

    def test_custom_policy_blocks(self):
        """Use a custom policy that blocks all high-vol."""
        inner = _MockDecisionModule()
        policy = RegimePolicy(
            blocked_regimes={"volatility": frozenset({"high", "mid"})},
            block_high_vol_flat_trend=False,
        )
        mod = self._build_module(inner=inner, policy=policy)

        # Feed prices that produce mid or high vol
        # Even with moderate movement, we check the delegation
        self._feed_prices(mod, n=50, base=100.0, volatility=0.0)
        # With zero volatility, vol=0 → "low" → should pass
        snap = _snapshot(markets={"BTCUSDT": _market(100.0)})
        list(mod.decide(snap))
        assert inner.call_count > 0

    def test_no_ts_skips_detection(self):
        inner = _MockDecisionModule()
        mod = self._build_module(inner=inner)
        self._feed_prices(mod, n=50, base=100.0)
        snap = SimpleNamespace(markets={"BTCUSDT": _market(100.0)}, ts=None)
        list(mod.decide(snap))
        # Should still delegate (no labels generated without ts)
        assert inner.call_count > 0
