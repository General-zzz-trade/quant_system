"""Order generation determinism test.

Runs 200 bars through MLSignalDecisionModule twice (fresh instances, same data).
Asserts identical order sequences (side, qty sign, timing).
Tests event-driven path determinism.

Marks: slow (requires _quant_hotpath and production models).
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, "/quant_system")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not Path("/quant_system/models_v8/BTCUSDT_gate_v2/config.json").exists(),
        reason="No production models available",
    ),
]

_hp = pytest.importorskip("_quant_hotpath")


def _load_config(symbol: str) -> Dict[str, Any]:
    cfg_path = Path(f"/quant_system/models_v8/{symbol}_gate_v2/config.json")
    with open(cfg_path) as f:
        return json.load(f)


def _make_snapshot(
    symbol: str,
    bar_index: int,
    features: Dict[str, float],
    close: float,
    position_qty: float = 0.0,
) -> SimpleNamespace:
    """Build a minimal pipeline snapshot for MLSignalDecisionModule."""
    market = SimpleNamespace(close=close, last_price=close)
    pos = SimpleNamespace(qty=Decimal(str(position_qty)), qty_f=position_qty)
    return SimpleNamespace(
        features=features,
        markets={symbol: market},
        market=market,
        positions={symbol: pos} if position_qty != 0 else {},
        event_index=bar_index,
        account=SimpleNamespace(balance=10000.0, equity=10000.0),
    )


def _generate_synthetic_bars(n: int, seed: int = 42) -> List[Dict[str, float]]:
    """Generate n synthetic feature dicts with stable randomness."""
    rng = np.random.RandomState(seed)
    bars = []
    close = 50000.0
    for i in range(n):
        close *= 1 + rng.randn() * 0.005
        features = {
            "close": close,
            "open": close * (1 + rng.randn() * 0.001),
            "high": close * (1 + abs(rng.randn()) * 0.002),
            "low": close * (1 - abs(rng.randn()) * 0.002),
            "volume": abs(rng.randn()) * 1e6,
            "atr_norm_14": abs(rng.randn()) * 0.02,
            "rsi_14": 50 + rng.randn() * 15,
            "ml_score": rng.randn() * 0.01,
            "_symbol": "BTCUSDT",
        }
        bars.append(features)
    return bars


def _extract_orders(events) -> List[Dict[str, Any]]:
    """Extract order info from event sequences."""
    orders = []
    for ev in events:
        et = getattr(ev, "event_type", None)
        et_str = str(getattr(et, "value", et)).upper() if et else ""
        if et_str == "ORDER":
            orders.append({
                "side": str(getattr(ev, "side", "")),
                "qty_sign": 1 if float(getattr(ev, "qty", 0)) > 0 else -1,
            })
    return orders


def _run_module(bars: List[Dict[str, float]], symbol: str = "BTCUSDT"):
    """Create fresh MLSignalDecisionModule and run bars through it."""
    from decision.backtest_module import MLSignalDecisionModule

    model_dir = f"/quant_system/models_v8/{symbol}_gate_v2"
    module = MLSignalDecisionModule(
        symbol=symbol,
        model_dir=model_dir,
        equity=10000.0,
        risk_fraction=0.05,
        min_hold=12,
    )

    all_orders = []
    position_qty = 0.0
    for i, feat in enumerate(bars):
        close = feat["close"]
        snapshot = _make_snapshot(symbol, i, feat, close, position_qty)
        events = list(module.decide(snapshot))
        orders = _extract_orders(events)
        for o in orders:
            o["bar"] = i
            all_orders.append(o)
            # Track position changes for subsequent snapshots
            if o["side"] == "buy":
                position_qty = abs(float(getattr(events[-1], "qty", 0)))
            elif o["side"] == "sell":
                position_qty = -abs(float(getattr(events[-1], "qty", 0)))
    return all_orders


class TestOrderGenerationParity:
    """Run same data twice, verify identical order sequences."""

    def test_deterministic_order_sequence(self):
        bars = _generate_synthetic_bars(200, seed=42)

        run1 = _run_module(bars)
        run2 = _run_module(bars)

        assert len(run1) == len(run2), (
            f"Order count mismatch: {len(run1)} vs {len(run2)}"
        )

        for i, (o1, o2) in enumerate(zip(run1, run2)):
            assert o1["bar"] == o2["bar"], (
                f"Order #{i} bar mismatch: {o1['bar']} vs {o2['bar']}"
            )
            assert o1["side"] == o2["side"], (
                f"Order #{i} side mismatch at bar {o1['bar']}: "
                f"{o1['side']} vs {o2['side']}"
            )
            assert o1["qty_sign"] == o2["qty_sign"], (
                f"Order #{i} qty sign mismatch at bar {o1['bar']}"
            )

    def test_different_seeds_differ(self):
        """Sanity check: different random data should produce different orders."""
        bars_a = _generate_synthetic_bars(200, seed=42)
        bars_b = _generate_synthetic_bars(200, seed=99)

        run_a = _run_module(bars_a)
        run_b = _run_module(bars_b)

        # At least one difference expected (order count or timing)
        if len(run_a) == len(run_b):
            bars_differ = any(
                a["bar"] != b["bar"] or a["side"] != b["side"]
                for a, b in zip(run_a, run_b)
            )
            assert bars_differ or len(run_a) == 0, (
                "Different inputs produced identical orders — unexpected"
            )
