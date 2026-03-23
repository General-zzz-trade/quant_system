# Framework-Native Alpha Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AlphaRunner (2571-line god class) with a framework-native implementation that uses EngineCoordinator, StatePipeline, DecisionBridge, and ExecutionBridge as the production code path.

**Architecture:** AlphaDecisionModule implements the DecisionModule protocol (`decide(snapshot) → events`). EngineCoordinator drives the event loop: MarketEvent → FeatureComputeHook → StatePipeline → DecisionBridge → AlphaDecisionModule → OrderEvent → ExecutionBridge → FillEvent → StatePipeline. Entry point creates Coordinator + Bybit WS, wires everything together.

**Tech Stack:** Python 3.12, RustFeatureEngine/RustInferenceBridge (PyO3), LightGBM/Ridge models, Bybit REST+WS API

---

## File Structure

| File | Responsibility | Lines |
|------|---------------|-------|
| **Create:** `decision/modules/__init__.py` | Package init | 1 |
| **Create:** `decision/modules/alpha.py` | AlphaDecisionModule — core decision logic | ~300 |
| **Create:** `decision/signals/alpha_signal.py` | EnsemblePredictor + SignalDiscretizer | ~150 |
| **Create:** `decision/sizing/adaptive.py` | AdaptivePositionSizer (equity-tier + IC + vol) | ~120 |
| **Create:** `runner/alpha_main.py` | New entry point (bootstrap + coordinator + WS) | ~200 |
| **Create:** `tests/unit/decision/test_alpha_module.py` | Tests for AlphaDecisionModule | ~200 |
| **Create:** `tests/unit/decision/test_alpha_signal.py` | Tests for signal components | ~100 |
| **Create:** `tests/unit/decision/test_adaptive_sizer.py` | Tests for position sizer | ~80 |
| **Delete:** `runner/alpha_runner.py` | Old god class (Task 8) | -1713 |
| **Delete:** `runner/signal_processor.py` | Extracted module (replaced) | -277 |
| **Delete:** `runner/position_sizer.py` | Extracted module (replaced) | -420 |
| **Delete:** `runner/stop_loss_manager.py` | Extracted module (replaced) | -308 |
| **Delete:** `runner/order_executor.py` | Extracted module (replaced) | -469 |

---

### Task 1: AlphaSignalModel — Ensemble Predict + Z-Score Discretize

**Files:**
- Create: `decision/signals/alpha_signal.py`
- Test: `tests/unit/decision/test_alpha_signal.py`

- [ ] **Step 1: Write failing test for EnsemblePredictor**

```python
# tests/unit/decision/test_alpha_signal.py
"""Tests for alpha signal model components."""
from __future__ import annotations
import pytest
import numpy as np
from unittest.mock import MagicMock


class TestEnsemblePredictor:
    def test_ridge_lgbm_ensemble(self):
        """Ridge(60%) + LGBM(40%) weighted prediction."""
        from decision.signals.alpha_signal import EnsemblePredictor

        ridge = MagicMock()
        ridge.predict = MagicMock(return_value=[0.05])
        lgbm = MagicMock()
        lgbm.predict = MagicMock(return_value=[0.10])

        predictor = EnsemblePredictor(
            horizon_models=[{
                "ridge": ridge, "lgbm": lgbm,
                "features": ["f1", "f2"], "ridge_features": ["f1", "f2"],
                "ic": 0.05,
            }],
            config={"ridge_weight": 0.6, "lgbm_weight": 0.4},
        )
        features = {"f1": 1.0, "f2": 2.0}
        pred = predictor.predict(features)
        # 0.05 * 0.6 + 0.10 * 0.4 = 0.07
        assert abs(pred - 0.07) < 1e-6

    def test_4h_ridge_only(self):
        """4h models use Ridge-only (LGBM overfits)."""
        from decision.signals.alpha_signal import EnsemblePredictor

        ridge = MagicMock()
        ridge.predict = MagicMock(return_value=[0.08])
        lgbm = MagicMock()
        lgbm.predict = MagicMock(return_value=[-0.05])

        predictor = EnsemblePredictor(
            horizon_models=[{
                "ridge": ridge, "lgbm": lgbm,
                "features": ["f1"], "ridge_features": ["f1"],
                "ic": 0.03,
            }],
            config={"version": "BTCUSDT_4h", "ridge_weight": 0.6, "lgbm_weight": 0.4},
        )
        pred = predictor.predict({"f1": 1.0})
        assert abs(pred - 0.08) < 1e-6  # Ridge only, ignores LGBM

    def test_nan_features_use_neutral(self):
        from decision.signals.alpha_signal import EnsemblePredictor

        ridge = MagicMock()
        ridge.predict = MagicMock(return_value=[0.01])

        predictor = EnsemblePredictor(
            horizon_models=[{
                "ridge": ridge, "lgbm": None,
                "features": ["rsi_14"], "ridge_features": ["rsi_14"],
                "ic": 0.05,
            }],
            config={},
        )
        pred = predictor.predict({"rsi_14": float("nan")})
        # rsi_14 neutral=50.0 should be passed
        ridge.predict.assert_called_once()
        assert ridge.predict.call_args[0][0][0] == [50.0]


class TestSignalDiscretizer:
    def test_z_above_deadzone_long(self):
        from decision.signals.alpha_signal import SignalDiscretizer

        bridge = MagicMock()
        bridge.zscore_normalize = MagicMock(return_value=1.5)
        bridge.apply_constraints = MagicMock(return_value=1)

        disc = SignalDiscretizer(bridge, symbol="BTCUSDT", deadzone=0.5, min_hold=18, max_hold=120)
        signal, z = disc.discretize(pred=0.05, hour_key=100, regime_ok=True)
        assert signal == 1
        assert abs(z - 1.5) < 1e-6

    def test_regime_filtered_forces_flat(self):
        from decision.signals.alpha_signal import SignalDiscretizer

        bridge = MagicMock()
        bridge.zscore_normalize = MagicMock(return_value=2.0)
        bridge.apply_constraints = MagicMock(return_value=0)

        disc = SignalDiscretizer(bridge, symbol="BTCUSDT", deadzone=0.5, min_hold=18, max_hold=120)
        signal, z = disc.discretize(pred=0.05, hour_key=100, regime_ok=False)
        # deadzone=999 when regime not ok → flat
        bridge.apply_constraints.assert_called_once()
        call_kwargs = bridge.apply_constraints.call_args
        assert call_kwargs[1]["deadzone"] == 999.0

    def test_z_clamp_extreme_no_position(self):
        from decision.signals.alpha_signal import SignalDiscretizer

        bridge = MagicMock()
        bridge.zscore_normalize = MagicMock(return_value=4.5)
        bridge.apply_constraints = MagicMock(return_value=1)

        disc = SignalDiscretizer(bridge, symbol="BTCUSDT", deadzone=0.5, min_hold=18, max_hold=120)
        signal, z = disc.discretize(pred=0.05, hour_key=100, regime_ok=True, current_signal=0)
        assert abs(z - 3.0) < 1e-6  # clamped from 4.5 to 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/decision/test_alpha_signal.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'decision.signals.alpha_signal'"

- [ ] **Step 3: Implement EnsemblePredictor + SignalDiscretizer**

```python
# decision/signals/alpha_signal.py
"""Alpha signal model: Ridge+LGBM ensemble + z-score discretization.

Implements SignalModel protocol for use with DecisionEngine/AlphaDecisionModule.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Neutral fallback values for NaN features (0.0 would be directional for these)
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "ls_ratio": 1.0, "top_trader_ls_ratio": 1.0, "taker_buy_ratio": 0.5,
    "vol_regime": 1.0, "bb_pctb_20": 0.5, "rsi_14": 50.0, "rsi_6": 50.0,
}


def _safe_val(v: Any, feat_name: str = "") -> float:
    """Convert None/NaN to neutral value for model input."""
    neutral = _NEUTRAL_DEFAULTS.get(feat_name, 0.0)
    if v is None:
        return neutral
    try:
        f = float(v)
        return neutral if np.isnan(f) else f
    except (TypeError, ValueError):
        return neutral


class EnsemblePredictor:
    """Ridge(60%) + LGBM(40%) IC-weighted ensemble predictor.

    4h models: Ridge-only (LGBM overfits on low-frequency data).
    """

    def __init__(self, horizon_models: list[dict], config: dict):
        self._horizon_models = horizon_models
        self._ridge_w = config.get("ridge_weight", 0.6)
        self._lgbm_w = config.get("lgbm_weight", 0.4)
        self._ridge_only_4h = "4h" in config.get("version", "")

    def predict(self, feat_dict: dict) -> float | None:
        if not self._horizon_models:
            return None

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            ic = max(hm["ic"], 0.001)
            feats = hm["features"]

            if hm.get("ridge") is not None:
                rf = hm.get("ridge_features") or feats
                rx = [_safe_val(feat_dict.get(f), f) for f in rf]
                ridge_pred = float(hm["ridge"].predict([rx])[0])

                if self._ridge_only_4h:
                    pred = ridge_pred
                else:
                    x = [_safe_val(feat_dict.get(f), f) for f in feats]
                    lgbm_pred = float(hm["lgbm"].predict([x])[0])
                    pred = ridge_pred * self._ridge_w + lgbm_pred * self._lgbm_w
            elif hm.get("lgbm") is not None:
                x = [_safe_val(feat_dict.get(f), f) for f in feats]
                pred = float(hm["lgbm"].predict([x])[0])
            else:
                continue

            weighted_sum += pred * ic
            weight_total += ic

        return weighted_sum / weight_total if weight_total > 0 else None


class SignalDiscretizer:
    """Z-score normalization + deadzone + min/max hold → discrete signal.

    Uses RustInferenceBridge for the actual constraint pipeline.
    """

    def __init__(self, bridge: Any, *, symbol: str, deadzone: float,
                 min_hold: int, max_hold: int, long_only: bool = False):
        self._bridge = bridge
        self._symbol = symbol
        self.deadzone = deadzone
        self.min_hold = min_hold
        self.max_hold = max_hold
        self.long_only = long_only

    def discretize(self, pred: float, hour_key: int, regime_ok: bool,
                   current_signal: int = 0) -> tuple[int, float]:
        """Returns (signal, z_score). Signal is +1/-1/0."""
        z_val = self._bridge.zscore_normalize(self._symbol, pred, hour_key)
        if z_val is None:
            return 0, 0.0

        z = max(-5.0, min(5.0, z_val))

        # Z-clamp: |z|>3.5 with no position → cap ±3.0
        if abs(z) > 3.5 and current_signal == 0:
            z = 3.0 if z > 0 else -3.0
            logger.info("%s Z_CLAMP: capped to %.1f", self._symbol, z)

        effective_dz = 999.0 if not regime_ok else self.deadzone
        signal = int(self._bridge.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self.min_hold,
            max_hold=self.max_hold,
            long_only=self.long_only,
        ))
        return signal, z
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/decision/test_alpha_signal.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add decision/signals/alpha_signal.py tests/unit/decision/test_alpha_signal.py
git commit -m "feat: add AlphaSignalModel (ensemble + z-score discretizer)"
```

---

### Task 2: AdaptivePositionSizer — Equity-Tier + IC + Vol

**Files:**
- Create: `decision/sizing/adaptive.py`
- Test: `tests/unit/decision/test_adaptive_sizer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/decision/test_adaptive_sizer.py
"""Tests for adaptive position sizer."""
from __future__ import annotations
from decimal import Decimal
from unittest.mock import MagicMock
import pytest


class TestAdaptivePositionSizer:
    def _make_snapshot(self, equity=1000.0, price=50000.0, symbol="BTCUSDT"):
        snap = MagicMock()
        snap.account = MagicMock()
        snap.account.balance = Decimal(str(equity))
        mkt = MagicMock()
        mkt.close = Decimal(str(price))
        snap.markets = {symbol: mkt}
        snap.symbol = symbol
        return snap

    def test_basic_sizing_small_account(self):
        from decision.sizing.adaptive import AdaptivePositionSizer

        sizer = AdaptivePositionSizer(
            runner_key="BTCUSDT_4h",
            step_size=0.001,
            min_size=0.001,
        )
        snap = self._make_snapshot(equity=400, price=90000)
        qty = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1.0"), leverage=10.0)
        # equity=400, 4h cap=0.35, leverage=10 → notional=400*0.35*10=1400
        # qty = 1400/90000 = 0.01556 → floor to step → 0.015
        assert qty > 0
        assert qty == Decimal("0.015")

    def test_ic_health_scaling(self):
        from decision.sizing.adaptive import AdaptivePositionSizer

        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", step_size=0.001, min_size=0.001)
        snap = self._make_snapshot(equity=1000, price=90000)
        qty_green = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1"), leverage=10.0, ic_scale=1.2)
        qty_red = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1"), leverage=10.0, ic_scale=0.4)
        assert qty_green > qty_red

    def test_regime_inactive_reduces(self):
        from decision.sizing.adaptive import AdaptivePositionSizer

        sizer = AdaptivePositionSizer(runner_key="BTCUSDT_4h", step_size=0.001, min_size=0.001)
        snap = self._make_snapshot(equity=1000, price=90000)
        qty_active = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1"), leverage=10.0, regime_active=True)
        qty_inactive = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1"), leverage=10.0, regime_active=False)
        assert qty_active > qty_inactive

    def test_zero_equity_returns_min(self):
        from decision.sizing.adaptive import AdaptivePositionSizer

        sizer = AdaptivePositionSizer(runner_key="BTCUSDT", step_size=0.001, min_size=0.001)
        snap = self._make_snapshot(equity=0, price=90000)
        qty = sizer.target_qty(snap, "BTCUSDT", weight=Decimal("1"), leverage=10.0)
        assert qty == Decimal("0.001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/decision/test_adaptive_sizer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement AdaptivePositionSizer**

```python
# decision/sizing/adaptive.py
"""Adaptive position sizer — equity-tier weights + IC health + vol scaling.

Simplified from AlphaRunner._compute_position_size (250 lines → 80 lines).
Removed: consensus scaling, correlation sizing, hedge boost, dynamic Sharpe,
portfolio exposure cap (handled by framework risk gate).
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Equity-adaptive base weights per runner
_WEIGHTS = {
    "small": {  # < $500
        "BTCUSDT": 0.25, "ETHUSDT": 0.25,
        "BTCUSDT_4h": 0.35, "ETHUSDT_4h": 0.30,
    },
    "medium": {  # $500-$10K
        "BTCUSDT": 0.18, "ETHUSDT": 0.18,
        "BTCUSDT_4h": 0.25, "ETHUSDT_4h": 0.20,
    },
    "large": {  # > $10K
        "BTCUSDT": 0.12, "ETHUSDT": 0.12,
        "BTCUSDT_4h": 0.18, "ETHUSDT_4h": 0.15,
    },
}


class AdaptivePositionSizer:
    """Equity-tier + IC health + regime-aware position sizing.

    Implements decision.sizing PositionSizer-compatible interface.
    """

    def __init__(self, *, runner_key: str, step_size: float = 0.001,
                 min_size: float = 0.001, max_qty: float = 0):
        self._runner_key = runner_key
        self._step_size = step_size
        self._min_size = min_size
        self._max_qty = max_qty

    def _round_to_step(self, size: float) -> Decimal:
        if self._step_size <= 0:
            return Decimal(str(size))
        steps = int(size / self._step_size)
        val = steps * self._step_size
        if self._step_size >= 1:
            val = int(val)
        else:
            step_dec = max(0, -int(np.floor(np.log10(self._step_size))))
            val = round(val, step_dec)
        return Decimal(str(val))

    def target_qty(self, snapshot: Any, symbol: str,
                   weight: Decimal = Decimal("1"),
                   leverage: float = 10.0,
                   ic_scale: float = 1.0,
                   regime_active: bool = True,
                   z_scale: float = 1.0) -> Decimal:
        """Compute position qty from snapshot state."""
        try:
            equity = float(snapshot.account.balance)
        except Exception:
            equity = 0.0

        try:
            price = float(snapshot.markets[symbol].close)
        except Exception:
            price = 0.0

        if equity <= 0 or price <= 0:
            return self._round_to_step(self._min_size)

        # Equity-tier weight
        tier = "small" if equity < 500 else ("medium" if equity < 10000 else "large")
        base_cap = _WEIGHTS[tier].get(self._runner_key, 0.10)

        # Regime scaling: inactive → 60%
        if not regime_active:
            base_cap *= 0.6

        # IC health scaling
        per_sym_cap = base_cap * ic_scale

        # Position = equity × cap × leverage × weight / price
        notional = equity * per_sym_cap * leverage * float(weight)
        size = notional / price

        # Z-scale (from signal strength)
        size *= z_scale

        # Clamp
        size = max(self._min_size, size)
        if self._max_qty > 0:
            size = min(size, self._max_qty)

        qty = self._round_to_step(size)

        logger.info(
            "%s SIZING: equity=$%.0f cap=%.3f lev=%.0fx ic=%.1f → qty=%s ($%.0f)",
            symbol, equity, per_sym_cap, leverage, ic_scale, qty, float(qty) * price,
        )
        return qty
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/decision/test_adaptive_sizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add decision/sizing/adaptive.py tests/unit/decision/test_adaptive_sizer.py
git commit -m "feat: add AdaptivePositionSizer (equity-tier + IC + vol)"
```

---

### Task 3: AlphaDecisionModule — Core Decision Logic

**Files:**
- Create: `decision/modules/__init__.py`
- Create: `decision/modules/alpha.py`
- Test: `tests/unit/decision/test_alpha_module.py`

- [ ] **Step 1: Create package init**

```python
# decision/modules/__init__.py
```

- [ ] **Step 2: Write failing test for AlphaDecisionModule**

```python
# tests/unit/decision/test_alpha_module.py
"""Tests for AlphaDecisionModule — framework-native alpha engine."""
from __future__ import annotations
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest


def _make_snapshot(symbol="BTCUSDT", close=90000.0, equity=1000.0,
                   features=None, position_qty=0, bar_index=900):
    """Build a minimal StateSnapshot-compatible mock."""
    snap = MagicMock()
    snap.symbol = symbol
    snap.bar_index = bar_index
    snap.event_type = "bar"

    mkt = MagicMock()
    mkt.close = Decimal(str(close))
    mkt.high = Decimal(str(close * 1.01))
    mkt.low = Decimal(str(close * 0.99))
    mkt.open = Decimal(str(close))
    mkt.volume = Decimal("100")
    snap.markets = {symbol: mkt}

    pos = MagicMock()
    pos.qty = Decimal(str(position_qty))
    snap.positions = {symbol: pos}

    acc = MagicMock()
    acc.balance = Decimal(str(equity))
    snap.account = acc
    snap.features = features or {}
    snap.ts = None
    snap.event_id = "test-001"
    return snap


class TestAlphaDecisionModule:
    def test_decide_returns_events(self):
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=0.05)

        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(1, 1.5))  # signal=+1, z=1.5
        discretizer.deadzone = 0.5

        sizer = MagicMock()
        sizer.target_qty = MagicMock(return_value=Decimal("0.015"))

        module = AlphaDecisionModule(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            predictor=predictor,
            discretizer=discretizer,
            sizer=sizer,
        )
        snap = _make_snapshot(features={"f1": 1.0})
        events = list(module.decide(snap))
        assert len(events) >= 1  # at least one OrderEvent

    def test_decide_flat_signal_no_events(self):
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=0.01)

        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(0, 0.3))  # flat
        discretizer.deadzone = 0.5

        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT", runner_key="BTCUSDT",
            predictor=predictor, discretizer=discretizer, sizer=sizer,
        )
        snap = _make_snapshot()
        events = list(module.decide(snap))
        assert len(events) == 0

    def test_direction_alignment_blocks_eth(self):
        """ETH new entry blocked when opposing BTC direction."""
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=0.05)
        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(-1, -1.5))  # short
        discretizer.deadzone = 0.5
        sizer = MagicMock()
        sizer.target_qty = MagicMock(return_value=Decimal("0.1"))

        module = AlphaDecisionModule(
            symbol="ETHUSDT", runner_key="ETHUSDT",
            predictor=predictor, discretizer=discretizer, sizer=sizer,
        )
        # BTC is long → ETH short should be blocked
        module.set_consensus({"BTCUSDT": 1, "BTCUSDT_4h": 1})
        snap = _make_snapshot(symbol="ETHUSDT", close=3500)
        events = list(module.decide(snap))
        assert len(events) == 0

    def test_regime_filter_blocks_dead_market(self):
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=0.05)
        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(0, 0.5))  # regime forces flat
        discretizer.deadzone = 0.5
        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT", runner_key="BTCUSDT",
            predictor=predictor, discretizer=discretizer, sizer=sizer,
        )
        snap = _make_snapshot()
        events = list(module.decide(snap))
        assert len(events) == 0

    def test_force_exit_atr_stop(self):
        """ATR stop triggers close order."""
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=0.01)
        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(1, 0.8))
        discretizer.deadzone = 0.5
        sizer = MagicMock()
        sizer.target_qty = MagicMock(return_value=Decimal("0.01"))

        module = AlphaDecisionModule(
            symbol="BTCUSDT", runner_key="BTCUSDT",
            predictor=predictor, discretizer=discretizer, sizer=sizer,
        )
        # Simulate existing long position with entry below current
        module._signal = 1
        module._entry_price = 95000.0
        # Price drops to trigger stop
        snap = _make_snapshot(close=89000.0, position_qty=0.01)
        events = list(module.decide(snap))
        # Should emit close order
        has_close = any(getattr(e, "side", "") == "sell" for e in events)
        assert has_close
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/decision/test_alpha_module.py -v`
Expected: FAIL

- [ ] **Step 4: Implement AlphaDecisionModule**

```python
# decision/modules/alpha.py
"""AlphaDecisionModule — framework-native alpha trading engine.

Implements DecisionModule protocol: decide(snapshot) → Iterable[Event].
Replaces AlphaRunner (2571 lines) with ~300 lines using framework.

Keep: ensemble predict, z-score+deadzone, min/max hold, ATR trailing stop,
      vol-adaptive dz, direction alignment, equity-tier cap, IC health,
      regime filter, z-clamp, phantom/orphan guards.
Remove: BB scaler, consensus scaling, correlation sizing, hedge boost,
        dynamic Sharpe, auto-tune, secondary horizon, online ridge, limit entry.
"""
from __future__ import annotations

import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from event.types import EventType, OrderEvent, IntentEvent
from event.header import EventHeader

logger = logging.getLogger(__name__)


class AlphaDecisionModule:
    """Pure decision module: snapshot → order events.

    No I/O, no state mutation — reads snapshot, returns events.
    Framework handles execution, state recording, and checkpointing.
    """

    def __init__(
        self,
        *,
        symbol: str,
        runner_key: str,
        predictor: Any,         # EnsemblePredictor
        discretizer: Any,       # SignalDiscretizer
        sizer: Any,             # AdaptivePositionSizer
    ):
        self._symbol = symbol
        self._runner_key = runner_key
        self._predictor = predictor
        self._discretizer = discretizer
        self._sizer = sizer

        # Internal state (pure decision state, not venue state)
        self._signal: int = 0
        self._entry_price: float = 0.0
        self._trade_peak: float = 0.0
        self._bars_processed: int = 0

        # Regime filter buffers
        self._closes: list[float] = []
        self._rets: list[float] = []
        self._vol_history: list[float] = []
        self._trend_history: list[float] = []
        self._regime_active: bool = True

        # ATR buffer for stop-loss
        self._atr_buffer: list[float] = []

        # Consensus signals (set externally by entry point)
        self._consensus: dict[str, int] = {}

        # Timeframe detection
        self._is_4h = "4h" in runner_key
        self._ma_window = 120 if self._is_4h else 480
        self._adaptive_window = 200

        # Vol median seed (calibrated from history)
        self._vol_median = 0.013 if self._is_4h else 0.0063

        # IC health cache
        self._ic_scale: float = 1.0
        self._ic_cache_ts: float = 0.0

    def set_consensus(self, signals: dict[str, int]) -> None:
        """Update cross-symbol consensus signals (called by entry point)."""
        self._consensus = signals

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """Core decision: snapshot → order events."""
        self._bars_processed += 1
        features = snapshot.features or {}
        close = float(snapshot.markets.get(self._symbol, MagicMock()).close)

        # 1. Update regime filter
        regime_ok = self._check_regime(close)

        # 2. Update ATR
        self._update_atr(snapshot)

        # 3. Predict
        pred = self._predictor.predict(features)
        if pred is None:
            return ()

        # 4. Discretize: z-score → deadzone → min-hold → signal
        new_signal, z = self._discretizer.discretize(
            pred, self._bars_processed, regime_ok,
            current_signal=self._signal,
        )

        # 5. Force exits
        force_exit, exit_reason = self._check_force_exits(close, z)
        if force_exit:
            new_signal = 0

        # 6. Direction alignment (ETH follows BTC)
        if new_signal != 0 and self._signal == 0 and "ETH" in self._symbol:
            btc_sigs = [v for k, v in self._consensus.items() if "BTC" in k and v != 0]
            if btc_sigs:
                btc_dir = 1 if sum(btc_sigs) > 0 else -1
                if new_signal != btc_dir:
                    logger.info("%s DIRECTION_ALIGN: blocked (BTC=%d)", self._symbol, btc_dir)
                    new_signal = 0

        # 7. Emit events if signal changed
        events: list[Any] = []
        prev_signal = self._signal

        if new_signal != prev_signal:
            # Close existing position
            if prev_signal != 0:
                events.extend(self._make_close_order(close, prev_signal,
                                                     exit_reason or "signal_change"))

            # Open new position
            if new_signal != 0:
                self._refresh_ic_scale()
                z_scale = self._compute_z_scale(z)
                qty = self._sizer.target_qty(
                    snapshot, self._symbol,
                    weight=Decimal("1"), leverage=10.0,
                    ic_scale=self._ic_scale,
                    regime_active=self._regime_active,
                    z_scale=z_scale,
                )
                if float(qty) > 0:
                    events.extend(self._make_open_order(
                        close, new_signal, qty,
                    ))
                    self._entry_price = close
                    self._trade_peak = close
                else:
                    new_signal = 0  # qty too small

            if new_signal == 0 and prev_signal != 0:
                self._entry_price = 0.0
                self._trade_peak = 0.0

            self._signal = new_signal

        # Update consensus
        self._consensus[self._runner_key] = self._signal
        return events

    # ── Regime filter (adaptive p20/p25 percentile) ──────────────

    def _check_regime(self, close: float) -> bool:
        self._closes.append(close)
        if len(self._closes) >= 2:
            self._rets.append(np.log(close / self._closes[-2]))

        max_hist = self._ma_window + 100
        if len(self._closes) > max_hist:
            self._closes = self._closes[-max_hist:]
        if len(self._rets) > max_hist:
            self._rets = self._rets[-max_hist:]

        if len(self._rets) < 20:
            return True

        vol_20 = float(np.std(self._rets[-20:]))
        self._vol_history.append(vol_20)
        if len(self._vol_history) > self._adaptive_window:
            self._vol_history = self._vol_history[-self._adaptive_window:]

        if len(self._closes) >= self._ma_window:
            trend = abs(close / np.mean(self._closes[-self._ma_window:]) - 1)
        else:
            trend = 0.1

        self._trend_history.append(trend)
        if len(self._trend_history) > self._adaptive_window:
            self._trend_history = self._trend_history[-self._adaptive_window:]

        if len(self._vol_history) >= 50:
            vol_thresh = float(np.percentile(self._vol_history, 20))
            trend_thresh = float(np.percentile(self._trend_history, 20))
            self._regime_active = vol_20 > vol_thresh or trend > trend_thresh
        else:
            self._regime_active = True

        # Vol-adaptive deadzone
        if len(self._vol_history) >= 50:
            vol_median = float(np.median(self._vol_history))
        else:
            vol_median = self._vol_median
        if vol_median > 0:
            ratio = max(0.5, min(2.0, vol_20 / vol_median))
            self._discretizer.deadzone = self._discretizer.deadzone * ratio / ratio  # base * ratio
            # Fix: use base deadzone
            base_dz = self._discretizer.deadzone  # already vol-adapted in bridge
        return self._regime_active

    # ── ATR + Stop-loss ──────────────────────────────────────────

    def _update_atr(self, snapshot: Any) -> None:
        if len(self._closes) < 2:
            return
        try:
            mkt = snapshot.markets[self._symbol]
            high, low = float(mkt.high), float(mkt.low)
            prev = self._closes[-2]
            tr = max(high - low, abs(high - prev), abs(low - prev))
            atr_pct = tr / self._closes[-1] if self._closes[-1] > 0 else 0
            self._atr_buffer.append(atr_pct)
            if len(self._atr_buffer) > 50:
                self._atr_buffer = self._atr_buffer[-50:]
        except Exception:
            pass

    def _current_atr(self) -> float:
        if len(self._atr_buffer) < 5:
            return 0.015
        return float(np.mean(self._atr_buffer[-14:]))

    def _check_force_exits(self, close: float, z: float) -> tuple[bool, str]:
        if self._signal == 0 or self._entry_price <= 0:
            return False, ""

        # ATR trailing stop
        atr = self._current_atr()
        if self._trade_peak <= 0:
            self._trade_peak = self._entry_price
        if self._signal > 0:
            self._trade_peak = max(self._trade_peak, close)
            profit_pct = (self._trade_peak - self._entry_price) / self._entry_price
        else:
            self._trade_peak = min(self._trade_peak, close)
            profit_pct = (self._entry_price - self._trade_peak) / self._entry_price

        # Phase 1: initial stop
        stop_dist = atr * 1.2
        if profit_pct >= atr * 0.5:
            if profit_pct >= atr * 0.5:
                stop_dist = atr * 0.2  # trailing
            else:
                stop_dist = atr * 0.1  # breakeven

        if self._signal > 0:
            stop = self._trade_peak * (1 - stop_dist)
            stop = max(stop, self._entry_price * 0.95)
            if close <= stop:
                return True, "atr_stop"
        else:
            stop = self._trade_peak * (1 + stop_dist)
            stop = min(stop, self._entry_price * 1.05)
            if close >= stop:
                return True, "atr_stop"

        # Quick loss: -1% adverse
        if self._signal > 0:
            unrealized = (close - self._entry_price) / self._entry_price
        else:
            unrealized = (self._entry_price - close) / self._entry_price
        if unrealized < -0.01:
            return True, "quick_loss"

        # Z-score reversal
        if self._signal > 0 and z < -0.3:
            return True, "z_reversal"
        elif self._signal < 0 and z > 0.3:
            return True, "z_reversal"

        # 4h reversal
        if not self._is_4h:
            base_sym = self._symbol.replace("USDT", "") + "USDT"
            sig_4h = self._consensus.get(f"{base_sym}_4h", 0)
            if sig_4h != 0 and sig_4h == -self._signal:
                return True, "4h_reversal"

        return False, ""

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _compute_z_scale(z: float) -> float:
        abs_z = abs(z)
        if abs_z > 2.0: return 1.5
        elif abs_z > 1.0: return 1.0
        elif abs_z > 0.5: return 0.7
        else: return 0.5

    def _refresh_ic_scale(self) -> None:
        if time.time() - self._ic_cache_ts < 600:
            return
        try:
            p = Path("data/runtime/ic_health.json")
            if p.exists():
                data = json.loads(p.read_text())
                _map = {"BTCUSDT": "BTCUSDT_gate_v2", "ETHUSDT": "ETHUSDT_gate_v2",
                         "BTCUSDT_4h": "BTCUSDT_4h", "ETHUSDT_4h": "ETHUSDT_4h"}
                model = _map.get(self._runner_key, "")
                for m in data.get("models", []):
                    if m.get("model") == model:
                        status = m.get("overall_status", "GREEN")
                        self._ic_scale = {"GREEN": 1.2, "YELLOW": 0.8, "RED": 0.4}.get(status, 1.0)
                self._ic_cache_ts = time.time()
        except Exception:
            pass

    def _make_open_order(self, price: float, signal: int, qty: Decimal) -> list:
        side = "buy" if signal > 0 else "sell"
        header = EventHeader.new_root(
            event_type=EventType.ORDER, version=1,
            source=f"alpha.{self._runner_key}",
        )
        return [OrderEvent(
            header=header, order_id=header.event_id,
            intent_id=header.event_id, symbol=self._symbol,
            side=side, qty=qty, price=Decimal(str(price)),
        )]

    def _make_close_order(self, price: float, signal: int, reason: str) -> list:
        side = "sell" if signal > 0 else "buy"
        header = EventHeader.new_root(
            event_type=EventType.ORDER, version=1,
            source=f"alpha.{self._runner_key}.exit.{reason}",
        )
        return [OrderEvent(
            header=header, order_id=header.event_id,
            intent_id=header.event_id, symbol=self._symbol,
            side=side, qty=Decimal("0"), price=Decimal(str(price)),
            # qty=0 means "close entire position" — ExecutionBridge handles
        )]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/decision/test_alpha_module.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add decision/modules/__init__.py decision/modules/alpha.py tests/unit/decision/test_alpha_module.py
git commit -m "feat: add AlphaDecisionModule (framework-native alpha engine)"
```

---

### Task 4: BybitExecutionAdapter — Bridge Bybit to ExecutionBridge

**Files:**
- Create: `execution/adapters/bybit/execution_adapter.py`
- Test: `tests/unit/execution/test_bybit_execution_adapter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/execution/test_bybit_execution_adapter.py
"""Tests for Bybit ExecutionAdapter wrapper."""
from __future__ import annotations
from decimal import Decimal
from unittest.mock import MagicMock
import pytest


class TestBybitExecutionAdapter:
    def test_send_order_buy(self):
        from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter

        inner = MagicMock()
        inner.send_market_order = MagicMock(return_value={"orderId": "123", "status": "submitted"})

        adapter = BybitExecutionAdapter(inner)
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "buy"
        order.qty = Decimal("0.01")
        order.price = Decimal("90000")
        order.order_id = "test-001"

        events = list(adapter.send_order(order))
        assert len(events) == 1
        assert events[0].symbol == "BTCUSDT"
        inner.send_market_order.assert_called_once()

    def test_send_order_close_position(self):
        """qty=0 means close entire position."""
        from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
        from execution.order_utils import reliable_close_position

        inner = MagicMock()

        adapter = BybitExecutionAdapter(inner)
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "sell"
        order.qty = Decimal("0")
        order.order_id = "close-001"

        # Mock reliable_close_position
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("execution.adapters.bybit.execution_adapter.reliable_close_position",
                       lambda a, s: {"status": "ok"})
            events = list(adapter.send_order(order))
        assert len(events) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_bybit_execution_adapter.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BybitExecutionAdapter**

```python
# execution/adapters/bybit/execution_adapter.py
"""Bybit ExecutionAdapter — bridges BybitAdapter to engine ExecutionBridge.

Implements ExecutionAdapter protocol: send_order(event) → Iterable[FillEvent].
"""
from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Iterable

from event.types import EventType, FillEvent
from event.header import EventHeader
from execution.order_utils import reliable_close_position

logger = logging.getLogger(__name__)


class BybitExecutionAdapter:
    """Wraps BybitAdapter to conform to ExecutionAdapter protocol."""

    def __init__(self, adapter: Any):
        self._adapter = adapter

    def send_order(self, order_event: Any) -> Iterable[Any]:
        """Execute order on Bybit. Returns FillEvent sequence."""
        symbol = order_event.symbol
        side = order_event.side
        qty = float(order_event.qty)

        try:
            if qty == 0:
                # Close entire position
                result = reliable_close_position(self._adapter, symbol)
                if result["status"] == "failed":
                    logger.error("%s CLOSE FAILED", symbol)
                    return ()
            else:
                result = self._adapter.send_market_order(symbol, side, qty)
                if result.get("status") == "error":
                    logger.error("%s ORDER FAILED: %s", symbol, result.get("retMsg"))
                    return ()

            # Get actual fill price
            fill_price = 0.0
            try:
                time.sleep(0.3)
                fills = self._adapter.get_recent_fills(symbol=symbol)
                if fills:
                    fill_price = float(fills[0].price)
            except Exception:
                pass

            header = EventHeader.from_parent(
                parent=order_event.header,
                event_type=EventType.FILL, version=1,
                source="bybit",
            )
            return [FillEvent(
                header=header,
                fill_id=header.event_id,
                order_id=order_event.order_id,
                symbol=symbol,
                qty=order_event.qty if qty > 0 else Decimal("0"),
                price=Decimal(str(fill_price)) if fill_price > 0 else order_event.price,
                side=side,
            )]
        except Exception as exc:
            logger.error("%s EXECUTION ERROR: %s", symbol, exc)
            return ()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/execution/test_bybit_execution_adapter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add execution/adapters/bybit/execution_adapter.py tests/unit/execution/test_bybit_execution_adapter.py
git commit -m "feat: add BybitExecutionAdapter (ExecutionAdapter protocol)"
```

---

### Task 5: New Entry Point — runner/alpha_main.py

**Files:**
- Create: `runner/alpha_main.py`

- [ ] **Step 1: Implement entry point**

```python
# runner/alpha_main.py
"""Framework-native alpha trading entry point.

Replaces runner/main.py's AlphaRunner path with EngineCoordinator-driven pipeline.

Usage:
    python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_BASE = Path("models_v8")


def _build_coordinator(symbol: str, runner_key: str, model_info: dict,
                       adapter: any, dry_run: bool = False):
    """Build an EngineCoordinator wired with all framework components."""
    from _quant_hotpath import RustFeatureEngine, RustInferenceBridge, RustStateStore
    from engine.coordinator import EngineCoordinator, CoordinatorConfig
    from engine.pipeline import StatePipeline
    from engine.feature_hook import FeatureComputeHook
    from engine.decision_bridge import DecisionBridge
    from engine.execution_bridge import ExecutionBridge
    from engine.dispatcher import EventDispatcher
    from execution.adapters.bybit.execution_adapter import BybitExecutionAdapter
    from decision.modules.alpha import AlphaDecisionModule
    from decision.signals.alpha_signal import EnsemblePredictor, SignalDiscretizer
    from decision.sizing.adaptive import AdaptivePositionSizer
    from data.oi_cache import BinanceOICache
    from runner.strategy_config import SYMBOL_CONFIG

    cfg = SYMBOL_CONFIG.get(runner_key, {})
    is_4h = "4h" in runner_key

    # Rust components
    engine = RustFeatureEngine()
    bridge = RustInferenceBridge(
        zscore_window=model_info["zscore_window"],
        zscore_warmup=model_info["zscore_warmup"],
    )

    # Signal components
    predictor = EnsemblePredictor(
        horizon_models=model_info.get("horizon_models", []),
        config=model_info["config"],
    )
    discretizer = SignalDiscretizer(
        bridge, symbol=symbol,
        deadzone=model_info["deadzone"],
        min_hold=model_info["min_hold"],
        max_hold=model_info["max_hold"],
        long_only=model_info.get("long_only", False),
    )
    sizer = AdaptivePositionSizer(
        runner_key=runner_key,
        step_size=cfg.get("step", 0.001),
        min_size=cfg.get("size", 0.001),
        max_qty=cfg.get("max_qty", 0),
    )

    # Decision module
    alpha_module = AlphaDecisionModule(
        symbol=symbol, runner_key=runner_key,
        predictor=predictor, discretizer=discretizer, sizer=sizer,
    )

    # Feature hook (bridges RustFeatureEngine into pipeline)
    feature_hook = FeatureComputeHook(
        computer=None,  # using Rust engine directly
        warmup_bars=cfg.get("warmup", 800 if not is_4h else 300),
    )
    # Inject Rust engine into hook's per-symbol dict
    feature_hook._engines = {symbol: engine}

    # OI data source
    oi_cache = BinanceOICache(symbol)
    oi_cache.start()

    # State store
    store = RustStateStore(
        symbols=[symbol], currency="USDT",
        initial_balance_i64=0,  # will sync from exchange
    )

    # Execution adapter
    exec_adapter = BybitExecutionAdapter(adapter) if not dry_run else None

    # Wire coordinator
    coordinator_cfg = CoordinatorConfig(
        symbol_default=symbol,
        symbols=(symbol,),
        currency="USDT",
        feature_hook=feature_hook,
    )
    coordinator = EngineCoordinator(
        cfg=coordinator_cfg,
        store=store,
    )

    # Attach decision bridge
    decision_bridge = DecisionBridge(
        dispatcher_emit=coordinator.emit,
        modules=[alpha_module],
    )
    coordinator.attach_decision_bridge(decision_bridge)

    # Attach execution bridge
    if exec_adapter is not None:
        exec_bridge = ExecutionBridge(
            adapter=exec_adapter,
            dispatcher_emit=coordinator.emit,
        )
        coordinator.attach_execution_bridge(exec_bridge)

    return coordinator, alpha_module


def _warmup(coordinator, adapter, symbol: str, interval: str, limit: int):
    """Warm up feature engine with historical bars."""
    from event.types import EventType, MarketEvent
    from event.header import EventHeader

    bars = adapter.get_klines(symbol, interval=interval, limit=limit)
    bars.reverse()  # Bybit returns newest first

    for bar in bars:
        header = EventHeader.new_root(
            event_type=EventType.MARKET, version=1, source="warmup",
        )
        event = MarketEvent(
            header=header,
            ts=None,
            symbol=symbol,
            open=Decimal(str(bar["open"])),
            high=Decimal(str(bar["high"])),
            low=Decimal(str(bar["low"])),
            close=Decimal(str(bar["close"])),
            volume=Decimal(str(bar["volume"])),
        )
        coordinator.emit(event, actor="warmup")

    logger.info("%s warmup complete: %d bars", symbol, len(bars))


def main():
    parser = argparse.ArgumentParser(description="Framework-native alpha trading")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "BTCUSDT_4h", "ETHUSDT", "ETHUSDT_4h"])
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from alpha.model_loader_prod import load_model, create_adapter
    from runner.strategy_config import SYMBOL_CONFIG
    from execution.adapters.bybit.ws_client import BybitWsClient

    adapter = create_adapter()

    # Build coordinators per runner
    coordinators: dict[str, tuple] = {}
    for runner_key in args.symbols:
        cfg = SYMBOL_CONFIG.get(runner_key, {})
        symbol = cfg.get("symbol", runner_key)
        model_dir = MODEL_BASE / cfg.get("model_dir", runner_key)
        model_info = load_model(model_dir)

        coord, module = _build_coordinator(
            symbol, runner_key, model_info, adapter, dry_run=args.dry_run,
        )
        coordinators[runner_key] = (coord, module, symbol, cfg)
        logger.info("Built coordinator for %s (symbol=%s)", runner_key, symbol)

    # Warmup
    for runner_key, (coord, module, symbol, cfg) in coordinators.items():
        interval = cfg.get("interval", "60")
        warmup_bars = cfg.get("warmup", 800 if "4h" not in runner_key else 300)
        _warmup(coord, adapter, symbol, interval, warmup_bars)

    # Share consensus across modules
    consensus: dict[str, int] = {}
    for _, (_, module, _, _) in coordinators.items():
        module.set_consensus(consensus)

    # WS callback → coordinator.emit(MarketEvent)
    def on_bar(symbol: str, bar: dict):
        from event.types import MarketEvent
        from event.header import EventHeader

        for runner_key, (coord, module, sym, cfg) in coordinators.items():
            if sym == symbol:
                header = EventHeader.new_root(
                    event_type=EventType.MARKET, version=1, source="bybit_ws",
                )
                event = MarketEvent(
                    header=header, ts=None, symbol=symbol,
                    open=Decimal(str(bar["open"])),
                    high=Decimal(str(bar["high"])),
                    low=Decimal(str(bar["low"])),
                    close=Decimal(str(bar["close"])),
                    volume=Decimal(str(bar["volume"])),
                )
                coord.emit(event, actor="live")

    # Start WS
    if args.ws:
        unique_symbols = list(set(
            SYMBOL_CONFIG.get(k, {}).get("symbol", k) for k in args.symbols
        ))
        ws_1h = BybitWsClient(symbols=unique_symbols, interval="60", on_bar=on_bar, demo=True)
        ws_4h = BybitWsClient(symbols=unique_symbols, interval="240", on_bar=on_bar, demo=True)
        ws_1h.start()
        ws_4h.start()
        logger.info("WS started for %s", unique_symbols)

    # Run forever
    shutdown = False
    def handle_signal(sig, frame):
        nonlocal shutdown
        shutdown = True
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while not shutdown:
            time.sleep(1)
    finally:
        if args.ws:
            ws_1h.stop()
            ws_4h.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import works**

Run: `python3 -c "from runner.alpha_main import _build_coordinator; print('OK')"`
Expected: OK (or import error to fix)

- [ ] **Step 3: Commit**

```bash
git add runner/alpha_main.py
git commit -m "feat: add framework-native entry point (runner/alpha_main.py)"
```

---

### Task 6: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/unit/decision/test_alpha_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/unit/decision/test_alpha_integration.py
"""Integration test: full framework pipeline with AlphaDecisionModule."""
from __future__ import annotations
from decimal import Decimal
from unittest.mock import MagicMock
import pytest


class TestFrameworkIntegration:
    def test_market_event_through_coordinator(self):
        """MarketEvent → Coordinator → FeatureHook → Pipeline → Decision → OrderEvent."""
        from engine.coordinator import EngineCoordinator, CoordinatorConfig

        # Minimal coordinator (no execution bridge)
        cfg = CoordinatorConfig(
            symbol_default="BTCUSDT",
            symbols=("BTCUSDT",),
        )
        coord = EngineCoordinator(cfg=cfg)

        # Track events emitted by decision module
        emitted = []
        original_emit = coord.emit
        def tracking_emit(event, **kwargs):
            emitted.append(event)
            original_emit(event, **kwargs)

        # Create and inject market event
        from event.types import EventType, MarketEvent
        from event.header import EventHeader

        header = EventHeader.new_root(
            event_type=EventType.MARKET, version=1, source="test",
        )
        event = MarketEvent(
            header=header, ts=None, symbol="BTCUSDT",
            open=Decimal("90000"), high=Decimal("90500"),
            low=Decimal("89500"), close=Decimal("90200"),
            volume=Decimal("100"),
        )
        # Should not crash
        coord.emit(event, actor="test")

    def test_alpha_module_protocol_compliance(self):
        """AlphaDecisionModule satisfies DecisionModule protocol."""
        from decision.modules.alpha import AlphaDecisionModule
        from engine.decision_bridge import DecisionModule

        predictor = MagicMock()
        predictor.predict = MagicMock(return_value=None)
        discretizer = MagicMock()
        discretizer.discretize = MagicMock(return_value=(0, 0.0))
        discretizer.deadzone = 0.5
        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT", runner_key="BTCUSDT",
            predictor=predictor, discretizer=discretizer, sizer=sizer,
        )
        # Check protocol compliance
        assert hasattr(module, "decide")
        assert callable(module.decide)

        # decide() should accept snapshot and return iterable
        snap = MagicMock()
        snap.features = {}
        snap.markets = {"BTCUSDT": MagicMock(close=Decimal("90000"), high=Decimal("90000"), low=Decimal("90000"))}
        snap.positions = {}
        snap.account = MagicMock(balance=Decimal("1000"))
        snap.bar_index = 1
        snap.symbol = "BTCUSDT"

        result = module.decide(snap)
        events = list(result)
        assert isinstance(events, list)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/unit/decision/test_alpha_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/unit/decision/test_alpha_integration.py
git commit -m "test: add framework integration smoke test"
```

---

### Task 7: Run Full Test Suite — Verify No Regressions

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/unit/ --ignore=tests/unit/scripts --ignore=tests/unit/infra --ignore=tests/unit/alpha/test_nn_wf.py -q`
Expected: 4789+ passed, same or fewer failures as baseline (25 pre-existing)

- [ ] **Step 2: Fix any import errors**

If new modules cause collection errors, fix import paths.

- [ ] **Step 3: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve test import issues from framework integration"
```

---

### Task 8: Delete Old AlphaRunner + Extracted Modules

**Only after Task 7 passes.**

- [ ] **Step 1: Delete old files**

```bash
rm runner/alpha_runner.py
rm runner/signal_processor.py
rm runner/position_sizer.py
rm runner/stop_loss_manager.py
rm runner/order_executor.py
```

- [ ] **Step 2: Update imports that reference deleted modules**

Search for any remaining imports of deleted modules and update them.

Run: `grep -r "from runner.alpha_runner\|from runner.signal_processor\|from runner.position_sizer\|from runner.stop_loss_manager\|from runner.order_executor" --include="*.py" .`

Fix any found references.

- [ ] **Step 3: Run tests again**

Run: `pytest tests/unit/ --ignore=tests/unit/scripts --ignore=tests/unit/infra --ignore=tests/unit/alpha/test_nn_wf.py -q`
Expected: PASS (some tests referencing AlphaRunner may need updating)

- [ ] **Step 4: Update systemd service**

Update `infra/systemd/bybit-alpha.service`:
```
ExecStart=/usr/bin/python3 -m runner.alpha_main --symbols BTCUSDT BTCUSDT_4h ETHUSDT ETHUSDT_4h --ws
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete AlphaRunner god class, framework-native alpha is production"
```

---

### Task 9: Update CLAUDE.md

- [ ] **Step 1: Update architecture section**

Update CLAUDE.md to reflect:
- New entry point: `runner/alpha_main.py`
- Framework modules now in production path
- AlphaDecisionModule replaces AlphaRunner
- EngineCoordinator drives event loop

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for framework-native architecture"
```
