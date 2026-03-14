# LiveRunner Decomposition Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 2,017-line LiveRunner God Object into 8 focused modules (~120-200 LOC each) with a transparent ~120-line assembly script.

**Architecture:** New modules live alongside the existing LiveRunner (no breaking changes). Each module wraps existing engine/execution/risk primitives behind a clean interface. The assembly script (`run_trading.py`) replaces `LiveRunner.build()` with explicit, visible wiring. Old LiveRunner remains functional for parallel comparison.

**Tech Stack:** Python 3.12, existing `engine/`, `execution/`, `risk/`, `decision/` modules (untouched). pytest for tests.

**Spec:** `docs/superpowers/specs/2026-03-15-liverunner-decomposition-design.md`

---

## File Map

| New File | LOC | Wraps/Uses |
|----------|-----|------------|
| `runner/trading_config.py` | ~80 | Simplified config (replaces 102-field LiveRunnerConfig for new runner) |
| `runner/trading_engine.py` | ~200 | `engine/feature_hook.py`, `RustInferenceBridge`, model reload logic |
| `runner/risk_manager.py` | ~120 | `risk/kill_switch.py`, `risk/kill_switch_bridge.py`, `RiskGate` |
| `runner/order_manager.py` | ~120 | `execution/order_state_machine.py`, `execution/timeout_tracker.py` |
| `runner/binance_executor.py` | ~180 | `execution/adapters/binance/`, `KillSwitchBridge`, `ShadowExecutionAdapter` |
| `runner/recovery_manager.py` | ~180 | `runner/recovery.py` (existing, wraps its functions) |
| `runner/lifecycle_manager.py` | ~150 | Signal handlers, shutdown sequencing, systemd watchdog |
| `runner/runner_loop.py` | ~120 | WS runtime, 1s poll, bar dispatch |
| `runner/run_trading.py` | ~120 | Assembly script (the new entry point) |
| `tests/unit/runner/test_trading_engine.py` | ~150 | |
| `tests/unit/runner/test_risk_manager.py` | ~100 | |
| `tests/unit/runner/test_order_manager.py` | ~100 | |
| `tests/unit/runner/test_binance_executor.py` | ~100 | |
| `tests/unit/runner/test_recovery_manager.py` | ~100 | |
| `tests/unit/runner/test_lifecycle_manager.py` | ~80 | |

**Existing files NOT modified:** `runner/live_runner.py`, `runner/config.py`, `runner/emit_handler.py`, `runner/graceful_shutdown.py`, `runner/recovery.py`, all `engine/`, `execution/`, `risk/`, `decision/` files.

---

## Chunk 1: Config + TradingEngine + RiskManager

### Task 1: TradingConfig (simplified config)

**Files:**
- Create: `runner/trading_config.py`
- Test: `tests/unit/runner/test_trading_config.py`

- [ ] **Step 1: Write test for TradingConfig defaults and factory methods**

```python
# tests/unit/runner/test_trading_config.py
"""Tests for simplified TradingConfig."""
import pytest
from runner.trading_config import TradingConfig


class TestTradingConfigDefaults:
    def test_default_symbols(self):
        cfg = TradingConfig(symbols=("BTCUSDT",))
        assert cfg.symbols == ("BTCUSDT",)
        assert cfg.testnet is True
        assert cfg.shadow_mode is True
        assert cfg.venue == "binance"

    def test_field_count_under_55(self):
        import dataclasses
        fields = dataclasses.fields(TradingConfig)
        assert len(fields) <= 55, f"TradingConfig has {len(fields)} fields, max 55"


class TestTradingConfigFactories:
    def test_paper(self):
        cfg = TradingConfig.paper(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.shadow_mode is True
        assert cfg.enable_reconcile is False

    def test_testnet(self):
        cfg = TradingConfig.testnet(symbols=["BTCUSDT"])
        assert cfg.testnet is True
        assert cfg.shadow_mode is False
        assert cfg.enable_reconcile is True

    def test_prod(self):
        cfg = TradingConfig.prod(symbols=["BTCUSDT"])
        assert cfg.testnet is False
        assert cfg.use_ws_orders is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/runner/test_trading_config.py -x -q`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement TradingConfig**

```python
# runner/trading_config.py
"""Simplified trading config — ~50 fields (down from 102)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union


@dataclass(frozen=True, slots=True)
class TradingConfig:
    """Lean config for the decomposed runner. No experimental flags."""

    # Identity
    symbols: tuple[str, ...]
    venue: str = "binance"
    testnet: bool = True
    shadow_mode: bool = True
    currency: str = "USDT"

    # Market data
    ws_base_url: str = "wss://fstream.binance.com/stream"
    kline_interval: str = "1m"

    # Feature / ML
    model_dir: str = "models_v8"

    # Signal / strategy
    deadzone: Union[float, Dict[str, float]] = 0.5
    min_hold_bars: Optional[Dict[str, int]] = None
    long_only_symbols: Optional[set] = None
    trend_follow: bool = False
    trend_indicator: str = "tf4h_close_vs_ma20"
    trend_threshold: float = 0.0
    max_hold: int = 120
    monthly_gate: bool = False
    monthly_gate_window: Union[int, Dict[str, int]] = 480
    vol_target: Union[None, float, Dict[str, Optional[float]]] = None
    vol_feature: Union[str, Dict[str, str]] = "atr_norm_14"

    # Risk / control
    max_gross_leverage: float = 3.0
    max_net_leverage: float = 1.0
    max_concentration: float = 0.4
    dd_warning_pct: float = 10.0
    dd_reduce_pct: float = 15.0
    dd_kill_pct: float = 20.0
    pending_order_timeout_sec: float = 30.0
    margin_check_interval_sec: float = 30.0
    margin_warning_ratio: float = 0.15
    margin_critical_ratio: float = 0.08

    # Execution
    use_ws_orders: bool = False
    enable_preflight: bool = True
    preflight_min_balance: float = 0.0

    # Recovery / persistence
    data_dir: str = "data/live"
    enable_persistent_stores: bool = True
    enable_reconcile: bool = True
    reconcile_interval_sec: float = 60.0
    reconcile_on_startup: bool = True
    checkpoint_interval_sec: float = 300.0

    # Monitoring
    enable_monitoring: bool = True
    health_port: Optional[int] = None
    health_host: str = "127.0.0.1"
    enable_alpha_health: bool = True
    enable_structured_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Model registry (optional)
    model_registry_db: Optional[str] = None
    model_names: tuple[str, ...] = ()

    @classmethod
    def paper(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=True, shadow_mode=True,
                   enable_reconcile=False, **kw)

    @classmethod
    def testnet(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=True, shadow_mode=False,
                   enable_reconcile=True, use_ws_orders=True, **kw)

    @classmethod
    def prod(cls, *, symbols: list[str], **kw) -> "TradingConfig":
        return cls(symbols=tuple(symbols), testnet=False, shadow_mode=False,
                   enable_reconcile=True, use_ws_orders=True, **kw)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/runner/test_trading_config.py -x -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add runner/trading_config.py tests/unit/runner/test_trading_config.py
git commit -m "feat(runner): add TradingConfig — simplified 50-field config"
```

---

### Task 2: TradingEngine (features + inference + model reload)

**Files:**
- Create: `runner/trading_engine.py`
- Test: `tests/unit/runner/test_trading_engine.py`
- Read: `runner/live_runner.py:397-488` (`_build_features_and_inference`), `runner/live_runner.py:1816-1897` (`_handle_model_reload`)

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_trading_engine.py
"""Tests for TradingEngine — uses mocks (no Rust build required)."""
from unittest.mock import MagicMock, patch
import pytest


class TestTradingEngineOnBar:
    def test_on_bar_returns_signal_when_prediction_nonzero(self):
        from runner.trading_engine import TradingEngine
        hook = MagicMock()
        bridge = MagicMock()
        bridge.predict.return_value = 0.8
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        result = engine.on_bar("BTCUSDT", {"close": 70000, "volume": 100})
        assert result is not None
        hook.on_bar.assert_called_once()

    def test_on_bar_returns_none_when_no_prediction(self):
        from runner.trading_engine import TradingEngine
        hook = MagicMock()
        bridge = MagicMock()
        bridge.predict.return_value = 0.0
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        result = engine.on_bar("BTCUSDT", {"close": 70000, "volume": 100})
        assert result is None


class TestTradingEngineReload:
    def test_reload_models_scans_model_dir(self):
        from runner.trading_engine import TradingEngine
        hook = MagicMock()
        bridge = MagicMock()
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        with patch("runner.trading_engine.Path") as mock_path:
            mock_path.return_value.glob.return_value = []
            result = engine.reload_models()
        assert isinstance(result, dict)


class TestTradingEngineCheckpoint:
    def test_checkpoint_returns_dict(self):
        from runner.trading_engine import TradingEngine
        hook = MagicMock()
        hook.checkpoint.return_value = {"features": {}}
        bridge = MagicMock()
        bridge.get_state.return_value = {"inference": {}}
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        state = engine.checkpoint()
        assert "features" in state or "inference" in state

    def test_restore_calls_hook_and_bridge(self):
        from runner.trading_engine import TradingEngine
        hook = MagicMock()
        bridge = MagicMock()
        engine = TradingEngine(feature_hook=hook, inference_bridge=bridge,
                               symbols=["BTCUSDT"], model_dir="models_v8")
        engine.restore({"features": {}, "inference": {}})
        # Should not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/runner/test_trading_engine.py -x -q`

- [ ] **Step 3: Implement TradingEngine**

Extract logic from `live_runner.py:397-488` (features/inference build) and `live_runner.py:1816-1897` (model reload) into a clean class. The class wraps `FeatureComputeHook` and `RustInferenceBridge` — it does NOT reimplement them.

Key methods:
- `on_bar(symbol, bar)` → calls `feature_hook.on_bar()`, then `inference_bridge.predict()`, returns Signal or None
- `reload_models()` → scans `model_dir/`, loads `.pkl` files, calls `inference_bridge.update_models()`
- `checkpoint()` / `restore()` → delegates to `feature_hook.checkpoint()` and inference bridge state

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/runner/test_trading_engine.py -x -q`

- [ ] **Step 5: Commit**

```bash
git add runner/trading_engine.py tests/unit/runner/test_trading_engine.py
git commit -m "feat(runner): add TradingEngine — features + inference + model reload"
```

---

### Task 3: RiskManager (kill switch + risk gate)

**Files:**
- Create: `runner/risk_manager.py`
- Test: `tests/unit/runner/test_risk_manager.py`
- Read: `runner/live_runner.py:181-230` (`_build_core_infra` for kill switch), `engine/guards.py` for RiskGate

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_risk_manager.py
"""Tests for RiskManager."""
from unittest.mock import MagicMock
import pytest


class TestRiskManagerCheck:
    def test_allows_normal_signal(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        allowed, reason = rm.check(signal=MagicMock(qty=0.1, notional=500), osm_open_count=0)
        assert allowed is True

    def test_blocks_when_kill_switch_active(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        ks.is_killed.return_value = True
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        allowed, reason = rm.check(signal=MagicMock(qty=0.1, notional=500), osm_open_count=0)
        assert allowed is False
        assert "kill" in reason.lower()

    def test_blocks_oversized_order(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=0.05,
                         max_notional=500.0, max_open_orders=5)
        allowed, reason = rm.check(signal=MagicMock(qty=1.0, notional=70000), osm_open_count=0)
        assert allowed is False

    def test_blocks_too_many_open_orders(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        ks.is_killed.return_value = False
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=3)
        allowed, reason = rm.check(signal=MagicMock(qty=0.1, notional=500), osm_open_count=3)
        assert allowed is False


class TestRiskManagerKill:
    def test_kill_activates_kill_switch(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        rm.kill("test reason")
        ks.activate.assert_called_once()


class TestRiskManagerCheckpoint:
    def test_checkpoint_returns_dict(self):
        from runner.risk_manager import RiskManager
        ks = MagicMock()
        ks.get_state.return_value = {"killed": False}
        rm = RiskManager(kill_switch=ks, max_position=1.0,
                         max_notional=10000.0, max_open_orders=5)
        state = rm.checkpoint()
        assert isinstance(state, dict)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/unit/runner/test_risk_manager.py -x -q`

- [ ] **Step 3: Implement RiskManager**

Wraps `KillSwitch` and provides `check()` / `kill()` / `checkpoint()`. The `check()` method performs: kill switch check → position limit → notional limit → open order count. No gate chain complexity (CorrelationGate, RegimeSizer removed per spec).

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/runner/test_risk_manager.py -x -q`

- [ ] **Step 5: Commit**

```bash
git add runner/risk_manager.py tests/unit/runner/test_risk_manager.py
git commit -m "feat(runner): add RiskManager — kill switch + position/notional limits"
```

---

## Chunk 2: OrderManager + BinanceExecutor

### Task 4: OrderManager (OSM + timeout tracking)

**Files:**
- Create: `runner/order_manager.py`
- Test: `tests/unit/runner/test_order_manager.py`
- Read: `runner/live_runner.py:367-394` (`_build_order_infra`), `execution/order_state_machine.py`

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_order_manager.py
"""Tests for OrderManager."""
import time
import pytest


class TestOrderManagerLifecycle:
    def test_submit_increments_open_count(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        assert om.open_count == 1

    def test_on_fill_decrements_open_count(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.on_fill("order-1")
        assert om.open_count == 0

    def test_on_cancel_decrements_open_count(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.on_cancel("order-1")
        assert om.open_count == 0


class TestOrderManagerTimeout:
    def test_check_timeouts_empty_when_fresh(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        assert om.check_timeouts() == []

    def test_check_timeouts_detects_stale(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=0.01)  # instant timeout
        om.submit("order-1", "BTCUSDT")
        time.sleep(0.02)
        timed_out = om.check_timeouts()
        assert "order-1" in timed_out


class TestOrderManagerDuplicates:
    def test_duplicate_submit_ignored(self):
        from runner.order_manager import OrderManager
        om = OrderManager(timeout_sec=30.0)
        om.submit("order-1", "BTCUSDT")
        om.submit("order-1", "BTCUSDT")  # duplicate
        assert om.open_count == 1
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement OrderManager**

Simple dict-based tracking: `_orders: dict[str, OrderEntry]` with submit time, symbol, status. `check_timeouts()` scans for orders older than `timeout_sec`. `open_count` is a property counting non-terminal orders.

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add runner/order_manager.py tests/unit/runner/test_order_manager.py
git commit -m "feat(runner): add OrderManager — order lifecycle + timeout tracking"
```

---

### Task 5: BinanceExecutor (order submission + user stream)

**Files:**
- Create: `runner/binance_executor.py`
- Test: `tests/unit/runner/test_binance_executor.py`
- Read: `runner/live_runner.py:602-694` (`_build_execution`), `runner/live_runner.py:797-873` (`_build_user_stream`), `runner/live_runner.py:1948-1961` (`_FillRecordingAdapter`)

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_binance_executor.py
"""Tests for BinanceExecutor — mocked venue client."""
from unittest.mock import MagicMock
import pytest


class TestBinanceExecutorSend:
    def test_send_delegates_to_venue_client(self):
        from runner.binance_executor import BinanceExecutor
        client = MagicMock()
        ks = MagicMock()
        ks.is_killed.return_value = False
        exe = BinanceExecutor(venue_client=client, kill_switch=ks)
        exe.send(MagicMock())
        client.send_order.assert_called_once()

    def test_send_blocked_when_killed(self):
        from runner.binance_executor import BinanceExecutor
        client = MagicMock()
        ks = MagicMock()
        ks.is_killed.return_value = True
        exe = BinanceExecutor(venue_client=client, kill_switch=ks)
        result = exe.send(MagicMock())
        client.send_order.assert_not_called()
        assert result.get("blocked") or result.get("status") == "killed"


class TestBinanceExecutorShadow:
    def test_shadow_mode_does_not_send_real_orders(self):
        from runner.binance_executor import BinanceExecutor
        client = MagicMock()
        ks = MagicMock()
        ks.is_killed.return_value = False
        exe = BinanceExecutor(venue_client=client, kill_switch=ks, shadow_mode=True)
        exe.send(MagicMock())
        client.send_order.assert_not_called()


class TestBinanceExecutorFillRecording:
    def test_on_fill_callback_called(self):
        from runner.binance_executor import BinanceExecutor
        client = MagicMock()
        ks = MagicMock()
        fills = []
        exe = BinanceExecutor(venue_client=client, kill_switch=ks, on_fill=fills.append)
        exe._handle_fill({"order_id": "1", "qty": 0.1})
        assert len(fills) == 1
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement BinanceExecutor**

Wraps venue client with kill switch check. In `shadow_mode`, logs orders but doesn't send. Manages user stream thread (start/stop). Includes `_handle_fill()` callback that invokes `on_fill` and records to internal list. Moves `_FillRecordingAdapter` logic inline.

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add runner/binance_executor.py tests/unit/runner/test_binance_executor.py
git commit -m "feat(runner): add BinanceExecutor — order submission + user stream + shadow mode"
```

---

## Chunk 3: RecoveryManager + LifecycleManager + RunnerLoop + Assembly

### Task 6: RecoveryManager (checkpoint + restore + reconcile)

**Files:**
- Create: `runner/recovery_manager.py`
- Test: `tests/unit/runner/test_recovery_manager.py`
- Read: `runner/recovery.py` (existing, 973 LOC — wraps its functions), `runner/live_runner.py:876-1025` (`_build_persistence_and_recovery`), `runner/live_runner.py:1914-1945` (`_reconcile_startup`)

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_recovery_manager.py
"""Tests for RecoveryManager."""
from unittest.mock import MagicMock, patch
import pytest


class TestRecoveryManagerSave:
    def test_save_calls_checkpoint_on_engine_and_risk(self):
        from runner.recovery_manager import RecoveryManager
        engine = MagicMock()
        engine.checkpoint.return_value = {"features": {}}
        risk = MagicMock()
        risk.checkpoint.return_value = {"kill_switch": {}}
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/test_state",
                             engine=engine, risk=risk, orders=orders)
        rm.save()
        engine.checkpoint.assert_called_once()
        risk.checkpoint.assert_called_once()


class TestRecoveryManagerRestore:
    def test_restore_returns_false_when_no_state(self):
        from runner.recovery_manager import RecoveryManager
        engine = MagicMock()
        risk = MagicMock()
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/nonexistent_state_dir_test",
                             engine=engine, risk=risk, orders=orders)
        assert rm.restore() is False


class TestRecoveryManagerReconcile:
    def test_reconcile_returns_empty_when_matching(self):
        from runner.recovery_manager import RecoveryManager
        engine = MagicMock()
        risk = MagicMock()
        orders = MagicMock()
        rm = RecoveryManager(state_dir="/tmp/test_state",
                             engine=engine, risk=risk, orders=orders)
        executor = MagicMock()
        executor.get_positions.return_value = []
        mismatches = rm.reconcile_startup(executor)
        assert isinstance(mismatches, list)
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement RecoveryManager**

Delegates to existing `runner/recovery.py` functions (`save_all_auxiliary_state`, `restore_all_auxiliary_state`, `reconcile_and_heal`). The class owns the `state_dir` path, `interval_sec` for periodic saves, and orchestrates the 8-component bundle by collecting state from `engine`, `risk`, and `orders`.

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add runner/recovery_manager.py tests/unit/runner/test_recovery_manager.py
git commit -m "feat(runner): add RecoveryManager — 8-component checkpoint/restore"
```

---

### Task 7: LifecycleManager (start/stop sequencing + signals)

**Files:**
- Create: `runner/lifecycle_manager.py`
- Test: `tests/unit/runner/test_lifecycle_manager.py`
- Read: `runner/live_runner.py:1545-1708` (`start()`), `runner/live_runner.py:1710-1756` (`stop()`), `runner/live_runner.py:1490-1517` (`_apply_perf_tuning`)

- [ ] **Step 1: Write tests**

```python
# tests/unit/runner/test_lifecycle_manager.py
"""Tests for LifecycleManager."""
from unittest.mock import MagicMock, patch
import signal
import pytest


class TestLifecycleSignals:
    def test_sighup_triggers_model_reload(self):
        from runner.lifecycle_manager import LifecycleManager
        engine = MagicMock()
        executor = MagicMock()
        recovery = MagicMock()
        loop = MagicMock()
        lm = LifecycleManager(engine=engine, executor=executor,
                               recovery=recovery, loop=loop)
        lm._handle_sighup()
        engine.reload_models.assert_called_once()


class TestLifecycleShutdownOrder:
    def test_stop_calls_subsystems_in_order(self):
        from runner.lifecycle_manager import LifecycleManager
        engine = MagicMock()
        executor = MagicMock()
        recovery = MagicMock()
        loop = MagicMock()
        lm = LifecycleManager(engine=engine, executor=executor,
                               recovery=recovery, loop=loop)
        lm._running = True
        lm.stop()
        # User stream stopped before loop
        executor.stop_user_stream.assert_called_once()
        loop.stop.assert_called_once()
        # Recovery save called during shutdown
        recovery.save.assert_called_once()
```

- [ ] **Step 2: Run to verify failure**
- [ ] **Step 3: Implement LifecycleManager**

Owns `start()` and `stop()`. `start()` sequence: perf tuning → recovery.restore() → executor.start_user_stream() → install signal handlers → loop.start(). `stop()` sequence: user stream → recovery.save() → loop.stop() → executor cleanup. SIGHUP → engine.reload_models(). SIGTERM/SIGINT → stop().

- [ ] **Step 4: Run tests**
- [ ] **Step 5: Commit**

```bash
git add runner/lifecycle_manager.py tests/unit/runner/test_lifecycle_manager.py
git commit -m "feat(runner): add LifecycleManager — start/stop ordering + signal handling"
```

---

### Task 8: RunnerLoop (event loop + bar dispatch)

**Files:**
- Create: `runner/runner_loop.py`
- Test: (covered by integration in Task 9)
- Read: `runner/live_runner.py:697-735` (`_build_decision`), `runner/live_runner.py:738-794` (`_build_market_data`)

- [ ] **Step 1: Implement RunnerLoop**

Minimal class: wraps existing `EngineLoop` and `WsRuntime`. `on_bar()` dispatches to engine → risk check → emit handler. `poll()` checks order timeouts. `start()` enters WS event loop. `stop()` sets `_running = False`.

- [ ] **Step 2: Commit**

```bash
git add runner/runner_loop.py
git commit -m "feat(runner): add RunnerLoop — WS event loop + bar dispatch"
```

---

### Task 9: Assembly Script (`run_trading.py`)

**Files:**
- Create: `runner/run_trading.py`
- Read: `runner/live_runner.py:1153-1343` (`build()`), `runner/live_runner.py:1346-1487` (`from_config()`)

- [ ] **Step 1: Implement assembly script**

~120 LOC entry point. Each step explicitly creates one module, wires it to the next. Uses `TradingConfig.from_yaml()` for config loading. Incorporates model auto-discovery from `from_config()`.

```python
#!/usr/bin/env python3
"""Start live trading — transparent assembly, every step visible."""
# See spec: docs/superpowers/specs/2026-03-15-liverunner-decomposition-design.md
```

- [ ] **Step 2: Smoke test with --dry-run flag**

Run: `python3 -m runner.run_trading --config config/production.yaml --dry-run`
Expected: assembles all modules, prints config summary, exits without connecting

- [ ] **Step 3: Commit**

```bash
git add runner/run_trading.py
git commit -m "feat(runner): add run_trading.py — transparent assembly script"
```

---

## Chunk 4: Integration Verification

### Task 10: Integration test — full assembly with mocks

**Files:**
- Create: `tests/unit/runner/test_assembly_integration.py`

- [ ] **Step 1: Write integration test**

Tests that `run_trading.py`'s assembly logic can construct all modules with mocked dependencies. Verifies the wiring is correct without connecting to any exchange.

- [ ] **Step 2: Run all new tests together**

Run: `pytest tests/unit/runner/ -x -q -v`
Expected: all tests pass

- [ ] **Step 3: Verify old LiveRunner still works**

Run: `pytest tests/unit/ execution/tests/ -q --ignore=tests/unit/data --ignore=tests/unit/execution/test_async_binance.py`
Expected: same pass count as before (no regressions)

- [ ] **Step 4: Run testnet smoke test**

Run: `python3 -m scripts.testnet_smoke --public-only`
Expected: 3/4+ pass (public endpoints)

- [ ] **Step 5: Final commit + push**

```bash
git add tests/unit/runner/test_assembly_integration.py
git commit -m "test(runner): add integration test for decomposed runner assembly"
git push origin master
```

---

## Summary

| Task | Module | Est LOC | Tests |
|------|--------|---------|-------|
| 1 | TradingConfig | ~80 | 5 |
| 2 | TradingEngine | ~200 | 5 |
| 3 | RiskManager | ~120 | 7 |
| 4 | OrderManager | ~120 | 6 |
| 5 | BinanceExecutor | ~180 | 4 |
| 6 | RecoveryManager | ~180 | 3 |
| 7 | LifecycleManager | ~150 | 2 |
| 8 | RunnerLoop | ~120 | 0 (integration) |
| 9 | run_trading.py | ~120 | 0 (integration) |
| 10 | Integration test | — | 1 |
| **Total** | | **~1,270** | **33** |

Old `LiveRunner` (2,017 LOC) remains untouched and functional throughout.
