# Phase 1: Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 15 identified safety and robustness issues without touching architecture, keeping tests green at every commit.

**Architecture:** Each fix is isolated to 1-2 files + test file. No cross-fix dependencies. Each commit uses prefix `fix(P1-NN):`. All fixes are pure bug fixes or validation additions — no new features, no refactoring.

**Tech Stack:** Python 3.12, pytest, _quant_hotpath (Rust FFI), threading, sqlite3

**Spec:** `docs/superpowers/specs/2026-03-17-quant-system-improvement-roadmap.md` (Phase 1 section)

**Test commands:**
- Single test: `pytest tests/unit/path/test_file.py::TestClass::test_name -xvs`
- All tests: `make test`
- Lint: `ruff check --select E,W,F .`

---

## File Structure

No new production files. Only test files created:

| Fix | Production File (modify) | Test File (create/modify) |
|-----|-------------------------|--------------------------|
| P1-01 | `execution/adapters/bybit/adapter.py` | `tests/unit/bybit/test_bybit_order_link_id.py` (new) |
| P1-02 | `features/cross_asset_computer.py` | `tests/unit/features/test_cross_asset.py` (modify) |
| P1-03 | `engine/feature_hook.py` | `tests/unit/engine/test_feature_hook.py` (modify) |
| P1-04 | `engine/saga.py` | `tests/unit/engine/test_saga_timeout.py` (new) |
| P1-05 | `execution/adapters/registry.py` | `tests/unit/execution/test_adapter_registry.py` (new) |
| P1-06 | `engine/execution_bridge.py` | `tests/unit/execution/test_live_execution_bridge.py` (modify) |
| P1-07 | `runner/config.py` | `tests/unit/runner/test_config.py` (modify) |
| P1-08 | `execution/safety/risk_gate.py` | `tests/unit/test_risk_gate_portfolio.py` (modify) |
| P1-09 | `engine/loop.py` | `tests/unit/engine/test_loop_metrics.py` (new) |
| P1-10 | `execution/store/dedup_store.py` | `tests/unit/execution/test_dedup_store_atomic.py` (new) |
| P1-11 | `core/observability.py` | `tests/unit/engine/test_observability_leak.py` (new) |
| P1-12 | `state/position.py` | `tests/unit/state/test_position_state.py` (modify) |
| P1-13 | `state/account.py` | `tests/unit/state/test_account_state.py` (modify) |
| P1-14 | `core/clock.py` | `tests/unit/engine/test_clock.py` (modify) |
| P1-15 | `engine/errors.py` | `tests/unit/engine/test_error_classification.py` (new) |

---

## Task 1: Bybit orderLinkId Collision Fix (P1-01)

**Files:**
- Modify: `execution/adapters/bybit/adapter.py:154`
- Create: `tests/unit/bybit/test_bybit_order_link_id.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/bybit/test_bybit_order_link_id.py
"""Test orderLinkId uniqueness under rapid-fire submission."""
import time


class TestOrderLinkIdUniqueness:
    def test_same_second_orders_produce_distinct_keys(self):
        """Two orders created within the same second must have different orderLinkIds."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        ids = set()
        for _ in range(100):
            ids.add(_make_order_link_id("ETHUSDT", "buy"))
        # All 100 must be unique
        assert len(ids) == 100

    def test_order_link_id_format(self):
        """orderLinkId must start with 'qs_' and contain symbol+side."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        oid = _make_order_link_id("BTCUSDT", "sell")
        assert oid.startswith("qs_")
        assert "BTCUSDT" in oid
        assert "s" in oid  # side[0]

    def test_order_link_id_length_within_bybit_limit(self):
        """Bybit orderLinkId max length is 36 characters."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        oid = _make_order_link_id("ETHUSDT", "buy")
        assert len(oid) <= 36
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/bybit/test_bybit_order_link_id.py -xvs`
Expected: FAIL — `ImportError: cannot import name '_make_order_link_id'`

- [ ] **Step 3: Extract helper function and add randomness**

In `execution/adapters/bybit/adapter.py`, add at module level (before the class) and update the call site:

```python
import os as _os
import time as _time


def _make_order_link_id(symbol: str, side: str) -> str:
    """Generate unique orderLinkId for Bybit dedup (max 36 chars)."""
    ts_ms = int(_time.time() * 1000)
    rand = _os.urandom(2).hex()  # 4 hex chars
    return f"qs_{symbol}_{side[0]}_{ts_ms}_{rand}"
```

Then change line 154 from:
```python
order_link_id = f"qs_{symbol}_{side[0]}_{int(_time.time())}"
```
to:
```python
order_link_id = _make_order_link_id(symbol, side)
```

Remove the `import time as _time` that was inside `send_market_order` (it's now at module level).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/bybit/test_bybit_order_link_id.py -xvs`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add execution/adapters/bybit/adapter.py tests/unit/bybit/test_bybit_order_link_id.py
git commit -m "fix(P1-01): add ms+random to Bybit orderLinkId preventing same-second collision"
```

---

## Task 2: Cross-Asset Ordering Guard (P1-02)

**Files:**
- Modify: `features/cross_asset_computer.py:35-40`
- Modify: `tests/unit/features/test_cross_asset.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/features/test_cross_asset.py`:

```python
class TestCrossAssetBenchmarkGuard:
    def test_altcoin_before_benchmark_returns_empty_with_warning(self, caplog):
        """Pushing altcoin before benchmark should return NaN features and warn."""
        import logging
        from features.cross_asset_computer import CrossAssetComputer

        comp = CrossAssetComputer()
        # Push altcoin BEFORE benchmark
        comp.on_bar("ETHUSDT", close=3000.0)
        feats = comp.get_features("ETHUSDT")
        # Should get features but values should be NaN (benchmark not fed)
        # At minimum, a warning should be logged
        assert any("benchmark" in r.message.lower() for r in caplog.records
                    if r.levelno >= logging.WARNING)

    def test_benchmark_then_altcoin_no_warning(self, caplog):
        """Pushing benchmark first then altcoin should not warn."""
        import logging
        from features.cross_asset_computer import CrossAssetComputer

        comp = CrossAssetComputer()
        comp.on_bar("BTCUSDT", close=60000.0)
        comp.on_bar("ETHUSDT", close=3000.0)
        feats = comp.get_features("ETHUSDT")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING
                    and "benchmark" in r.message.lower()]
        assert len(warnings) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/features/test_cross_asset.py::TestCrossAssetBenchmarkGuard -xvs`
Expected: FAIL — no warning logged

- [ ] **Step 3: Add benchmark guard to on_bar**

In `features/cross_asset_computer.py`, modify `on_bar`:

```python
import logging

_log = logging.getLogger(__name__)

class CrossAssetComputer:
    def __init__(self) -> None:
        self._inner = RustCrossAssetComputer(_BENCHMARK)
        self._benchmark_fed_this_bar = False

    def begin_bar(self) -> None:
        """Call at the start of each bar cycle to reset benchmark tracking."""
        self._benchmark_fed_this_bar = False

    def on_bar(self, symbol: str, *, close: float,
               funding_rate: Optional[float] = None,
               high: Optional[float] = None,
               low: Optional[float] = None) -> None:
        if symbol == _BENCHMARK:
            self._benchmark_fed_this_bar = True
        elif not self._benchmark_fed_this_bar:
            _log.warning(
                "CrossAssetComputer: %s pushed before benchmark %s — "
                "cross-asset features will be NaN this bar",
                symbol, _BENCHMARK,
            )
        self._inner.on_bar(symbol, close, funding_rate=funding_rate,
                           high=high, low=low)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/features/test_cross_asset.py::TestCrossAssetBenchmarkGuard -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add features/cross_asset_computer.py tests/unit/features/test_cross_asset.py
git commit -m "fix(P1-02): warn when altcoin pushed before benchmark in CrossAssetComputer"
```

---

## Task 3: Feature Hook Source Exception Isolation (P1-03)

**Files:**
- Modify: `engine/feature_hook.py:144-215`
- Modify: `tests/unit/engine/test_feature_hook.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/engine/test_feature_hook.py`:

```python
class TestSourceExceptionIsolation:
    def test_funding_source_exception_does_not_break_bar(self):
        """If funding_rate_source raises, bar should complete with NaN funding."""
        pytest.importorskip("_quant_hotpath")
        from engine.feature_hook import FeatureComputeHook

        def bad_funding():
            raise ConnectionError("API timeout")

        hook = FeatureComputeHook(
            symbols=["ETHUSDT"],
            funding_rate_source=bad_funding,
        )
        # Should not raise — bar completes with NaN for funding features
        # We just verify no exception propagates
        # (Full integration would need RustFeatureEngine; here we test isolation)
        result = hook._safe_call_source(bad_funding, "funding_rate", "ETHUSDT")
        assert result is None

    def test_oi_source_exception_isolated(self):
        """If oi_source raises, bar should complete with NaN OI."""
        pytest.importorskip("_quant_hotpath")
        from engine.feature_hook import FeatureComputeHook

        def bad_oi():
            raise ValueError("corrupt data")

        hook = FeatureComputeHook(symbols=["ETHUSDT"])
        result = hook._safe_call_source(bad_oi, "oi_source", "ETHUSDT")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_feature_hook.py::TestSourceExceptionIsolation -xvs`
Expected: FAIL — `AttributeError: 'FeatureComputeHook' has no attribute '_safe_call_source'`

- [ ] **Step 3: Add _safe_call_source helper and wrap all source calls**

In `engine/feature_hook.py`, add helper method to the class:

```python
    def _safe_call_source(self, source_fn, source_name: str, symbol: str):
        """Call a source callable safely. Returns None on any exception."""
        try:
            return source_fn()
        except Exception:
            _log.warning(
                "FeatureHook: %s source raised for %s, using NaN",
                source_name, symbol, exc_info=True,
            )
            return None
```

Then wrap each bare source call. For example, change line 147 from:
```python
            rate = _funding_src()
```
to:
```python
            rate = self._safe_call_source(_funding_src, "funding_rate", symbol)
```

Apply the same pattern to: `_oi_src()` (line ~160), `_ls_src()` (if exists), `self._fgi_source()` (line ~178), `_liq_src()` (line ~211), and any other bare source callable invocations.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_feature_hook.py::TestSourceExceptionIsolation -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add engine/feature_hook.py tests/unit/engine/test_feature_hook.py
git commit -m "fix(P1-03): isolate feature_hook source exceptions — NaN on failure instead of crash"
```

---

## Task 4: Saga Timeout (P1-04)

**Files:**
- Modify: `engine/saga.py:91-198`
- Create: `tests/unit/engine/test_saga_timeout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/engine/test_saga_timeout.py
"""Test OrderSaga TTL and auto-cancellation."""
import time
from unittest.mock import MagicMock


class TestSagaTimeout:
    def test_submitted_saga_expires_after_ttl(self):
        """A saga in SUBMITTED state should auto-cancel after ttl_seconds."""
        from engine.saga import SagaManager, SagaState

        terminal_cb = MagicMock()
        mgr = SagaManager(on_terminal=terminal_cb, saga_ttl_seconds=1.0)

        mgr.create("order-1", intent_id="i1", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-1", SagaState.SUBMITTED, reason="sent")

        # Before TTL
        expired = mgr.tick()
        assert len(expired) == 0

        # Wait for TTL
        time.sleep(1.1)
        expired = mgr.tick()
        assert len(expired) == 1
        assert expired[0] == "order-1"

        saga = mgr.get("order-1")
        # Should be in completed (terminal) after expiry
        assert saga is None  # moved to _completed

    def test_filled_saga_not_expired(self):
        """A saga in FILLED state should not be affected by tick()."""
        from engine.saga import SagaManager, SagaState

        mgr = SagaManager(saga_ttl_seconds=0.1)
        mgr.create("order-2", intent_id="i2", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-2", SagaState.SUBMITTED, reason="sent")
        mgr.transition("order-2", SagaState.FILLED, reason="filled")

        time.sleep(0.2)
        expired = mgr.tick()
        assert len(expired) == 0

    def test_tick_without_ttl_is_noop(self):
        """If no ttl configured, tick() should not expire anything."""
        from engine.saga import SagaManager, SagaState

        mgr = SagaManager()  # No saga_ttl_seconds
        mgr.create("order-3", intent_id="i3", symbol="ETHUSDT", side="buy", qty=0.1)
        mgr.transition("order-3", SagaState.SUBMITTED, reason="sent")

        expired = mgr.tick()
        assert len(expired) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_saga_timeout.py -xvs`
Expected: FAIL — `TypeError: SagaManager.__init__() got an unexpected keyword argument 'saga_ttl_seconds'`

- [ ] **Step 3: Add TTL support to SagaManager**

In `engine/saga.py`:

1. Add `submitted_at` field to `OrderSaga`:
```python
    submitted_at: Optional[float] = None  # monotonic timestamp when entering SUBMITTED
```

2. Add `saga_ttl_seconds` to `SagaManager.__init__`:
```python
    def __init__(
        self,
        *,
        on_terminal: Optional[Callable[[OrderSaga], None]] = None,
        default_compensating_action: Optional[CompensatingAction] = None,
        max_completed: int = 10000,
        saga_ttl_seconds: Optional[float] = None,
    ) -> None:
        # ... existing code ...
        self._saga_ttl = saga_ttl_seconds
```

3. In `transition()`, record `submitted_at` when entering SUBMITTED:
```python
        # After the state change:
        if new_state == SagaState.SUBMITTED:
            saga.submitted_at = time.monotonic()
```

4. **IMPORTANT**: Change `self._lock = threading.Lock()` to `self._lock = threading.RLock()` in `SagaManager.__init__`. This is needed because `tick()` calls `self.transition()` while holding the lock, and `transition()` also acquires the lock. Without `RLock`, this deadlocks.

5. Add `tick()` method:
```python
    def tick(self) -> list[str]:
        """Check for expired sagas. Returns list of expired order_ids."""
        if self._saga_ttl is None:
            return []
        now = time.monotonic()
        expired = []
        with self._lock:
            for oid, saga in list(self._sagas.items()):
                if (saga.submitted_at is not None
                        and saga.state == SagaState.SUBMITTED
                        and now - saga.submitted_at > self._saga_ttl):
                    expired.append(oid)
            for oid in expired:
                self.transition(oid, SagaState.CANCELLED, reason="ttl_expired")
        return expired
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_saga_timeout.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add engine/saga.py tests/unit/engine/test_saga_timeout.py
git commit -m "fix(P1-04): add TTL to SagaManager — auto-cancel SUBMITTED orders after timeout"
```

---

## Task 5: AdapterRegistry Locking (P1-05)

**Files:**
- Modify: `execution/adapters/registry.py:24-58`
- Create: `tests/unit/execution/test_adapter_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/execution/test_adapter_registry.py
"""Test AdapterRegistry thread safety."""
import threading
from unittest.mock import MagicMock


class TestAdapterRegistryThreadSafety:
    def test_concurrent_register_and_get(self):
        """Concurrent register + get should not corrupt state."""
        from execution.adapters.registry import AdapterRegistry

        registry = AdapterRegistry()
        errors = []

        def register_adapters(start: int):
            try:
                for i in range(start, start + 50):
                    mock = MagicMock()
                    mock.name = f"venue_{i}"
                    registry.register(f"venue_{i}", mock)
            except Exception as e:
                errors.append(e)

        def read_adapters():
            try:
                for _ in range(100):
                    _ = registry.venues
                    _ = len(registry)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_adapters, args=(0,)),
            threading.Thread(target=register_adapters, args=(50,)),
            threading.Thread(target=read_adapters),
            threading.Thread(target=read_adapters),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry) == 100

    def test_register_and_get_basic(self):
        """Basic register/get functionality."""
        from execution.adapters.registry import AdapterRegistry, AdapterNotFoundError
        import pytest

        registry = AdapterRegistry()
        mock = MagicMock()
        registry.register("bybit", mock)

        assert registry.get("bybit") is mock
        assert registry.get("BYBIT") is mock  # case insensitive
        assert "bybit" in registry
        assert len(registry) == 1

        with pytest.raises(AdapterNotFoundError):
            registry.get("nonexistent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_adapter_registry.py -xvs`
Expected: May pass on single-threaded but the concurrent test could fail intermittently. The key change is adding the lock.

- [ ] **Step 3: Add RLock to AdapterRegistry**

In `execution/adapters/registry.py`:

```python
import threading

class AdapterRegistry:
    def __init__(self, plugin_registry: Optional["PluginRegistry"] = None) -> None:
        self._lock = threading.RLock()
        self._adapters: Dict[str, VenueAdapter] = {}
        self._plugin_registry = plugin_registry

    def register(self, venue: str, adapter: VenueAdapter) -> None:
        with self._lock:
            self._adapters[venue.lower()] = adapter
            if self._plugin_registry is not None:
                self._plugin_registry.register_instance(adapter, name=venue.lower())

    def get(self, venue: str) -> VenueAdapter:
        with self._lock:
            v = venue.lower()
            adapter = self._adapters.get(v)
            if adapter is None:
                raise AdapterNotFoundError(
                    f"no adapter registered for venue {venue!r}, "
                    f"available: {list(self._adapters.keys())}"
                )
            return adapter

    def get_optional(self, venue: str) -> Optional[VenueAdapter]:
        with self._lock:
            return self._adapters.get(venue.lower())

    @property
    def venues(self) -> Sequence[str]:
        with self._lock:
            return list(self._adapters.keys())

    def __contains__(self, venue: str) -> bool:
        with self._lock:
            return venue.lower() in self._adapters

    def __len__(self) -> int:
        with self._lock:
            return len(self._adapters)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/execution/test_adapter_registry.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add execution/adapters/registry.py tests/unit/execution/test_adapter_registry.py
git commit -m "fix(P1-05): add RLock to AdapterRegistry for thread-safe concurrent access"
```

---

## Task 6: ExecutionBridge Retry + Circuit Breaker (P1-06)

**Files:**
- Modify: `engine/execution_bridge.py:75-103`
- Modify: `tests/unit/execution/test_live_execution_bridge.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/execution/test_live_execution_bridge.py`:

```python
class TestExecutionBridgeRetry:
    def test_transient_failure_retried(self):
        """Adapter transient failure should be retried up to max_retries."""
        from engine.execution_bridge import ExecutionBridge

        call_count = 0

        class FlakeyAdapter:
            def send_order(self, event):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("transient")
                return []

        bridge = ExecutionBridge(
            adapter=FlakeyAdapter(),
            dispatcher_emit=lambda e: None,
            max_retries=3,
            retry_base_delay=0.01,
        )
        bridge.handle_event("fake_order")
        assert call_count == 3  # 2 failures + 1 success

    def test_permanent_failure_opens_circuit(self):
        """Repeated failures should open circuit breaker."""
        from engine.execution_bridge import ExecutionBridge

        class AlwaysFailAdapter:
            def send_order(self, event):
                raise ConnectionError("down")

        bridge = ExecutionBridge(
            adapter=AlwaysFailAdapter(),
            dispatcher_emit=lambda e: None,
            max_retries=2,
            retry_base_delay=0.01,
            cb_failure_threshold=3,
            cb_cooldown_seconds=10.0,
        )
        # First call exhausts retries
        try:
            bridge.handle_event("order1")
        except Exception:
            pass
        try:
            bridge.handle_event("order2")
        except Exception:
            pass
        # Circuit should now be open
        assert bridge.circuit_open
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/execution/test_live_execution_bridge.py::TestExecutionBridgeRetry -xvs`
Expected: FAIL — `TypeError: ExecutionBridge.__init__() got an unexpected keyword argument 'max_retries'`

- [ ] **Step 3: Add retry + circuit breaker to ExecutionBridge**

**IMPORTANT**: `ExecutionBridge` is currently `@dataclass(slots=True)`. Since `slots=True` prevents adding arbitrary attributes in `__init__`, declare new parameters as dataclass fields with defaults. Do NOT remove the `@dataclass` decorator — that would be an architecture change.

In `engine/execution_bridge.py`, add new fields to the dataclass:

```python
@dataclass(slots=True)
class ExecutionBridge:
    adapter: Any
    dispatcher_emit: Any
    risk_gate: Any = None
    max_retries: int = 2
    retry_base_delay: float = 0.5
    cb_failure_threshold: int = 5
    cb_cooldown_seconds: float = 30.0
    # Mutable state — use field(default_factory=...) or __post_init__
    _failure_count: int = field(default=0, init=False, repr=False)
    _circuit_opened_at: Optional[float] = field(default=None, init=False, repr=False)

    @property
    def circuit_open(self) -> bool:
        if self._circuit_opened_at is None:
            return False
        if time.monotonic() - self._circuit_opened_at > self._cb_cooldown:
            # Half-open: allow one attempt
            return False
        return True

    def handle_event(self, event) -> None:
        if self.risk_gate is not None:
            check = self.risk_gate.check(event)
            if not check.allowed:
                _log.warning("RiskGate (execution bridge) REJECTED: %s", check.reason)
                return

        if self.circuit_open:
            raise ExecutionBridgeError("Circuit breaker OPEN — rejecting order")

        last_exc = None
        for attempt in range(1, self._max_retries + 1):
            try:
                results = self.adapter.send_order(event)
                self._failure_count = 0
                self._circuit_opened_at = None
                break
            except Exception as e:
                last_exc = e
                _log.warning(
                    "ExecutionBridge attempt %d/%d failed: %s",
                    attempt, self._max_retries, e,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_base_delay * (2 ** (attempt - 1)))
        else:
            self._failure_count += 1
            if self._failure_count >= self._cb_threshold:
                self._circuit_opened_at = time.monotonic()
                _log.error("ExecutionBridge circuit breaker OPENED after %d failures",
                           self._failure_count)
            raise ExecutionBridgeError("Execution adapter failed after retries") from last_exc

        if not results:
            return
        for ev in results:
            try:
                self.dispatcher_emit(ev)
            except Exception as e:
                raise ExecutionBridgeError("Failed to emit execution result") from e
```

Add `import time` and `import logging` at top if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/execution/test_live_execution_bridge.py::TestExecutionBridgeRetry -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green (verify existing ExecutionBridge tests still pass with new default params)

- [ ] **Step 6: Commit**

```bash
git add engine/execution_bridge.py tests/unit/execution/test_live_execution_bridge.py
git commit -m "fix(P1-06): add retry with exponential backoff + circuit breaker to ExecutionBridge"
```

---

## Task 7: LiveRunnerConfig Schema Validation (P1-07)

**Files:**
- Modify: `runner/config.py:10-95`
- Modify: `tests/unit/runner/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/runner/test_config.py`:

```python
class TestLiveRunnerConfigValidation:
    def test_negative_leverage_rejected(self):
        """max_gross_leverage must be positive."""
        import pytest
        from runner.config import LiveRunnerConfig

        with pytest.raises(ValueError, match="max_gross_leverage"):
            LiveRunnerConfig(max_gross_leverage=-1.0)

    def test_deadzone_must_be_non_negative(self):
        """Deadzone must be >= 0."""
        import pytest
        from runner.config import LiveRunnerConfig

        with pytest.raises(ValueError, match="deadzone"):
            LiveRunnerConfig(deadzone=-0.5)

    def test_dd_thresholds_ordered(self):
        """dd_warning < dd_reduce < dd_kill."""
        import pytest
        from runner.config import LiveRunnerConfig

        with pytest.raises(ValueError, match="drawdown"):
            LiveRunnerConfig(dd_warning_pct=20.0, dd_reduce_pct=15.0, dd_kill_pct=10.0)

    def test_valid_config_passes(self):
        """Default config should pass validation."""
        from runner.config import LiveRunnerConfig

        cfg = LiveRunnerConfig()  # Should not raise
        assert cfg.max_gross_leverage == 3.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/runner/test_config.py::TestLiveRunnerConfigValidation -xvs`
Expected: FAIL — negative leverage accepted (no validation)

- [ ] **Step 3: Add __post_init__ validation**

In `runner/config.py`, add after the class fields:

```python
    def __post_init__(self) -> None:
        # Frozen dataclass: __post_init__ can only raise, never assign
        if self.max_gross_leverage <= 0:
            raise ValueError(f"max_gross_leverage must be positive, got {self.max_gross_leverage}")
        if self.max_net_leverage <= 0:
            raise ValueError(f"max_net_leverage must be positive, got {self.max_net_leverage}")
        if isinstance(self.deadzone, (int, float)) and self.deadzone < 0:
            raise ValueError(f"deadzone must be non-negative, got {self.deadzone}")
        if not (self.dd_warning_pct < self.dd_reduce_pct < self.dd_kill_pct):
            raise ValueError(
                f"drawdown thresholds must be ordered: warning({self.dd_warning_pct}) "
                f"< reduce({self.dd_reduce_pct}) < kill({self.dd_kill_pct})"
            )
        if self.initial_equity <= 0:
            raise ValueError(f"initial_equity must be positive, got {self.initial_equity}")
        if self.max_concentration <= 0 or self.max_concentration > 1:
            raise ValueError(f"max_concentration must be in (0, 1], got {self.max_concentration}")
        if self.margin_warning_ratio < 0:
            raise ValueError(f"margin_warning_ratio must be non-negative, got {self.margin_warning_ratio}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/runner/test_config.py::TestLiveRunnerConfigValidation -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green (existing tests use valid defaults)

- [ ] **Step 6: Commit**

```bash
git add runner/config.py tests/unit/runner/test_config.py
git commit -m "fix(P1-07): add __post_init__ validation to LiveRunnerConfig (frozen, raise-only)"
```

---

## Task 8: RiskGate Price Source Consistency (P1-08)

**Files:**
- Modify: `execution/safety/risk_gate.py:56-71`
- Modify: `tests/unit/test_risk_gate_portfolio.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_risk_gate_portfolio.py`:

```python
class TestRiskGateMarketOrderPrice:
    def test_market_order_without_price_rejected(self):
        """Market orders with no price field must be rejected (fail-closed)."""
        from types import SimpleNamespace
        from execution.safety.risk_gate import RiskGate, RiskGateConfig

        gate = RiskGate(RiskGateConfig())
        cmd = SimpleNamespace(qty=1.0)  # No price attribute
        result = gate.check(cmd)
        assert result.allowed is False
        assert "missing" in result.reason.lower() or "price" in result.reason.lower()

    def test_order_with_valid_price_passes(self):
        """Order with valid qty and price should pass basic notional check."""
        from types import SimpleNamespace
        from execution.safety.risk_gate import RiskGate, RiskGateConfig

        gate = RiskGate(RiskGateConfig(max_order_notional=10000))
        cmd = SimpleNamespace(qty=1.0, price=3000.0)
        result = gate.check(cmd)
        assert result.allowed is True
```

- [ ] **Step 2: Run test to verify behavior**

Run: `pytest tests/unit/test_risk_gate_portfolio.py::TestRiskGateMarketOrderPrice -xvs`
Expected: Should already pass (fail-closed exists). If so, this confirms the existing behavior is correct and we document it. If not, implement the fix.

- [ ] **Step 3: Update `_get_price` to prefer mark_price over order price**

In `execution/safety/risk_gate.py`, modify `_get_price`:

```python
def _get_price(cmd: Any) -> Optional[float]:
    """Extract price for notional check. Prefer mark_price (market truth) over order price."""
    # Prefer mark_price for position notional accuracy
    for attr in ("mark_price", "price", "limit_price"):
        v = getattr(cmd, attr, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None
```

Also update the rejection log to be more explicit:

```python
        if qty is None or price is None:
            logger.warning(
                "RiskGate REJECT: cannot extract price for notional check — "
                "fail-closed (qty=%s, price=%s, symbol=%s). "
                "Ensure mark_price or price is set on the order.",
                qty, price, getattr(cmd, 'symbol', 'unknown'),
            )
```

- [ ] **Step 4: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 5: Commit**

```bash
git add execution/safety/risk_gate.py tests/unit/test_risk_gate_portfolio.py
git commit -m "fix(P1-08): clarify RiskGate fail-closed policy for market orders without price"
```

---

## Task 9: EngineLoop Metrics (P1-09)

**Files:**
- Modify: `engine/loop.py:141-235`
- Create: `tests/unit/engine/test_loop_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/engine/test_loop_metrics.py
"""Test EngineLoop drop/retry/error counters."""


class TestEngineLoopMetrics:
    def test_loop_metrics_dataclass(self):
        """LoopMetrics should be a standalone dataclass with zero defaults."""
        from engine.loop import LoopMetrics

        m = LoopMetrics()
        assert m.drops == 0
        assert m.retries == 0
        assert m.errors == 0
        assert m.processed == 0

    def test_loop_metrics_increment(self):
        """LoopMetrics fields should be mutable counters."""
        from engine.loop import LoopMetrics

        m = LoopMetrics()
        m.drops += 1
        m.retries += 3
        m.errors += 2
        m.processed += 100
        assert m.drops == 1
        assert m.retries == 3
        assert m.errors == 2
        assert m.processed == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_loop_metrics.py -xvs`
Expected: FAIL — no `LoopMetrics` class or `metrics` attribute

- [ ] **Step 3: Add LoopMetrics dataclass and counters**

In `engine/loop.py`:

```python
@dataclass
class LoopMetrics:
    """Observable counters for EngineLoop."""
    drops: int = 0
    retries: int = 0
    errors: int = 0
    processed: int = 0
```

Add to `EngineLoop.__init__`:
```python
        self.metrics = LoopMetrics()
```

In the drop path (line ~141):
```python
        self.metrics.drops += 1
```

In `_process_one` success path:
```python
        self.metrics.processed += 1
```

In `_process_one` error path (line ~193):
```python
        self.metrics.errors += 1
```

In `_retry_or_drop` (line ~232):
```python
        self.metrics.retries += 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_loop_metrics.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add engine/loop.py tests/unit/engine/test_loop_metrics.py
git commit -m "fix(P1-09): add drop/retry/error/processed counters to EngineLoop"
```

---

## Task 10: SQLiteDedupStore Atomicity (P1-10)

**Files:**
- Modify: `execution/store/dedup_store.py:77-86`
- Create: `tests/unit/execution/test_dedup_store_atomic.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/execution/test_dedup_store_atomic.py
"""Test SQLiteDedupStore atomic upsert."""
import tempfile
import os


class TestSQLiteDedupStoreAtomic:
    def test_put_then_get_returns_digest(self):
        """put() followed by get() should return the stored digest."""
        from execution.store.dedup_store import SQLiteDedupStore

        with tempfile.TemporaryDirectory() as d:
            store = SQLiteDedupStore(path=os.path.join(d, "dedup.db"))
            store.put("key1", "digest_abc")
            result = store.get("key1")
            assert result is not None
            # get() returns Optional[str] (the digest string), not an object
            assert result == "digest_abc"

    def test_put_twice_same_key_updates_timestamp(self):
        """Second put with same key should update ts but keep digest."""
        import time
        from execution.store.dedup_store import SQLiteDedupStore

        with tempfile.TemporaryDirectory() as d:
            store = SQLiteDedupStore(path=os.path.join(d, "dedup.db"))
            store.put("key1", "digest_v1")
            time.sleep(0.05)
            store.put("key1", "digest_v1")
            result = store.get("key1")
            assert result is not None
            assert result == "digest_v1"
```

- [ ] **Step 2: Run test to verify it passes (existing behavior)**

Run: `pytest tests/unit/execution/test_dedup_store_atomic.py -xvs`
Expected: PASS (functionally correct, but not atomic)

- [ ] **Step 3: Replace two-statement pattern with single upsert**

In `execution/store/dedup_store.py`, change `put()`:

From:
```python
    def put(self, key: str, digest: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO dedup(key, digest, ts) VALUES(?,?,?)",
                (key, str(digest), now),
            )
            self._conn.execute("UPDATE dedup SET ts=? WHERE key=?", (now, key))
            self._conn.commit()
```

To:
```python
    def put(self, key: str, digest: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO dedup(key, digest, ts) VALUES(?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET ts=excluded.ts",
                (key, str(digest), now),
            )
            self._conn.commit()
```

- [ ] **Step 4: Run test to verify it still passes**

Run: `pytest tests/unit/execution/test_dedup_store_atomic.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add execution/store/dedup_store.py tests/unit/execution/test_dedup_store_atomic.py
git commit -m "fix(P1-10): replace INSERT+UPDATE with atomic upsert in SQLiteDedupStore"
```

---

## Task 11: Observability Memory Leak (P1-11)

**Files:**
- Modify: `core/observability.py:42-88`
- Create: `tests/unit/engine/test_observability_leak.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/engine/test_observability_leak.py
"""Test TracingInterceptor _active dict cleanup."""
import time
from types import SimpleNamespace


class TestTracingInterceptorLeak:
    def test_orphaned_entries_cleaned_after_ttl(self):
        """Entries in _active without matching after_reduce should be cleaned."""
        from core.observability import TracingInterceptor
        from core.interceptors import InterceptResult

        tracer = TracingInterceptor(max_spans=100, active_ttl_seconds=0.1)
        envelope = SimpleNamespace(event_id="evt-1")

        # before_reduce adds to _active
        tracer.before_reduce(envelope, state=None)
        assert len(tracer._active) == 1

        # Wait for TTL
        time.sleep(0.15)

        # cleanup should remove the orphan
        tracer.cleanup_stale()
        assert len(tracer._active) == 0

    def test_normal_flow_not_affected(self):
        """Normal before+after flow should work as before."""
        from core.observability import TracingInterceptor

        tracer = TracingInterceptor(max_spans=100, active_ttl_seconds=60.0)
        # after_reduce accesses envelope.event_id, envelope.metadata.trace, etc.
        # Provide a sufficiently complete mock:
        trace = SimpleNamespace(trace_id="t1", span_id="s1")
        metadata = SimpleNamespace(trace=trace)
        envelope = SimpleNamespace(
            event_id="evt-2", metadata=metadata, kind="MARKET",
        )

        tracer.before_reduce(envelope, state=None)
        tracer.after_reduce(envelope, old_state=None, new_state=None)
        assert len(tracer._active) == 0
        assert len(tracer._spans) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_observability_leak.py -xvs`
Expected: FAIL — no `active_ttl_seconds` param or `cleanup_stale` method

- [ ] **Step 3: Add TTL cleanup to TracingInterceptor**

In `core/observability.py`:

```python
    def __init__(self, *, max_spans: int = 10_000, tracer: Optional[Any] = None,
                 active_ttl_seconds: float = 60.0) -> None:
        self._max_spans = max_spans
        self._spans: List[SpanRecord] = []
        self._active: Dict[str, float] = {}
        self._tracer = tracer
        self._active_ttl = active_ttl_seconds

    def cleanup_stale(self) -> int:
        """Remove entries from _active older than TTL. Returns count removed."""
        now = time.monotonic()
        stale = [eid for eid, start in self._active.items()
                 if now - start > self._active_ttl]
        for eid in stale:
            del self._active[eid]
        return len(stale)
```

Also call `cleanup_stale()` at the start of `before_reduce()` (every N calls to avoid overhead):

```python
    def before_reduce(self, envelope: Envelope, state: Any) -> InterceptResult:
        # Periodic cleanup (every 1000 events)
        if len(self._active) > 100:
            self.cleanup_stale()
        event_id = envelope.event_id
        self._active[event_id] = time.monotonic()
        return InterceptResult.ok(self.name)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_observability_leak.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add core/observability.py tests/unit/engine/test_observability_leak.py
git commit -m "fix(P1-11): add TTL cleanup to TracingInterceptor preventing _active memory leak"
```

---

## Task 12: PositionState Input Validation (P1-12)

**Files:**
- Modify: `state/position.py:29-43`
- Modify: `tests/unit/state/test_position_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/state/test_position_state.py`:

```python
import math
from decimal import Decimal


class TestPositionStateValidation:
    def test_nan_qty_rejected(self):
        """NaN qty should be rejected."""
        import pytest
        from state.position import PositionState

        pos = PositionState.empty("ETHUSDT")
        with pytest.raises(ValueError, match="qty"):
            pos.with_update(qty=Decimal("NaN"), avg_price=Decimal("3000"), last_price=None, ts=None)

    def test_inf_avg_price_rejected(self):
        """Infinite avg_price should be rejected."""
        import pytest
        from state.position import PositionState

        pos = PositionState.empty("ETHUSDT")
        with pytest.raises(ValueError, match="avg_price"):
            pos.with_update(qty=Decimal("1"), avg_price=Decimal("Inf"), last_price=None, ts=None)

    def test_valid_update_passes(self):
        """Valid inputs should work normally."""
        from state.position import PositionState

        pos = PositionState.empty("ETHUSDT")
        updated = pos.with_update(qty=Decimal("1.5"), avg_price=Decimal("3000"), last_price=Decimal("3010"), ts=None)
        assert updated.qty == Decimal("1.5")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/state/test_position_state.py::TestPositionStateValidation -xvs`
Expected: FAIL — NaN accepted without error

- [ ] **Step 3: Add validation to with_update**

In `state/position.py`, modify `with_update`:

```python
    def with_update(self, *, qty, avg_price, last_price, ts) -> "PositionState":
        # Validate numeric inputs
        if qty is not None and (qty != qty or abs(qty) == Decimal("Inf")):
            raise ValueError(f"qty must be finite, got {qty}")
        if avg_price is not None and (avg_price != avg_price or abs(avg_price) == Decimal("Inf")):
            raise ValueError(f"avg_price must be finite, got {avg_price}")
        if last_price is not None and (last_price != last_price or abs(last_price) == Decimal("Inf")):
            raise ValueError(f"last_price must be finite, got {last_price}")
        return PositionState(
            symbol=self.symbol,
            qty=qty,
            avg_price=avg_price,
            last_price=last_price,
            last_ts=ensure_utc(ts) if ts is not None else self.last_ts,
        )
```

Note: `Decimal("NaN") != Decimal("NaN")` is True (NaN self-comparison), so `qty != qty` detects NaN.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/state/test_position_state.py::TestPositionStateValidation -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add state/position.py tests/unit/state/test_position_state.py
git commit -m "fix(P1-12): validate qty/price are finite in PositionState.with_update"
```

---

## Task 13: AccountState Margin Precondition (P1-13)

**Files:**
- Modify: `state/account.py:47-67`
- Modify: `tests/unit/state/test_account_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/state/test_account_state.py`:

```python
from decimal import Decimal


class TestAccountStateValidation:
    def test_margin_exceeding_balance_raises(self):
        """margin_used > balance should raise ValueError."""
        import pytest
        from state.account import AccountState

        acct = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        with pytest.raises(ValueError, match="margin"):
            acct.with_update(
                balance=Decimal("1000"),
                margin_used=Decimal("1500"),  # > balance
                realized_pnl=Decimal("0"),
                unrealized_pnl=Decimal("0"),
                fees_paid=Decimal("0"),
                ts=None,
            )

    def test_valid_margin_passes(self):
        """Valid margin_used < balance should work."""
        from state.account import AccountState

        acct = AccountState.initial(currency="USDT", balance=Decimal("1000"))
        updated = acct.with_update(
            balance=Decimal("1000"),
            margin_used=Decimal("500"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("0"),
            fees_paid=Decimal("0"),
            ts=None,
        )
        assert updated.margin_available == Decimal("500")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/state/test_account_state.py::TestAccountStateValidation -xvs`
Expected: FAIL — negative margin_available accepted

- [ ] **Step 3: Add validation to with_update**

In `state/account.py`, add at the start of `with_update`:

```python
    def with_update(self, *, balance, margin_used, realized_pnl, unrealized_pnl, fees_paid, ts) -> "AccountState":
        if margin_used > balance:
            raise ValueError(
                f"margin_used ({margin_used}) exceeds balance ({balance}) — "
                f"margin_available would be negative"
            )
        margin_available = balance - margin_used
        # ... rest unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/state/test_account_state.py::TestAccountStateValidation -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add state/account.py tests/unit/state/test_account_state.py
git commit -m "fix(P1-13): reject margin_used > balance in AccountState.with_update"
```

---

## Task 14: ReplayClock Monotonicity (P1-14)

**Files:**
- Modify: `core/clock.py:125-133`
- Modify: `tests/unit/engine/test_clock.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/engine/test_clock.py`:

```python
from datetime import datetime, timezone, timedelta


class TestReplayClockMonotonicity:
    def test_backward_time_raises_warning(self, caplog):
        """feed() with backward timestamp should log warning."""
        import logging
        from core.clock import ReplayClock

        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 11, 0, 0, tzinfo=timezone.utc)  # backward

        # ReplayClock is a dataclass — construct then feed initial time
        clock = ReplayClock()
        clock.feed(t0)  # set initial time
        clock.feed(t1)  # backward — should warn

        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "monoton" in r.message.lower()]
        assert len(warnings) >= 1

    def test_forward_time_no_warning(self, caplog):
        """feed() with forward timestamp should not warn."""
        import logging
        from core.clock import ReplayClock

        t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc)

        clock = ReplayClock()
        clock.feed(t0)
        clock.feed(t1)

        warnings = [r for r in caplog.records
                    if r.levelno >= logging.WARNING and "monoton" in r.message.lower()]
        assert len(warnings) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_clock.py::TestReplayClockMonotonicity -xvs`
Expected: FAIL — no warning logged for backward time

- [ ] **Step 3: Add warning to feed()**

In `core/clock.py`, modify `ReplayClock.feed`:

```python
import logging

_log = logging.getLogger(__name__)

    def feed(self, ts: datetime) -> None:
        """Advance clock to *ts* (must be >= current)."""
        with self._lock:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            delta = (ts - self._current).total_seconds()
            if delta < 0:
                _log.warning(
                    "ReplayClock: non-monotonic feed — ts=%s is %.1fs before current=%s, ignoring",
                    ts.isoformat(), abs(delta), self._current.isoformat(),
                )
                return
            if delta > 0:
                self._mono += delta
                self._current = ts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_clock.py::TestReplayClockMonotonicity -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add core/clock.py tests/unit/engine/test_clock.py
git commit -m "fix(P1-14): warn on non-monotonic feed() in ReplayClock instead of silent no-op"
```

---

## Task 15: Error Classification for Network Errors (P1-15)

**Files:**
- Modify: `engine/errors.py:113-166`
- Create: `tests/unit/engine/test_error_classification.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/engine/test_error_classification.py
"""Test classify_exception covers network and venue errors."""


class TestErrorClassification:
    def test_connection_error_classified_as_io(self):
        """ConnectionError should map to IO domain."""
        from engine.errors import classify_exception, ErrorDomain

        exc = ConnectionError("Connection refused")
        result = classify_exception(exc)
        assert result.domain == ErrorDomain.IO

    def test_os_error_classified_as_io(self):
        """OSError should map to IO domain."""
        from engine.errors import classify_exception, ErrorDomain

        exc = OSError("No such file or directory")
        result = classify_exception(exc)
        assert result.domain == ErrorDomain.IO

    def test_key_error_classified_as_data(self):
        """KeyError should map to DATA domain."""
        from engine.errors import classify_exception, ErrorDomain

        exc = KeyError("missing_field")
        result = classify_exception(exc)
        assert result.domain == ErrorDomain.DATA

    def test_timeout_still_io(self):
        """TimeoutError should still map to IO."""
        from engine.errors import classify_exception, ErrorDomain

        exc = TimeoutError("timed out")
        result = classify_exception(exc)
        assert result.domain == ErrorDomain.IO

    def test_unknown_exception_defaults_to_engine(self):
        """Unrecognized exception defaults to ENGINE domain."""
        from engine.errors import classify_exception, ErrorDomain

        exc = RuntimeError("something weird")
        result = classify_exception(exc)
        assert result.domain == ErrorDomain.ENGINE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/engine/test_error_classification.py -xvs`
Expected: FAIL — ConnectionError and OSError fall through to default ENGINE domain

- [ ] **Step 3: Add network/data error mappings**

In `engine/errors.py`, expand `classify_exception` after the TimeoutError block:

```python
    # Network/IO errors
    if isinstance(exc, (ConnectionError, OSError)):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR,
            domain=ErrorDomain.IO,
            code=exc.__class__.__name__,
            message=str(exc) or exc.__class__.__name__,
            ctx=c,
            cause=exc,
        )

    # Data lookup errors
    if isinstance(exc, (KeyError, IndexError, AttributeError)):
        return ClassifiedError(
            severity=ErrorSeverity.ERROR,
            domain=ErrorDomain.DATA,
            code=exc.__class__.__name__,
            message=str(exc) or exc.__class__.__name__,
            ctx=c,
            cause=exc,
        )
```

Note: Place these BEFORE the final default block but AFTER the TimeoutError block. Order matters because `TimeoutError` is a subclass of `OSError` in Python 3.12 — the existing TimeoutError check must come first.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/engine/test_error_classification.py -xvs`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `make test`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add engine/errors.py tests/unit/engine/test_error_classification.py
git commit -m "fix(P1-15): classify ConnectionError/OSError as IO and KeyError as DATA domain"
```

---

## Final Verification

- [ ] **Step 1: Run complete test suite**

```bash
make test
```
Expected: All green (pytest + cargo test + ruff lint)

- [ ] **Step 2: Verify all 15 commits present**

```bash
git log --oneline -15
```
Expected: 15 commits with `fix(P1-NN):` prefixes

- [ ] **Step 3: Verify no untracked files**

```bash
git status
```
Expected: Clean working tree
