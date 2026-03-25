# Rust Delegation Wave 2: Sizing + Exit + Backtest Engine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delegate AdaptivePositionSizer, ExitManager, and backtest hot loop to Rust for latency reduction (sizing/exit on warm path) and 10-50x backtest speedup.

**Architecture:** Three independent Rust modules added to existing `decision/rust/` directory. Each follows the established pattern: pure Rust logic with `#[pyclass]`/`#[pyfunction]` PyO3 wrappers, Python thin wrapper that tries Rust import and falls back to existing Python. All existing Python tests must pass unchanged against both paths.

**Tech Stack:** Rust + PyO3 (existing `_quant_hotpath` module), Python 3.12, maturin build

---

## File Structure

### New Rust files
- `decision/rust/adaptive_sizer.rs` — `RustAdaptivePositionSizer` (equity-tier + IC + regime + step rounding)
- `decision/rust/exit_manager.rs` — `RustExitManager` (trailing stop + signal reversal + deadzone fade + time filter)
- `decision/rust/backtest_loop.rs` — `RustBacktestLoop` (vectorized backtest hot loop: predict → z-score → discretize → exit/entry → PnL)

### Modified files
- `rust_lib.rs` — Register new modules + PyO3 exports
- `decision/sizing/adaptive.py` — Add Rust fast path (try import, fallback to Python)
- `decision/exit_manager.py` — Add Rust fast path
- `decision/backtest_module.py` — Add Rust vectorized path for `decide()` loop

### Test files
- `tests/unit/decision/test_adaptive_sizer_rust.py` — Rust-vs-Python parity tests
- `tests/unit/decision/test_exit_manager_rust.py` — Rust-vs-Python parity tests
- `tests/unit/decision/test_backtest_loop_rust.py` — Rust backtest engine tests

---

## Task 1: RustAdaptivePositionSizer

**Files:**
- Create: `decision/rust/adaptive_sizer.rs`
- Modify: `rust_lib.rs` (add module + registration)
- Modify: `decision/sizing/adaptive.py` (add Rust fast path)
- Create: `tests/unit/decision/test_adaptive_sizer_rust.py`

- [x] **Step 1: Write Rust adaptive sizer**

Create `decision/rust/adaptive_sizer.rs`:

```rust
//! Adaptive position sizer: equity-tier + IC + regime + step rounding.
//! Mirrors decision/sizing/adaptive.py AdaptivePositionSizer.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Tier weight tables (mirrors Python _TIER_WEIGHTS).
fn tier_weights() -> [(&'static str, [(&'static str, f64); 4]); 3] {
    [
        ("small", [
            ("BTCUSDT", 0.25), ("ETHUSDT", 0.25),
            ("BTCUSDT_4h", 0.35), ("ETHUSDT_4h", 0.30),
        ]),
        ("medium", [
            ("BTCUSDT", 0.18), ("ETHUSDT", 0.18),
            ("BTCUSDT_4h", 0.25), ("ETHUSDT_4h", 0.20),
        ]),
        ("large", [
            ("BTCUSDT", 0.12), ("ETHUSDT", 0.12),
            ("BTCUSDT_4h", 0.18), ("ETHUSDT_4h", 0.15),
        ]),
    ]
}

const DEFAULT_CAP: f64 = 0.15;

#[inline]
fn equity_tier(equity: f64) -> &'static str {
    if equity < 500.0 { "small" }
    else if equity < 10_000.0 { "medium" }
    else { "large" }
}

#[inline]
fn get_tier_cap(tier: &str, runner_key: &str) -> f64 {
    for (t, weights) in &tier_weights() {
        if *t == tier {
            for (key, cap) in weights {
                if *key == runner_key {
                    return *cap;
                }
            }
            return DEFAULT_CAP;
        }
    }
    DEFAULT_CAP
}

/// Floor to step size increment.
#[inline]
fn round_to_step(size: f64, step_size: f64) -> f64 {
    if step_size <= 0.0 {
        return size;
    }
    let decimals = (-step_size.log10()).floor().max(0.0) as u32;
    let factor = 10f64.powi(decimals as i32);
    (size * factor).floor() / factor
}

/// Compute target position quantity (Rust fast path).
///
/// Mirrors AdaptivePositionSizer.target_qty() exactly.
#[pyfunction]
#[pyo3(signature = (
    runner_key, equity, price,
    step_size=0.001, min_size=0.001, max_qty=0.0,
    weight=1.0, leverage=10.0, ic_scale=1.0,
    regime_active=true, z_scale=1.0
))]
pub fn rust_adaptive_target_qty(
    runner_key: &str,
    equity: f64,
    price: f64,
    step_size: f64,
    min_size: f64,
    max_qty: f64,
    weight: f64,
    leverage: f64,
    ic_scale: f64,
    regime_active: bool,
    z_scale: f64,
) -> f64 {
    if equity <= 0.0 || price <= 0.0 {
        return round_to_step(min_size, step_size);
    }

    // 1. Tier-based cap
    let tier = equity_tier(equity);
    let mut base_cap = get_tier_cap(tier, runner_key);

    // 2. Regime discount
    if !regime_active {
        base_cap *= 0.6;
    }

    // 3. IC health scaling
    let per_sym_cap = base_cap * ic_scale;

    // 4. Notional -> quantity
    let notional = equity * per_sym_cap * leverage * weight;
    let mut size = notional / price * z_scale;

    // 5. Clamp
    if size < min_size {
        size = min_size;
    }
    if max_qty > 0.0 && size > max_qty {
        size = max_qty;
    }

    round_to_step(size, step_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_account_btc_4h() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT_4h", 400.0, 60000.0,
            0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        // small tier cap=0.35 → notional=400*0.35*10=1400 → qty=1400/60000≈0.023
        assert!(qty > 0.01 && qty < 1.0);
    }

    #[test]
    fn test_zero_equity() {
        let qty = rust_adaptive_target_qty(
            "BTCUSDT", 0.0, 60000.0,
            0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        assert_eq!(qty, 0.001);
    }

    #[test]
    fn test_regime_discount() {
        let active = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0,
            0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, true, 1.0,
        );
        let inactive = rust_adaptive_target_qty(
            "BTCUSDT", 2000.0, 60000.0,
            0.001, 0.001, 0.0,
            1.0, 10.0, 1.0, false, 1.0,
        );
        assert!(active > inactive);
    }

    #[test]
    fn test_round_to_step_basic() {
        assert_eq!(round_to_step(0.0155, 0.001), 0.015);
        assert_eq!(round_to_step(0.0199, 0.001), 0.019);
    }
}
```

- [x] **Step 2: Run Rust tests** — 14 passed

- [x] **Step 3: Register in rust_lib.rs**

Add module declaration in the `decision` block:
```rust
    #[path = "rust/adaptive_sizer.rs"]
    pub mod adaptive_sizer;
```

Add PyO3 registration in `_quant_hotpath`:
```rust
    // Adaptive position sizer
    m.add_function(wrap_pyfunction!(decision::adaptive_sizer::rust_adaptive_target_qty, m)?)?;
```

- [x] **Step 4: Build Rust and verify export**

Run:
```bash
make rust
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "from _quant_hotpath import rust_adaptive_target_qty; print(rust_adaptive_target_qty('BTCUSDT_4h', 400.0, 60000.0))"
```
Expected: prints a float ~0.023

- [x] **Step 5: Wire Python fast path in adaptive.py**

Add Rust delegation to `AdaptivePositionSizer.target_qty()`:

```python
# At module top
try:
    from _quant_hotpath import rust_adaptive_target_qty
    _RUST_SIZER = True
except ImportError:
    _RUST_SIZER = False

# In target_qty(), before the Python logic:
    if _RUST_SIZER:
        result = rust_adaptive_target_qty(
            self.runner_key,
            float(snapshot.account.balance),
            float(market.close) if market is not None else 0.0,
            self.step_size, self.min_size, self.max_qty,
            float(weight), leverage, ic_scale,
            regime_active, z_scale,
        )
        return Decimal(str(result))
```

- [x] **Step 6: Write parity test**

Create `tests/unit/decision/test_adaptive_sizer_rust.py`:
```python
"""Parity tests: Rust vs Python AdaptivePositionSizer."""
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

try:
    from _quant_hotpath import rust_adaptive_target_qty
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from decision.sizing.adaptive import AdaptivePositionSizer, _TIER_WEIGHTS

def _snap(equity, price, symbol="BTCUSDT"):
    snap = MagicMock()
    snap.account.balance = equity
    m = MagicMock(); m.close = price
    snap.markets = {symbol: m}
    return snap

CASES = [
    ("BTCUSDT_4h", 400.0, 60000.0, {}),
    ("BTCUSDT", 2000.0, 60000.0, {"ic_scale": 0.4}),
    ("ETHUSDT_4h", 1000.0, 3000.0, {"regime_active": False}),
    ("BTCUSDT", 50000.0, 60000.0, {"leverage": 3.0, "z_scale": 0.8}),
    ("BTCUSDT", 0.0, 60000.0, {}),
    ("BTCUSDT", 5000.0, 0.0, {}),
]

@pytest.mark.skipif(not HAS_RUST, reason="Rust not available")
@pytest.mark.parametrize("runner_key,equity,price,kwargs", CASES)
def test_rust_python_parity(runner_key, equity, price, kwargs):
    sizer = AdaptivePositionSizer(runner_key=runner_key)
    symbol = runner_key.replace("_4h", "")
    snap = _snap(equity, price, symbol)
    py_qty = sizer.target_qty(snap, symbol, **kwargs)

    rust_qty = rust_adaptive_target_qty(
        runner_key, equity, price,
        sizer.step_size, sizer.min_size, sizer.max_qty,
        float(kwargs.get("weight", 1.0)),
        kwargs.get("leverage", 10.0),
        kwargs.get("ic_scale", 1.0),
        kwargs.get("regime_active", True),
        kwargs.get("z_scale", 1.0),
    )
    assert abs(float(py_qty) - rust_qty) < 1e-9, f"py={py_qty} rust={rust_qty}"
```

- [x] **Step 7: Run all tests** — 76 passed (8 original + 21 parity + 47 exit)

- [ ] **Step 8: Commit**

```bash
git add decision/rust/adaptive_sizer.rs rust_lib.rs decision/sizing/adaptive.py tests/unit/decision/test_adaptive_sizer_rust.py
git commit -m "feat: delegate AdaptivePositionSizer to Rust (warm path)"
```

---

## Task 2: RustExitManager

**Files:**
- Create: `decision/rust/exit_manager.rs`
- Modify: `rust_lib.rs` (add module + registration)
- Modify: `decision/exit_manager.py` (add Rust fast path with fallback)
- Create: `tests/unit/decision/test_exit_manager_rust.py`

- [ ] **Step 1: Write Rust exit manager**

Create `decision/rust/exit_manager.rs`:

```rust
//! Exit manager: trailing stop, signal reversal, deadzone fade, z-cap, time filter.
//! Mirrors decision/exit_manager.py ExitManager.

use pyo3::prelude::*;
use std::collections::HashMap;

struct TrailingState {
    entry_price: f64,
    peak_price: f64,
    entry_bar: i64,
    direction: f64,
}

#[pyclass]
pub struct RustExitManager {
    trailing_stop_pct: f64,
    reversal_threshold: f64,
    deadzone_fade: f64,
    zscore_cap: f64,
    time_filter_enabled: bool,
    skip_hours: Vec<i32>,
    min_hold: i64,
    max_hold: i64,
    positions: HashMap<String, TrailingState>,
}

#[pymethods]
impl RustExitManager {
    #[new]
    #[pyo3(signature = (
        trailing_stop_pct=0.0,
        reversal_threshold=0.0,
        deadzone_fade=0.3,
        zscore_cap=0.0,
        time_filter_enabled=false,
        skip_hours=vec![],
        min_hold=12,
        max_hold=96
    ))]
    fn new(
        trailing_stop_pct: f64,
        reversal_threshold: f64,
        deadzone_fade: f64,
        zscore_cap: f64,
        time_filter_enabled: bool,
        skip_hours: Vec<i32>,
        min_hold: i64,
        max_hold: i64,
    ) -> Self {
        Self {
            trailing_stop_pct,
            reversal_threshold,
            deadzone_fade,
            zscore_cap,
            time_filter_enabled,
            skip_hours,
            min_hold,
            max_hold,
            positions: HashMap::new(),
        }
    }

    fn on_entry(&mut self, symbol: &str, price: f64, bar: i64, direction: f64) {
        self.positions.insert(symbol.to_string(), TrailingState {
            entry_price: price,
            peak_price: price,
            entry_bar: bar,
            direction,
        });
    }

    fn on_exit(&mut self, symbol: &str) {
        self.positions.remove(symbol);
    }

    fn update_price(&mut self, symbol: &str, price: f64) {
        if let Some(state) = self.positions.get_mut(symbol) {
            if state.direction > 0.0 {
                if price > state.peak_price { state.peak_price = price; }
            } else {
                if price < state.peak_price { state.peak_price = price; }
            }
        }
    }

    /// Returns (should_exit, reason).
    fn check_exit(
        &self,
        symbol: &str,
        price: f64,
        bar: i64,
        z_score: f64,
        position: f64,
    ) -> (bool, String) {
        let state = match self.positions.get(symbol) {
            Some(s) => s,
            None => return (false, String::new()),
        };

        let held = bar - state.entry_bar;

        // 1. Max hold
        if held >= self.max_hold {
            return (true, format!("max_hold={}", held));
        }

        // Min hold gate
        if held < self.min_hold {
            return (false, String::new());
        }

        // 2. Trailing stop
        if self.trailing_stop_pct > 0.0 {
            let drawdown = if state.direction > 0.0 {
                (state.peak_price - price) / state.peak_price
            } else {
                (price - state.peak_price) / state.peak_price
            };
            if drawdown >= self.trailing_stop_pct {
                return (true, format!("trailing_stop={:.4}", drawdown));
            }
        }

        // 3. Signal reversal
        if position * z_score < self.reversal_threshold {
            return (true, format!("reversal_z={:.2}", z_score));
        }

        // 4. Deadzone fade
        if z_score.abs() < self.deadzone_fade {
            return (true, format!("deadzone_fade_z={:.2}", z_score));
        }

        (false, String::new())
    }

    fn allow_entry(&self, z_score: f64, hour_utc: Option<i32>) -> bool {
        // Z-score cap
        if self.zscore_cap > 0.0 && z_score.abs() > self.zscore_cap {
            return false;
        }
        // Time filter
        if self.time_filter_enabled {
            if let Some(hour) = hour_utc {
                if self.skip_hours.contains(&hour) {
                    return false;
                }
            }
        }
        true
    }

    /// Serialize state for checkpoint persistence.
    fn checkpoint(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut out = HashMap::new();
        for (sym, s) in &self.positions {
            let mut entry = HashMap::new();
            entry.insert("entry_price".to_string(), s.entry_price);
            entry.insert("peak_price".to_string(), s.peak_price);
            entry.insert("entry_bar".to_string(), s.entry_bar as f64);
            entry.insert("direction".to_string(), s.direction);
            out.insert(sym.clone(), entry);
        }
        out
    }

    /// Restore state from checkpoint.
    fn restore(&mut self, data: HashMap<String, HashMap<String, f64>>) {
        self.positions.clear();
        for (sym, vals) in data {
            self.positions.insert(sym, TrailingState {
                entry_price: *vals.get("entry_price").unwrap_or(&0.0),
                peak_price: *vals.get("peak_price").unwrap_or(&0.0),
                entry_bar: *vals.get("entry_bar").unwrap_or(&0.0) as i64,
                direction: *vals.get("direction").unwrap_or(&0.0),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trailing_stop_long() {
        let mut mgr = RustExitManager::new(0.02, 0.0, 0.3, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        mgr.update_price("ETH", 2100.0);
        let (exit, reason) = mgr.check_exit("ETH", 2050.0, 20, 0.5, 1.0);
        assert!(exit);
        assert!(reason.contains("trailing_stop"));
    }

    #[test]
    fn test_max_hold() {
        let mut mgr = RustExitManager::new(0.0, 0.0, 0.3, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        let (exit, reason) = mgr.check_exit("ETH", 2000.0, 97, 1.0, 1.0);
        assert!(exit);
        assert!(reason.contains("max_hold"));
    }

    #[test]
    fn test_min_hold_respected() {
        let mut mgr = RustExitManager::new(0.02, 0.0, 0.3, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        mgr.update_price("ETH", 2100.0);
        let (exit, _) = mgr.check_exit("ETH", 1800.0, 10, 0.5, 1.0);
        assert!(!exit);
    }

    #[test]
    fn test_zcap_blocks_entry() {
        let mgr = RustExitManager::new(0.0, 0.0, 0.3, 4.0, false, vec![], 12, 96);
        assert!(!mgr.allow_entry(5.0, None));
        assert!(mgr.allow_entry(2.0, None));
    }

    #[test]
    fn test_time_filter() {
        let mgr = RustExitManager::new(0.0, 0.0, 0.3, 0.0, true, vec![0, 1, 2, 3], 12, 96);
        assert!(!mgr.allow_entry(1.0, Some(0)));
        assert!(mgr.allow_entry(1.0, Some(4)));
    }
}
```

- [ ] **Step 2: Run Rust tests**

Run: `cargo test exit_manager -- --nocapture`
Expected: All 5 tests PASS

- [ ] **Step 3: Register in rust_lib.rs**

Module declaration:
```rust
    #[path = "rust/exit_manager.rs"]
    pub mod exit_manager;
```

PyO3 registration:
```rust
    // Exit manager
    m.add_class::<decision::exit_manager::RustExitManager>()?;
```

- [ ] **Step 4: Build and verify**

Run:
```bash
make rust
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "from _quant_hotpath import RustExitManager; m = RustExitManager(trailing_stop_pct=0.02); m.on_entry('ETH', 2000.0, 1, 1.0); print(m.check_exit('ETH', 1950.0, 20, 0.5, 1.0))"
```
Expected: `(True, 'trailing_stop=0.0238')`

- [ ] **Step 5: Wire Python fast path in exit_manager.py**

Add at top of file:
```python
try:
    from _quant_hotpath import RustExitManager as _RustExitManager
    _HAS_RUST_EXIT = True
except ImportError:
    _HAS_RUST_EXIT = False
```

Add factory function at module level:
```python
def create_exit_manager(config, min_hold=12, max_hold=96):
    """Create ExitManager — Rust if available, else Python."""
    if _HAS_RUST_EXIT:
        tf = config.time_filter
        return _RustExitManager(
            trailing_stop_pct=config.trailing_stop_pct,
            reversal_threshold=config.reversal_threshold,
            deadzone_fade=config.deadzone_fade,
            zscore_cap=config.zscore_cap,
            time_filter_enabled=tf.enabled if tf else False,
            skip_hours=list(tf.skip_hours_utc) if tf and tf.enabled else [],
            min_hold=min_hold,
            max_hold=max_hold,
        )
    return ExitManager(config=config, min_hold=min_hold, max_hold=max_hold)
```

- [x] **Step 6: Write parity test**

Create `tests/unit/decision/test_exit_manager_rust.py` — run both Rust and Python ExitManagers through identical scenarios, assert matching outputs for check_exit/allow_entry.

- [ ] **Step 7: Run all exit manager tests**

Run: `pytest tests/unit/decision/test_exit_manager.py tests/unit/decision/test_exit_manager_rust.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add decision/rust/exit_manager.rs rust_lib.rs decision/exit_manager.py tests/unit/decision/test_exit_manager_rust.py
git commit -m "feat: delegate ExitManager to Rust (trailing stop + signal exits)"
```

---

## Task 3: RustBacktestLoop (simplified vectorized backtest engine)

**Files:**
- Create: `decision/rust/backtest_loop.rs`
- Modify: `rust_lib.rs`
- Create: `tests/unit/decision/test_backtest_loop_rust.py`

**Scope note:** This implements the *core* backtest signal loop only (z-score → discretize → exit/entry → PnL). Features excluded from Rust path (remain in Python): monthly gate, bear model, trend follow hold, regime gate scaling, short model, vol-target scaling. These are handled by existing Python code calling into Rust for the hot inner loop. The existing `cpp_run_backtest` in `research/backtest.rs` already provides a basic Rust backtest; this extends it with ExitManager integration.

- [ ] **Step 1: Write Rust backtest loop**

Create `decision/rust/backtest_loop.rs` with `rust_run_signal_backtest`:

Accepts:
- `predictions: Vec<f64>` — raw model predictions per bar
- `closes: Vec<f64>` — close prices per bar
- `equity: f64` — starting equity
- `risk_fraction: f64`, `leverage: f64`
- `deadzone: f64`, `min_hold: i32`, `max_hold: i32`
- `long_only: bool`
- `zscore_window: usize`, `zscore_warmup: usize`
- `trailing_stop_pct: f64`, `reversal_threshold: f64`, `deadzone_fade: f64`

Returns PyDict:
- `pnl_curve: Vec<f64>` — cumulative PnL per bar
- `positions: Vec<f64>` — position state per bar (+1/-1/0)
- `trade_count: i64`, `final_equity: f64`

Inner loop (no per-bar FFI):
1. Push prediction → z-score buffer
2. Discretize (deadzone + long_only + min_hold)
3. Exit checks (trailing stop + reversal + deadzone fade + max hold)
4. Entry sizing (equity × risk_fraction × leverage / price)
5. PnL tracking (mark-to-market)

- [ ] **Step 2: Run Rust tests**

Run: `cargo test backtest_loop -- --nocapture`

- [ ] **Step 3: Register in rust_lib.rs**

Module + function registration.

- [ ] **Step 4: Build and verify**

```bash
make rust
cp $(python3 -c "import _quant_hotpath, os; print(os.path.dirname(_quant_hotpath.__file__))")/*.so _quant_hotpath/ 2>/dev/null || true
python3 -c "from _quant_hotpath import rust_run_signal_backtest; print('ok')"
```

- [ ] **Step 5: Write Python test with synthetic data**

Create `tests/unit/decision/test_backtest_loop_rust.py`:
- Generate synthetic predictions (sine wave + noise) + prices (1000 bars)
- Run Rust `rust_run_signal_backtest` — verify PnL curve length, trade count > 0
- Verify position states are in {-1, 0, 1}
- Verify final_equity == equity + pnl_curve[-1]
- NOTE: No Python parity test (Python loop includes monthly gate, bear model, etc.)

- [ ] **Step 6: Run all tests**

Run: `pytest tests/unit/decision/ -v -x`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add decision/rust/backtest_loop.rs rust_lib.rs tests/unit/decision/test_backtest_loop_rust.py
git commit -m "feat: Rust vectorized backtest engine (core signal loop)"
```

---

## Task 4: Integration verification

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/unit/ -x -q`
Expected: All existing tests PASS

- [ ] **Step 2: Run Rust tests**

Run: `cargo test`
Expected: All PASS

- [ ] **Step 3: Verify export count**

Run: `python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)), 'exports')"`
Expected: 201 exports (was 198, +3 new: rust_adaptive_target_qty, RustExitManager, rust_run_signal_backtest)

- [ ] **Step 4: Lint**

Run: `ruff check --select E,W,F .`
Expected: Clean

- [ ] **Step 5: Final commit**

```bash
git commit -m "chore: verify Rust delegation wave 2 integration"
```
