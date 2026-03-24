// adaptive_stop.rs — RustAdaptiveStopGate: ATR 3-phase adaptive stop-loss
//
// Python source: runner/gates/adaptive_stop_gate.py
//
// Three phases (monotonically advancing):
//   INITIAL   : stop = entry ± atr × initial_mult
//   BREAKEVEN : stop = entry ± atr × 0.1 buffer  (after ≥ breakeven_trigger × ATR profit)
//   TRAILING  : stop = peak ∓ atr × trailing_mult (after ≥ trail_trigger × ATR profit)
//
// Hard limits:
//   max 5% loss from entry
//   min 0.3% distance from current price (avoids noise stops)
//
// ATR is stored as fraction of price (tr / prev_close). All stop computations
// treat ATR as a fraction and multiply by price where absolute distance is needed.

use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;

// Phase constants
const PHASE_INITIAL: u8 = 0;
const PHASE_BREAKEVEN: u8 = 1;
const PHASE_TRAILING: u8 = 2;

struct SymbolStopState {
    entry_price: f64,
    side: i32,        // +1 long, -1 short, 0 flat
    peak_price: f64,
    stop_phase: u8,   // 0=INITIAL, 1=BREAKEVEN, 2=TRAILING
    atr_buffer: VecDeque<f64>, // true range fractions, maxlen 50
}

impl SymbolStopState {
    fn new() -> Self {
        Self {
            entry_price: 0.0,
            side: 0,
            peak_price: 0.0,
            stop_phase: PHASE_INITIAL,
            atr_buffer: VecDeque::with_capacity(50),
        }
    }

    fn reset(&mut self) {
        self.entry_price = 0.0;
        self.side = 0;
        self.peak_price = 0.0;
        self.stop_phase = PHASE_INITIAL;
        // NOTE: atr_buffer is NOT cleared on reset — ATR history persists across positions
    }

    /// Returns ATR as a fraction. If < 5 samples, returns fallback directly (already a fraction).
    fn current_atr(&self, fallback: f64) -> f64 {
        if self.atr_buffer.len() < 5 {
            return fallback;
        }
        let window: Vec<f64> = self.atr_buffer.iter().copied().rev().take(14).collect();
        window.iter().sum::<f64>() / window.len() as f64
    }

    fn push_true_range(&mut self, high: f64, low: f64, prev_close: f64) {
        if prev_close <= 0.0 {
            return;
        }
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());
        let atr_frac = tr / prev_close;
        if self.atr_buffer.len() == 50 {
            self.atr_buffer.pop_front();
        }
        self.atr_buffer.push_back(atr_frac);
    }
}

#[pyclass(name = "RustAdaptiveStopGate")]
pub struct RustAdaptiveStopGate {
    atr_initial_mult: f64,  // default 2.0
    breakeven_trigger: f64, // default 1.0 (in ATR multiples)
    trailing_mult: f64,     // default 0.3
    atr_fallback: f64,      // default 0.015 (fraction of price)
    // Phase threshold constants (matching Python class vars)
    trail_trigger: f64,     // 0.8 (profit in ATR multiples to enter trailing)
    breakeven_buffer: f64,  // 0.1 (ATR buffer above entry)
    max_loss_pct: f64,      // 0.05
    min_dist_pct: f64,      // 0.003
    states: HashMap<String, SymbolStopState>,
}

#[pymethods]
impl RustAdaptiveStopGate {
    #[new]
    #[pyo3(signature = (atr_initial_mult=2.0, atr_breakeven_trigger=1.0, atr_trailing_mult=0.3, atr_fallback=0.015))]
    fn new(
        atr_initial_mult: f64,
        atr_breakeven_trigger: f64,
        atr_trailing_mult: f64,
        atr_fallback: f64,
    ) -> Self {
        Self {
            atr_initial_mult,
            breakeven_trigger: atr_breakeven_trigger,
            trailing_mult: atr_trailing_mult,
            atr_fallback,
            trail_trigger: 0.8,
            breakeven_buffer: 0.1,
            max_loss_pct: 0.05,
            min_dist_pct: 0.003,
            states: HashMap::new(),
        }
    }

    /// Record a new position. Resets state (but not ATR buffer) for symbol.
    fn on_new_position(&mut self, symbol: String, side: i32, entry_price: f64) -> PyResult<()> {
        if !entry_price.is_finite() || entry_price <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "entry_price must be finite and positive",
            ));
        }
        if side != -1 && side != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "side must be +1 (long) or -1 (short)",
            ));
        }
        let state = self.states.entry(symbol).or_insert_with(SymbolStopState::new);
        state.reset();
        state.entry_price = entry_price;
        state.side = side;
        state.peak_price = entry_price;
        state.stop_phase = PHASE_INITIAL;
        Ok(())
    }

    /// Compute and buffer true range as fraction of close.
    fn push_true_range(
        &mut self,
        symbol: String,
        high: f64,
        low: f64,
        prev_close: f64,
    ) -> PyResult<()> {
        if !high.is_finite() || !low.is_finite() || !prev_close.is_finite() || prev_close <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "push_true_range: invalid inputs high={} low={} prev_close={}", high, low, prev_close
            )));
        }
        let state = self.states.entry(symbol).or_insert_with(SymbolStopState::new);
        state.push_true_range(high, low, prev_close);
        Ok(())
    }

    /// Real-time tick check. Returns True (and resets state) if stopped out.
    /// Updates peak BEFORE computing stop (same order as Python).
    fn check_stop(&mut self, symbol: String, price: f64) -> bool {
        let state = match self.states.get_mut(&symbol) {
            Some(s) => s,
            None => return false,
        };

        if state.side == 0 || state.entry_price <= 0.0 {
            return false;
        }

        if !price.is_finite() || price <= 0.0 {
            return false;
        }

        // Update peak
        if state.side > 0 {
            if price > state.peak_price {
                state.peak_price = price;
            }
        } else {
            if price < state.peak_price {
                state.peak_price = price;
            }
        }

        // Compute stop and check breach (with phase mutation)
        let stop = Self::compute_stop_internal(
            state,
            price,
            self.atr_fallback,
            self.trail_trigger,
            self.breakeven_trigger,
            self.breakeven_buffer,
            self.atr_initial_mult,
            self.trailing_mult,
            self.max_loss_pct,
            self.min_dist_pct,
            true, // mutate_phase
        );

        let breached = if state.side > 0 {
            price <= stop
        } else {
            price >= stop
        };

        if breached {
            state.reset();
            return true;
        }

        false
    }

    /// Return the current stop price without mutating phase or peak (read-only).
    /// Returns 0.0 if no active position.
    fn compute_stop_price(&self, symbol: String, price: f64) -> f64 {
        let state = match self.states.get(&symbol) {
            Some(s) => s,
            None => return 0.0,
        };

        if state.side == 0 || state.entry_price <= 0.0 {
            return 0.0;
        }

        // We need a mutable borrow for phase mutation flag=false, but we pass
        // state as a shared reference. We create a temporary copy of needed fields.
        Self::compute_stop_readonly(
            state,
            price,
            self.atr_fallback,
            self.trail_trigger,
            self.breakeven_trigger,
            self.breakeven_buffer,
            self.atr_initial_mult,
            self.trailing_mult,
            self.max_loss_pct,
            self.min_dist_pct,
        )
    }

    /// Return current stop phase: "INITIAL", "BREAKEVEN", or "TRAILING".
    fn get_phase(&self, symbol: String) -> String {
        match self.states.get(&symbol) {
            None => "INITIAL".to_string(),
            Some(s) => match s.stop_phase {
                PHASE_INITIAL => "INITIAL".to_string(),
                PHASE_BREAKEVEN => "BREAKEVEN".to_string(),
                PHASE_TRAILING => "TRAILING".to_string(),
                _ => "INITIAL".to_string(),
            },
        }
    }

    /// Reset position fields for symbol, preserving ATR buffer (matches Python behavior).
    fn reset_symbol(&mut self, symbol: String) {
        if let Some(state) = self.states.get_mut(&symbol) {
            state.reset();
        }
    }

    /// Force the stop phase for a symbol (for testing/state sync).
    /// phase_str: "INITIAL", "BREAKEVEN", or "TRAILING"
    fn set_phase(&mut self, symbol: String, phase_str: String) -> PyResult<()> {
        let state = self.states.entry(symbol).or_insert_with(SymbolStopState::new);
        state.stop_phase = match phase_str.as_str() {
            "INITIAL" => PHASE_INITIAL,
            "BREAKEVEN" => PHASE_BREAKEVEN,
            "TRAILING" => PHASE_TRAILING,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown phase: {}",
                    other
                )))
            }
        };
        Ok(())
    }

    /// Force the peak price for a symbol (for testing/state sync).
    fn set_peak(&mut self, symbol: String, peak: f64) -> PyResult<()> {
        let state = self.states.entry(symbol).or_insert_with(SymbolStopState::new);
        state.peak_price = peak;
        Ok(())
    }
}

impl RustAdaptiveStopGate {
    /// Core stop computation with mutable phase (used in check_stop hot path).
    #[allow(clippy::too_many_arguments)]
    fn compute_stop_internal(
        state: &mut SymbolStopState,
        current_price: f64,
        atr_fallback: f64,
        trail_trigger: f64,
        breakeven_trigger: f64,
        breakeven_buffer: f64,
        atr_initial_mult: f64,
        trailing_mult: f64,
        max_loss_pct: f64,
        min_dist_pct: f64,
        mutate_phase: bool,
    ) -> f64 {
        let atr = state.current_atr(atr_fallback);
        let side = state.side;
        let entry = state.entry_price;
        let peak = state.peak_price;

        // Profit fraction from best price seen since entry
        let profit_pct = if side > 0 {
            if entry > 0.0 { (peak - entry) / entry } else { 0.0 }
        } else {
            if entry > 0.0 { (entry - peak) / entry } else { 0.0 }
        };

        // Phase selection
        // Note: trail_trigger (0.8) < breakeven_trigger (1.0) by default,
        // so TRAILING fires before BREAKEVEN. BREAKEVEN is only reachable
        // if breakeven_trigger is configured lower than trail_trigger.
        let new_phase = if profit_pct >= atr * trail_trigger {
            PHASE_TRAILING
        } else if profit_pct >= atr * breakeven_trigger {
            PHASE_BREAKEVEN
        } else {
            PHASE_INITIAL
        };

        // Monotonic advance only
        let effective_phase = if new_phase > state.stop_phase {
            if mutate_phase {
                state.stop_phase = new_phase;
            }
            new_phase
        } else {
            state.stop_phase
        };

        // Compute raw stop
        let stop = if effective_phase == PHASE_TRAILING {
            let trail_dist = atr * trailing_mult;
            if side > 0 {
                peak * (1.0 - trail_dist)
            } else {
                peak * (1.0 + trail_dist)
            }
        } else if effective_phase == PHASE_BREAKEVEN {
            let buffer = atr * breakeven_buffer;
            if side > 0 {
                entry * (1.0 + buffer)
            } else {
                entry * (1.0 - buffer)
            }
        } else {
            // INITIAL
            let initial_dist = atr * atr_initial_mult;
            if side > 0 {
                entry * (1.0 - initial_dist)
            } else {
                entry * (1.0 + initial_dist)
            }
        };

        // Hard limits
        // 1. Max loss 5% from entry
        let stop = if side > 0 {
            let floor = entry * (1.0 - max_loss_pct);
            stop.max(floor)
        } else {
            let ceil = entry * (1.0 + max_loss_pct);
            stop.min(ceil)
        };

        // 2. Min distance 0.3% from current price (only when stop is on correct side)
        let min_dist = current_price * min_dist_pct;
        let stop = if side > 0 && stop < current_price {
            if current_price - stop < min_dist {
                current_price - min_dist
            } else {
                stop
            }
        } else if side < 0 && stop > current_price {
            if stop - current_price < min_dist {
                current_price + min_dist
            } else {
                stop
            }
        } else {
            stop
        };

        stop
    }

    /// Read-only stop computation (no phase mutation).
    #[allow(clippy::too_many_arguments)]
    fn compute_stop_readonly(
        state: &SymbolStopState,
        current_price: f64,
        atr_fallback: f64,
        trail_trigger: f64,
        breakeven_trigger: f64,
        breakeven_buffer: f64,
        atr_initial_mult: f64,
        trailing_mult: f64,
        max_loss_pct: f64,
        min_dist_pct: f64,
    ) -> f64 {
        let atr = state.current_atr(atr_fallback);
        let side = state.side;
        let entry = state.entry_price;
        let peak = state.peak_price;

        let profit_pct = if side > 0 {
            if entry > 0.0 { (peak - entry) / entry } else { 0.0 }
        } else {
            if entry > 0.0 { (entry - peak) / entry } else { 0.0 }
        };

        let new_phase = if profit_pct >= atr * trail_trigger {
            PHASE_TRAILING
        } else if profit_pct >= atr * breakeven_trigger {
            PHASE_BREAKEVEN
        } else {
            PHASE_INITIAL
        };

        // Monotonic: use max of computed and stored, but do NOT mutate
        let effective_phase = new_phase.max(state.stop_phase);

        let stop = if effective_phase == PHASE_TRAILING {
            let trail_dist = atr * trailing_mult;
            if side > 0 {
                peak * (1.0 - trail_dist)
            } else {
                peak * (1.0 + trail_dist)
            }
        } else if effective_phase == PHASE_BREAKEVEN {
            let buffer = atr * breakeven_buffer;
            if side > 0 {
                entry * (1.0 + buffer)
            } else {
                entry * (1.0 - buffer)
            }
        } else {
            let initial_dist = atr * atr_initial_mult;
            if side > 0 {
                entry * (1.0 - initial_dist)
            } else {
                entry * (1.0 + initial_dist)
            }
        };

        // Hard limits
        let stop = if side > 0 {
            let floor = entry * (1.0 - max_loss_pct);
            stop.max(floor)
        } else {
            let ceil = entry * (1.0 + max_loss_pct);
            stop.min(ceil)
        };

        let min_dist = current_price * min_dist_pct;
        let stop = if side > 0 && stop < current_price {
            if current_price - stop < min_dist {
                current_price - min_dist
            } else {
                stop
            }
        } else if side < 0 && stop > current_price {
            if stop - current_price < min_dist {
                current_price + min_dist
            } else {
                stop
            }
        } else {
            stop
        };

        stop
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gate() -> RustAdaptiveStopGate {
        RustAdaptiveStopGate::new(2.0, 1.0, 0.3, 0.015)
    }

    fn feed_atr(gate: &mut RustAdaptiveStopGate, sym: &str, n: usize) {
        // 1% TR per bar: high=101, low=99, prev=100
        for _ in 0..n {
            gate.push_true_range(sym.to_string(), 101.0, 99.0, 100.0).unwrap();
        }
    }

    #[test]
    fn test_long_initial_stop() {
        // ATR = 0.02 (mean of last 14 of 20 bars, each 2/100=0.02)
        // initial stop = entry * (1 - 2.0 * 0.02) = 100 * 0.96 = 96.0
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 100.0);
        // Expected: 100 * (1 - 2.0 * 0.02) = 96.0
        // But min_dist = 100 * 0.003 = 0.3, so stop pushed to 100 - 0.3 = 99.7
        // Wait: 96.0 < 99.7 → current_price - stop = 4.0 > min_dist(0.3) → no adjustment
        assert!((stop - 96.0).abs() < 1e-9, "Expected ~96.0, got {}", stop);
    }

    #[test]
    fn test_long_trailing_phase() {
        // ATR ≈ 0.02, trail_trigger=0.8 → need profit_pct >= 0.016
        // entry=100, peak=102 → profit=(102-100)/100=0.02 >= 0.8*0.02=0.016 → TRAILING
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        // push price to 102.0 to cross trail_trigger
        gate.check_stop(sym.to_string(), 102.0);
        assert_eq!(gate.get_phase(sym.to_string()), "TRAILING");
    }

    #[test]
    fn test_short_initial_stop() {
        // Short: stop = entry * (1 + atr * initial_mult) > entry
        let mut gate = make_gate();
        let sym = "BTCUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), -1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 100.0);
        // ATR=0.02, stop = 100*(1+2*0.02) = 104.0
        // min_dist: stop(104) > current_price(100) → correct side → check dist
        // 104 - 100 = 4 > 0.3 → no adjustment
        assert!((stop - 104.0).abs() < 1e-9, "Expected ~104.0, got {}", stop);
    }

    #[test]
    fn test_stop_breach_long() {
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 95.0);
        // Price well below stop → check_stop should return true
        let breached = gate.check_stop(sym.to_string(), stop - 0.01);
        assert!(breached, "Long breach should trigger");
        // State should be reset: side==0
        assert!(!gate.check_stop(sym.to_string(), 90.0), "After reset, no trigger");
    }

    #[test]
    fn test_stop_breach_short() {
        let mut gate = make_gate();
        let sym = "BTCUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), -1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 105.0);
        // For short, stop is above entry (104.0), price above stop triggers
        let breached = gate.check_stop(sym.to_string(), stop + 0.01);
        assert!(breached, "Short breach should trigger");
    }

    #[test]
    fn test_max_loss_limit() {
        // High ATR → raw stop would be far below entry, should be capped at 5%
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        // TR = 30% each bar
        for _ in 0..20 {
            gate.push_true_range(sym.to_string(), 115.0, 85.0, 100.0).unwrap();
        }
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 100.0);
        // Max loss: 100 * (1 - 0.05) = 95.0
        assert!(stop >= 95.0, "Stop should be at least 95.0, got {}", stop);
    }

    #[test]
    fn test_min_distance() {
        // Very small ATR → stop would be very close to price, should be pushed back
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        // TR = 0.001% (tiny)
        for _ in 0..20 {
            gate.push_true_range(sym.to_string(), 100.0001, 99.9999, 100.0).unwrap();
        }
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        let price = 100.0;
        let stop = gate.compute_stop_price(sym.to_string(), price);
        // Min dist = 100 * 0.003 = 0.3; stop must be <= price - 0.3 = 99.7
        assert!(stop <= price - price * 0.003 + 1e-9,
            "Stop must respect min distance, got {}", stop);
    }

    #[test]
    fn test_atr_fallback() {
        // With < 5 samples, fallback=0.015 should be used directly
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        // Only 3 ATR samples
        for _ in 0..3 {
            gate.push_true_range(sym.to_string(), 101.0, 99.0, 100.0).unwrap();
        }
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        let stop = gate.compute_stop_price(sym.to_string(), 100.0);
        // ATR = fallback = 0.015; stop = 100 * (1 - 2.0*0.015) = 97.0
        // dist = 3.0 > min_dist(0.3) → no adjustment
        assert!((stop - 97.0).abs() < 1e-9, "Expected 97.0 with fallback ATR, got {}", stop);
    }

    #[test]
    fn test_phase_monotonic() {
        // Phase should never regress
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();

        // Advance to trailing: ATR=0.02, trail_trigger=0.8 → need profit >= 0.016
        // push price to 102: profit=2% >= trail_trigger(0.8)*ATR(0.02)=1.6%
        gate.check_stop(sym.to_string(), 102.0);
        assert_eq!(gate.get_phase(sym.to_string()), "TRAILING");

        // Even with further gains, phase stays TRAILING (monotonic).
        // trailing stop = peak * (1 - 0.3*0.02) = 102 * 0.994 = 101.388
        // Use price 103 (well above stop) so no breach occurs.
        gate.check_stop(sym.to_string(), 103.0);
        assert_eq!(gate.get_phase(sym.to_string()), "TRAILING");
    }

    #[test]
    fn test_no_position_no_stop() {
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        // No position set → should not stop
        assert!(!gate.check_stop(sym.to_string(), 100.0));
    }

    #[test]
    fn test_reset_symbol() {
        let mut gate = make_gate();
        let sym = "ETHUSDT";
        feed_atr(&mut gate, sym, 20);
        gate.on_new_position(sym.to_string(), 1, 100.0).unwrap();
        gate.reset_symbol(sym.to_string());
        assert!(!gate.check_stop(sym.to_string(), 90.0));
        assert_eq!(gate.compute_stop_price(sym.to_string(), 100.0), 0.0);
    }
}
