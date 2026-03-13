// inference_bridge.rs — Per-symbol signal processing for LiveInferenceBridge
//
// Migrates hot-path computation from Python:
//   - Rolling z-score normalization (O(n) sum over zscore_window)
//   - Discretization + min-hold enforcement
//   - Monthly gate (rolling MA over close history)
//   - Short model signal processing

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, VecDeque};

use crate::constraint_pipeline::{
    discretize, enforce_hold_step, enforce_short_hold_step,
    long_only_clip, discretize_short, update_monthly_gate,
};

/// Per-symbol signal processing state.
pub(crate) struct SymbolState {
    pub(crate) zscore_buf: VecDeque<f64>,
    pub(crate) zscore_last_hour: i64,
    pub(crate) position: f64,
    pub(crate) hold_counter: i32,
    pub(crate) close_history: VecDeque<f64>,
    pub(crate) gate_last_hour: i64,
    pub(crate) gate_window: usize,
    pub(crate) short_zscore_buf: VecDeque<f64>,
    pub(crate) short_zscore_last_hour: i64,
    pub(crate) short_position: f64,
    pub(crate) short_hold_counter: i32,
}

impl SymbolState {
    pub(crate) fn new(zscore_window: usize, gate_window: usize) -> Self {
        Self {
            zscore_buf: VecDeque::with_capacity(zscore_window),
            zscore_last_hour: -1,
            position: 0.0,
            hold_counter: 0,
            close_history: VecDeque::with_capacity(gate_window),
            gate_last_hour: -1,
            gate_window,
            short_zscore_buf: VecDeque::with_capacity(zscore_window),
            short_zscore_last_hour: -1,
            short_position: 0.0,
            short_hold_counter: 0,
        }
    }
}

/// Compute mean and std from a VecDeque.
#[inline]
pub(crate) fn mean_std(buf: &VecDeque<f64>) -> (f64, f64) {
    let n = buf.len() as f64;
    if n < 1.0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &v in buf.iter() {
        sum += v;
        sum_sq += v * v;
    }
    let mu = sum / n;
    let var = sum_sq / n - mu * mu;
    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
    (mu, std)
}

/// Append value to buffer on new hour, return z-score if enough data.
#[inline]
pub(crate) fn zscore_from_buf(
    buf: &mut VecDeque<f64>,
    last_hour: &mut i64,
    raw_score: f64,
    hour_key: i64,
    window: usize,
    warmup: usize,
) -> Option<f64> {
    if hour_key != *last_hour {
        if buf.len() == window {
            buf.pop_front();
        }
        buf.push_back(raw_score);
        *last_hour = hour_key;
    }
    if buf.len() < warmup {
        return None;
    }
    let (mu, std) = mean_std(buf);
    if std < 1e-12 {
        return None;
    }
    Some((raw_score - mu) / std)
}

#[pyclass]
pub struct RustInferenceBridge {
    pub(crate) symbols: HashMap<String, SymbolState>,
    pub(crate) zscore_window: usize,
    pub(crate) zscore_warmup: usize,
    pub(crate) default_gate_window: usize,
}

#[pymethods]
impl RustInferenceBridge {
    #[new]
    #[pyo3(signature = (zscore_window=720, zscore_warmup=180, default_gate_window=480))]
    fn new(zscore_window: usize, zscore_warmup: usize, default_gate_window: usize) -> Self {
        // Clamp warmup to window size, matching backtest (backtest_engine.rs:110).
        // When warmup > window the buffer can never reach warmup threshold,
        // so signals would never be produced.
        let clamped_warmup = std::cmp::min(zscore_warmup, zscore_window);
        Self {
            symbols: HashMap::new(),
            zscore_window,
            zscore_warmup: clamped_warmup,
            default_gate_window,
        }
    }

    /// Rolling z-score normalization. Returns None during warmup.
    /// Only appends to buffer on new hour boundary.
    fn zscore_normalize(&mut self, symbol: &str, raw_score: f64, hour_key: i64) -> Option<f64> {
        let window = self.zscore_window;
        let warmup = self.zscore_warmup;
        let state = self.get_or_create(symbol);
        zscore_from_buf(
            &mut state.zscore_buf, &mut state.zscore_last_hour,
            raw_score, hour_key, window, warmup,
        )
    }

    /// Apply signal constraints: z-score → discretize → min-hold → trend-hold.
    /// Returns the constrained signal value.
    ///
    /// If min_hold <= 0, returns raw_score unchanged (no constraints).
    /// trend_val: value of trend indicator (only used if trend_follow=true).
    ///   Pass f64::NAN if not available.
    #[pyo3(signature = (symbol, raw_score, hour_key, deadzone=0.5, min_hold=0, long_only=false, trend_follow=false, trend_val=f64::NAN, trend_threshold=0.0, max_hold=120))]
    fn apply_constraints(
        &mut self,
        symbol: &str,
        raw_score: f64,
        hour_key: i64,
        deadzone: f64,
        min_hold: i32,
        long_only: bool,
        trend_follow: bool,
        trend_val: f64,
        trend_threshold: f64,
        max_hold: i32,
    ) -> f64 {
        if min_hold <= 0 {
            return raw_score;
        }

        let window = self.zscore_window;
        let warmup = self.zscore_warmup;
        let state = self.get_or_create(symbol);

        // Rolling z-score
        let z = if warmup > 0 {
            match zscore_from_buf(
                &mut state.zscore_buf, &mut state.zscore_last_hour,
                raw_score, hour_key, window, warmup,
            ) {
                Some(z) => z,
                None => {
                    // Warmup: increment hold counter to match backtest behavior.
                    // In backtest, min-hold runs over warmup bars (raw=0.0) starting
                    // at hold_count=1, so by bar k the count is k+1. Replicating
                    // that here ensures the first post-warmup bar has the same
                    // hold state in both paths.
                    if state.hold_counter == 0 {
                        state.hold_counter = 1;
                    } else {
                        state.hold_counter += 1;
                    }
                    return state.position;
                }
            }
        } else {
            raw_score
        };

        // Long-only clip + discretize (shared)
        let z = long_only_clip(z, long_only);
        let desired = discretize(z, deadzone);

        let prev_pos = state.position;
        let hold_count = if state.hold_counter == 0 { min_hold } else { state.hold_counter };

        // Min-hold + trend-hold (shared)
        let (output, new_hold) = enforce_hold_step(
            desired, prev_pos, hold_count, min_hold,
            trend_follow, trend_val, trend_threshold, max_hold,
        );
        state.hold_counter = new_hold;
        if output != prev_pos {
            state.position = output;
        }
        output
    }

    /// Check monthly gate: close > SMA(window). Returns true if signal allowed.
    /// Only appends close on new hour boundary.
    fn check_monthly_gate(&mut self, symbol: &str, close: f64, hour_key: i64, window: usize) -> bool {
        let state = self.get_or_create(symbol);
        let w = if window > 0 { window } else { state.gate_window };
        if state.gate_window != w {
            state.gate_window = w;
        }
        update_monthly_gate(
            &mut state.close_history, &mut state.gate_last_hour,
            close, hour_key, w,
        )
    }

    /// Process short model signal: z-score → short-only clip → min-hold.
    /// Returns constrained short signal.
    #[pyo3(signature = (symbol, raw_score, hour_key, deadzone=0.5, min_hold=0))]
    fn process_short_signal(
        &mut self,
        symbol: &str,
        raw_score: f64,
        hour_key: i64,
        deadzone: f64,
        min_hold: i32,
    ) -> f64 {
        if min_hold <= 0 {
            // No constraints — clip to short-only
            return raw_score.min(0.0);
        }

        let window = self.zscore_window;
        let warmup = self.zscore_warmup;
        let state = self.get_or_create(symbol);

        // Z-score normalization for short buffer
        let z = match zscore_from_buf(
            &mut state.short_zscore_buf, &mut state.short_zscore_last_hour,
            raw_score, hour_key, window, warmup,
        ) {
            Some(z) => z,
            None => {
                // Warmup: increment hold counter (same fix as apply_constraints)
                if state.short_hold_counter == 0 {
                    state.short_hold_counter = 1;
                } else {
                    state.short_hold_counter += 1;
                }
                return state.short_position;
            }
        };

        // Short-only discretize (shared)
        let desired = discretize_short(z, deadzone);

        // Min-hold enforcement (shared)
        let prev = state.short_position;
        let hold = if state.short_hold_counter == 0 { min_hold } else { state.short_hold_counter };

        let (output, new_hold) = enforce_short_hold_step(desired, prev, hold, min_hold);
        state.short_hold_counter = new_hold;
        if output != prev {
            state.short_position = output;
        }
        output
    }

    /// Force-set position and hold state (for regime switch sync).
    fn set_position(&mut self, symbol: &str, position: f64, hold: i32) {
        let state = self.get_or_create(symbol);
        state.position = position;
        state.hold_counter = hold;
    }

    /// Get current position for a symbol.
    fn get_position(&self, symbol: &str) -> f64 {
        self.symbols.get(symbol).map_or(0.0, |s| s.position)
    }

    /// Serialize state to Python dict for checkpointing.
    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);

        let position = PyDict::new(py);
        let hold_counter = PyDict::new(py);
        let zscore_buf = PyDict::new(py);
        let zscore_last_hour = PyDict::new(py);
        let close_history = PyDict::new(py);
        let gate_last_hour = PyDict::new(py);
        let short_position = PyDict::new(py);
        let short_hold_counter = PyDict::new(py);
        let short_zscore_buf = PyDict::new(py);
        let short_zscore_last_hour = PyDict::new(py);

        for (sym, state) in &self.symbols {
            position.set_item(sym, state.position)?;
            hold_counter.set_item(sym, state.hold_counter)?;
            let buf_vec: Vec<f64> = state.zscore_buf.iter().copied().collect();
            zscore_buf.set_item(sym, buf_vec)?;
            zscore_last_hour.set_item(sym, state.zscore_last_hour)?;
            let hist_vec: Vec<f64> = state.close_history.iter().copied().collect();
            close_history.set_item(sym, hist_vec)?;
            gate_last_hour.set_item(sym, state.gate_last_hour)?;
            short_position.set_item(sym, state.short_position)?;
            short_hold_counter.set_item(sym, state.short_hold_counter)?;
            let sbuf_vec: Vec<f64> = state.short_zscore_buf.iter().copied().collect();
            short_zscore_buf.set_item(sym, sbuf_vec)?;
            short_zscore_last_hour.set_item(sym, state.short_zscore_last_hour)?;
        }

        d.set_item("position", position)?;
        d.set_item("hold_counter", hold_counter)?;
        d.set_item("zscore_buf", zscore_buf)?;
        d.set_item("zscore_last_hour", zscore_last_hour)?;
        d.set_item("close_history", close_history)?;
        d.set_item("gate_last_hour", gate_last_hour)?;
        d.set_item("short_position", short_position)?;
        d.set_item("short_hold_counter", short_hold_counter)?;
        d.set_item("short_zscore_buf", short_zscore_buf)?;
        d.set_item("short_zscore_last_hour", short_zscore_last_hour)?;
        Ok(d)
    }

    /// Restore state from Python dict.
    fn restore(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        self.symbols.clear();

        let position: HashMap<String, f64> = data.get_item("position")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let hold_counter: HashMap<String, i32> = data.get_item("hold_counter")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let zscore_buf: HashMap<String, Vec<f64>> = data.get_item("zscore_buf")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let zscore_last_hour: HashMap<String, i64> = data.get_item("zscore_last_hour")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let close_history: HashMap<String, Vec<f64>> = data.get_item("close_history")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let gate_last_hour: HashMap<String, i64> = data.get_item("gate_last_hour")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let short_position: HashMap<String, f64> = data.get_item("short_position")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let short_hold_counter: HashMap<String, i32> = data.get_item("short_hold_counter")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let short_zscore_buf: HashMap<String, Vec<f64>> = data.get_item("short_zscore_buf")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();
        let short_zscore_last_hour: HashMap<String, i64> = data.get_item("short_zscore_last_hour")?
            .map(|v| v.extract()).transpose()?.unwrap_or_default();

        // Collect all symbol names
        let mut all_syms: std::collections::HashSet<String> = std::collections::HashSet::new();
        for k in position.keys() { all_syms.insert(k.clone()); }
        for k in zscore_buf.keys() { all_syms.insert(k.clone()); }
        for k in close_history.keys() { all_syms.insert(k.clone()); }
        for k in short_position.keys() { all_syms.insert(k.clone()); }

        for sym in all_syms {
            let mut state = SymbolState::new(self.zscore_window, self.default_gate_window);
            state.position = position.get(&sym).copied().unwrap_or(0.0);
            state.hold_counter = hold_counter.get(&sym).copied().unwrap_or(0);
            if let Some(buf) = zscore_buf.get(&sym) {
                for &v in buf {
                    state.zscore_buf.push_back(v);
                }
            }
            state.zscore_last_hour = zscore_last_hour.get(&sym).copied().unwrap_or(-1);
            if let Some(hist) = close_history.get(&sym) {
                for &v in hist {
                    state.close_history.push_back(v);
                }
            }
            state.gate_last_hour = gate_last_hour.get(&sym).copied().unwrap_or(-1);
            state.short_position = short_position.get(&sym).copied().unwrap_or(0.0);
            state.short_hold_counter = short_hold_counter.get(&sym).copied().unwrap_or(0);
            if let Some(buf) = short_zscore_buf.get(&sym) {
                for &v in buf {
                    state.short_zscore_buf.push_back(v);
                }
            }
            state.short_zscore_last_hour = short_zscore_last_hour.get(&sym).copied().unwrap_or(-1);
            self.symbols.insert(sym, state);
        }
        Ok(())
    }

    /// Reset all state (e.g., on model hot-swap).
    fn reset(&mut self) {
        self.symbols.clear();
    }
}

impl RustInferenceBridge {
    pub(crate) fn get_or_create(&mut self, symbol: &str) -> &mut SymbolState {
        if !self.symbols.contains_key(symbol) {
            self.symbols.insert(
                symbol.to_string(),
                SymbolState::new(self.zscore_window, self.default_gate_window),
            );
        }
        self.symbols.get_mut(symbol).unwrap()
    }
}
