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
        reversal_threshold=-0.3,
        deadzone_fade=0.2,
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

        // 1. Max hold — always enforced
        if held >= self.max_hold {
            return (true, format!("max_hold={}", held));
        }

        // Min hold gate
        if held < self.min_hold {
            return (false, String::new());
        }

        // 2. Trailing stop (if enabled)
        if self.trailing_stop_pct > 0.0 && !price.is_nan() && state.peak_price != 0.0 {
            let drawdown = if state.direction > 0.0 {
                (state.peak_price - price) / state.peak_price
            } else {
                (price - state.peak_price) / state.peak_price
            };
            if drawdown >= self.trailing_stop_pct {
                return (true, format!("trailing_stop={:.4}", drawdown));
            }
        }

        // NaN guard: if z_score or position is NaN, skip signal-based exits
        if z_score.is_nan() || position.is_nan() {
            return (false, String::new());
        }

        // 3. Signal reversal
        if position * z_score < self.reversal_threshold {
            return (true, format!("reversal_z={:.2}", z_score));
        }

        // 4. Deadzone fade (signal too weak to hold)
        if z_score.abs() < self.deadzone_fade {
            return (true, format!("deadzone_fade_z={:.2}", z_score));
        }

        (false, String::new())
    }

    fn allow_entry(&self, z_score: f64, hour_utc: Option<i32>) -> bool {
        // NaN guard
        if z_score.is_nan() {
            return false;
        }

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
        let mut mgr = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        mgr.update_price("ETH", 2100.0);
        // 2100 -> 2050 = 2.38% drawdown > 2% threshold
        let (exit, reason) = mgr.check_exit("ETH", 2050.0, 20, 0.5, 1.0);
        assert!(exit);
        assert!(reason.contains("trailing_stop"));
    }

    #[test]
    fn test_trailing_stop_short() {
        let mut mgr = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, -1.0);
        mgr.update_price("ETH", 1900.0);  // peak for short
        // Price rises > 2% from trough: (1940 - 1900) / 1900 = 2.1%
        let (exit, reason) = mgr.check_exit("ETH", 1940.0, 20, -0.5, -1.0);
        assert!(exit);
        assert!(reason.contains("trailing_stop"));
    }

    #[test]
    fn test_max_hold() {
        let mut mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        let (exit, reason) = mgr.check_exit("ETH", 2000.0, 97, 1.0, 1.0);
        assert!(exit);
        assert!(reason.contains("max_hold"));
    }

    #[test]
    fn test_min_hold_respected() {
        let mut mgr = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        mgr.update_price("ETH", 2100.0);
        let (exit, _) = mgr.check_exit("ETH", 1800.0, 10, 0.5, 1.0);
        assert!(!exit);
    }

    #[test]
    fn test_reversal_exit() {
        let mut mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        // Long position with z=-0.5: position * z = 1.0 * (-0.5) = -0.5 < -0.3
        let (exit, reason) = mgr.check_exit("ETH", 2000.0, 20, -0.5, 1.0);
        assert!(exit);
        assert!(reason.contains("reversal"));
    }

    #[test]
    fn test_deadzone_fade_exit() {
        let mut mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        // |z_score| = 0.1 < 0.2 deadzone_fade
        let (exit, reason) = mgr.check_exit("ETH", 2000.0, 20, 0.1, 1.0);
        assert!(exit);
        assert!(reason.contains("deadzone_fade"));
    }

    #[test]
    fn test_no_exit_strong_signal() {
        let mut mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        // z=0.8, position=1.0: product=0.8 > -0.3, |z|=0.8 > 0.2
        let (exit, _) = mgr.check_exit("ETH", 2000.0, 20, 0.8, 1.0);
        assert!(!exit);
    }

    #[test]
    fn test_zcap_blocks_entry() {
        let mgr = RustExitManager::new(0.0, -0.3, 0.2, 4.0, false, vec![], 12, 96);
        assert!(!mgr.allow_entry(5.0, None));
        assert!(mgr.allow_entry(2.0, None));
    }

    #[test]
    fn test_time_filter() {
        let mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, true, vec![0, 1, 2, 3], 12, 96);
        assert!(!mgr.allow_entry(1.0, Some(0)));
        assert!(mgr.allow_entry(1.0, Some(4)));
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut mgr = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("BTC", 40000.0, 10, 1.0);
        mgr.update_price("BTC", 41000.0);
        mgr.on_entry("ETH", 3000.0, 15, -1.0);
        mgr.update_price("ETH", 2900.0);

        let data = mgr.checkpoint();
        assert!(data.contains_key("BTC"));
        assert!(data.contains_key("ETH"));
        assert_eq!(*data["BTC"].get("peak_price").unwrap(), 41000.0);
        assert_eq!(*data["ETH"].get("direction").unwrap(), -1.0);

        let mut mgr2 = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr2.restore(data);

        // Verify restored state works for exit checks
        mgr2.update_price("BTC", 42000.0);
        let cp2 = mgr2.checkpoint();
        assert_eq!(*cp2["BTC"].get("peak_price").unwrap(), 42000.0);
    }

    #[test]
    fn test_on_exit_clears() {
        let mut mgr = RustExitManager::new(0.02, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        mgr.on_exit("ETH");
        let (exit, _) = mgr.check_exit("ETH", 1800.0, 100, -2.0, 0.0);
        assert!(!exit);
    }

    #[test]
    fn test_nan_z_score_no_exit() {
        let mut mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        mgr.on_entry("ETH", 2000.0, 1, 1.0);
        let (exit, _) = mgr.check_exit("ETH", 2000.0, 20, f64::NAN, 1.0);
        assert!(!exit);
    }

    #[test]
    fn test_nan_z_score_blocks_entry() {
        let mgr = RustExitManager::new(0.0, -0.3, 0.2, 0.0, false, vec![], 12, 96);
        assert!(!mgr.allow_entry(f64::NAN, None));
    }
}
