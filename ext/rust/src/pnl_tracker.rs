// pnl_tracker.rs — RustPnLTracker: Rust backend for scripts/ops/pnl_tracker.py
//
// Tracks trade P&L, win rate, peak equity, and drawdown.
// Mirrors Python PnLTracker 1:1 for parity.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::VecDeque;

// ============================================================
// Internal trade record (no PyO3 dependency)
// ============================================================

struct TradeRecord {
    symbol: String,
    side: i32,
    entry: f64,
    exit_price: f64,
    size: f64,
    pnl_usd: f64,
    pnl_pct: f64,
    reason: String,
    total_pnl: f64,
    trade_count: u64,
}

// ============================================================
// Pure Rust computation helpers (testable without PyO3)
// ============================================================

/// Compute pnl_pct and pnl_usd for a position close.
/// Returns (pnl_pct, pnl_usd).
pub(crate) fn compute_pnl(side: i32, entry_price: f64, exit_price: f64, size: f64) -> (f64, f64) {
    let pnl_pct = if side == 1 {
        (exit_price - entry_price) / entry_price * 100.0
    } else {
        (entry_price - exit_price) / entry_price * 100.0
    };
    let pnl_usd = pnl_pct / 100.0 * entry_price * size;
    (pnl_pct, pnl_usd)
}

/// Compute drawdown percentage from peak and current total PnL.
pub(crate) fn compute_drawdown(peak_equity: f64, total_pnl: f64) -> f64 {
    if peak_equity <= 0.0 {
        0.0
    } else {
        (peak_equity - total_pnl) / peak_equity * 100.0
    }
}

/// Compute win rate percentage.
pub(crate) fn compute_win_rate(win_count: u64, trade_count: u64) -> f64 {
    if trade_count == 0 {
        0.0
    } else {
        win_count as f64 / trade_count as f64 * 100.0
    }
}

// ============================================================
// RustPnLTracker
// ============================================================

#[pyclass(name = "RustPnLTracker")]
pub struct RustPnLTracker {
    total_pnl: f64,
    peak_equity: f64,
    trade_count: u64,
    win_count: u64,
    trades: VecDeque<TradeRecord>,
}

#[pymethods]
impl RustPnLTracker {
    #[new]
    pub fn new() -> Self {
        Self {
            total_pnl: 0.0,
            peak_equity: 0.0,
            trade_count: 0,
            win_count: 0,
            trades: VecDeque::new(),
        }
    }

    /// Record a position close. Returns trade info dict.
    ///
    /// Args:
    ///   symbol: exchange symbol (e.g. "ETHUSDT")
    ///   side: +1 = was long, -1 = was short
    ///   entry_price: average entry price
    ///   exit_price: exit/close price
    ///   size: position size in base asset
    ///   reason: close reason
    #[pyo3(signature = (symbol, side, entry_price, exit_price, size, reason = "".to_string()))]
    pub fn record_close(
        &mut self,
        symbol: String,
        side: i32,
        entry_price: f64,
        exit_price: f64,
        size: f64,
        reason: String,
        py: Python<'_>,
    ) -> PyResult<Py<PyDict>> {
        if !entry_price.is_finite() {
            return Err(PyValueError::new_err(format!(
                "entry_price must be finite, got {entry_price}"
            )));
        }
        if !exit_price.is_finite() {
            return Err(PyValueError::new_err(format!(
                "exit_price must be finite, got {exit_price}"
            )));
        }
        if size <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "size must be > 0, got {size}"
            )));
        }

        let (pnl_pct, pnl_usd) = compute_pnl(side, entry_price, exit_price, size);

        self.total_pnl += pnl_usd;
        self.trade_count += 1;
        if pnl_usd > 0.0 {
            self.win_count += 1;
        }
        self.peak_equity = f64::max(self.peak_equity, self.total_pnl);

        let record = TradeRecord {
            symbol: symbol.clone(),
            side,
            entry: entry_price,
            exit_price,
            size,
            pnl_usd,
            pnl_pct,
            reason: reason.clone(),
            total_pnl: self.total_pnl,
            trade_count: self.trade_count,
        };

        self.trades.push_back(record);
        if self.trades.len() > 100 {
            self.trades.pop_front();
        }

        let dict = PyDict::new(py);
        dict.set_item("symbol", &symbol)?;
        dict.set_item("side", side)?;
        dict.set_item("entry", entry_price)?;
        dict.set_item("exit", exit_price)?;
        dict.set_item("size", size)?;
        dict.set_item("pnl_usd", pnl_usd)?;
        dict.set_item("pnl_pct", pnl_pct)?;
        dict.set_item("reason", &reason)?;
        dict.set_item("total_pnl", self.total_pnl)?;
        dict.set_item("trade_count", self.trade_count)?;
        Ok(dict.into())
    }

    /// Win rate as percentage (0-100).
    #[getter]
    pub fn win_rate(&self) -> f64 {
        compute_win_rate(self.win_count, self.trade_count)
    }

    /// Current drawdown from peak equity as percentage.
    #[getter]
    pub fn drawdown_pct(&self) -> f64 {
        compute_drawdown(self.peak_equity, self.total_pnl)
    }

    #[getter]
    pub fn total_pnl(&self) -> f64 {
        self.total_pnl
    }

    #[getter]
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    #[getter]
    pub fn trade_count(&self) -> u64 {
        self.trade_count
    }

    #[getter]
    pub fn win_count(&self) -> u64 {
        self.win_count
    }

    /// Return summary dict for logging/monitoring.
    pub fn summary(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_pnl", self.total_pnl)?;
        dict.set_item("trades", self.trade_count)?;
        dict.set_item("wins", self.win_count)?;
        dict.set_item("win_rate", self.win_rate())?;
        dict.set_item("peak", self.peak_equity)?;
        dict.set_item("drawdown", self.drawdown_pct())?;
        Ok(dict.into())
    }
}

// ============================================================
// Unit tests (pure Rust, no PyO3)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_win() {
        // entry=100, exit=110, side=1, size=1 → pnl_pct=10, pnl_usd=10
        let (pnl_pct, pnl_usd) = compute_pnl(1, 100.0, 110.0, 1.0);
        assert!((pnl_pct - 10.0).abs() < 1e-9, "pnl_pct={pnl_pct}");
        assert!((pnl_usd - 10.0).abs() < 1e-9, "pnl_usd={pnl_usd}");

        // Simulate tracker state
        let mut total_pnl = 0.0_f64;
        let mut peak_equity = 0.0_f64;
        let mut trade_count = 0_u64;
        let mut win_count = 0_u64;

        total_pnl += pnl_usd;
        trade_count += 1;
        if pnl_usd > 0.0 {
            win_count += 1;
        }
        peak_equity = f64::max(peak_equity, total_pnl);

        assert!((total_pnl - 10.0).abs() < 1e-9);
        assert_eq!(win_count, 1);
        assert_eq!(trade_count, 1);
        assert!((peak_equity - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_short_loss() {
        // entry=100, exit=110, side=-1 → price moved against short → loss
        let (pnl_pct, pnl_usd) = compute_pnl(-1, 100.0, 110.0, 1.0);
        assert!((pnl_pct - (-10.0)).abs() < 1e-9, "pnl_pct={pnl_pct}");
        assert!((pnl_usd - (-10.0)).abs() < 1e-9, "pnl_usd={pnl_usd}");

        let mut win_count = 0_u64;
        let mut trade_count = 0_u64;
        let mut total_pnl = 0.0_f64;
        total_pnl += pnl_usd;
        trade_count += 1;
        if pnl_usd > 0.0 {
            win_count += 1;
        }
        assert_eq!(win_count, 0);
        assert_eq!(trade_count, 1);
        assert!((total_pnl - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn test_short_win() {
        // entry=100, exit=90, side=-1 → price fell → short wins
        let (pnl_pct, pnl_usd) = compute_pnl(-1, 100.0, 90.0, 1.0);
        assert!((pnl_pct - 10.0).abs() < 1e-9, "pnl_pct={pnl_pct}");
        assert!((pnl_usd - 10.0).abs() < 1e-9, "pnl_usd={pnl_usd}");
    }

    #[test]
    fn test_drawdown() {
        // Win then partial loss → peak > current → 0 < drawdown < 100
        // Win: entry=100, exit=110, size=1 → +10
        // Partial loss: entry=110, exit=105, size=1 → -5/110*100 * (1/100) * 110 * 1 = -5.0
        let (_, pnl1) = compute_pnl(1, 100.0, 110.0, 1.0); // +10
        let (_, pnl2) = compute_pnl(1, 110.0, 105.0, 1.0); // ≈ -4.545...

        let mut total_pnl = 0.0_f64;
        let mut peak_equity = 0.0_f64;

        total_pnl += pnl1;
        peak_equity = f64::max(peak_equity, total_pnl);

        total_pnl += pnl2;
        peak_equity = f64::max(peak_equity, total_pnl);

        let dd = compute_drawdown(peak_equity, total_pnl);
        // peak=10.0, current≈5.455, dd ≈ (10-5.455)/10*100 ≈ 45.5%
        assert!(dd > 0.0, "drawdown should be positive, got {dd}");
        assert!(dd < 100.0, "drawdown should be < 100%, got {dd}");
    }

    #[test]
    fn test_win_rate() {
        assert!((compute_win_rate(0, 0) - 0.0).abs() < 1e-9);
        assert!((compute_win_rate(3, 4) - 75.0).abs() < 1e-9);
        assert!((compute_win_rate(1, 1) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_drawdown_zero_peak() {
        // If peak <= 0, drawdown is 0
        assert!((compute_drawdown(0.0, -5.0) - 0.0).abs() < 1e-9);
        assert!((compute_drawdown(-1.0, -5.0) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_cap_100_trades() {
        // Simulate pushing 105 trades into a VecDeque capped at 100
        let mut trades: VecDeque<u64> = VecDeque::new();
        for i in 0..105_u64 {
            trades.push_back(i);
            if trades.len() > 100 {
                trades.pop_front();
            }
        }
        assert_eq!(trades.len(), 100);
        // The oldest 5 should have been evicted; first remaining = 5
        assert_eq!(*trades.front().unwrap(), 5_u64);
    }
}
