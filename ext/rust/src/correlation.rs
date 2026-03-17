// correlation.rs — RustCorrelationComputer: Rust backend for risk/correlation_computer.py
//
// Rolling pairwise Pearson correlation across portfolio symbols.
// Mirrors Python CorrelationComputer 1:1 for parity.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, VecDeque};

// ============================================================
// Pure Rust computation (testable without PyO3)
// ============================================================

/// Pearson correlation between two VecDeque<f64> series.
/// Uses the last `min(a.len(), b.len())` elements aligned from the tail.
fn pearson_corr(a: &VecDeque<f64>, b: &VecDeque<f64>) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 2 {
        return None;
    }

    // Align from the tail: take the last n elements of each
    let a_offset = a.len() - n;
    let b_offset = b.len() - n;

    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n {
        sum_a += a[a_offset + i];
        sum_b += b[b_offset + i];
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[a_offset + i] - mean_a;
        let db = b[b_offset + i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-15 {
        return None;
    }

    let r = cov / denom;
    if r.is_finite() {
        Some(r)
    } else {
        None
    }
}

// ============================================================
// PyO3 wrapper
// ============================================================

#[pyclass(name = "RustCorrelationComputer")]
pub struct RustCorrelationComputer {
    returns: HashMap<String, VecDeque<f64>>,
    last_prices: HashMap<String, f64>,
    window: usize,
}

#[pymethods]
impl RustCorrelationComputer {
    #[new]
    #[pyo3(signature = (window=60))]
    fn new(window: usize) -> Self {
        Self {
            returns: HashMap::new(),
            last_prices: HashMap::new(),
            window,
        }
    }

    /// Push a new close price. Computes log return internally.
    fn update(&mut self, symbol: &str, close: f64) -> PyResult<()> {
        if !close.is_finite() || close <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "close must be positive finite, got {}",
                close
            )));
        }

        if let Some(&prev) = self.last_prices.get(symbol) {
            if prev > 0.0 {
                let log_ret = (close / prev).ln();
                if log_ret.is_finite() {
                    let buf = self
                        .returns
                        .entry(symbol.to_string())
                        .or_insert_with(|| VecDeque::with_capacity(self.window));
                    if buf.len() >= self.window {
                        buf.pop_front();
                    }
                    buf.push_back(log_ret);
                }
            }
        }
        self.last_prices.insert(symbol.to_string(), close);
        Ok(())
    }

    /// Push a pre-computed return directly.
    fn push_return(&mut self, symbol: &str, ret: f64) -> PyResult<()> {
        if !ret.is_finite() {
            return Err(PyValueError::new_err("return must be finite"));
        }
        let buf = self
            .returns
            .entry(symbol.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.window));
        if buf.len() >= self.window {
            buf.pop_front();
        }
        buf.push_back(ret);
        Ok(())
    }

    /// Compute pairwise Pearson correlation between two symbols.
    fn pairwise_correlation(&self, sym_a: &str, sym_b: &str) -> Option<f64> {
        let a = self.returns.get(sym_a)?;
        let b = self.returns.get(sym_b)?;
        pearson_corr(a, b)
    }

    /// Compute average absolute pairwise correlation for given symbols.
    fn avg_correlation(&self, symbols: Vec<String>) -> Option<f64> {
        if symbols.len() < 2 {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0u32;
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                if let Some(corr) = self.pairwise_correlation(&symbols[i], &symbols[j]) {
                    total += corr.abs();
                    count += 1;
                }
            }
        }
        if count == 0 {
            None
        } else {
            Some(total / count as f64)
        }
    }

    /// Check if avg correlation exceeds threshold.
    fn exceeds_limit(&self, symbols: Vec<String>, max_corr: f64) -> bool {
        self.avg_correlation(symbols)
            .map_or(false, |c| c > max_corr)
    }

    /// Compute average absolute correlation of a new symbol against existing portfolio.
    fn position_correlation(&self, new_symbol: &str, existing: Vec<String>) -> Option<f64> {
        if existing.is_empty() {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0u32;
        for sym in &existing {
            if let Some(corr) = self.pairwise_correlation(new_symbol, sym) {
                total += corr.abs();
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(total / count as f64)
        }
    }

    /// Get correlation matrix as dict of dicts.
    fn correlation_matrix(&self, py: Python<'_>, symbols: Vec<String>) -> PyResult<PyObject> {
        let outer = PyDict::new(py);
        for i in 0..symbols.len() {
            let inner = PyDict::new(py);
            for j in 0..symbols.len() {
                if i == j {
                    inner.set_item(&symbols[j], 1.0)?;
                } else {
                    let corr = self
                        .pairwise_correlation(&symbols[i], &symbols[j])
                        .unwrap_or(f64::NAN);
                    inner.set_item(&symbols[j], corr)?;
                }
            }
            outer.set_item(&symbols[i], inner)?;
        }
        Ok(outer.into_any().unbind())
    }

    /// Number of symbols being tracked.
    fn symbol_count(&self) -> usize {
        self.returns.len()
    }

    /// Number of returns stored for a symbol.
    fn data_points(&self, symbol: &str) -> usize {
        self.returns.get(symbol).map_or(0, |v| v.len())
    }

    /// Checkpoint state as dict for persistence.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Serialize returns: {symbol: [float, ...]}
        let returns_dict = PyDict::new(py);
        for (sym, buf) in &self.returns {
            let py_list = PyList::new(py, buf.iter().copied())?;
            returns_dict.set_item(sym, py_list)?;
        }
        dict.set_item("returns", returns_dict)?;

        // Serialize last_prices: {symbol: float}
        let prices_dict = PyDict::new(py);
        for (sym, price) in &self.last_prices {
            prices_dict.set_item(sym, *price)?;
        }
        dict.set_item("last_prices", prices_dict)?;

        Ok(dict.into())
    }

    /// Restore from checkpoint dict.
    fn restore(&mut self, data: &Bound<'_, PyDict>) -> PyResult<()> {
        // Restore returns
        if let Some(returns_obj) = data.get_item("returns")? {
            let returns_dict: &Bound<'_, PyDict> = returns_obj.downcast()?;
            self.returns.clear();
            for (key, value) in returns_dict.iter() {
                let sym: String = key.extract()?;
                let py_list: &Bound<'_, PyList> = value.downcast()?;
                let mut buf = VecDeque::with_capacity(self.window);
                for item in py_list.iter() {
                    let val: f64 = item.extract()?;
                    buf.push_back(val);
                }
                // Trim to window size if checkpoint had more
                while buf.len() > self.window {
                    buf.pop_front();
                }
                self.returns.insert(sym, buf);
            }
        }

        // Restore last_prices
        if let Some(prices_obj) = data.get_item("last_prices")? {
            let prices_dict: &Bound<'_, PyDict> = prices_obj.downcast()?;
            self.last_prices.clear();
            for (key, value) in prices_dict.iter() {
                let sym: String = key.extract()?;
                let price: f64 = value.extract()?;
                self.last_prices.insert(sym, price);
            }
        }

        Ok(())
    }
}

// ============================================================
// Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_perfect_positive() {
        let a: VecDeque<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0].into();
        let b: VecDeque<f64> = vec![2.0, 4.0, 6.0, 8.0, 10.0].into();
        let r = pearson_corr(&a, &b).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let a: VecDeque<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0].into();
        let b: VecDeque<f64> = vec![10.0, 8.0, 6.0, 4.0, 2.0].into();
        let r = pearson_corr(&a, &b).unwrap();
        assert!((r + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_insufficient_data() {
        let a: VecDeque<f64> = vec![1.0].into();
        let b: VecDeque<f64> = vec![2.0].into();
        assert!(pearson_corr(&a, &b).is_none());
    }

    #[test]
    fn test_pearson_constant_series() {
        let a: VecDeque<f64> = vec![5.0, 5.0, 5.0].into();
        let b: VecDeque<f64> = vec![1.0, 2.0, 3.0].into();
        // Constant series → zero variance → None
        assert!(pearson_corr(&a, &b).is_none());
    }

    #[test]
    fn test_pearson_different_lengths() {
        // Uses last min(len) elements aligned from tail
        let a: VecDeque<f64> = vec![99.0, 1.0, 2.0, 3.0].into();
        let b: VecDeque<f64> = vec![1.0, 2.0, 3.0].into();
        let r = pearson_corr(&a, &b).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_computer_update_and_corr() {
        let mut comp = RustCorrelationComputer::new(100);
        // Feed perfectly correlated prices
        for i in 0..10 {
            let price = 100.0 + i as f64;
            comp.update("A", price).unwrap();
            comp.update("B", price * 2.0).unwrap();
        }
        // 9 returns each (first price establishes baseline)
        assert_eq!(comp.data_points("A"), 9);
        assert_eq!(comp.data_points("B"), 9);

        let corr = comp.pairwise_correlation("A", "B").unwrap();
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_push_return() {
        let mut comp = RustCorrelationComputer::new(5);
        for i in 0..7 {
            comp.push_return("X", i as f64 * 0.01).unwrap();
        }
        // Window is 5, so only last 5 kept
        assert_eq!(comp.data_points("X"), 5);
    }

    #[test]
    fn test_avg_correlation_single_symbol() {
        let comp = RustCorrelationComputer::new(60);
        assert!(comp.avg_correlation(vec!["A".to_string()]).is_none());
    }

    #[test]
    fn test_exceeds_limit() {
        let mut comp = RustCorrelationComputer::new(100);
        for i in 0..20 {
            let p = 100.0 + i as f64;
            comp.update("A", p).unwrap();
            comp.update("B", p * 2.0).unwrap();
        }
        let syms = vec!["A".to_string(), "B".to_string()];
        // Perfect correlation ≈ 1.0, so exceeds 0.5
        assert!(comp.exceeds_limit(syms.clone(), 0.5));
        // Does not exceed 1.5
        assert!(!comp.exceeds_limit(syms, 1.5));
    }

    #[test]
    fn test_position_correlation_empty() {
        let comp = RustCorrelationComputer::new(60);
        assert!(comp
            .position_correlation("A", vec![])
            .is_none());
    }

    #[test]
    fn test_symbol_count() {
        let mut comp = RustCorrelationComputer::new(60);
        assert_eq!(comp.symbol_count(), 0);
        comp.update("A", 100.0).unwrap();
        comp.update("A", 101.0).unwrap();
        assert_eq!(comp.symbol_count(), 1);
        comp.update("B", 50.0).unwrap();
        comp.update("B", 51.0).unwrap();
        assert_eq!(comp.symbol_count(), 2);
    }

    #[test]
    fn test_update_rejects_invalid() {
        let mut comp = RustCorrelationComputer::new(60);
        assert!(comp.update("A", -1.0).is_err());
        assert!(comp.update("A", 0.0).is_err());
        assert!(comp.update("A", f64::NAN).is_err());
        assert!(comp.update("A", f64::INFINITY).is_err());
    }

    #[test]
    fn test_push_return_rejects_nan() {
        let mut comp = RustCorrelationComputer::new(60);
        assert!(comp.push_return("A", f64::NAN).is_err());
        assert!(comp.push_return("A", f64::INFINITY).is_err());
    }

    #[test]
    fn test_window_eviction() {
        let mut comp = RustCorrelationComputer::new(3);
        // Push 5 returns; only last 3 kept
        for i in 1..=5 {
            comp.push_return("A", i as f64 * 0.01).unwrap();
        }
        assert_eq!(comp.data_points("A"), 3);
    }
}
