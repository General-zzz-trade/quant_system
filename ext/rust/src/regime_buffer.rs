//! Regime price buffer — rolling volatility and moving averages in Rust.
//!
//! Replaces `_PriceBuffer` from `decision/regime_bridge.py`.

use pyo3::prelude::*;
use std::collections::VecDeque;

/// Rolling price buffer for regime detection features.
///
/// Maintains a fixed-capacity ring buffer of prices and provides
/// O(1) moving average and O(n) rolling volatility.
#[pyclass]
pub struct RustRegimeBuffer {
    prices: VecDeque<f64>,
    maxlen: usize,
}

#[pymethods]
impl RustRegimeBuffer {
    #[new]
    #[pyo3(signature = (maxlen=100))]
    fn new(maxlen: usize) -> Self {
        Self {
            prices: VecDeque::with_capacity(maxlen),
            maxlen,
        }
    }

    fn push(&mut self, price: f64) {
        if self.prices.len() == self.maxlen {
            self.prices.pop_front();
        }
        self.prices.push_back(price);
    }

    #[getter]
    fn n(&self) -> usize {
        self.prices.len()
    }

    /// Rolling volatility (std of returns) over last `window` prices.
    ///
    /// Returns None if fewer than window+1 prices available.
    #[pyo3(signature = (window=20))]
    fn rolling_vol(&self, window: usize) -> Option<f64> {
        let n = self.prices.len();
        if n < window + 1 {
            return None;
        }

        let start = n - window;
        let mut rets = Vec::with_capacity(window);
        for i in start..n {
            let prev = self.prices[i - 1];
            if prev != 0.0 {
                rets.push((self.prices[i] - prev) / prev);
            }
        }

        if rets.len() < 2 {
            return None;
        }

        let mean = rets.iter().sum::<f64>() / rets.len() as f64;
        let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rets.len() as f64;
        Some(var.max(0.0).sqrt())
    }

    /// Simple moving average of last `window` prices.
    ///
    /// Returns None if fewer than `window` prices.
    fn ma(&self, window: usize) -> Option<f64> {
        let n = self.prices.len();
        if n < window {
            return None;
        }
        let sum: f64 = self.prices.iter().skip(n - window).sum();
        Some(sum / window as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_n() {
        let mut buf = RustRegimeBuffer::new(5);
        assert_eq!(buf.n(), 0);
        buf.push(100.0);
        buf.push(101.0);
        assert_eq!(buf.n(), 2);
    }

    #[test]
    fn test_maxlen() {
        let mut buf = RustRegimeBuffer::new(3);
        for i in 0..10 {
            buf.push(i as f64);
        }
        assert_eq!(buf.n(), 3);
        // Should have [7, 8, 9]
        assert_eq!(buf.prices[0], 7.0);
    }

    #[test]
    fn test_ma() {
        let mut buf = RustRegimeBuffer::new(100);
        for i in 1..=10 {
            buf.push(i as f64);
        }
        let ma5 = buf.ma(5).unwrap();
        // MA of [6,7,8,9,10] = 8.0
        assert!((ma5 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_ma_insufficient() {
        let mut buf = RustRegimeBuffer::new(100);
        buf.push(1.0);
        buf.push(2.0);
        assert!(buf.ma(5).is_none());
    }

    #[test]
    fn test_rolling_vol() {
        let mut buf = RustRegimeBuffer::new(100);
        // Push constant prices → vol = 0
        for _ in 0..25 {
            buf.push(100.0);
        }
        let vol = buf.rolling_vol(20).unwrap();
        assert!((vol - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_vol_insufficient() {
        let mut buf = RustRegimeBuffer::new(100);
        for i in 0..10 {
            buf.push(100.0 + i as f64);
        }
        assert!(buf.rolling_vol(20).is_none());
    }

    #[test]
    fn test_rolling_vol_positive() {
        let mut buf = RustRegimeBuffer::new(100);
        // Alternating prices → positive vol
        for i in 0..25 {
            let price = if i % 2 == 0 { 100.0 } else { 102.0 };
            buf.push(price);
        }
        let vol = buf.rolling_vol(20).unwrap();
        assert!(vol > 0.0);
    }
}
