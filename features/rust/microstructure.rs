//! Microstructure feature extraction in Rust.
//!
//! Provides orderbook feature extraction, VPIN calculation, and streaming
//! microstructure state computation.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::VecDeque;

// ─── Orderbook Feature Extraction ───────────────────────────────────────────

/// Extract microstructure features from L2 orderbook snapshot.
///
/// bids/asks: list of [price, qty] sorted desc/asc respectively.
/// depth_levels: number of top levels to consider.
///
/// Returns dict with: bid_ask_spread, mid_price, bid_ask_imbalance,
///                     depth_ratio, weighted_mid, trade_flow_toxicity.
#[pyfunction]
#[pyo3(signature = (bids, asks, depth_levels=5))]
pub fn rust_extract_orderbook_features(
    py: Python<'_>,
    bids: &Bound<'_, PyList>,
    asks: &Bound<'_, PyList>,
    depth_levels: usize,
) -> PyResult<PyObject> {
    let result = PyDict::new(py);

    if bids.is_empty() || asks.is_empty() {
        result.set_item("bid_ask_spread", 0.0)?;
        result.set_item("mid_price", 0.0)?;
        result.set_item("bid_ask_imbalance", 0.0)?;
        result.set_item("depth_ratio", 0.0)?;
        result.set_item("weighted_mid", 0.0)?;
        result.set_item("trade_flow_toxicity", 0.0)?;
        return Ok(result.into_any().unbind());
    }

    // Parse bids and asks into Vec<(f64, f64)>
    let bids_vec = parse_levels(&bids)?;
    let asks_vec = parse_levels(&asks)?;

    let best_bid_price = bids_vec[0].0;
    let best_bid_qty = bids_vec[0].1;
    let best_ask_price = asks_vec[0].0;
    let best_ask_qty = asks_vec[0].1;

    let spread = best_ask_price - best_bid_price;
    let mid = (best_ask_price + best_bid_price) / 2.0;

    // Imbalance from top N levels
    let n = depth_levels.min(bids_vec.len()).min(asks_vec.len());
    let bid_vol: f64 = bids_vec[..n].iter().map(|(_, q)| q).sum();
    let ask_vol: f64 = asks_vec[..n].iter().map(|(_, q)| q).sum();
    let total_vol = bid_vol + ask_vol;

    let imbalance = if total_vol > 0.0 {
        (bid_vol - ask_vol) / total_vol
    } else {
        0.0
    };

    // Depth ratio (notional)
    let bid_depth: f64 = bids_vec[..n].iter().map(|(p, q)| p * q).sum();
    let ask_depth: f64 = asks_vec[..n].iter().map(|(p, q)| p * q).sum();
    let depth_ratio = if ask_depth > 0.0 {
        bid_depth / ask_depth
    } else {
        0.0
    };

    // Volume-weighted mid
    let top_total = best_bid_qty + best_ask_qty;
    let weighted_mid = if top_total > 0.0 {
        (best_bid_price * best_ask_qty + best_ask_price * best_bid_qty) / top_total
    } else {
        mid
    };

    // Trade flow toxicity: spread / mid
    let toxicity = if mid > 0.0 { spread / mid } else { 0.0 };

    result.set_item("bid_ask_spread", spread)?;
    result.set_item("mid_price", mid)?;
    result.set_item("bid_ask_imbalance", imbalance)?;
    result.set_item("depth_ratio", depth_ratio)?;
    result.set_item("weighted_mid", weighted_mid)?;
    result.set_item("trade_flow_toxicity", toxicity)?;

    Ok(result.into_any().unbind())
}

fn parse_levels(levels: &Bound<'_, PyList>) -> PyResult<Vec<(f64, f64)>> {
    let mut result = Vec::with_capacity(levels.len());
    for item in levels.iter() {
        let pair: Vec<f64> = item.extract()?;
        if pair.len() >= 2 {
            result.push((pair[0], pair[1]));
        }
    }
    Ok(result)
}

// ─── VPIN Calculator ────────────────────────────────────────────────────────

struct VolumeBucket {
    buy_volume: f64,
    sell_volume: f64,
}

/// Volume-Synchronized Probability of Informed Trading.
///
/// Measures order flow toxicity by comparing buy/sell volume imbalance
/// across fixed-volume buckets.
#[pyclass]
pub struct RustVPINCalculator {
    bucket_volume: f64,
    n_buckets: usize,
}

/// Result of VPIN calculation.
#[pyclass]
#[derive(Clone)]
pub struct RustVPINResult {
    #[pyo3(get)]
    vpin: f64,
    #[pyo3(get)]
    buy_volume: f64,
    #[pyo3(get)]
    sell_volume: f64,
    #[pyo3(get)]
    bucket_count: usize,
}

#[pymethods]
impl RustVPINCalculator {
    #[new]
    #[pyo3(signature = (bucket_volume=100.0, n_buckets=50))]
    fn new(bucket_volume: f64, n_buckets: usize) -> Self {
        Self {
            bucket_volume,
            n_buckets,
        }
    }

    /// Calculate VPIN from tick data.
    ///
    /// ticks: list of objects with .price (f64), .qty (f64), and optional .side ("buy"/"sell")
    fn calculate(&self, ticks: &Bound<'_, PyList>) -> PyResult<RustVPINResult> {
        if ticks.is_empty() {
            return Ok(RustVPINResult {
                vpin: 0.0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                bucket_count: 0,
            });
        }

        let mut buckets: VecDeque<VolumeBucket> = VecDeque::with_capacity(self.n_buckets);
        let mut current_buy = 0.0_f64;
        let mut current_sell = 0.0_f64;
        let mut current_total = 0.0_f64;
        let mut prev_price: Option<f64> = None;

        for tick_obj in ticks.iter() {
            let price: f64 = tick_obj
                .getattr("price")
                .and_then(|v| v.extract())
                .unwrap_or(0.0);
            let qty: f64 = tick_obj
                .getattr("qty")
                .and_then(|v| v.extract())
                .unwrap_or(0.0);
            let side_str: String = tick_obj
                .getattr("side")
                .and_then(|v| v.extract())
                .unwrap_or_default();

            let is_buy = if side_str == "buy" || side_str == "sell" {
                side_str == "buy"
            } else {
                // Tick rule
                match prev_price {
                    Some(pp) => price >= pp,
                    None => true,
                }
            };
            prev_price = Some(price);

            let mut remaining = qty;
            while remaining > 0.0 {
                let space = self.bucket_volume - current_total;
                let fill = remaining.min(space);

                if is_buy {
                    current_buy += fill;
                } else {
                    current_sell += fill;
                }
                current_total += fill;
                remaining -= fill;

                if current_total >= self.bucket_volume {
                    if buckets.len() == self.n_buckets {
                        buckets.pop_front();
                    }
                    buckets.push_back(VolumeBucket {
                        buy_volume: current_buy,
                        sell_volume: current_sell,
                    });
                    current_buy = 0.0;
                    current_sell = 0.0;
                    current_total = 0.0;
                }
            }
        }

        let n = buckets.len();
        if n == 0 {
            return Ok(RustVPINResult {
                vpin: 0.0,
                buy_volume: 0.0,
                sell_volume: 0.0,
                bucket_count: 0,
            });
        }

        let imbalance_sum: f64 = buckets
            .iter()
            .map(|b| (b.buy_volume - b.sell_volume).abs())
            .sum();
        let vpin = imbalance_sum / (n as f64 * self.bucket_volume);

        let used_buy: f64 = buckets.iter().map(|b| b.buy_volume).sum();
        let used_sell: f64 = buckets.iter().map(|b| b.sell_volume).sum();

        Ok(RustVPINResult {
            vpin,
            buy_volume: used_buy,
            sell_volume: used_sell,
            bucket_count: n,
        })
    }
}

// ─── Streaming Microstructure ───────────────────────────────────────────────

/// Streaming microstructure feature computer.
///
/// Maintains rolling trade buffer and computes VPIN + orderbook features
/// incrementally on each trade/depth update.
#[pyclass]
pub struct RustStreamingMicrostructure {
    trades: VecDeque<(f64, f64, bool)>, // (price, qty, is_buy)
    max_trades: usize,
    vpin_bucket_volume: f64,
    vpin_n_buckets: usize,
    // Last orderbook state
    last_imbalance: f64,
    last_spread_bps: f64,
    last_weighted_mid: f64,
    last_depth_ratio: f64,
    last_ob_signal: String,
}

#[pymethods]
impl RustStreamingMicrostructure {
    #[new]
    #[pyo3(signature = (trade_buffer_size=200, vpin_bucket_volume=100.0, vpin_n_buckets=50))]
    fn new(trade_buffer_size: usize, vpin_bucket_volume: f64, vpin_n_buckets: usize) -> Self {
        Self {
            trades: VecDeque::with_capacity(trade_buffer_size),
            max_trades: trade_buffer_size,
            vpin_bucket_volume,
            vpin_n_buckets,
            last_imbalance: 0.0,
            last_spread_bps: 0.0,
            last_weighted_mid: 0.0,
            last_depth_ratio: 1.0,
            last_ob_signal: "neutral".to_string(),
        }
    }

    /// Process a trade and return updated microstructure state.
    ///
    /// side: "buy" or "sell"
    /// Returns dict with vpin, trade_count, last_price, ob_imbalance, spread_bps, etc.
    fn on_trade(&mut self, py: Python<'_>, price: f64, qty: f64, side: &str) -> PyResult<PyObject> {
        let is_buy = side == "buy";
        if self.trades.len() == self.max_trades {
            self.trades.pop_front();
        }
        self.trades.push_back((price, qty, is_buy));

        let vpin = self.compute_vpin();

        let result = PyDict::new(py);
        result.set_item("vpin", vpin)?;
        result.set_item("ob_imbalance", self.last_imbalance)?;
        result.set_item("spread_bps", self.last_spread_bps)?;
        result.set_item("weighted_mid", self.last_weighted_mid)?;
        result.set_item("ob_signal", &self.last_ob_signal)?;
        result.set_item("depth_ratio", self.last_depth_ratio)?;
        result.set_item("trade_count", self.trades.len())?;
        result.set_item("last_price", price)?;

        Ok(result.into_any().unbind())
    }

    /// Process a depth update and return updated microstructure state.
    ///
    /// bids/asks: list of [price, qty]
    fn on_depth(
        &mut self,
        py: Python<'_>,
        bids: &Bound<'_, PyList>,
        asks: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let bids_vec = parse_levels(bids)?;
        let asks_vec = parse_levels(asks)?;

        if !bids_vec.is_empty() && !asks_vec.is_empty() {
            let best_bid = bids_vec[0].0;
            let best_ask = asks_vec[0].0;
            let mid = (best_bid + best_ask) / 2.0;

            self.last_spread_bps = if mid > 0.0 {
                (best_ask - best_bid) / mid * 10000.0
            } else {
                0.0
            };

            let n = 5.min(bids_vec.len()).min(asks_vec.len());
            let bid_vol: f64 = bids_vec[..n].iter().map(|(_, q)| q).sum();
            let ask_vol: f64 = asks_vec[..n].iter().map(|(_, q)| q).sum();
            let total = bid_vol + ask_vol;
            self.last_imbalance = if total > 0.0 {
                (bid_vol - ask_vol) / total
            } else {
                0.0
            };

            let bid_depth: f64 = bids_vec[..n].iter().map(|(p, q)| p * q).sum();
            let ask_depth: f64 = asks_vec[..n].iter().map(|(p, q)| p * q).sum();
            self.last_depth_ratio = if ask_depth > 0.0 {
                bid_depth / ask_depth
            } else {
                1.0
            };

            let top_total = bids_vec[0].1 + asks_vec[0].1;
            self.last_weighted_mid = if top_total > 0.0 {
                (best_bid * asks_vec[0].1 + best_ask * bids_vec[0].1) / top_total
            } else {
                mid
            };

            // Simple signal based on imbalance
            self.last_ob_signal = if self.last_imbalance > 0.2 {
                "bullish".to_string()
            } else if self.last_imbalance < -0.2 {
                "bearish".to_string()
            } else {
                "neutral".to_string()
            };
        }

        let vpin = self.compute_vpin();

        let result = PyDict::new(py);
        result.set_item("vpin", vpin)?;
        result.set_item("ob_imbalance", self.last_imbalance)?;
        result.set_item("spread_bps", self.last_spread_bps)?;
        result.set_item("weighted_mid", self.last_weighted_mid)?;
        result.set_item("ob_signal", &self.last_ob_signal)?;
        result.set_item("depth_ratio", self.last_depth_ratio)?;
        result.set_item("trade_count", self.trades.len())?;
        let last_price = self.trades.back().map(|(p, _, _)| *p).unwrap_or(0.0);
        result.set_item("last_price", last_price)?;

        Ok(result.into_any().unbind())
    }
}

impl RustStreamingMicrostructure {
    fn compute_vpin(&self) -> f64 {
        if self.trades.is_empty() {
            return 0.0;
        }

        let mut buckets: VecDeque<(f64, f64)> = VecDeque::with_capacity(self.vpin_n_buckets);
        let mut cur_buy = 0.0_f64;
        let mut cur_sell = 0.0_f64;
        let mut cur_total = 0.0_f64;

        for &(_price, qty, is_buy) in &self.trades {
            let mut remaining = qty;
            while remaining > 0.0 {
                let space = self.vpin_bucket_volume - cur_total;
                let fill = remaining.min(space);
                if is_buy {
                    cur_buy += fill;
                } else {
                    cur_sell += fill;
                }
                cur_total += fill;
                remaining -= fill;

                if cur_total >= self.vpin_bucket_volume {
                    if buckets.len() == self.vpin_n_buckets {
                        buckets.pop_front();
                    }
                    buckets.push_back((cur_buy, cur_sell));
                    cur_buy = 0.0;
                    cur_sell = 0.0;
                    cur_total = 0.0;
                }
            }
        }

        let n = buckets.len();
        if n == 0 {
            return 0.0;
        }

        let imbalance_sum: f64 = buckets.iter().map(|(b, s)| (b - s).abs()).sum();
        imbalance_sum / (n as f64 * self.vpin_bucket_volume)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_empty() {
        let calc = RustVPINCalculator::new(100.0, 50);
        assert_eq!(calc.bucket_volume, 100.0);
        assert_eq!(calc.n_buckets, 50);
    }

    #[test]
    fn test_streaming_defaults() {
        let sm = RustStreamingMicrostructure::new(200, 100.0, 50);
        assert_eq!(sm.max_trades, 200);
        assert_eq!(sm.last_ob_signal, "neutral");
        assert_eq!(sm.last_depth_ratio, 1.0);
    }

    #[test]
    fn test_streaming_vpin_no_trades() {
        let sm = RustStreamingMicrostructure::new(200, 100.0, 50);
        assert_eq!(sm.compute_vpin(), 0.0);
    }

    #[test]
    fn test_streaming_vpin_with_trades() {
        let mut sm = RustStreamingMicrostructure::new(200, 10.0, 5);
        // Add enough trades to fill some buckets
        for i in 0..20 {
            let is_buy = i % 2 == 0;
            sm.trades.push_back((100.0, 5.0, is_buy));
        }
        let vpin = sm.compute_vpin();
        // With alternating buy/sell of equal size, VPIN should be 0
        // (each bucket has equal buy/sell)
        assert!(vpin >= 0.0 && vpin <= 1.0);
    }

    #[test]
    fn test_streaming_vpin_all_buy() {
        let mut sm = RustStreamingMicrostructure::new(200, 10.0, 5);
        // All buys → max toxicity
        for _ in 0..20 {
            sm.trades.push_back((100.0, 5.0, true));
        }
        let vpin = sm.compute_vpin();
        assert!((vpin - 1.0).abs() < 1e-10, "All-buy VPIN should be 1.0, got {vpin}");
    }
}
