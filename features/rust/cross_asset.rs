//! Cross-asset feature computation in Rust.
//!
//! Maintains per-symbol return/funding state and computes inter-asset features:
//! rolling beta, relative strength, rolling correlation, funding spread, etc.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{HashMap, VecDeque};

const BENCHMARK: &str = "BTCUSDT";

// ── Helper types: AssetState, EMA, RollingStats, PairState, utility fns ──
include!("cross_asset_helpers.inc.rs");

// ── PyO3 class ──────────────────────────────────────────────

#[pyclass]
pub struct RustCrossAssetComputer {
    assets: HashMap<String, AssetState>,
    pairs: HashMap<String, PairState>,
    benchmark: String,
}

#[pymethods]
impl RustCrossAssetComputer {
    #[new]
    #[pyo3(signature = (benchmark=None))]
    fn new(benchmark: Option<&str>) -> Self {
        Self {
            assets: HashMap::new(),
            pairs: HashMap::new(),
            benchmark: benchmark.unwrap_or(BENCHMARK).to_string(),
        }
    }

    fn set_benchmark(&mut self, symbol: &str) {
        self.benchmark = symbol.to_string();
    }

    #[pyo3(signature = (symbol, close, funding_rate=None, high=None, low=None))]
    fn on_bar(
        &mut self,
        symbol: &str,
        close: f64,
        funding_rate: Option<f64>,
        high: Option<f64>,
        low: Option<f64>,
    ) {
        let sym_key = symbol.to_string();
        let asset = self.assets.entry(sym_key.clone()).or_insert_with(AssetState::new);
        asset.push(close, funding_rate, high, low);

        if symbol != self.benchmark {
            // Gather data from immutable refs before mutable pair access
            let (sym_ret, bench_ret, f_diff) = {
                let sym_state = match self.assets.get(symbol) {
                    Some(s) => s,
                    None => return,
                };
                let bench_state = match self.assets.get(&self.benchmark) {
                    Some(bs) => bs,
                    None => return,
                };
                let sr = sym_state.ret(1);
                let br = bench_state.ret(1);
                let fd = match (sym_state.last_funding_rate, bench_state.last_funding_rate) {
                    (Some(sf), Some(bf)) => Some(sf - bf),
                    _ => None,
                };
                (sr, br, fd)
            };
            if let (Some(sr), Some(br)) = (sym_ret, bench_ret) {
                let pair_key = [symbol, "_", &self.benchmark].concat();
                let pair = self.pairs.entry(pair_key).or_insert_with(PairState::new);
                pair.push(sr, br, f_diff);
            }
        }
    }

    #[pyo3(signature = (symbol, benchmark=None))]
    fn get_features<'py>(
        &self,
        py: Python<'py>,
        symbol: &str,
        benchmark: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let bench = benchmark.unwrap_or(&self.benchmark);
        let dict = PyDict::new(py);

        // Initialize all to None
        for name in FEATURE_NAMES {
            dict.set_item(*name, py.None())?;
        }

        if symbol == bench {
            return Ok(dict);
        }

        let bench_state = match self.assets.get(bench) {
            Some(bs) => bs,
            None => return Ok(dict),
        };
        if !self.assets.contains_key(symbol) {
            return Ok(dict);
        }

        set_opt(&dict, "btc_ret_1", bench_state.ret(1))?;
        set_opt(&dict, "btc_ret_3", bench_state.ret(3))?;
        set_opt(&dict, "btc_ret_6", bench_state.ret(6))?;
        set_opt(&dict, "btc_ret_12", bench_state.ret(12))?;
        set_opt(&dict, "btc_ret_24", bench_state.ret(24))?;
        set_opt(&dict, "btc_rsi_14", bench_state.rsi_14())?;
        set_opt(&dict, "btc_macd_line", bench_state.macd_line())?;
        set_opt(&dict, "btc_mean_reversion_20", bench_state.mean_reversion_20())?;
        set_opt(&dict, "btc_atr_norm_14", bench_state.atr_norm_14())?;
        set_opt(&dict, "btc_bb_width_20", bench_state.bb_width_20())?;

        let pair_key = [symbol, "_", bench].concat();
        let pair = match self.pairs.get(&pair_key) {
            Some(p) => p,
            None => return Ok(dict),
        };

        set_opt(&dict, "rolling_beta_30",
            beta_from_deques(&pair.sym_rets_30, &pair.bench_rets_30, 30))?;
        set_opt(&dict, "rolling_beta_60",
            beta_from_deques(&pair.sym_rets_60, &pair.bench_rets_60, 60))?;

        // Relative strength 20
        if pair.sym_cum_20.len() >= 20 {
            let mut sym_cum = 1.0;
            let mut bench_cum = 1.0;
            for (sr, br) in pair.sym_cum_20.iter().zip(pair.bench_cum_20.iter()) {
                sym_cum *= 1.0 + sr;
                bench_cum *= 1.0 + br;
            }
            if bench_cum != 0.0 {
                dict.set_item("relative_strength_20", sym_cum / bench_cum)?;
            }
        }

        // Rolling correlation 30
        if pair.sym_rets_30.len() >= 30 {
            set_opt(&dict, "rolling_corr_30",
                pearson(&pair.sym_rets_30, &pair.bench_rets_30))?;
        }

        // Funding diff
        if let Some(fd) = pair.last_funding_diff {
            dict.set_item("funding_diff", fd)?;
        }
        if pair.funding_diff_ema.ready() {
            dict.set_item("funding_diff_ma8", pair.funding_diff_ema.value)?;
        }

        // Spread z-score
        if pair.spread_window_20.full() {
            let mean_s = pair.spread_window_20.mean();
            let std_s = pair.spread_window_20.std();
            let beta30 = beta_from_deques(&pair.sym_rets_30, &pair.bench_rets_30, 30);
            let sym_ret_1 = self.assets.get(symbol).and_then(|s| s.ret(1));
            let bench_ret_1 = bench_state.ret(1);
            if let (Some(b30), Some(sr1), Some(br1)) = (beta30, sym_ret_1, bench_ret_1) {
                if std_s > 1e-12 {
                    let spread = sr1 - b30 * br1;
                    dict.set_item("spread_zscore_20", (spread - mean_s) / std_s)?;
                }
            }
        }

        Ok(dict)
    }
}

fn set_opt(dict: &Bound<'_, PyDict>, key: &str, val: Option<f64>) -> PyResult<()> {
    if let Some(v) = val {
        dict.set_item(key, v)?;
    }
    Ok(())
}

// ── Unit tests ──
include!("cross_asset_tests.inc.rs");
