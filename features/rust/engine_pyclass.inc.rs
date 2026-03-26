// pyclass.inc.rs — Included by engine.rs via include!() macro.
// ============================================================
// RustFeatureEngine — Incremental feature computer for live trading
// ============================================================
// Wraps BarState as a PyO3 class. Holds all rolling state on the Rust heap.
// Call push_bar() once per bar, then get_features() to read the 105-feature vector.

#[pyclass(name = "RustFeatureEngine")]
pub struct RustFeatureEngine {
    pub(crate) state: BarState,
    pub(crate) cached_features: [f64; N_FEATURES],
    pub(crate) prev_momentum_val: f64,

    // V14 dominance state (used by push_dominance)
    dom_ratio_buf: Vec<f64>,
    dom_btc_ret_buf: Vec<f64>,
    dom_eth_ret_buf: Vec<f64>,
    dom_last_btc: f64,
    dom_last_eth: f64,
}

#[pymethods]
impl RustFeatureEngine {
    #[new]
    fn new() -> Self {
        Self {
            state: BarState::new(),
            cached_features: [f64::NAN; N_FEATURES],
            prev_momentum_val: f64::NAN,
            dom_ratio_buf: Vec::with_capacity(75),
            dom_btc_ret_buf: Vec::with_capacity(25),
            dom_eth_ret_buf: Vec::with_capacity(25),
            dom_last_btc: 0.0,
            dom_last_eth: 0.0,
        }
    }

    /// Push a new bar and update all rolling state.
    ///
    /// Args:
    ///   close, volume, high, low, open: OHLCV data
    ///   hour, dow: cyclical time features (-1 if unknown)
    ///   funding_rate: NaN if not available
    ///   trades: trade count (0 if unknown)
    ///   taker_buy_volume, quote_volume, taker_buy_quote_volume: microstructure
    ///   open_interest, ls_ratio, spot_close, fear_greed: NaN if unavailable
    ///   implied_vol, put_call_ratio: NaN if unavailable
    ///   All on-chain/liquidation/mempool/macro/social: NaN if unavailable
    #[pyo3(signature = (
        close, volume, high, low, open,
        hour=-1, dow=-1,
        funding_rate=f64::NAN,
        trades=0.0,
        taker_buy_volume=0.0,
        quote_volume=0.0,
        taker_buy_quote_volume=0.0,
        open_interest=f64::NAN,
        ls_ratio=f64::NAN,
        spot_close=f64::NAN,
        fear_greed=f64::NAN,
        implied_vol=f64::NAN,
        put_call_ratio=f64::NAN,
        oc_flow_in=f64::NAN, oc_flow_out=f64::NAN,
        oc_supply=f64::NAN, oc_addr=f64::NAN,
        oc_tx=f64::NAN, oc_hashrate=f64::NAN,
        liq_total_vol=f64::NAN, liq_buy_vol=f64::NAN, liq_sell_vol=f64::NAN,
        liq_count=f64::NAN,
        mempool_fastest_fee=f64::NAN, mempool_economy_fee=f64::NAN,
        mempool_size=f64::NAN,
        macro_dxy=f64::NAN, macro_spx=f64::NAN, macro_vix=f64::NAN,
        macro_day=-1_i64,
        social_volume=f64::NAN, sentiment_score=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn push_bar(
        &mut self,
        close: f64, volume: f64, high: f64, low: f64, open: f64,
        hour: i32, dow: i32,
        funding_rate: f64,
        trades: f64,
        taker_buy_volume: f64,
        quote_volume: f64,
        taker_buy_quote_volume: f64,
        open_interest: f64,
        ls_ratio: f64,
        spot_close: f64,
        fear_greed: f64,
        implied_vol: f64,
        put_call_ratio: f64,
        oc_flow_in: f64, oc_flow_out: f64,
        oc_supply: f64, oc_addr: f64,
        oc_tx: f64, oc_hashrate: f64,
        liq_total_vol: f64, liq_buy_vol: f64, liq_sell_vol: f64,
        liq_count: f64,
        mempool_fastest_fee: f64, mempool_economy_fee: f64,
        mempool_size: f64,
        macro_dxy: f64, macro_spx: f64, macro_vix: f64,
        macro_day: i64,
        social_volume: f64, sentiment_score: f64,
    ) {
        self.state.push(
            close, volume, high, low, open,
            hour, dow,
            funding_rate, trades,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio,
            oc_flow_in, oc_flow_out,
            oc_supply, oc_addr,
            oc_tx, oc_hashrate,
            liq_total_vol, liq_buy_vol, liq_sell_vol,
            liq_count,
            mempool_fastest_fee, mempool_economy_fee,
            mempool_size,
            macro_dxy, macro_spx, macro_vix,
            macro_day,
            social_volume, sentiment_score,
        );
        // Ensure BarState uses our tracked prev_momentum
        self.state.prev_momentum = self.prev_momentum_val;
        // Compute features and cache them (same sequence as batch mode)
        let mut out = [f64::NAN; N_FEATURES];
        self.state.get_features(&mut out);
        // Read momentum BEFORE caching (avoid optimizer issues)
        let new_mom = out[F_MA_CROSS_10_30];
        // Cache the feature output
        self.cached_features.copy_from_slice(&out);
        // Update prev_momentum AFTER get_features (same as batch mode)
        self.state.prev_momentum = new_mom;
        self.prev_momentum_val = new_mom;
    }

    /// Get the current feature vector as a dict {name: value}.
    /// NaN values are converted to None. Returns cached features from last push_bar().
    #[pyo3(signature = ())]
    fn get_features(&self) -> std::collections::HashMap<String, Option<f64>> {
        let mut result = std::collections::HashMap::with_capacity(N_FEATURES);
        for (i, name) in FEATURE_NAMES.iter().enumerate() {
            let val = self.cached_features[i];
            result.insert(name.to_string(), if val.is_nan() { None } else { Some(val) });
        }
        result
    }

    /// Get features as a flat list of f64 (NaN for unavailable).
    /// Returns cached features from last push_bar().
    #[pyo3(signature = ())]
    fn get_features_array(&self) -> Vec<f64> {
        self.cached_features.to_vec()
    }

    /// Get feature names in order.
    #[pyo3(signature = ())]
    fn feature_names(&self) -> Vec<String> {
        FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
    }

    /// Number of bars pushed so far.
    #[getter]
    fn bar_count(&self) -> i32 {
        self.state.bar_count
    }

    /// Whether warmup is complete (enough bars for all features).
    #[getter]
    fn warmed_up(&self) -> bool {
        self.state.bar_count >= 65  // _WARMUP_BARS
    }


    /// Serialize bar history to JSON for checkpoint persistence.
    ///
    /// Stores up to 720 raw bars (same format as TickProcessor.checkpoint_native).
    /// On restore, bars are replayed through push() to rebuild all rolling state.
    #[pyo3(signature = ())]
    fn checkpoint(&self) -> PyResult<String> {
        use serde_json::{json, Value};

        let bars = self.state.get_bar_history();
        let bars_json: Vec<Value> = bars.iter().map(|b| {
            json!({
                "c": b.close, "v": b.volume, "h": b.high,
                "l": b.low, "o": b.open,
                "hour": b.hour, "dow": b.dow,
                "fr": b.funding_rate, "trades": b.trades,
                "tbv": b.taker_buy_volume, "qv": b.quote_volume,
                "tbqv": b.taker_buy_quote_volume,
                "oi": b.open_interest, "ls": b.ls_ratio,
                "spot": b.spot_close, "fg": b.fear_greed,
                "iv": b.implied_vol, "pcr": b.put_call_ratio,
                "oc_fi": b.oc_flow_in, "oc_fo": b.oc_flow_out,
                "oc_s": b.oc_supply, "oc_a": b.oc_addr,
                "oc_t": b.oc_tx, "oc_h": b.oc_hashrate,
                "liq_tv": b.liq_total_vol, "liq_bv": b.liq_buy_vol,
                "liq_sv": b.liq_sell_vol, "liq_c": b.liq_count,
                "mp_ff": b.mempool_fastest_fee, "mp_ef": b.mempool_economy_fee,
                "mp_s": b.mempool_size,
                "m_dxy": b.macro_dxy, "m_spx": b.macro_spx,
                "m_vix": b.macro_vix, "m_day": b.macro_day,
                "sv": b.social_volume, "ss": b.sentiment_score,
            })
        }).collect();

        let checkpoint = json!({
            "version": 1,
            "bar_count": self.state.bar_count,
            "bars": bars_json,
        });

        serde_json::to_string(&checkpoint)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Checkpoint serialize error: {}", e)
            ))
    }

    /// Restore from checkpoint JSON. Replays stored bars to rebuild all rolling state.
    ///
    /// Returns the number of bars replayed.
    #[pyo3(signature = (json_str))]
    fn restore_checkpoint(&mut self, json_str: &str) -> PyResult<usize> {
        let checkpoint: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(
                format!("Checkpoint parse error: {}", e)
            ))?;

        let bars = checkpoint.get("bars")
            .and_then(|v| v.as_array())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No 'bars' in checkpoint"))?;

        let mut total_replayed = 0usize;

        for bar in bars {
            let close = bar.get("c").and_then(|v| v.as_f64()).unwrap_or(0.0);
            if close == 0.0 { continue; }

            self.state.push(
                close,
                bar.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("h").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("l").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("o").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("hour").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                bar.get("dow").and_then(|v| v.as_i64()).unwrap_or(-1) as i32,
                bar.get("fr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("trades").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("qv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("tbqv").and_then(|v| v.as_f64()).unwrap_or(0.0),
                bar.get("oi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ls").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("spot").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("fg").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("iv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("pcr").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fi").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_fo").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_a").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_t").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("oc_h").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_tv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_bv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("liq_c").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ff").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_ef").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("mp_s").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_dxy").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_spx").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_vix").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("m_day").and_then(|v| v.as_i64()).unwrap_or(-1),
                bar.get("sv").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
                bar.get("ss").and_then(|v| v.as_f64()).unwrap_or(f64::NAN),
            );

            // Maintain prev_momentum tracking (same as push_bar)
            let mut out = [f64::NAN; N_FEATURES];
            self.state.prev_momentum = self.prev_momentum_val;
            self.state.get_features(&mut out);
            let new_mom = out[F_MA_CROSS_10_30];
            self.cached_features.copy_from_slice(&out);
            self.state.prev_momentum = new_mom;
            self.prev_momentum_val = new_mom;

            total_replayed += 1;
        }

        Ok(total_replayed)
    }

    fn __repr__(&self) -> String {
        format!("RustFeatureEngine(bar_count={}, warmed_up={})", self.state.bar_count, self.state.bar_count >= 65)
    }

    /// Push cross-market daily data (ETF closes, Treasury yield, USDT dominance).
    ///
    /// Call this once per bar (or when new daily data arrives) AFTER push_bar().
    /// Computes derived features: spy_ret_1d, spy_ret_5d, spy_ret_10d,
    /// tlt_ret_5d, uso_ret_5d, xlf_ret_5d (also as gold_ret_5d legacy),
    /// ethe_ret_1d (also as etf_premium legacy), gbtc_vol_zscore, treasury chg_5d,
    /// usdt_dominance passthrough, spy_extreme (|ret|>2% → sign).
    #[pyo3(signature = (spy_close=f64::NAN, tlt_close=f64::NAN, uso_close=f64::NAN,
                         xlf_close=f64::NAN, ethe_close=f64::NAN, gbtc_vol=f64::NAN,
                         treasury_10y=f64::NAN, usdt_dominance=f64::NAN))]
    fn push_cross_market(
        &mut self,
        spy_close: f64,
        tlt_close: f64,
        uso_close: f64,
        xlf_close: f64,
        ethe_close: f64,
        gbtc_vol: f64,
        treasury_10y: f64,
        usdt_dominance: f64,
    ) {
        let st = &mut self.state;

        // Store raw values and push to buffers
        if !spy_close.is_nan() {
            st.cm_spy_close = spy_close;
            st.cm_spy_buf.push(spy_close);
        }
        if !tlt_close.is_nan() {
            st.cm_tlt_close = tlt_close;
            st.cm_tlt_buf.push(tlt_close);
        }
        if !uso_close.is_nan() {
            st.cm_uso_close = uso_close;
            st.cm_uso_buf.push(uso_close);
        }
        if !xlf_close.is_nan() {
            st.cm_xlf_close = xlf_close;
            st.cm_xlf_buf.push(xlf_close);
        }
        if !ethe_close.is_nan() {
            st.cm_ethe_close = ethe_close;
            st.cm_ethe_buf.push(ethe_close);
        }
        if !gbtc_vol.is_nan() {
            st.cm_gbtc_vol = gbtc_vol;
            st.cm_gbtc_vol_buf.push(gbtc_vol);
        }
        if !treasury_10y.is_nan() {
            st.cm_treasury_10y = treasury_10y;
            st.cm_treasury_buf.push(treasury_10y);
        }
        if !usdt_dominance.is_nan() {
            st.cm_usdt_dominance = usdt_dominance;
        }

        // Compute derived features and update cached_features

        // spy_ret_1d = SPY[0]/SPY[-1] - 1
        if st.cm_spy_buf.size() >= 2 {
            let prev = st.cm_spy_buf.back_n(1);
            if prev > 0.0 {
                let ret_1d = st.cm_spy_buf.back() / prev - 1.0;
                self.cached_features[F_SPY_RET_1D] = ret_1d;
                // spy_extreme: |ret|>2% → sign, else 0
                // (mapped to spy_vix_change slot as an extreme indicator)
                self.cached_features[F_SPY_VIX_CHANGE] = if ret_1d.abs() > 0.02 {
                    ret_1d.signum()
                } else {
                    0.0
                };
            }
        }

        // spy_ret_5d
        if st.cm_spy_buf.size() >= 6 {
            let old = st.cm_spy_buf.back_n(5);
            if old > 0.0 {
                self.cached_features[F_SPY_RET_5D] = st.cm_spy_buf.back() / old - 1.0;
            }
        }

        // spy_ret_10d
        if st.cm_spy_buf.size() >= 11 {
            let old = st.cm_spy_buf.back_n(10);
            if old > 0.0 {
                self.cached_features[F_SPY_RET_10D] = st.cm_spy_buf.back() / old - 1.0;
            }
        }

        // tnx_change_5d (treasury yield 5-day change)
        if st.cm_treasury_buf.size() >= 6 {
            let old = st.cm_treasury_buf.back_n(5);
            self.cached_features[F_TNX_CHANGE_5D] = st.cm_treasury_buf.back() - old;
        }

        // gold_ret_5d (using XLF as proxy for gold/macro sentiment — legacy name)
        if st.cm_xlf_buf.size() >= 6 {
            let old = st.cm_xlf_buf.back_n(5);
            if old > 0.0 {
                let ret = st.cm_xlf_buf.back() / old - 1.0;
                self.cached_features[F_GOLD_RET_5D] = ret;
                self.cached_features[F_XLF_RET_5D] = ret;
            }
        }

        // tlt_ret_5d (independent TLT 5-day return)
        if st.cm_tlt_buf.size() >= 6 {
            let old = st.cm_tlt_buf.back_n(5);
            if old > 0.0 {
                self.cached_features[F_TLT_RET_5D] = st.cm_tlt_buf.back() / old - 1.0;
            }
        }

        // uso_ret_5d (independent USO 5-day return)
        if st.cm_uso_buf.size() >= 6 {
            let old = st.cm_uso_buf.back_n(5);
            if old > 0.0 {
                self.cached_features[F_USO_RET_5D] = st.cm_uso_buf.back() / old - 1.0;
            }
        }

        // ibit_flow_zscore (using GBTC volume z-score as ETF flow proxy)
        if st.cm_gbtc_vol_buf.size() >= 14 {
            let mut tmp = [0.0_f64; 14];
            for i in 0..14 {
                tmp[i] = st.cm_gbtc_vol_buf.get(i);
            }
            self.cached_features[F_IBIT_FLOW_ZSCORE] = zscore_buf(&tmp, 1e-8);
        }

        // etf_premium / ethe_ret_1d: ETHE 1-day return (crypto ETF premium proxy)
        if st.cm_ethe_buf.size() >= 2 {
            let prev = st.cm_ethe_buf.back_n(1);
            if prev > 0.0 {
                let ret = st.cm_ethe_buf.back() / prev - 1.0;
                self.cached_features[F_ETF_PREMIUM] = ret;
                self.cached_features[F_ETHE_RET_1D] = ret;
            }
        }

        // USDT dominance passthrough
        if !usdt_dominance.is_nan() {
            self.cached_features[F_USDT_DOMINANCE] = usdt_dominance;
        }
    }

    /// Push BTC and ETH closes to compute V14 dominance features.
    ///
    /// Returns a dict with 4 keys:
    ///   btc_dom_ratio_dev_20   — ratio deviation from 20-bar MA (None until 20 bars)
    ///   btc_dom_ratio_mom_10   — 10-bar ratio momentum          (None until 11 bars)
    ///   btc_dom_return_diff_6h  — BTC-ETH 6-bar return diff    (None until 6 return bars)
    ///   btc_dom_return_diff_24h — BTC-ETH 24-bar return diff   (None until 24 return bars)
    ///
    /// This method maintains its own independent state from push_bar().
    #[pyo3(signature = (btc_close, eth_close))]
    fn push_dominance(
        &mut self,
        btc_close: f64,
        eth_close: f64,
    ) -> std::collections::HashMap<String, Option<f64>> {
        let mut result = std::collections::HashMap::with_capacity(4);

        if eth_close <= 0.0 || btc_close <= 0.0 {
            result.insert("btc_dom_ratio_dev_20".to_string(), None);
            result.insert("btc_dom_ratio_mom_10".to_string(), None);
            result.insert("btc_dom_return_diff_6h".to_string(), None);
            result.insert("btc_dom_return_diff_24h".to_string(), None);
            return result;
        }

        let ratio = btc_close / eth_close;

        // Maintain circular buffer capped at 75
        if self.dom_ratio_buf.len() >= 75 {
            self.dom_ratio_buf.remove(0);
        }
        self.dom_ratio_buf.push(ratio);

        // Compute and store returns (only once we have a previous close)
        if self.dom_last_btc > 0.0 {
            let btc_ret = btc_close / self.dom_last_btc - 1.0;
            if self.dom_btc_ret_buf.len() >= 25 {
                self.dom_btc_ret_buf.remove(0);
            }
            self.dom_btc_ret_buf.push(btc_ret);
        }
        if self.dom_last_eth > 0.0 {
            let eth_ret = eth_close / self.dom_last_eth - 1.0;
            if self.dom_eth_ret_buf.len() >= 25 {
                self.dom_eth_ret_buf.remove(0);
            }
            self.dom_eth_ret_buf.push(eth_ret);
        }
        self.dom_last_btc = btc_close;
        self.dom_last_eth = eth_close;

        let n_ratio = self.dom_ratio_buf.len();

        // btc_dom_ratio_dev_20: ratio / MA(20) - 1
        if n_ratio >= 20 {
            let start = n_ratio - 20;
            let ma20: f64 = self.dom_ratio_buf[start..].iter().sum::<f64>() / 20.0;
            result.insert(
                "btc_dom_ratio_dev_20".to_string(),
                if ma20 > 0.0 { Some(ratio / ma20 - 1.0) } else { None },
            );
        } else {
            result.insert("btc_dom_ratio_dev_20".to_string(), None);
        }

        // btc_dom_ratio_mom_10: ratio / ratio[10 bars ago] - 1
        if n_ratio >= 11 {
            let prev = self.dom_ratio_buf[n_ratio - 11];
            result.insert(
                "btc_dom_ratio_mom_10".to_string(),
                if prev > 0.0 { Some(ratio / prev - 1.0) } else { None },
            );
        } else {
            result.insert("btc_dom_ratio_mom_10".to_string(), None);
        }

        // btc_dom_return_diff_6h: sum(btc_ret[-6:]) - sum(eth_ret[-6:])
        let n_btc = self.dom_btc_ret_buf.len();
        let n_eth = self.dom_eth_ret_buf.len();
        if n_btc >= 6 && n_eth >= 6 {
            let btc_sum: f64 = self.dom_btc_ret_buf[n_btc - 6..].iter().sum();
            let eth_sum: f64 = self.dom_eth_ret_buf[n_eth - 6..].iter().sum();
            result.insert("btc_dom_return_diff_6h".to_string(), Some(btc_sum - eth_sum));
        } else {
            result.insert("btc_dom_return_diff_6h".to_string(), None);
        }

        // btc_dom_return_diff_24h: sum(btc_ret[-24:]) - sum(eth_ret[-24:])
        if n_btc >= 24 && n_eth >= 24 {
            let btc_sum: f64 = self.dom_btc_ret_buf[n_btc - 24..].iter().sum();
            let eth_sum: f64 = self.dom_eth_ret_buf[n_eth - 24..].iter().sum();
            result.insert("btc_dom_return_diff_24h".to_string(), Some(btc_sum - eth_sum));
            // Also set cached feature for btc_dom_ret_24
            self.cached_features[F_BTC_DOM_RET_24] = btc_sum - eth_sum;
        } else {
            result.insert("btc_dom_return_diff_24h".to_string(), None);
        }

        // Set btc_dom_dev_20 in cached features
        if let Some(&Some(dev)) = result.get("btc_dom_ratio_dev_20") {
            self.cached_features[F_BTC_DOM_DEV_20] = dev;
        }

        result
    }
}
