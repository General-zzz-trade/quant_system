// batch.inc.rs — Included by engine.rs via include!() macro.
// ============================================================
// PyO3 exported functions
// ============================================================

#[pyfunction]
#[pyo3(signature = (
    timestamps, opens, highs, lows, closes, volumes,
    trades, taker_buy_volumes, quote_volumes, taker_buy_quote_volumes,
    funding_sched, oi_sched, ls_sched, spot_sched, fgi_sched,
    iv_sched, pcr_sched, onchain_sched,
    liq_sched, mempool_sched, macro_sched
))]
pub fn cpp_compute_all_features(
    timestamps: Vec<f64>,
    opens: Vec<f64>,
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    trades: Vec<f64>,
    taker_buy_volumes: Vec<f64>,
    quote_volumes: Vec<f64>,
    taker_buy_quote_volumes: Vec<f64>,
    funding_sched: Vec<Vec<f64>>,
    oi_sched: Vec<Vec<f64>>,
    ls_sched: Vec<Vec<f64>>,
    spot_sched: Vec<Vec<f64>>,
    fgi_sched: Vec<Vec<f64>>,
    iv_sched: Vec<Vec<f64>>,
    pcr_sched: Vec<Vec<f64>>,
    onchain_sched: Vec<Vec<f64>>,
    liq_sched: Vec<Vec<f64>>,
    mempool_sched: Vec<Vec<f64>>,
    macro_sched: Vec<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let n_bars = timestamps.len();

    let mut funding_cur = StridedCursor::new(&funding_sched);
    let mut oi_cur = StridedCursor::new(&oi_sched);
    let mut ls_cur = StridedCursor::new(&ls_sched);
    let mut spot_cur = StridedCursor::new(&spot_sched);
    let mut fgi_cur = StridedCursor::new(&fgi_sched);
    let mut iv_cur = StridedCursor::new(&iv_sched);
    let mut pcr_cur = StridedCursor::new(&pcr_sched);
    let mut oc_cur = OnchainCursor::new(&onchain_sched);
    let mut liq_cur = LiqCursor::new(&liq_sched);
    let mut mp_cur = MempoolCursor::new(&mempool_sched);
    let mut macro_cur = MacroCursor::new(&macro_sched);

    let mut state = BarState::new();
    let mut result = Vec::with_capacity(n_bars);

    for i in 0..n_bars {
        let ts = timestamps[i];
        let mut close = closes[i];
        let volume = volumes[i];
        let mut high = highs[i];
        let mut low = lows[i];
        let mut open_ = opens[i];
        let trades_val = trades[i];
        let taker_buy_volume = taker_buy_volumes[i];
        let quote_volume = quote_volumes[i];
        let taker_buy_quote_volume = taker_buy_quote_volumes[i];

        // Default open/high/low to close if zero
        if open_ == 0.0 { open_ = close; }
        if high == 0.0 { high = close; }
        if low == 0.0 { low = close; }
        let _ = close; // suppress unused warning

        // Parse hour and dow from timestamp (ms)
        let mut hour: i32 = -1;
        let mut dow: i32 = -1;
        if ts > 0.0 {
            let ts_sec = (ts / 1000.0) as i64;
            let mut days = ts_sec / 86400;
            let mut day_sec = ts_sec % 86400;
            if day_sec < 0 {
                days -= 1;
                day_sec += 86400;
            }
            hour = (day_sec / 3600) as i32;
            dow = ((days + 3) % 7) as i32;
            if dow < 0 {
                dow += 7;
            }
        }

        // Advance schedule cursors
        let funding_rate = funding_cur.advance(ts);
        let open_interest = oi_cur.advance(ts);
        let ls_ratio = ls_cur.advance(ts);
        let spot_close = spot_cur.advance(ts);
        let fear_greed = fgi_cur.advance(ts);
        let implied_vol = iv_cur.advance(ts);
        let put_call_ratio_val = pcr_cur.advance(ts);

        oc_cur.advance(ts);
        liq_cur.advance(ts);
        mp_cur.advance(ts);
        macro_cur.advance(ts);

        close = closes[i];

        state.push(
            close, volume, high, low, open_,
            hour, dow,
            funding_rate, trades_val,
            taker_buy_volume, quote_volume, taker_buy_quote_volume,
            open_interest, ls_ratio, spot_close, fear_greed,
            implied_vol, put_call_ratio_val,
            oc_cur.flow_in, oc_cur.flow_out,
            oc_cur.supply, oc_cur.addr,
            oc_cur.tx, oc_cur.hashrate,
            // V11: Liquidation
            liq_cur.total_vol, liq_cur.buy_vol, liq_cur.sell_vol,
            if !liq_cur.total_vol.is_nan() { 1.0 } else { f64::NAN },
            // V11: Mempool
            mp_cur.fastest_fee, mp_cur.economy_fee, mp_cur.mempool_size,
            // V11: Macro
            macro_cur.dxy, macro_cur.spx, macro_cur.vix, macro_cur.day,
            // V11: Sentiment (no historical data in batch mode)
            f64::NAN, f64::NAN,
        );

        let mut row_out = [f64::NAN; N_FEATURES];
        state.get_features(&mut row_out);

        // Update prev_momentum (Python does this after get_features)
        state.prev_momentum = row_out[F_MA_CROSS_10_30];

        result.push(row_out.to_vec());
    }

    result
}

#[pyfunction]
pub fn cpp_feature_names() -> Vec<String> {
    FEATURE_NAMES.iter().map(|s| s.to_string()).collect()
}
