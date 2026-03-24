// backtest_pyo3.inc.rs — PyO3 entry points for backtest engine.
// Included by backtest.rs via include!() macro.

#[pyfunction]
#[pyo3(signature = (timestamps, closes, volumes, vol_20, y_pred, bear_probs, vol_values, funding_rates, funding_ts, config_json, trend_values=vec![]))]
pub fn cpp_run_backtest(
    timestamps: Vec<i64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    vol_20: Vec<f64>,
    y_pred: Vec<f64>,
    bear_probs: Vec<f64>,
    vol_values: Vec<f64>,
    funding_rates: Vec<f64>,
    funding_ts: Vec<i64>,
    config_json: String,
    trend_values: Vec<f64>,
) -> PyResult<PyObject> {
    let cfg = parse_config(&config_json);

    let vol_opt = if volumes.is_empty() { None } else { Some(volumes.as_slice()) };
    let v20_opt = if vol_20.is_empty() { None } else { Some(vol_20.as_slice()) };
    let bp_opt = if bear_probs.is_empty() { None } else { Some(bear_probs.as_slice()) };
    let vv_opt = if vol_values.is_empty() { None } else { Some(vol_values.as_slice()) };
    let fr_opt = if funding_rates.is_empty() { None } else { Some(funding_rates.as_slice()) };
    let ft_opt = if funding_ts.is_empty() { None } else { Some(funding_ts.as_slice()) };
    let tv_opt = if trend_values.is_empty() { None } else { Some(trend_values.as_slice()) };

    let (signal, equity, net_pnl, metrics) = run_backtest_impl(
        &timestamps, &closes, vol_opt, v20_opt, &y_pred, bp_opt, vv_opt, fr_opt, ft_opt, tv_opt, &cfg,
    );

    Python::with_gil(|py| {
        let dict = PyDict::new(py);
        dict.set_item("signal", signal.clone())?;
        dict.set_item("equity", equity.clone())?;
        dict.set_item("net_pnl", net_pnl.clone())?;
        dict.set_item("sharpe", metrics.sharpe)?;
        dict.set_item("max_drawdown", metrics.max_drawdown)?;
        dict.set_item("total_return", metrics.total_return)?;
        dict.set_item("annual_return", metrics.annual_return)?;
        dict.set_item("win_rate", metrics.win_rate)?;
        dict.set_item("profit_factor", metrics.profit_factor)?;
        dict.set_item("n_trades", metrics.n_trades)?;
        dict.set_item("avg_holding", metrics.avg_holding)?;
        dict.set_item("total_turnover", metrics.total_turnover)?;
        dict.set_item("total_cost", metrics.total_cost)?;
        dict.set_item("n_active", metrics.n_active)?;

        // Monthly breakdown as list of dicts
        let monthly_list = PyList::empty(py);
        for ms in &metrics.monthly {
            let md = PyDict::new(py);
            md.set_item("month", format!("{:04}-{:02}", ms.year, ms.month))?;
            md.set_item("return", ms.total_return)?;
            md.set_item("sharpe", ms.sharpe)?;
            md.set_item("active_pct", ms.active_pct)?;
            md.set_item("bars", ms.bars)?;
            monthly_list.append(md)?;
        }
        dict.set_item("monthly", monthly_list)?;

        Ok(dict.into())
    })
}

#[pyfunction]
#[pyo3(signature = (y_pred, deadzone, min_hold, zscore_window, zscore_warmup, long_only=false, trend_follow=false, trend_values=vec![], trend_threshold=0.0, max_hold=120))]
pub fn cpp_pred_to_signal(
    y_pred: Vec<f64>,
    deadzone: f64,
    min_hold: i32,
    zscore_window: i32,
    zscore_warmup: i32,
    long_only: bool,
    trend_follow: bool,
    trend_values: Vec<f64>,
    trend_threshold: f64,
    max_hold: i32,
) -> Vec<f64> {
    let tv_opt = if trend_values.is_empty() { None } else { Some(trend_values.as_slice()) };
    pred_to_signal_impl(&y_pred, deadzone, min_hold, zscore_window, zscore_warmup,
                        long_only, trend_follow, tv_opt, trend_threshold, max_hold)
}
