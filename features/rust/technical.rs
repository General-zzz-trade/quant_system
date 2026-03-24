use pyo3::prelude::*;

use crate::features::indicators;

// ── Technical indicator wrappers ──

#[pyfunction]
#[pyo3(signature = (values, window))]
pub fn cpp_sma(values: Vec<f64>, window: i32) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::sma(&values, window as usize))
}

#[pyfunction]
#[pyo3(signature = (values, window))]
pub fn cpp_ema(values: Vec<f64>, window: i32) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::ema(&values, window as usize))
}

#[pyfunction]
#[pyo3(signature = (values, log_ret = false))]
pub fn cpp_returns(values: Vec<f64>, log_ret: bool) -> Vec<Option<f64>> {
    indicators::returns(&values, log_ret)
}

#[pyfunction]
#[pyo3(signature = (rets, window))]
pub fn cpp_volatility(rets: Vec<Option<f64>>, window: i32) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::volatility(&rets, window as usize))
}

#[pyfunction]
#[pyo3(signature = (values, window = 14))]
pub fn cpp_rsi(values: Vec<f64>, window: i32) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::rsi(&values, window as usize))
}

#[pyfunction]
#[pyo3(signature = (values, fast = 12, slow = 26, signal = 9))]
pub fn cpp_macd(
    values: Vec<f64>,
    fast: i32,
    slow: i32,
    signal: i32,
) -> PyResult<(Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>)> {
    if fast <= 0 || slow <= 0 || signal <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "MACD windows must be positive",
        ));
    }
    if fast >= slow {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fast window must be smaller than slow window",
        ));
    }
    Ok(indicators::macd(
        &values,
        fast as usize,
        slow as usize,
        signal as usize,
    ))
}

#[pyfunction]
#[pyo3(signature = (values, window = 20, num_std = 2.0))]
pub fn cpp_bollinger_bands(
    values: Vec<f64>,
    window: i32,
    num_std: f64,
) -> PyResult<(Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>)> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    if num_std <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_std must be positive",
        ));
    }
    Ok(indicators::bollinger_bands(
        &values,
        window as usize,
        num_std,
    ))
}

#[pyfunction]
#[pyo3(signature = (highs, lows, closes, window = 14))]
pub fn cpp_atr(
    highs: Vec<f64>,
    lows: Vec<f64>,
    closes: Vec<f64>,
    window: i32,
) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    let n = highs.len();
    if lows.len() != n || closes.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "highs, lows, closes must have same length",
        ));
    }
    Ok(indicators::atr(&highs, &lows, &closes, window as usize))
}

// ── OLS and microstructure batch functions ──

#[pyfunction]
#[pyo3(signature = (x, y))]
pub fn cpp_ols(x: Vec<f64>, y: Vec<f64>) -> PyResult<(f64, f64)> {
    if x.len() != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must have same length",
        ));
    }
    Ok(indicators::ols(&x, &y))
}

#[pyfunction]
#[pyo3(signature = (closes, volumes, window))]
pub fn cpp_vwap(
    closes: Vec<f64>,
    volumes: Vec<f64>,
    window: i32,
) -> PyResult<Vec<Option<f64>>> {
    if closes.len() != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "closes and volumes must have same length",
        ));
    }
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::vwap_batch(&closes, &volumes, window as usize))
}

#[pyfunction]
#[pyo3(signature = (opens, closes, volumes, window))]
pub fn cpp_order_flow_imbalance(
    opens: Vec<f64>,
    closes: Vec<f64>,
    volumes: Vec<f64>,
    window: i32,
) -> PyResult<Vec<Option<f64>>> {
    let n = closes.len();
    if opens.len() != n || volumes.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opens, closes, volumes must have same length",
        ));
    }
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::order_flow_imbalance(
        &opens,
        &closes,
        &volumes,
        window as usize,
    ))
}

#[pyfunction]
#[pyo3(signature = (rets, window))]
pub fn cpp_rolling_volatility(
    rets: Vec<Option<f64>>,
    window: i32,
) -> PyResult<Vec<Option<f64>>> {
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::rolling_volatility(&rets, window as usize))
}

#[pyfunction]
#[pyo3(signature = (closes, volumes, window))]
pub fn cpp_price_impact(
    closes: Vec<f64>,
    volumes: Vec<f64>,
    window: i32,
) -> PyResult<Vec<Option<f64>>> {
    if closes.len() != volumes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "closes and volumes must have same length",
        ));
    }
    if window <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be positive",
        ));
    }
    Ok(indicators::price_impact(&closes, &volumes, window as usize))
}
