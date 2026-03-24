/// Pure Rust indicator implementations — no PyO3 annotations.
/// Shared by technical.rs, multi_timeframe.rs, feature_engine.rs.

/// Simple Moving Average using rolling window.
pub fn sma(vals: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = vals.len();
    let mut out = Vec::with_capacity(n);
    let mut sum = 0.0;
    let mut buf = vec![0.0; window];
    let mut head = 0usize;
    let mut count = 0usize;

    for &x in vals {
        if count < window {
            buf[count] = x;
            count += 1;
        } else {
            let old = buf[head];
            sum -= old;
            buf[head] = x;
            head = (head + 1) % window;
        }
        sum += x;

        if count == window {
            out.push(Some(sum / count as f64));
        } else {
            out.push(None);
        }
    }
    out
}

/// Exponential Moving Average.
pub fn ema(vals: &[f64], window: usize) -> Vec<Option<f64>> {
    let alpha = 2.0 / (window as f64 + 1.0);
    let n = vals.len();
    let mut out = Vec::with_capacity(n);
    let mut prev = 0.0;
    let mut started = false;

    for &x in vals {
        if !started {
            prev = x;
            started = true;
        } else {
            prev = alpha * x + (1.0 - alpha) * prev;
        }
        out.push(Some(prev));
    }
    out
}

/// Simple/log returns.
pub fn returns(vals: &[f64], log_ret: bool) -> Vec<Option<f64>> {
    let n = vals.len();
    let mut out = vec![None; n];
    for i in 1..n {
        let p0 = vals[i - 1];
        let p1 = vals[i];
        if p0 == 0.0 {
            continue;
        }
        let r = p1 / p0;
        out[i] = Some(if log_ret { r.ln() } else { r - 1.0 });
    }
    out
}

/// Rolling volatility (std of returns).
pub fn volatility(rets: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let n = rets.len();
    let mut out = Vec::with_capacity(n);
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut buf = vec![0.0; window];
    let mut head = 0usize;
    let mut count = 0usize;

    for r in rets {
        let val = r.unwrap_or(0.0);
        if count < window {
            buf[count] = val;
            count += 1;
        } else {
            let old = buf[head];
            sum -= old;
            sumsq -= old * old;
            buf[head] = val;
            head = (head + 1) % window;
        }
        sum += val;
        sumsq += val * val;

        if count == window {
            let mu = sum / count as f64;
            let v = (sumsq / count as f64 - mu * mu).max(0.0);
            out.push(Some(v.sqrt()));
        } else {
            out.push(None);
        }
    }
    out
}

/// Relative Strength Index.
pub fn rsi(vals: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = vals.len();
    let mut out = vec![None; n];
    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    for i in 1..n {
        let change = vals[i] - vals[i - 1];
        let gain = change.max(0.0);
        let loss = (-change).max(0.0);

        if i < window {
            avg_gain += gain;
            avg_loss += loss;
            continue;
        }

        if i == window {
            avg_gain /= window as f64;
            avg_loss /= window as f64;
        } else {
            avg_gain = (avg_gain * (window as f64 - 1.0) + gain) / window as f64;
            avg_loss = (avg_loss * (window as f64 - 1.0) + loss) / window as f64;
        }

        if avg_loss == 0.0 {
            out[i] = Some(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            out[i] = Some(100.0 - (100.0 / (1.0 + rs)));
        }
    }
    out
}

/// MACD: returns (macd_line, signal_line, histogram).
pub fn macd(
    vals: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let fast_ema = ema(vals, fast);
    let slow_ema = ema(vals, slow);
    let n = vals.len();

    let mut macd_line: Vec<Option<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        match (fast_ema[i], slow_ema[i]) {
            (Some(f), Some(s)) => macd_line.push(Some(f - s)),
            _ => macd_line.push(None),
        }
    }

    // Signal line: EMA of macd_line values (substitute 0.0 for None during warmup)
    let macd_values: Vec<f64> = macd_line.iter().map(|v| v.unwrap_or(0.0)).collect();
    let signal_raw = ema(&macd_values, signal);

    // Find first valid macd_line index
    let first_valid = macd_line.iter().position(|v| v.is_some()).unwrap_or(n);

    let mut signal_line: Vec<Option<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        if i < first_valid + signal - 1 {
            signal_line.push(None);
        } else {
            signal_line.push(signal_raw[i]);
        }
    }

    let mut histogram: Vec<Option<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        match (macd_line[i], signal_line[i]) {
            (Some(m), Some(s)) => histogram.push(Some(m - s)),
            _ => histogram.push(None),
        }
    }

    (macd_line, signal_line, histogram)
}

/// Bollinger Bands: returns (upper, middle, lower).
pub fn bollinger_bands(
    vals: &[f64],
    window: usize,
    num_std: f64,
) -> (Vec<Option<f64>>, Vec<Option<f64>>, Vec<Option<f64>>) {
    let n = vals.len();
    let mut upper = Vec::with_capacity(n);
    let mut middle = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);

    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut buf = vec![0.0; window];
    let mut head = 0usize;
    let mut count = 0usize;

    for &x in vals {
        if count < window {
            buf[count] = x;
            count += 1;
        } else {
            let old = buf[head];
            sum -= old;
            sumsq -= old * old;
            buf[head] = x;
            head = (head + 1) % window;
        }
        sum += x;
        sumsq += x * x;

        if count < window {
            upper.push(None);
            middle.push(None);
            lower.push(None);
        } else {
            let mid = sum / count as f64;
            let v = (sumsq / count as f64 - mid * mid).max(0.0);
            let sd = v.sqrt();
            upper.push(Some(mid + num_std * sd));
            middle.push(Some(mid));
            lower.push(Some(mid - num_std * sd));
        }
    }

    (upper, middle, lower)
}

/// Average True Range.
pub fn atr(highs: &[f64], lows: &[f64], closes: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = highs.len();
    let mut trs = Vec::with_capacity(n);

    for i in 0..n {
        let tr = if i == 0 {
            highs[i] - lows[i]
        } else {
            (highs[i] - lows[i])
                .max((highs[i] - closes[i - 1]).abs())
                .max((lows[i] - closes[i - 1]).abs())
        };
        trs.push(tr);
    }

    let mut out = vec![None; n];
    let mut atr_prev = 0.0;

    for i in 0..n {
        if i < window {
            atr_prev += trs[i];
            continue;
        }
        if i == window {
            atr_prev /= window as f64;
            out[i] = Some(atr_prev);
            continue;
        }
        atr_prev = (atr_prev * (window as f64 - 1.0) + trs[i]) / window as f64;
        out[i] = Some(atr_prev);
    }
    out
}

/// OLS regression: y = slope * x + intercept. Returns (slope, r_squared).
/// Single-pass Welford's algorithm.
pub fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut m2_x = 0.0;
    let mut m2_y = 0.0;
    let mut co = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        let w = 1.0 / (i as f64 + 1.0);
        mean_x += dx * w;
        mean_y += dy * w;
        let dx2 = x[i] - mean_x;
        let dy2 = y[i] - mean_y;
        m2_x += dx * dx2;
        m2_y += dy * dy2;
        co += dx * dy2;
    }

    let var_x = m2_x / n as f64;
    let var_y = m2_y / n as f64;
    let cov_xy = co / n as f64;

    if var_x < 1e-15 {
        return (0.0, 0.0);
    }

    let slope = cov_xy / var_x;
    let r_squared = if var_y < 1e-15 {
        if var_x < 1e-15 { 1.0 } else { 0.0 }
    } else {
        (cov_xy * cov_xy) / (var_x * var_y)
    };

    (slope, r_squared)
}

/// Batch VWAP over bar series.
pub fn vwap_batch(closes: &[f64], volumes: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = Vec::with_capacity(n);
    let mut sum_pv = 0.0;
    let mut sum_v = 0.0;

    for i in 0..n {
        sum_pv += closes[i] * volumes[i];
        sum_v += volumes[i];

        if i >= window {
            let drop = i - window;
            sum_pv -= closes[drop] * volumes[drop];
            sum_v -= volumes[drop];
        }

        if i < window - 1 {
            out.push(None);
        } else if sum_v > 0.0 {
            out.push(Some(sum_pv / sum_v));
        } else {
            out.push(None);
        }
    }
    out
}

/// Order flow imbalance over bar series.
pub fn order_flow_imbalance(
    opens: &[f64],
    closes: &[f64],
    volumes: &[f64],
    window: usize,
) -> Vec<Option<f64>> {
    let n = closes.len();
    let sv: Vec<f64> = (0..n)
        .map(|i| {
            let dir = if closes[i] >= opens[i] { 1.0 } else { -1.0 };
            dir * volumes[i]
        })
        .collect();

    let mut out = Vec::with_capacity(n);
    let mut sum_sv = 0.0;
    let mut sum_abs = 0.0;

    for i in 0..n {
        sum_sv += sv[i];
        sum_abs += sv[i].abs();

        if i >= window {
            let drop = i - window;
            sum_sv -= sv[drop];
            sum_abs -= sv[drop].abs();
        }

        if i < window - 1 {
            out.push(None);
        } else if sum_abs > 0.0 {
            out.push(Some(sum_sv / sum_abs));
        } else {
            out.push(Some(0.0));
        }
    }
    out
}

/// Rolling realized volatility (annualized).
pub fn rolling_volatility(rets: &[Option<f64>], window: usize) -> Vec<Option<f64>> {
    let n = rets.len();
    let mut out = Vec::with_capacity(n);
    let annualize = (252.0_f64).sqrt();

    let mut sum = 0.0;
    let mut sumsq = 0.0;
    let mut buf = vec![0.0; window];
    let mut head = 0usize;
    let mut count = 0usize;

    for i in 0..n {
        let val = rets[i].unwrap_or(0.0);

        if count < window {
            buf[count] = val;
            count += 1;
        } else {
            let old = buf[head];
            sum -= old;
            sumsq -= old * old;
            buf[head] = val;
            head = (head + 1) % window;
        }
        sum += val;
        sumsq += val * val;

        if count < window || rets[i].is_none() {
            out.push(None);
        } else {
            let mean = sum / count as f64;
            let var = (sumsq / count as f64 - mean * mean).max(0.0);
            let sample_var = var * count as f64 / (count as f64 - 1.0).max(1.0);
            out.push(Some(sample_var.sqrt() * annualize));
        }
    }
    out
}

/// Price impact proxy: mean(|delta_price|) / sum(volume).
pub fn price_impact(closes: &[f64], volumes: &[f64], window: usize) -> Vec<Option<f64>> {
    let n = closes.len();
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        if i < window {
            out.push(None);
            continue;
        }

        let mut sum_dc = 0.0;
        let mut sum_vol = 0.0;
        for j in (i - window + 1)..=i {
            if j > 0 {
                sum_dc += (closes[j] - closes[j - 1]).abs();
                sum_vol += volumes[j];
            }
        }

        if sum_vol > 0.0 {
            out.push(Some(sum_dc / sum_vol));
        } else {
            out.push(None);
        }
    }
    out
}
