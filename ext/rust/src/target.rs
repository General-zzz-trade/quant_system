use pyo3::prelude::*;

/// Vol-normalized forward return target.
/// Returns NaN where target is undefined.
#[pyfunction]
#[pyo3(signature = (closes, horizon = 5, vol_window = 20))]
pub fn cpp_vol_normalized_target(
    closes: Vec<f64>,
    horizon: i32,
    vol_window: i32,
) -> Vec<f64> {
    let n = closes.len();
    let hz = horizon as usize;
    let vw = vol_window as usize;
    let nan = f64::NAN;
    let mut result = vec![nan; n];

    if n < vw + hz {
        return result;
    }

    // 1. Forward returns: raw_ret[i] = closes[i+horizon]/closes[i] - 1
    let mut raw_ret = vec![nan; n];
    for i in 0..n - hz {
        raw_ret[i] = closes[i + hz] / closes[i] - 1.0;
    }

    // 2. Pct change
    let mut pct = vec![nan; n];
    for i in 1..n {
        pct[i] = closes[i] / closes[i - 1] - 1.0;
    }

    // 3. Rolling std of pct (ddof=1)
    let mut vol = vec![nan; n];
    let half_window = vw / 2;
    for i in vw..n {
        // window: pct[i - vw + 1 .. i] inclusive
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for j in (i - vw + 1)..=i {
            if !pct[j].is_nan() {
                sum += pct[j];
                count += 1;
            }
        }
        if count < half_window {
            continue;
        }

        let mean = sum / count as f64;
        let mut sumsq = 0.0f64;
        for j in (i - vw + 1)..=i {
            if !pct[j].is_nan() {
                let d = pct[j] - mean;
                sumsq += d * d;
            }
        }
        vol[i] = (sumsq / (count as f64 - 1.0)).sqrt();
    }

    // 4. Clip vol at 5th percentile floor
    let mut vol_valid: Vec<f64> = vol.iter().filter(|v| !v.is_nan()).cloned().collect();

    if vol_valid.is_empty() {
        return raw_ret;
    }

    vol_valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx5 = (vol_valid.len() as f64 * 0.05) as usize;
    let floor = vol_valid[idx5.min(vol_valid.len() - 1)];

    for i in 0..n {
        if !vol[i].is_nan() && vol[i] < floor {
            vol[i] = floor;
        }
    }

    // 5. Normalize: target = raw_ret / vol
    for i in 0..n {
        if !raw_ret[i].is_nan() && !vol[i].is_nan() && vol[i] > 0.0 {
            result[i] = raw_ret[i] / vol[i];
        }
    }

    result
}
