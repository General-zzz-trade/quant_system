use pyo3::prelude::*;

// ── Internal helpers (not exported to Python) ──

/// Rank data with average tie-breaking
fn rankdata(arr: &[f64], ranks: &mut [f64]) {
    let n = arr.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks 1..N
    for i in 0..n {
        ranks[order[i]] = (i + 1) as f64;
    }

    // Average ties
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && arr[order[j]] == arr[order[i]] {
            j += 1;
        }
        if j > i + 1 {
            let mut avg = 0.0;
            for k in i..j {
                avg += ranks[order[k]];
            }
            avg /= (j - i) as f64;
            for k in i..j {
                ranks[order[k]] = avg;
            }
        }
        i = j;
    }
}

/// Pearson correlation (population)
fn pearson_ic(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    for i in 0..n {
        sx += x[i];
        sy += y[i];
        sxx += x[i] * x[i];
        syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    let nf = n as f64;
    let mx = sx / nf;
    let my = sy / nf;
    let vx = sxx / nf - mx * mx;
    let vy = syy / nf - my * my;
    let cov = sxy / nf - mx * my;
    let den = (vx * vy).sqrt();
    if den < 1e-15 {
        0.0
    } else {
        cov / den
    }
}

/// Spearman IC = Pearson of ranks
fn spearman_ic(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let mut rx = vec![0.0f64; n];
    let mut ry = vec![0.0f64; n];
    rankdata(x, &mut rx);
    rankdata(y, &mut ry);
    pearson_ic(&rx, &ry)
}

struct ICResult {
    ic: f64,
    valid: bool,
}

/// Compute IC for one feature column vs y, handling NaN mask
fn compute_ic_masked(
    x_flat: &[f64],
    n_features: usize,
    col_j: usize,
    y: &[f64],
    start: usize,
    end: usize,
    use_spearman: bool,
) -> ICResult {
    let mut x_buf = Vec::new();
    let mut y_buf = Vec::new();

    for i in start..end {
        let xv = x_flat[i * n_features + col_j];
        let yv = y[i];
        if xv.is_nan() || yv.is_nan() {
            continue;
        }
        x_buf.push(xv);
        y_buf.push(yv);
    }

    let valid_count = x_buf.len();
    if valid_count < 30 {
        return ICResult { ic: 0.0, valid: false };
    }

    // Check variance
    let nf = valid_count as f64;
    let mut sx = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sy = 0.0f64;
    let mut syy = 0.0f64;
    for i in 0..valid_count {
        sx += x_buf[i];
        sxx += x_buf[i] * x_buf[i];
        sy += y_buf[i];
        syy += y_buf[i] * y_buf[i];
    }
    let vx = sxx / nf - (sx / nf) * (sx / nf);
    let vy = syy / nf - (sy / nf) * (sy / nf);
    if vx < 1e-24 || vy < 1e-24 {
        return ICResult { ic: 0.0, valid: false };
    }

    let ic = if use_spearman {
        spearman_ic(&x_buf, &y_buf)
    } else {
        pearson_ic(&x_buf, &y_buf)
    };

    ICResult { ic, valid: true }
}

/// Flatten 2D Vec<Vec<f64>> to row-major Vec<f64> and extract dimensions
fn flatten_2d(x: &[Vec<f64>]) -> (Vec<f64>, usize, usize) {
    if x.is_empty() {
        return (vec![], 0, 0);
    }
    let n_samples = x.len();
    let n_features = x[0].len();
    let mut flat = Vec::with_capacity(n_samples * n_features);
    for row in x {
        flat.extend_from_slice(row);
    }
    (flat, n_samples, n_features)
}

// ── Exported pyfunctions ──

/// Rolling IC select: Pearson IC on last ic_window bars, top_k by |IC|
#[pyfunction]
#[pyo3(signature = (x, y, top_k = 20, ic_window = 500))]
pub fn cpp_rolling_ic_select(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    top_k: i32,
    ic_window: i32,
) -> Vec<i32> {
    let (x_flat, n_samples, n_features) = flatten_2d(&x);
    if n_samples < 50 {
        return vec![];
    }

    let start = if n_samples > ic_window as usize {
        n_samples - ic_window as usize
    } else {
        0
    };
    let end = n_samples;

    let mut scores: Vec<(f64, i32)> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let r = compute_ic_masked(&x_flat, n_features, j, &y, start, end, false);
        let score = if r.valid { r.ic.abs() } else { 0.0 };
        scores.push((score, j as i32));
    }

    let k = (top_k as usize).min(n_features);
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scores[..k].iter().map(|s| s.1).collect()
}

/// Spearman IC select: Spearman IC on last ic_window bars, top_k by |IC|
#[pyfunction]
#[pyo3(signature = (x, y, top_k = 20, ic_window = 500))]
pub fn cpp_spearman_ic_select(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    top_k: i32,
    ic_window: i32,
) -> Vec<i32> {
    let (x_flat, n_samples, n_features) = flatten_2d(&x);
    if n_samples < 50 {
        return vec![];
    }

    let start = if n_samples > ic_window as usize {
        n_samples - ic_window as usize
    } else {
        0
    };
    let end = n_samples;

    let mut scores: Vec<(f64, i32)> = Vec::with_capacity(n_features);
    for j in 0..n_features {
        let r = compute_ic_masked(&x_flat, n_features, j, &y, start, end, true);
        let score = if r.valid { r.ic.abs() } else { 0.0 };
        scores.push((score, j as i32));
    }

    let k = (top_k as usize).min(n_features);
    scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    scores[..k].iter().map(|s| s.1).collect()
}

/// ICIR select: ICIR = mean(|IC|) / std(IC) across n_windows, with consecutive negative filter
#[pyfunction]
#[pyo3(signature = (x, y, top_k = 20, ic_window = 200, n_windows = 5, min_icir = 0.3, max_consec_neg = 3))]
pub fn cpp_icir_select(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    top_k: i32,
    ic_window: i32,
    n_windows: i32,
    min_icir: f64,
    max_consec_neg: i32,
) -> Vec<i32> {
    let (x_flat, n_samples, n_features) = flatten_2d(&x);
    let total_needed = ic_window as usize * n_windows as usize;
    if n_samples < total_needed || n_samples < 50 {
        return vec![];
    }

    let start_offset = n_samples - total_needed;
    let nw = n_windows as usize;

    // Precompute window boundaries
    let windows: Vec<(usize, usize)> = (0..nw)
        .map(|w| {
            let ws = start_offset + w * ic_window as usize;
            (ws, ws + ic_window as usize)
        })
        .collect();

    let mut results: Vec<(f64, i32)> = Vec::new();
    let mut ics = vec![0.0f64; nw];

    for j in 0..n_features {
        // Compute Spearman IC per window
        for w in 0..nw {
            let r = compute_ic_masked(
                &x_flat, n_features, j, &y,
                windows[w].0, windows[w].1, true,
            );
            ics[w] = if r.valid { r.ic } else { 0.0 };
        }

        // Check consecutive negative
        let mut max_neg = 0i32;
        let mut cur_neg = 0i32;
        for w in 0..nw {
            if ics[w] < 0.0 {
                cur_neg += 1;
                if cur_neg > max_neg {
                    max_neg = cur_neg;
                }
            } else {
                cur_neg = 0;
            }
        }
        if max_neg >= max_consec_neg {
            continue;
        }

        // ICIR = mean(|IC|) / std(IC, ddof=1)
        let mut sum_abs = 0.0f64;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for w in 0..nw {
            sum_abs += ics[w].abs();
            sum += ics[w];
            sumsq += ics[w] * ics[w];
        }
        let mean_abs = sum_abs / nw as f64;
        let mut ic_std = 0.0f64;
        if nw > 1 {
            let mean_ic = sum / nw as f64;
            let var = (sumsq - nw as f64 * mean_ic * mean_ic) / (nw as f64 - 1.0);
            ic_std = if var > 0.0 { var.sqrt() } else { 0.0 };
        }

        let icir = if ic_std < 1e-12 {
            if mean_abs > 0.0 { mean_abs * 100.0 } else { 0.0 }
        } else {
            mean_abs / ic_std
        };

        if icir < min_icir {
            continue;
        }

        results.push((icir, j as i32));
    }

    // Sort by descending ICIR
    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let k = (top_k as usize).min(results.len());
    results[..k].iter().map(|s| s.1).collect()
}

/// Stable ICIR select: Jackknife stability + sign consistency + ICIR
#[pyfunction]
#[pyo3(signature = (x, y, top_k = 20, ic_window = 200, n_windows = 5, min_icir = 0.3, min_stable_folds = 4, sign_consistency = 0.8))]
pub fn cpp_stable_icir_select(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    top_k: i32,
    ic_window: i32,
    n_windows: i32,
    min_icir: f64,
    min_stable_folds: i32,
    sign_consistency: f64,
) -> Vec<i32> {
    let (x_flat, n_samples, n_features) = flatten_2d(&x);
    let nw = n_windows as usize;
    let total_needed = ic_window as usize * nw;
    if n_samples < total_needed || n_samples < 50 {
        return vec![];
    }

    let start_offset = n_samples - total_needed;

    let windows: Vec<(usize, usize)> = (0..nw)
        .map(|w| {
            let ws = start_offset + w * ic_window as usize;
            (ws, ws + ic_window as usize)
        })
        .collect();

    let mut candidates: Vec<(f64, i32)> = Vec::new();
    let mut ics = vec![0.0f64; nw];

    for j in 0..n_features {
        for w in 0..nw {
            let r = compute_ic_masked(
                &x_flat, n_features, j, &y,
                windows[w].0, windows[w].1, true,
            );
            ics[w] = if r.valid { r.ic } else { 0.0 };
        }

        // Jackknife: for fold i, compute std of OTHER folds
        let mut folds_above = 0i32;
        for iw in 0..nw {
            let n_other = nw - 1;
            let mut other_sum = 0.0f64;
            let mut other_sumsq = 0.0f64;
            for k in 0..nw {
                if k == iw {
                    continue;
                }
                other_sum += ics[k];
                other_sumsq += ics[k] * ics[k];
            }
            let mut std_other = 1e-12f64;
            if n_other > 1 {
                let mean_other = other_sum / n_other as f64;
                let var_other =
                    (other_sumsq - n_other as f64 * mean_other * mean_other) / (n_other as f64 - 1.0);
                if var_other > 0.0 {
                    std_other = var_other.sqrt();
                }
                if std_other < 1e-12 {
                    std_other = 1e-12;
                }
            }
            let window_icir = ics[iw].abs() / std_other;
            if window_icir > min_icir {
                folds_above += 1;
            }
        }

        if folds_above < min_stable_folds {
            continue;
        }

        // Sign consistency
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        for w in 0..nw {
            if ics[w] > 0.0 {
                n_pos += 1;
            } else if ics[w] < 0.0 {
                n_neg += 1;
            }
        }
        let dominant = n_pos.max(n_neg) as f64 / nw.max(1) as f64;
        if dominant < sign_consistency {
            continue;
        }

        // Overall ICIR
        let mut sum_abs = 0.0f64;
        let mut sum = 0.0f64;
        let mut sumsq = 0.0f64;
        for w in 0..nw {
            sum_abs += ics[w].abs();
            sum += ics[w];
            sumsq += ics[w] * ics[w];
        }
        let mean_abs = sum_abs / nw as f64;
        let mut ic_std = 0.0f64;
        if nw > 1 {
            let mean_ic = sum / nw as f64;
            let var = (sumsq - nw as f64 * mean_ic * mean_ic) / (nw as f64 - 1.0);
            ic_std = if var > 0.0 { var.sqrt() } else { 0.0 };
        }
        let icir = if ic_std < 1e-12 {
            if mean_abs > 0.0 { mean_abs * 100.0 } else { 0.0 }
        } else {
            mean_abs / ic_std
        };

        candidates.push((icir, j as i32));
    }

    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let k = (top_k as usize).min(candidates.len());

    // Return empty if < 5 pass (Python will fallback to greedy)
    if k < 5 {
        return vec![];
    }

    candidates[..k].iter().map(|s| s.1).collect()
}

/// Feature ICIR report: returns flat (n_features x 5) array
/// [mean_ic, std_ic, icir, max_consec_neg, pct_positive] per feature
#[pyfunction]
#[pyo3(signature = (x, y, ic_window = 200, n_windows = 5))]
pub fn cpp_feature_icir_report(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    ic_window: i32,
    n_windows: i32,
) -> Vec<f64> {
    let (x_flat, n_samples, n_features) = flatten_2d(&x);
    let nw = n_windows as usize;
    let total_needed = ic_window as usize * nw;

    let mut out = vec![0.0f64; n_features * 5];
    if n_samples < total_needed {
        return out;
    }

    let start_offset = n_samples - total_needed;

    let windows: Vec<(usize, usize)> = (0..nw)
        .map(|w| {
            let ws = start_offset + w * ic_window as usize;
            (ws, ws + ic_window as usize)
        })
        .collect();

    let mut ics = vec![0.0f64; nw];

    for j in 0..n_features {
        for w in 0..nw {
            let r = compute_ic_masked(
                &x_flat, n_features, j, &y,
                windows[w].0, windows[w].1, true,
            );
            ics[w] = if r.valid { r.ic } else { 0.0 };
        }

        let mut sum = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sumsq = 0.0f64;
        let mut n_pos = 0i32;
        let mut max_neg = 0i32;
        let mut cur_neg = 0i32;

        for w in 0..nw {
            sum += ics[w];
            sum_abs += ics[w].abs();
            sumsq += ics[w] * ics[w];
            if ics[w] > 0.0 {
                n_pos += 1;
            }
            if ics[w] < 0.0 {
                cur_neg += 1;
                if cur_neg > max_neg {
                    max_neg = cur_neg;
                }
            } else {
                cur_neg = 0;
            }
        }

        let mean_ic = sum / nw as f64;
        let mean_abs = sum_abs / nw as f64;
        let mut ic_std = 0.0f64;
        if nw > 1 {
            let var = (sumsq - nw as f64 * mean_ic * mean_ic) / (nw as f64 - 1.0);
            ic_std = if var > 0.0 { var.sqrt() } else { 0.0 };
        }
        let icir = if ic_std < 1e-12 {
            if mean_abs > 0.0 { mean_abs * 100.0 } else { 0.0 }
        } else {
            mean_abs / ic_std
        };

        let base = j * 5;
        out[base] = mean_ic;
        out[base + 1] = ic_std;
        out[base + 2] = icir;
        out[base + 3] = max_neg as f64;
        out[base + 4] = n_pos as f64 / nw as f64;
    }

    out
}
