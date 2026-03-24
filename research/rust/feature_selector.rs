use pyo3::prelude::*;

// ── Internal helpers (rankdata, pearson_ic, spearman_ic, compute_ic_masked, flatten_2d) ──
include!("feature_selector_helpers.inc.rs");

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
