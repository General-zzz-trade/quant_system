// feature_selector_helpers.inc.rs — Internal helpers for IC computation.
// Included by feature_selector.rs via include!().

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
