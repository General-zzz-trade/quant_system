use pyo3::prelude::*;

/// Batch Pearson correlation: |cor(feature_i, target)| for each feature.
/// features: F x T matrix (F features, T observations)
/// Returns: F vector of absolute correlations
#[pyfunction]
pub fn cpp_correlation_select(
    features: Vec<Vec<f64>>,
    target: Vec<f64>,
) -> Vec<f64> {
    let f_count = features.len();
    let t_len = target.len();
    let mut result = vec![0.0; f_count];

    if t_len < 2 || f_count == 0 {
        return result;
    }

    // Precompute target mean and variance once
    let t_sum: f64 = target.iter().sum();
    let t_mean = t_sum / t_len as f64;

    let t_var: f64 = target.iter().map(|&v| (v - t_mean) * (v - t_mean)).sum();
    if t_var < 1e-12 {
        return result;
    }

    for f in 0..f_count {
        let n = features[f].len().min(t_len);
        if n < 2 {
            continue;
        }

        let f_sum: f64 = features[f][..n].iter().sum();
        let f_mean = f_sum / n as f64;

        let mut f_var = 0.0;
        let mut cov = 0.0;
        for t in 0..n {
            let fd = features[f][t] - f_mean;
            let td = target[t] - t_mean;
            f_var += fd * fd;
            cov += fd * td;
        }

        if f_var < 1e-12 {
            continue;
        }
        result[f] = (cov / (f_var * t_var).sqrt()).abs();
    }

    result
}

/// Batch mutual information: MI(feature_i, target) for each feature.
/// features: F x T matrix, target: T vector, n_bins: discretization bins
/// Returns: F vector of mutual information scores
#[pyfunction]
#[pyo3(signature = (features, target, n_bins))]
pub fn cpp_mutual_info_select(
    features: Vec<Vec<f64>>,
    target: Vec<f64>,
    n_bins: i32,
) -> Vec<f64> {
    let f_count = features.len();
    let t_len = target.len();
    let mut result = vec![0.0; f_count];

    if t_len < 2 || f_count == 0 || n_bins < 2 {
        return result;
    }

    let nb = n_bins as usize;
    let inv_n = 1.0 / t_len as f64;

    // Bin the target once
    let t_min = target.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = target.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_range = t_max - t_min;
    let t_scale = if t_range < 1e-12 {
        0.0
    } else {
        (nb as f64 - 1.0) / t_range
    };

    let mut t_bins = vec![0usize; t_len];
    let mut y_counts = vec![0i32; nb];
    for t in 0..t_len {
        let b = if t_scale > 0.0 {
            ((target[t] - t_min) * t_scale) as usize
        } else {
            0
        };
        let b = b.min(nb - 1);
        t_bins[t] = b;
        y_counts[b] += 1;
    }

    // Flat 2D joint histogram (reused per feature)
    let mut joint = vec![0i32; nb * nb];
    let mut x_counts = vec![0i32; nb];

    for f in 0..f_count {
        let n = features[f].len().min(t_len);
        if n < 2 {
            continue;
        }

        // Bin this feature
        let f_min = features[f][..n]
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let f_max = features[f][..n]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let f_range = f_max - f_min;
        let f_scale = if f_range < 1e-12 {
            0.0
        } else {
            (nb as f64 - 1.0) / f_range
        };

        // Reset histograms
        joint.iter_mut().for_each(|v| *v = 0);
        x_counts.iter_mut().for_each(|v| *v = 0);

        for t in 0..n {
            let xb = if f_scale > 0.0 {
                ((features[f][t] - f_min) * f_scale) as usize
            } else {
                0
            };
            let xb = xb.min(nb - 1);
            x_counts[xb] += 1;
            joint[xb * nb + t_bins[t]] += 1;
        }

        // Compute MI
        let mut mi = 0.0;
        for xi in 0..nb {
            if x_counts[xi] == 0 {
                continue;
            }
            let p_x = x_counts[xi] as f64 * inv_n;
            for yi in 0..nb {
                let jcount = joint[xi * nb + yi];
                if jcount == 0 {
                    continue;
                }
                let p_xy = jcount as f64 * inv_n;
                let p_y = y_counts[yi] as f64 * inv_n;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        result[f] = mi.max(0.0);
    }

    result
}
