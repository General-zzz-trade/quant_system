#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

/**
 * Batch Pearson correlation: |cor(feature_i, target)| for each feature.
 *
 * features: F x T matrix (F features, T observations)
 * target: T vector
 * Returns: F vector of absolute correlations
 */
inline std::vector<double> cpp_correlation_select(
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& target
) {
    size_t F = features.size();
    size_t T = target.size();
    std::vector<double> result(F, 0.0);

    if (T < 2 || F == 0) return result;

    // Precompute target mean and variance once
    double t_sum = 0.0;
    for (size_t t = 0; t < T; ++t)
        t_sum += target[t];
    double t_mean = t_sum / static_cast<double>(T);

    double t_var = 0.0;
    for (size_t t = 0; t < T; ++t) {
        double d = target[t] - t_mean;
        t_var += d * d;
    }

    if (t_var < 1e-12) return result;

    for (size_t f = 0; f < F; ++f) {
        size_t n = std::min(features[f].size(), T);
        if (n < 2) continue;

        double f_sum = 0.0;
        for (size_t t = 0; t < n; ++t)
            f_sum += features[f][t];
        double f_mean = f_sum / static_cast<double>(n);

        double f_var = 0.0;
        double cov = 0.0;
        for (size_t t = 0; t < n; ++t) {
            double fd = features[f][t] - f_mean;
            double td = target[t] - t_mean;
            f_var += fd * fd;
            cov += fd * td;
        }

        if (f_var < 1e-12) continue;
        result[f] = std::abs(cov / std::sqrt(f_var * t_var));
    }

    return result;
}

/**
 * Batch mutual information: MI(feature_i, target) for each feature.
 *
 * features: F x T matrix
 * target: T vector
 * n_bins: number of bins for discretization
 * Returns: F vector of mutual information scores
 */
inline std::vector<double> cpp_mutual_info_select(
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& target,
    int n_bins
) {
    size_t F = features.size();
    size_t T = target.size();
    std::vector<double> result(F, 0.0);

    if (T < 2 || F == 0 || n_bins < 2) return result;

    int nb = n_bins;
    double inv_n = 1.0 / static_cast<double>(T);

    // Bin the target once
    double t_min = target[0], t_max = target[0];
    for (size_t t = 1; t < T; ++t) {
        if (target[t] < t_min) t_min = target[t];
        if (target[t] > t_max) t_max = target[t];
    }
    double t_range = t_max - t_min;
    double t_scale = (t_range < 1e-12) ? 0.0 : (nb - 1) / t_range;

    std::vector<int> t_bins(T);
    std::vector<int> y_counts(nb, 0);
    for (size_t t = 0; t < T; ++t) {
        int b = (t_scale > 0.0) ? std::min(static_cast<int>((target[t] - t_min) * t_scale), nb - 1) : 0;
        t_bins[t] = b;
        y_counts[b]++;
    }

    // Flat 2D joint histogram (reused per feature)
    std::vector<int> joint(static_cast<size_t>(nb) * nb);
    std::vector<int> x_counts(nb);

    for (size_t f = 0; f < F; ++f) {
        size_t n = std::min(features[f].size(), T);
        if (n < 2) continue;

        // Bin this feature
        double f_min = features[f][0], f_max = features[f][0];
        for (size_t t = 1; t < n; ++t) {
            if (features[f][t] < f_min) f_min = features[f][t];
            if (features[f][t] > f_max) f_max = features[f][t];
        }
        double f_range = f_max - f_min;
        double f_scale = (f_range < 1e-12) ? 0.0 : (nb - 1) / f_range;

        // Reset histograms
        std::fill(joint.begin(), joint.end(), 0);
        std::fill(x_counts.begin(), x_counts.end(), 0);

        for (size_t t = 0; t < n; ++t) {
            int xb = (f_scale > 0.0) ? std::min(static_cast<int>((features[f][t] - f_min) * f_scale), nb - 1) : 0;
            x_counts[xb]++;
            joint[static_cast<size_t>(xb) * nb + t_bins[t]]++;
        }

        // Compute MI
        double mi = 0.0;
        for (int xi = 0; xi < nb; ++xi) {
            if (x_counts[xi] == 0) continue;
            double p_x = x_counts[xi] * inv_n;
            for (int yi = 0; yi < nb; ++yi) {
                int jcount = joint[static_cast<size_t>(xi) * nb + yi];
                if (jcount == 0) continue;
                double p_xy = jcount * inv_n;
                double p_y = y_counts[yi] * inv_n;
                mi += p_xy * std::log(p_xy / (p_x * p_y));
            }
        }

        result[f] = std::max(mi, 0.0);
    }

    return result;
}
