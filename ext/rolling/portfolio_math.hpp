#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * Sample covariance matrix.
 *
 * Input: returns_matrix[m][t] = return of symbol m at time t (M symbols × T observations)
 * Output: M × M covariance matrix (nested vector)
 * Exploits symmetry: only computes upper triangle, mirrors.
 */
inline std::vector<std::vector<double>> cpp_sample_covariance(
    const std::vector<std::vector<double>>& returns_matrix
) {
    size_t M = returns_matrix.size();
    if (M == 0) return {};

    size_t n_obs = returns_matrix[0].size();
    for (size_t m = 1; m < M; ++m)
        n_obs = std::min(n_obs, returns_matrix[m].size());

    std::vector<std::vector<double>> result(M, std::vector<double>(M, 0.0));
    if (n_obs < 2) return result;

    // Compute means
    std::vector<double> means(M, 0.0);
    for (size_t m = 0; m < M; ++m) {
        double s = 0.0;
        for (size_t t = 0; t < n_obs; ++t)
            s += returns_matrix[m][t];
        means[m] = s / static_cast<double>(n_obs);
    }

    // Precompute demeaned returns for cache-friendly access
    // Layout: demeaned[m][t]
    std::vector<std::vector<double>> dm(M, std::vector<double>(n_obs));
    for (size_t m = 0; m < M; ++m)
        for (size_t t = 0; t < n_obs; ++t)
            dm[m][t] = returns_matrix[m][t] - means[m];

    double inv_n1 = 1.0 / static_cast<double>(n_obs - 1);

    // Upper triangle covariance
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = i; j < M; ++j) {
            double cov = 0.0;
            for (size_t t = 0; t < n_obs; ++t)
                cov += dm[i][t] * dm[j][t];
            cov *= inv_n1;
            result[i][j] = cov;
            result[j][i] = cov;
        }
    }

    return result;
}

/**
 * EWMA covariance matrix.
 *
 * Recursive: cov[i,j] = alpha * r_i * r_j + (1-alpha) * cov[i,j]
 * Initialized with first observation outer product.
 * Uses flat array + symmetry for cache locality.
 */
inline std::vector<std::vector<double>> cpp_ewma_covariance(
    const std::vector<std::vector<double>>& returns_matrix,
    double alpha
) {
    size_t M = returns_matrix.size();
    if (M == 0) return {};

    size_t n_obs = returns_matrix[0].size();
    for (size_t m = 1; m < M; ++m)
        n_obs = std::min(n_obs, returns_matrix[m].size());

    std::vector<std::vector<double>> result(M, std::vector<double>(M, 0.0));
    if (n_obs < 2) return result;

    // Flat M×M state for cache locality
    std::vector<double> cov(M * M, 0.0);

    // Initialize with first observation outer product (symmetric)
    for (size_t i = 0; i < M; ++i) {
        double ri = returns_matrix[i][0];
        for (size_t j = i; j < M; ++j) {
            double val = ri * returns_matrix[j][0];
            cov[i * M + j] = val;
            cov[j * M + i] = val;
        }
    }

    // EWMA recursion
    double one_minus_alpha = 1.0 - alpha;
    for (size_t t = 1; t < n_obs; ++t) {
        for (size_t i = 0; i < M; ++i) {
            double ari = alpha * returns_matrix[i][t];
            for (size_t j = i; j < M; ++j) {
                double rj = returns_matrix[j][t];
                double val = ari * rj + one_minus_alpha * cov[i * M + j];
                cov[i * M + j] = val;
                cov[j * M + i] = val;
            }
        }
    }

    // Copy to nested vector
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < M; ++j)
            result[i][j] = cov[i * M + j];

    return result;
}

/**
 * Rolling Pearson correlation matrix.
 *
 * Uses the last `window` observations from each symbol.
 * Exploits symmetry and precomputes demeaned data + std devs.
 */
inline std::vector<std::vector<double>> cpp_rolling_correlation(
    const std::vector<std::vector<double>>& returns_matrix,
    int window
) {
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t M = returns_matrix.size();
    if (M == 0) return {};

    // Extract last `window` observations from each symbol
    size_t w = static_cast<size_t>(window);
    std::vector<std::vector<double>> data(M);
    for (size_t m = 0; m < M; ++m) {
        size_t n = returns_matrix[m].size();
        size_t actual_w = std::min(n, w);
        data[m].assign(returns_matrix[m].end() - actual_w, returns_matrix[m].end());
    }

    // Use minimum length across all symbols
    size_t n = data[0].size();
    for (size_t m = 1; m < M; ++m)
        n = std::min(n, data[m].size());

    std::vector<std::vector<double>> result(M, std::vector<double>(M, 0.0));
    if (n < 2) {
        for (size_t i = 0; i < M; ++i) result[i][i] = 1.0;
        return result;
    }

    double inv_n1 = 1.0 / static_cast<double>(n - 1);

    // Precompute means, demeaned data, and std devs
    std::vector<double> means(M, 0.0);
    std::vector<double> stds(M, 0.0);
    std::vector<std::vector<double>> dm(M, std::vector<double>(n));

    for (size_t m = 0; m < M; ++m) {
        double s = 0.0;
        for (size_t t = 0; t < n; ++t)
            s += data[m][t];
        means[m] = s / static_cast<double>(n);

        double var_sum = 0.0;
        for (size_t t = 0; t < n; ++t) {
            double d = data[m][t] - means[m];
            dm[m][t] = d;
            var_sum += d * d;
        }
        stds[m] = std::sqrt(var_sum * inv_n1);
    }

    // Correlation: upper triangle + diagonal
    for (size_t i = 0; i < M; ++i) {
        result[i][i] = 1.0;
        for (size_t j = i + 1; j < M; ++j) {
            if (stds[i] < 1e-12 || stds[j] < 1e-12) {
                // result[i][j] = 0.0 already
                continue;
            }
            double cov = 0.0;
            for (size_t t = 0; t < n; ++t)
                cov += dm[i][t] * dm[j][t];
            cov *= inv_n1;
            double corr = cov / (stds[i] * stds[j]);
            corr = std::max(-1.0, std::min(1.0, corr));
            result[i][j] = corr;
            result[j][i] = corr;
        }
    }

    return result;
}

/**
 * Portfolio variance: w' * Cov * w.
 *
 * Exploits symmetry: diagonal + 2 * upper triangle.
 */
inline double cpp_portfolio_variance(
    const std::vector<double>& weights,
    const std::vector<std::vector<double>>& cov
) {
    size_t n = weights.size();
    if (n == 0 || cov.size() < n) return 0.0;

    double variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        variance += weights[i] * weights[i] * cov[i][i];
        for (size_t j = i + 1; j < n; ++j) {
            variance += 2.0 * weights[i] * weights[j] * cov[i][j];
        }
    }
    return variance;
}
