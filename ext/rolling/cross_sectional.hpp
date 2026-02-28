#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

using OptVec = std::vector<std::optional<double>>;

/**
 * Cross-sectional momentum rank (NaN-sentinel fast path).
 *
 * For each timestep, ranks all symbols by lookback-period cumulative return.
 * Rank in [0, 1]: 0 = worst performer, 1 = best performer.
 *
 * returns_matrix: M x T matrix of doubles (NaN = missing value)
 * lookback: lookback period
 * Returns: M x T matrix of doubles (NaN = missing value)
 */
inline std::vector<std::vector<double>> cpp_momentum_rank(
    const std::vector<std::vector<double>>& returns_matrix,
    int lookback
) {
    if (lookback <= 0)
        throw std::invalid_argument("lookback must be positive");

    size_t M = returns_matrix.size();
    if (M == 0) return {};

    size_t T = returns_matrix[0].size();
    for (size_t m = 1; m < M; ++m)
        T = std::min(T, returns_matrix[m].size());

    const double NAN_VAL = std::numeric_limits<double>::quiet_NaN();

    // Output: M x T, initialized to NaN
    std::vector<std::vector<double>> result(M, std::vector<double>(T, NAN_VAL));

    // Reusable per-timestep buffers
    std::vector<double> cum_rets(M);
    std::vector<size_t> indices(M);
    int half_lookback = lookback / 2;

    for (size_t t = 0; t < T; ++t) {
        if (static_cast<int>(t) < lookback)
            continue;

        int valid_count = 0;

        for (size_t m = 0; m < M; ++m) {
            // Cumulative return for symbol m over [t-lookback+1, t]
            double cum = 1.0;
            int n_valid = 0;
            size_t start = t - lookback + 1;
            for (size_t j = start; j <= t; ++j) {
                double r = returns_matrix[m][j];
                if (!std::isnan(r)) {
                    cum *= (1.0 + r);
                    ++n_valid;
                }
            }

            if (n_valid >= half_lookback) {
                cum_rets[m] = cum - 1.0;
                indices[valid_count] = m;
                ++valid_count;
            }
        }

        if (valid_count < 2)
            continue;

        // Sort valid symbols by cumulative return (ascending)
        std::sort(indices.begin(), indices.begin() + valid_count,
                  [&](size_t a, size_t b) { return cum_rets[a] < cum_rets[b]; });

        double denom = static_cast<double>(std::max(valid_count - 1, 1));
        for (int rank = 0; rank < valid_count; ++rank) {
            result[indices[rank]][t] = static_cast<double>(rank) / denom;
        }
    }

    return result;
}

/**
 * Rolling beta: beta = cov(asset, market) / var(market).
 *
 * O(n) sliding window using running sums of a, m, a*m, m*m, and valid count.
 * NaN values in either series are excluded from the window computation.
 */
inline std::vector<double> cpp_rolling_beta(
    const std::vector<double>& asset_returns,
    const std::vector<double>& market_returns,
    int window
) {
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t n = std::min(asset_returns.size(), market_returns.size());
    const double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(n, NAN_VAL);
    int half_window = window / 2;

    // Precompute validity and raw values for O(1) access
    std::vector<bool> valid(n, false);
    std::vector<double> a_vals(n, 0.0);
    std::vector<double> m_vals(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        if (!std::isnan(asset_returns[i]) && !std::isnan(market_returns[i])) {
            valid[i] = true;
            a_vals[i] = asset_returns[i];
            m_vals[i] = market_returns[i];
        }
    }

    // Sliding window sums for O(n) total work
    double sum_a = 0.0, sum_m = 0.0, sum_am = 0.0, sum_mm = 0.0;
    int count = 0;

    for (size_t i = 0; i < n; ++i) {
        // Add new element
        if (valid[i]) {
            sum_a += a_vals[i];
            sum_m += m_vals[i];
            sum_am += a_vals[i] * m_vals[i];
            sum_mm += m_vals[i] * m_vals[i];
            ++count;
        }

        // Evict old element when window is full
        if (i >= static_cast<size_t>(window)) {
            size_t drop = i - window;
            if (valid[drop]) {
                sum_a -= a_vals[drop];
                sum_m -= m_vals[drop];
                sum_am -= a_vals[drop] * m_vals[drop];
                sum_mm -= m_vals[drop] * m_vals[drop];
                --count;
            }
        }

        if (static_cast<int>(i) < window - 1) continue;
        if (count < half_window) continue;

        double mean_a = sum_a / count;
        double mean_m = sum_m / count;
        double cov = sum_am / count - mean_a * mean_m;
        double var_m = sum_mm / count - mean_m * mean_m;

        if (var_m > 0.0) {
            out[i] = cov / var_m;
        }
    }

    return out;
}

/**
 * Relative strength vs benchmark over a rolling window.
 *
 * RS = cumulative_return(target) / cumulative_return(benchmark).
 * Values > 1 mean outperforming, < 1 mean underperforming.
 * Any NaN in the window produces NaN output.
 */
inline std::vector<double> cpp_relative_strength(
    const std::vector<double>& target_returns,
    const std::vector<double>& benchmark_returns,
    int window
) {
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t n = std::min(target_returns.size(), benchmark_returns.size());
    const double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> out(n, NAN_VAL);

    for (size_t i = 0; i < n; ++i) {
        if (static_cast<int>(i) < window - 1) continue;

        double t_cum = 1.0;
        double b_cum = 1.0;
        bool all_valid = true;

        size_t start = i - window + 1;
        for (size_t j = start; j <= i; ++j) {
            if (std::isnan(target_returns[j]) || std::isnan(benchmark_returns[j])) {
                all_valid = false;
                break;
            }
            t_cum *= (1.0 + target_returns[j]);
            b_cum *= (1.0 + benchmark_returns[j]);
        }

        if (all_valid && b_cum != 0.0) {
            out[i] = t_cum / b_cum;
        }
    }

    return out;
}
