#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

/**
 * Simple OLS regression: y = slope * x + intercept.
 *
 * Single-pass Welford-style computation for numerical stability.
 * Returns (slope, r_squared).
 */
inline std::tuple<double, double> cpp_ols(
    const std::vector<double>& x,
    const std::vector<double>& y
) {
    if (x.size() != y.size())
        throw std::invalid_argument("x and y must have same length");

    int n = static_cast<int>(x.size());
    if (n == 0)
        return {0.0, 0.0};

    // Single-pass: Welford's online algorithm for mean, var, cov
    double mean_x = 0.0;
    double mean_y = 0.0;
    double m2_x = 0.0;   // sum of squared deviations from mean_x
    double m2_y = 0.0;   // sum of squared deviations from mean_y
    double co = 0.0;      // sum of cross-deviations

    for (int i = 0; i < n; ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        double w = 1.0 / (i + 1);
        mean_x += dx * w;
        mean_y += dy * w;
        double dx2 = x[i] - mean_x;  // updated mean
        double dy2 = y[i] - mean_y;
        m2_x += dx * dx2;
        m2_y += dy * dy2;
        co += dx * dy2;
    }

    double var_x = m2_x / n;
    double var_y = m2_y / n;
    double cov_xy = co / n;

    if (var_x < 1e-15)
        return {0.0, 0.0};

    double slope = cov_xy / var_x;

    double r_squared;
    if (var_y < 1e-15) {
        r_squared = (var_x < 1e-15) ? 1.0 : 0.0;
    } else {
        r_squared = (cov_xy * cov_xy) / (var_x * var_y);
    }

    return {slope, r_squared};
}

/**
 * Batch VWAP: rolling VWAP over a bar series.
 *
 * Takes parallel arrays of closes and volumes, returns VWAP series.
 * Uses O(1) rolling window internally.
 */
inline std::vector<std::optional<double>> cpp_vwap(
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    int window
) {
    if (closes.size() != volumes.size())
        throw std::invalid_argument("closes and volumes must have same length");
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t n = closes.size();
    std::vector<std::optional<double>> out;
    out.reserve(n);

    // Use simple sliding window sums for O(n) total work
    double sum_pv = 0.0;
    double sum_v = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_pv += closes[i] * volumes[i];
        sum_v += volumes[i];

        if (i >= static_cast<size_t>(window)) {
            size_t drop = i - window;
            sum_pv -= closes[drop] * volumes[drop];
            sum_v -= volumes[drop];
        }

        if (static_cast<int>(i) < window - 1) {
            out.push_back(std::nullopt);
        } else if (sum_v > 0.0) {
            out.push_back(sum_pv / sum_v);
        } else {
            out.push_back(std::nullopt);
        }
    }
    return out;
}

/**
 * Batch order flow imbalance over a bar series.
 *
 * OFI = sum(signed_volume) / sum(|signed_volume|) over rolling window.
 * Direction: close >= open → buy (+volume), else sell (-volume).
 */
inline std::vector<std::optional<double>> cpp_order_flow_imbalance(
    const std::vector<double>& opens,
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    int window
) {
    size_t n = closes.size();
    if (opens.size() != n || volumes.size() != n)
        throw std::invalid_argument("opens, closes, volumes must have same length");
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    // Pre-compute signed volumes
    std::vector<double> sv(n);
    for (size_t i = 0; i < n; ++i) {
        double dir = (closes[i] >= opens[i]) ? 1.0 : -1.0;
        sv[i] = dir * volumes[i];
    }

    std::vector<std::optional<double>> out;
    out.reserve(n);

    double sum_sv = 0.0;
    double sum_abs = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_sv += sv[i];
        sum_abs += std::abs(sv[i]);

        if (i >= static_cast<size_t>(window)) {
            size_t drop = i - window;
            sum_sv -= sv[drop];
            sum_abs -= std::abs(sv[drop]);
        }

        if (static_cast<int>(i) < window - 1) {
            out.push_back(std::nullopt);
        } else if (sum_abs > 0.0) {
            out.push_back(sum_sv / sum_abs);
        } else {
            out.push_back(0.0);
        }
    }
    return out;
}

/**
 * Rolling realized volatility (annualized) over a returns series.
 *
 * For each position, computes std(returns[i-w+1:i+1]) * sqrt(252).
 * None values in input are treated as 0.0.
 */
inline std::vector<std::optional<double>> cpp_rolling_volatility(
    const std::vector<std::optional<double>>& rets,
    int window
) {
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t n = rets.size();
    std::vector<std::optional<double>> out;
    out.reserve(n);

    double sum = 0.0;
    double sumsq = 0.0;
    double annualize = std::sqrt(252.0);

    // Circular buffer for eviction
    std::vector<double> buf(window, 0.0);
    int head = 0;
    int count = 0;

    for (size_t i = 0; i < n; ++i) {
        double val = rets[i].has_value() ? *rets[i] : 0.0;

        if (count < window) {
            buf[count] = val;
            ++count;
        } else {
            double old = buf[head];
            sum -= old;
            sumsq -= old * old;
            buf[head] = val;
            head = (head + 1) % window;
        }
        sum += val;
        sumsq += val * val;

        if (count < window || rets[i] == std::nullopt) {
            out.push_back(std::nullopt);
        } else {
            double mean = sum / count;
            double var = sumsq / count - mean * mean;
            if (var < 0.0) var = 0.0;
            // Use sample variance (n-1) like the Python version
            double sample_var = var * count / std::max(count - 1, 1);
            out.push_back(std::sqrt(sample_var) * annualize);
        }
    }
    return out;
}

/**
 * Price impact proxy (Kyle's lambda): mean(|delta_price|) / sum(volume).
 */
inline std::vector<std::optional<double>> cpp_price_impact(
    const std::vector<double>& closes,
    const std::vector<double>& volumes,
    int window
) {
    if (closes.size() != volumes.size())
        throw std::invalid_argument("closes and volumes must have same length");
    if (window <= 0)
        throw std::invalid_argument("window must be positive");

    size_t n = closes.size();
    std::vector<std::optional<double>> out;
    out.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (static_cast<int>(i) < window) {
            out.push_back(std::nullopt);
            continue;
        }

        double sum_dc = 0.0;
        double sum_vol = 0.0;
        for (size_t j = i - window + 1; j <= i; ++j) {
            if (j > 0) {
                sum_dc += std::abs(closes[j] - closes[j - 1]);
                sum_vol += volumes[j];
            }
        }

        if (sum_vol > 0.0) {
            out.push_back(sum_dc / sum_vol);
        } else {
            out.push_back(std::nullopt);
        }
    }
    return out;
}
