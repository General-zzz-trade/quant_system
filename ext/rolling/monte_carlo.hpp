#pragma once

#include "fast_rng.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

struct MCResult {
    int paths;
    double mean_final;
    double median_final;
    double percentile_5;
    double percentile_95;
    double prob_loss;
    double prob_target;
    double max_drawdown_mean;
    double max_drawdown_95;
};

inline double _mc_max_drawdown(const double* equity, int len) {
    double peak = equity[0];
    double max_dd = 0.0;
    for (int i = 1; i < len; ++i) {
        if (equity[i] > peak) peak = equity[i];
        double dd = (peak - equity[i]) / peak;
        if (dd > max_dd) max_dd = dd;
    }
    return max_dd;
}

inline double _mc_percentile(const std::vector<double>& sorted_vals, double p) {
    int n = static_cast<int>(sorted_vals.size());
    if (n == 0) return 0.0;
    if (n == 1) return sorted_vals[0];
    double k = (n - 1) * p / 100.0;
    int lo = static_cast<int>(k);
    int hi = std::min(lo + 1, n - 1);
    double frac = k - lo;
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]);
}

inline MCResult cpp_simulate_paths(
    const std::vector<double>& returns,
    int n_paths = 1000,
    int horizon = 252,
    bool parametric = false,
    double target_return = 0.0,
    int block_size = 5,
    uint64_t seed = 42
) {
    const int n = static_cast<int>(returns.size());
    if (n == 0 || n_paths < 1) {
        return {0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    }

    FastRNG rng(seed);
    const double target_wealth = 1.0 + target_return;

    // Pre-compute parametric params
    double mu = 0.0, sigma = 0.0;
    if (parametric) {
        for (int i = 0; i < n; ++i) mu += returns[i];
        mu /= n;
        double var = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = returns[i] - mu;
            var += d * d;
        }
        sigma = std::sqrt(var / std::max(n - 1, 1));
    }

    std::vector<double> finals(n_paths);
    std::vector<double> drawdowns(n_paths);
    std::vector<double> equity(horizon + 1);

    for (int p = 0; p < n_paths; ++p) {
        equity[0] = 1.0;

        if (parametric) {
            for (int t = 0; t < horizon; ++t) {
                double r = rng.gauss(mu, sigma);
                equity[t + 1] = equity[t] * (1.0 + r);
            }
        } else {
            // Block bootstrap
            int pos = 0;
            while (pos < horizon) {
                int start = rng.randint(n);
                for (int j = 0; j < block_size && pos < horizon; ++j, ++pos) {
                    double r = returns[(start + j) % n];
                    equity[pos + 1] = equity[pos] * (1.0 + r);
                }
            }
        }

        finals[p] = equity[horizon];
        drawdowns[p] = _mc_max_drawdown(equity.data(), horizon + 1);
    }

    std::vector<double> finals_sorted(finals);
    std::vector<double> dd_sorted(drawdowns);
    std::sort(finals_sorted.begin(), finals_sorted.end());
    std::sort(dd_sorted.begin(), dd_sorted.end());

    double sum_f = 0.0, sum_dd = 0.0;
    int n_loss = 0, n_target = 0;
    for (int i = 0; i < n_paths; ++i) {
        sum_f += finals[i];
        sum_dd += drawdowns[i];
        if (finals[i] < 1.0) ++n_loss;
        if (finals[i] >= target_wealth) ++n_target;
    }

    return {
        n_paths,
        sum_f / n_paths,
        _mc_percentile(finals_sorted, 50),
        _mc_percentile(finals_sorted, 5),
        _mc_percentile(finals_sorted, 95),
        static_cast<double>(n_loss) / n_paths,
        static_cast<double>(n_target) / n_paths,
        sum_dd / n_paths,
        _mc_percentile(dd_sorted, 95),
    };
}
