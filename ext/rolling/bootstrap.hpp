#pragma once

#include "fast_rng.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

struct BootstrapResult {
    double sharpe_mean;
    double sharpe_95ci_lo;
    double sharpe_95ci_hi;
    double p_sharpe_gt_0;
    double p_sharpe_gt_05;
};

inline BootstrapResult cpp_bootstrap_sharpe_ci(
    const std::vector<double>& returns,
    int n_bootstrap = 10000,
    int block_size = 5,
    uint64_t seed = 42
) {
    const int n = static_cast<int>(returns.size());
    if (n < 10) {
        return {0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double annualize = std::sqrt(8760.0);
    FastRNG rng(seed);

    std::vector<double> sharpes(n_bootstrap);
    std::vector<double> sample(n);

    for (int b = 0; b < n_bootstrap; ++b) {
        int pos = 0;
        while (pos < n) {
            int start = rng.randint(n);
            for (int j = 0; j < block_size && pos < n; ++j, ++pos) {
                sample[pos] = returns[(start + j) % n];
            }
        }

        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += sample[i];
        double mu = sum / n;

        double sumsq = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = sample[i] - mu;
            sumsq += d * d;
        }
        double std_dev = std::sqrt(sumsq / (n - 1));
        sharpes[b] = (std_dev > 1e-15) ? (mu / std_dev * annualize) : 0.0;
    }

    std::vector<double> sorted_sharpes(sharpes);
    std::sort(sorted_sharpes.begin(), sorted_sharpes.end());

    double s_sum = 0.0;
    for (double s : sharpes) s_sum += s;
    double s_mean = s_sum / n_bootstrap;

    auto percentile = [&](double p) -> double {
        double k = (n_bootstrap - 1) * p / 100.0;
        int lo = static_cast<int>(k);
        int hi = std::min(lo + 1, n_bootstrap - 1);
        double frac = k - lo;
        return sorted_sharpes[lo] + frac * (sorted_sharpes[hi] - sorted_sharpes[lo]);
    };

    int gt0 = 0, gt05 = 0;
    for (double s : sharpes) {
        if (s > 0.0) ++gt0;
        if (s > 0.5) ++gt05;
    }

    return {
        s_mean,
        percentile(2.5),
        percentile(97.5),
        static_cast<double>(gt0) / n_bootstrap,
        static_cast<double>(gt05) / n_bootstrap,
    };
}
