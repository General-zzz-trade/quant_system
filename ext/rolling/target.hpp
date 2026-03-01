#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Vol-normalized forward return target.
// Returns NaN where target is undefined.
inline std::vector<double> cpp_vol_normalized_target(
    const std::vector<double>& closes,
    int horizon = 5,
    int vol_window = 20
) {
    const int n = static_cast<int>(closes.size());
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(n, NaN);

    if (n < vol_window + horizon) return result;

    // 1. Forward returns: raw_ret[i] = closes[i+horizon]/closes[i] - 1
    std::vector<double> raw_ret(n, NaN);
    for (int i = 0; i < n - horizon; ++i) {
        raw_ret[i] = closes[i + horizon] / closes[i] - 1.0;
    }

    // 2. Pct change
    std::vector<double> pct(n, NaN);
    for (int i = 1; i < n; ++i) {
        pct[i] = closes[i] / closes[i - 1] - 1.0;
    }

    // 3. Rolling std of pct (ddof=1)
    std::vector<double> vol(n, NaN);
    int half_window = vol_window / 2;
    for (int i = vol_window; i < n; ++i) {
        // window: pct[i - vol_window + 1 .. i]
        double sum = 0.0;
        int count = 0;
        for (int j = i - vol_window + 1; j <= i; ++j) {
            if (!std::isnan(pct[j])) {
                sum += pct[j];
                ++count;
            }
        }
        if (count < half_window) continue;

        double mean = sum / count;
        double sumsq = 0.0;
        for (int j = i - vol_window + 1; j <= i; ++j) {
            if (!std::isnan(pct[j])) {
                double d = pct[j] - mean;
                sumsq += d * d;
            }
        }
        vol[i] = std::sqrt(sumsq / (count - 1));
    }

    // 4. Clip vol at 5th percentile floor
    std::vector<double> vol_valid;
    vol_valid.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(vol[i])) vol_valid.push_back(vol[i]);
    }

    if (vol_valid.empty()) return raw_ret;

    std::sort(vol_valid.begin(), vol_valid.end());
    int idx5 = static_cast<int>(vol_valid.size() * 0.05);
    double floor = vol_valid[std::min(idx5, static_cast<int>(vol_valid.size()) - 1)];

    for (int i = 0; i < n; ++i) {
        if (!std::isnan(vol[i]) && vol[i] < floor) {
            vol[i] = floor;
        }
    }

    // 5. Normalize: target = raw_ret / vol
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(raw_ret[i]) && !std::isnan(vol[i]) && vol[i] > 0) {
            result[i] = raw_ret[i] / vol[i];
        }
    }

    return result;
}
