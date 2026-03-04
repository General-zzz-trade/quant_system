#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include <pybind11/numpy.h>

namespace py = pybind11;

namespace mtf {

// EMA: skip NaN, first non-NaN initializes
inline void ema(const double* arr, int n, int span, double* out) {
    double alpha = 2.0 / (span + 1);
    bool started = false;
    double prev = 0.0;
    for (int i = 0; i < n; ++i) {
        out[i] = std::nan("");
        if (std::isnan(arr[i])) continue;
        if (!started) {
            out[i] = arr[i];
            prev = arr[i];
            started = true;
        } else {
            out[i] = alpha * arr[i] + (1.0 - alpha) * prev;
            prev = out[i];
        }
    }
}

// SMA via nancumsum (NaN treated as 0, divide by window not valid count)
inline void sma(const double* arr, int n, int window, double* out) {
    std::vector<double> cumsum(n);
    double cs = 0;
    for (int i = 0; i < n; ++i) {
        cs += std::isnan(arr[i]) ? 0.0 : arr[i];
        cumsum[i] = cs;
    }
    for (int i = 0; i < window - 1; ++i)
        out[i] = std::nan("");
    for (int i = window - 1; i < n; ++i) {
        double s = cumsum[i];
        if (i >= window) s -= cumsum[i - window];
        out[i] = s / window;
    }
}

// Rolling std (ddof=1, filter NaN, valid >= window//2)
inline void rolling_std(const double* arr, int n, int window, double* out) {
    int min_valid = window / 2;
    for (int i = 0; i < n; ++i) {
        out[i] = std::nan("");
        if (i < window - 1) continue;
        // Collect valid values in window
        double sum = 0, sumsq = 0;
        int valid = 0;
        for (int k = i - window + 1; k <= i; ++k) {
            if (!std::isnan(arr[k])) {
                sum += arr[k];
                sumsq += arr[k] * arr[k];
                ++valid;
            }
        }
        if (valid < min_valid) continue;
        double mean = sum / valid;
        double var = (sumsq - valid * mean * mean) / (valid - 1);
        out[i] = (var > 0) ? std::sqrt(var) : 0.0;
    }
}

struct Group4H {
    double open, high, low, close, volume;
    int64_t group_id;
};

// Main entry: compute 10 4h features from 1h OHLCV, return (n, 10) array
inline py::array_t<double> cpp_compute_4h_features(
    py::array_t<int64_t, py::array::c_style> ts_arr,
    py::array_t<double, py::array::c_style> opens_arr,
    py::array_t<double, py::array::c_style> highs_arr,
    py::array_t<double, py::array::c_style> lows_arr,
    py::array_t<double, py::array::c_style> closes_arr,
    py::array_t<double, py::array::c_style> volumes_arr
) {
    auto ts_buf = ts_arr.request();
    auto opens_buf = opens_arr.request();
    auto highs_buf = highs_arr.request();
    auto lows_buf = lows_arr.request();
    auto closes_buf = closes_arr.request();
    auto volumes_buf = volumes_arr.request();

    int n = static_cast<int>(ts_buf.shape[0]);
    const int64_t* ts = static_cast<const int64_t*>(ts_buf.ptr);
    const double* opens = static_cast<const double*>(opens_buf.ptr);
    const double* highs = static_cast<const double*>(highs_buf.ptr);
    const double* lows = static_cast<const double*>(lows_buf.ptr);
    const double* closes = static_cast<const double*>(closes_buf.ptr);
    const double* volumes = static_cast<const double*>(volumes_buf.ptr);

    constexpr int64_t FOUR_HOURS_MS = 4LL * 3600 * 1000;
    constexpr int N_FEAT = 10;

    // Step 1: Resample to 4h bars
    // group_id = ts[i] // FOUR_HOURS_MS
    std::vector<int64_t> group_keys(n);
    for (int i = 0; i < n; ++i)
        group_keys[i] = ts[i] / FOUR_HOURS_MS;

    // Build ordered 4h bars
    std::vector<Group4H> bars;
    std::map<int64_t, int> group_to_idx;

    int i = 0;
    while (i < n) {
        int64_t gid = group_keys[i];
        double o = opens[i], h = highs[i], l = lows[i], c = closes[i], v = volumes[i];
        int j = i + 1;
        while (j < n && group_keys[j] == gid) {
            if (highs[j] > h) h = highs[j];
            if (lows[j] < l) l = lows[j];
            c = closes[j];
            v += volumes[j];
            ++j;
        }
        group_to_idx[gid] = static_cast<int>(bars.size());
        bars.push_back({o, h, l, c, v, gid});
        i = j;
    }

    int n4 = static_cast<int>(bars.size());

    // Extract close, high, low arrays
    std::vector<double> close_4h(n4), high_4h(n4), low_4h(n4);
    for (int i = 0; i < n4; ++i) {
        close_4h[i] = bars[i].close;
        high_4h[i] = bars[i].high;
        low_4h[i] = bars[i].low;
    }

    // Step 2: Compute 10 features on 4h bars
    // Feature arrays
    std::vector<double> ret_1(n4, std::nan("")), ret_3(n4, std::nan("")), ret_6(n4, std::nan(""));
    std::vector<double> rsi_14(n4, std::nan(""));
    std::vector<double> macd_hist(n4, std::nan(""));
    std::vector<double> bb_pctb(n4, std::nan(""));
    std::vector<double> atr_norm(n4, std::nan(""));
    std::vector<double> vol_20(n4, std::nan(""));
    std::vector<double> close_vs_ma20(n4, std::nan(""));
    std::vector<double> mean_rev(n4, std::nan(""));

    // Returns
    for (int i = 1; i < n4; ++i)
        ret_1[i] = close_4h[i] / close_4h[i - 1] - 1.0;
    for (int i = 3; i < n4; ++i)
        ret_3[i] = close_4h[i] / close_4h[i - 3] - 1.0;
    for (int i = 6; i < n4; ++i)
        ret_6[i] = close_4h[i] / close_4h[i - 6] - 1.0;

    // RSI-14
    std::vector<double> pct(n4, std::nan(""));
    for (int i = 1; i < n4; ++i)
        pct[i] = close_4h[i] / close_4h[i - 1] - 1.0;

    std::vector<double> gains(n4, 0.0), losses(n4, 0.0);
    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(pct[i])) {
            if (pct[i] > 0) gains[i] = pct[i];
            else if (pct[i] < 0) losses[i] = -pct[i];
        }
    }
    std::vector<double> avg_gain(n4), avg_loss(n4);
    ema(gains.data(), n4, 14, avg_gain.data());
    ema(losses.data(), n4, 14, avg_loss.data());

    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(avg_gain[i]) && !std::isnan(avg_loss[i])) {
            if (avg_loss[i] < 1e-15)
                rsi_14[i] = 100.0;
            else {
                double rs = avg_gain[i] / avg_loss[i];
                rsi_14[i] = 100.0 - 100.0 / (1.0 + rs);
            }
        }
    }

    // MACD (12, 26, 9)
    std::vector<double> ema12(n4), ema26(n4), macd_line(n4), signal_line(n4);
    ema(close_4h.data(), n4, 12, ema12.data());
    ema(close_4h.data(), n4, 26, ema26.data());
    for (int i = 0; i < n4; ++i)
        macd_line[i] = ema12[i] - ema26[i];
    ema(macd_line.data(), n4, 9, signal_line.data());
    for (int i = 0; i < n4; ++i) {
        macd_hist[i] = macd_line[i] - signal_line[i];
        if (close_4h[i] > 0 && !std::isnan(macd_hist[i]))
            macd_hist[i] /= close_4h[i];
    }

    // Bollinger %B (20, 2)
    std::vector<double> ma20(n4), std20(n4);
    sma(close_4h.data(), n4, 20, ma20.data());
    rolling_std(close_4h.data(), n4, 20, std20.data());
    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(ma20[i]) && !std::isnan(std20[i]) && std20[i] > 1e-15) {
            double upper = ma20[i] + 2.0 * std20[i];
            double lower = ma20[i] - 2.0 * std20[i];
            bb_pctb[i] = (close_4h[i] - lower) / (upper - lower);
        }
    }

    // ATR normalized (14)
    std::vector<double> tr(n4, std::nan("")), atr_raw(n4);
    for (int i = 1; i < n4; ++i) {
        double hl = high_4h[i] - low_4h[i];
        double hc = std::abs(high_4h[i] - close_4h[i - 1]);
        double lc = std::abs(low_4h[i] - close_4h[i - 1]);
        tr[i] = std::max({hl, hc, lc});
    }
    ema(tr.data(), n4, 14, atr_raw.data());
    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(atr_raw[i]) && close_4h[i] > 0)
            atr_norm[i] = atr_raw[i] / close_4h[i];
    }

    // Volatility (rolling_std of pct, 20)
    rolling_std(pct.data(), n4, 20, vol_20.data());

    // Close vs MA20
    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(ma20[i]) && ma20[i] > 0)
            close_vs_ma20[i] = close_4h[i] / ma20[i] - 1.0;
    }

    // Mean reversion z-score
    for (int i = 0; i < n4; ++i) {
        if (!std::isnan(ma20[i]) && !std::isnan(std20[i]) && std20[i] > 1e-15)
            mean_rev[i] = (close_4h[i] - ma20[i]) / std20[i];
    }

    // Step 3: Map back to 1h (anti-lookahead: use G-1)
    // Build feature matrix for 4h bars
    std::vector<double*> feat_ptrs = {
        ret_1.data(), ret_3.data(), ret_6.data(), rsi_14.data(),
        macd_hist.data(), bb_pctb.data(), atr_norm.data(),
        vol_20.data(), close_vs_ma20.data(), mean_rev.data()
    };

    // Allocate output (n x 10)
    auto result = py::array_t<double>({n, N_FEAT});
    auto rbuf = result.mutable_unchecked<2>();

    for (int i = 0; i < n; ++i) {
        int64_t g = group_keys[i];
        auto it = group_to_idx.find(g - 1);
        if (it != group_to_idx.end()) {
            int idx = it->second;
            for (int f = 0; f < N_FEAT; ++f)
                rbuf(i, f) = feat_ptrs[f][idx];
        } else {
            for (int f = 0; f < N_FEAT; ++f)
                rbuf(i, f) = std::nan("");
        }
    }

    return result;
}

inline std::vector<std::string> cpp_4h_feature_names() {
    return {
        "tf4h_ret_1", "tf4h_ret_3", "tf4h_ret_6", "tf4h_rsi_14",
        "tf4h_macd_hist", "tf4h_bb_pctb_20", "tf4h_atr_norm_14",
        "tf4h_vol_20", "tf4h_close_vs_ma20", "tf4h_mean_reversion_20"
    };
}

} // namespace mtf
