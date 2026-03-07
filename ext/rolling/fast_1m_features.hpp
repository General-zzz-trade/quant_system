// fast_1m_features.hpp — C++ accelerated 1-minute feature computation
// Single-pass over 2.2M+ bars, replaces pandas vectorized _compute_fast_features()
// Output: N x 15 column-major double array (matching FAST_FEATURE_NAMES order)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace fast1m {

static constexpr int N_FAST = 15;
static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// Feature column indices (match FAST_FEATURE_NAMES order in multi_resolution.py)
enum F1m {
    F_ret_1 = 0, F_ret_3, F_ret_5, F_ret_10,
    F_rsi_6,
    F_vol_5, F_vol_20,
    F_taker_imbalance,
    F_trade_intensity,
    F_cvd_10,
    F_body_ratio, F_upper_shadow, F_lower_shadow,
    F_vol_ratio_20,
    F_aggressive_flow_zscore,
};

static const std::vector<std::string> FAST_1M_NAMES = {
    "ret_1", "ret_3", "ret_5", "ret_10",
    "rsi_6",
    "vol_5", "vol_20",
    "taker_imbalance",
    "trade_intensity",
    "cvd_10",
    "body_ratio", "upper_shadow", "lower_shadow",
    "vol_ratio_20",
    "aggressive_flow_zscore",
};

// ── Circular buffer for rolling stats ────────────────────────
template<int N>
struct CircBuf {
    double buf[N];
    double sum = 0.0;
    double sq_sum = 0.0;
    int head = 0;
    int count = 0;

    void push(double x) {
        if (count >= N) {
            double old = buf[head];
            sum -= old;
            sq_sum -= old * old;
        } else {
            count++;
        }
        buf[head] = x;
        sum += x;
        sq_sum += x * x;
        head = (head + 1) % N;
    }

    bool full() const { return count >= N; }
    double mean() const { return sum / count; }
    double var() const {
        double m = mean();
        return sq_sum / count - m * m;
    }
    double std() const {
        double v = var();
        return v > 0 ? std::sqrt(v) : 0.0;
    }
};

// ── Ring buffer for rolling sum (no variance needed) ─────────
template<int N>
struct RingSum {
    double buf[N];
    double sum = 0.0;
    int head = 0;
    int count = 0;

    void push(double x) {
        if (count >= N) {
            sum -= buf[head];
        } else {
            count++;
        }
        buf[head] = x;
        sum += x;
        head = (head + 1) % N;
    }
    bool full() const { return count >= N; }
    double total() const { return sum; }
};

// ── Main computation ─────────────────────────────────────────

inline py::array_t<double> cpp_compute_fast_1m_features(
    py::array_t<double, py::array::c_style | py::array::forcecast> opens_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> highs_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> lows_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> closes_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> volumes_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> trades_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> tbv_arr
) {
    auto cb = closes_arr.request();
    const int n = static_cast<int>(cb.shape[0]);
    const double* open = static_cast<const double*>(opens_arr.request().ptr);
    const double* high = static_cast<const double*>(highs_arr.request().ptr);
    const double* low = static_cast<const double*>(lows_arr.request().ptr);
    const double* close = static_cast<const double*>(cb.ptr);
    const double* volume = static_cast<const double*>(volumes_arr.request().ptr);
    const double* trades = static_cast<const double*>(trades_arr.request().ptr);
    const double* tbv = static_cast<const double*>(tbv_arr.request().ptr);

    // Output: row-major N x N_FAST
    auto result = py::array_t<double>({n, N_FAST});
    auto rbuf = result.request();
    double* out = static_cast<double*>(rbuf.ptr);

    // Initialize all to NaN
    for (int i = 0; i < n * N_FAST; ++i) out[i] = NaN;

    // State variables for EMA-based RSI (span=6 → alpha = 2/7)
    const double rsi_alpha = 2.0 / 7.0;
    double avg_gain = 0.0, avg_loss = 0.0;
    bool rsi_started = false;

    // EMA for trades (span=20 → alpha = 2/21)
    const double ema_trades_alpha = 2.0 / 21.0;
    double ema_trades = 0.0;
    bool ema_trades_started = false;

    // EMA for volume (span=20)
    const double ema_vol_alpha = 2.0 / 21.0;
    double ema_vol = 0.0;
    bool ema_vol_started = false;

    // Rolling windows for vol_5, vol_20 (rolling std of pct_change)
    CircBuf<5> vol5_buf;
    CircBuf<20> vol20_buf;

    // Rolling sum for CVD-10
    RingSum<10> cvd10_buf;

    // Rolling window for aggressive_flow_zscore (24-bar)
    CircBuf<24> afs_buf;

    for (int i = 0; i < n; ++i) {
        double* row = out + i * N_FAST;
        double c = close[i];
        double o = open[i];
        double h = high[i];
        double l = low[i];
        double v = volume[i];
        double t = trades[i];
        double tb = tbv[i];

        // ── Returns ──────────────────────────────────
        // ret_1
        if (i >= 1) {
            double prev = close[i - 1];
            if (prev != 0.0) row[F_ret_1] = c / prev - 1.0;
        }
        // ret_3
        if (i >= 3) {
            double prev = close[i - 3];
            if (prev != 0.0) row[F_ret_3] = c / prev - 1.0;
        }
        // ret_5
        if (i >= 5) {
            double prev = close[i - 5];
            if (prev != 0.0) row[F_ret_5] = c / prev - 1.0;
        }
        // ret_10
        if (i >= 10) {
            double prev = close[i - 10];
            if (prev != 0.0) row[F_ret_10] = c / prev - 1.0;
        }

        // ── RSI-6 (EWM, span=6) ─────────────────────
        if (i >= 1) {
            double pct = (close[i - 1] != 0.0) ? (c / close[i - 1] - 1.0) : 0.0;
            double gain = pct > 0 ? pct : 0.0;
            double loss = pct < 0 ? -pct : 0.0;
            if (!rsi_started) {
                avg_gain = gain;
                avg_loss = loss;
                rsi_started = true;
            } else {
                avg_gain = rsi_alpha * gain + (1.0 - rsi_alpha) * avg_gain;
                avg_loss = rsi_alpha * loss + (1.0 - rsi_alpha) * avg_loss;
            }
            if (avg_loss > 0.0) {
                double rs = avg_gain / avg_loss;
                row[F_rsi_6] = 100.0 - 100.0 / (1.0 + rs);
            } else if (avg_gain > 0.0) {
                row[F_rsi_6] = 100.0;
            } else {
                row[F_rsi_6] = 50.0;  // no movement
            }
        }

        // ── Volatility (rolling std of pct_change) ───
        if (i >= 1) {
            double pct = (close[i - 1] != 0.0) ? (c / close[i - 1] - 1.0) : 0.0;
            vol5_buf.push(pct);
            vol20_buf.push(pct);
            if (vol5_buf.count >= 3) row[F_vol_5] = vol5_buf.std();
            if (vol20_buf.count >= 10) row[F_vol_20] = vol20_buf.std();
        }

        // ── Taker imbalance ──────────────────────────
        double taker_ratio = (v > 0) ? tb / v : 0.5;
        row[F_taker_imbalance] = 2.0 * taker_ratio - 1.0;

        // ── Trade intensity ──────────────────────────
        if (!ema_trades_started) {
            ema_trades = t;
            ema_trades_started = true;
        } else {
            ema_trades = ema_trades_alpha * t + (1.0 - ema_trades_alpha) * ema_trades;
        }
        if (ema_trades > 0.0) {
            row[F_trade_intensity] = t / ema_trades;
        }

        // ── CVD-10 ───────────────────────────────────
        double delta = tb - (v - tb);  // buy - sell volume
        cvd10_buf.push(delta);

        if (!ema_vol_started) {
            ema_vol = v;
            ema_vol_started = true;
        } else {
            ema_vol = ema_vol_alpha * v + (1.0 - ema_vol_alpha) * ema_vol;
        }

        if (cvd10_buf.full()) {
            double denom = ema_vol * 10.0;
            if (denom > 0.0) {
                row[F_cvd_10] = cvd10_buf.total() / denom;
            }
        }

        // ── Candle structure ─────────────────────────
        double body = std::abs(c - o);
        double full_range = h - l;
        if (full_range > 0.0) {
            row[F_body_ratio] = body / full_range;
            row[F_upper_shadow] = (h - std::max(c, o)) / full_range;
            row[F_lower_shadow] = (std::min(c, o) - l) / full_range;
        } else {
            row[F_body_ratio] = 0.0;
            row[F_upper_shadow] = 0.0;
            row[F_lower_shadow] = 0.0;
        }

        // ── Vol ratio 20 ────────────────────────────
        if (ema_vol > 0.0) {
            row[F_vol_ratio_20] = v / ema_vol;
        }

        // ── Aggressive flow z-score (24-bar) ─────────
        double tbr = (v > 0) ? tb / v : 0.5;
        afs_buf.push(tbr);
        if (afs_buf.full()) {
            double s = afs_buf.std();
            if (s > 0.0) {
                row[F_aggressive_flow_zscore] = (tbr - afs_buf.mean()) / s;
            }
        }
    }

    return result;
}

inline std::vector<std::string> cpp_fast_1m_feature_names() {
    return FAST_1M_NAMES;
}

}  // namespace fast1m
