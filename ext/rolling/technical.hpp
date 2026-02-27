#pragma once

#include "rolling_window.hpp"
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

using Opt = std::optional<double>;
using Vec = std::vector<Opt>;

inline Vec cpp_sma(const std::vector<double>& vals, int window) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    RollingWindow rw(window);
    Vec out;
    out.reserve(vals.size());
    for (double x : vals) {
        rw.push(x);
        out.push_back(rw.full() ? rw.mean() : std::nullopt);
    }
    return out;
}

inline Vec cpp_ema(const std::vector<double>& vals, int window) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    double alpha = 2.0 / (window + 1.0);
    Vec out;
    out.reserve(vals.size());
    double prev = 0.0;
    bool started = false;
    for (double x : vals) {
        if (!started) {
            prev = x;
            started = true;
        } else {
            prev = alpha * x + (1.0 - alpha) * prev;
        }
        out.push_back(prev);
    }
    return out;
}

inline Vec cpp_returns(const std::vector<double>& vals, bool log_ret) {
    Vec out(vals.size(), std::nullopt);
    for (size_t i = 1; i < vals.size(); ++i) {
        double p0 = vals[i - 1];
        double p1 = vals[i];
        if (p0 == 0.0) {
            out[i] = std::nullopt;
            continue;
        }
        double r = p1 / p0;
        out[i] = log_ret ? std::log(r) : (r - 1.0);
    }
    return out;
}

inline Vec cpp_volatility(const std::vector<Opt>& rets, int window) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    RollingWindow rw(window);
    Vec out;
    out.reserve(rets.size());
    for (const auto& r : rets) {
        rw.push(r.has_value() ? *r : 0.0);
        out.push_back(rw.full() ? rw.std_dev() : std::nullopt);
    }
    return out;
}

inline Vec cpp_rsi(const std::vector<double>& vals, int window) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    Vec out(vals.size(), std::nullopt);
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    for (size_t i = 1; i < vals.size(); ++i) {
        double change = vals[i] - vals[i - 1];
        double gain = std::max(change, 0.0);
        double loss = std::max(-change, 0.0);

        if (static_cast<int>(i) < window) {
            avg_gain += gain;
            avg_loss += loss;
            continue;
        }

        if (static_cast<int>(i) == window) {
            avg_gain /= window;
            avg_loss /= window;
        } else {
            avg_gain = (avg_gain * (window - 1) + gain) / window;
            avg_loss = (avg_loss * (window - 1) + loss) / window;
        }

        if (avg_loss == 0.0) {
            out[i] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            out[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    return out;
}

inline std::tuple<Vec, Vec, Vec> cpp_macd(
    const std::vector<double>& vals, int fast, int slow, int signal
) {
    if (fast <= 0 || slow <= 0 || signal <= 0)
        throw std::invalid_argument("MACD windows must be positive");
    if (fast >= slow)
        throw std::invalid_argument("fast window must be smaller than slow window");

    Vec fast_ema = cpp_ema(vals, fast);
    Vec slow_ema = cpp_ema(vals, slow);

    Vec macd_line;
    macd_line.reserve(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        if (fast_ema[i] && slow_ema[i]) {
            macd_line.push_back(*fast_ema[i] - *slow_ema[i]);
        } else {
            macd_line.push_back(std::nullopt);
        }
    }

    // Signal line: EMA of macd_line values (substitute 0.0 for None during warmup)
    std::vector<double> macd_values;
    macd_values.reserve(vals.size());
    for (const auto& v : macd_line) {
        macd_values.push_back(v.has_value() ? *v : 0.0);
    }
    Vec signal_raw = cpp_ema(macd_values, signal);

    // Find first valid macd_line index
    size_t first_valid = vals.size();
    for (size_t i = 0; i < macd_line.size(); ++i) {
        if (macd_line[i].has_value()) { first_valid = i; break; }
    }

    Vec signal_line;
    signal_line.reserve(vals.size());
    for (size_t i = 0; i < signal_raw.size(); ++i) {
        if (i < first_valid + signal - 1) {
            signal_line.push_back(std::nullopt);
        } else {
            signal_line.push_back(signal_raw[i]);
        }
    }

    Vec histogram;
    histogram.reserve(vals.size());
    for (size_t i = 0; i < macd_line.size(); ++i) {
        if (macd_line[i] && signal_line[i]) {
            histogram.push_back(*macd_line[i] - *signal_line[i]);
        } else {
            histogram.push_back(std::nullopt);
        }
    }

    return {macd_line, signal_line, histogram};
}

inline std::tuple<Vec, Vec, Vec> cpp_bollinger_bands(
    const std::vector<double>& vals, int window, double num_std
) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    if (num_std <= 0) throw std::invalid_argument("num_std must be positive");

    RollingWindow rw(window);
    Vec upper, middle, lower;
    upper.reserve(vals.size());
    middle.reserve(vals.size());
    lower.reserve(vals.size());

    for (double x : vals) {
        rw.push(x);
        if (!rw.full()) {
            upper.push_back(std::nullopt);
            middle.push_back(std::nullopt);
            lower.push_back(std::nullopt);
        } else {
            auto mid = rw.mean();
            auto sd = rw.std_dev();
            if (mid && sd) {
                upper.push_back(*mid + num_std * *sd);
                middle.push_back(*mid);
                lower.push_back(*mid - num_std * *sd);
            } else {
                upper.push_back(std::nullopt);
                middle.push_back(std::nullopt);
                lower.push_back(std::nullopt);
            }
        }
    }
    return {upper, middle, lower};
}

inline Vec cpp_atr(
    const std::vector<double>& highs,
    const std::vector<double>& lows,
    const std::vector<double>& closes,
    int window
) {
    if (window <= 0) throw std::invalid_argument("window must be positive");
    size_t n = highs.size();
    if (lows.size() != n || closes.size() != n)
        throw std::invalid_argument("highs, lows, closes must have same length");

    std::vector<double> trs;
    trs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        double tr;
        if (i == 0) {
            tr = highs[i] - lows[i];
        } else {
            tr = std::max({
                highs[i] - lows[i],
                std::abs(highs[i] - closes[i - 1]),
                std::abs(lows[i] - closes[i - 1])
            });
        }
        trs.push_back(tr);
    }

    Vec out(n, std::nullopt);
    double atr_prev = 0.0;

    for (size_t i = 0; i < n; ++i) {
        int ii = static_cast<int>(i);
        if (ii < window) {
            atr_prev += trs[i];
            continue;
        }
        if (ii == window) {
            atr_prev /= window;
            out[i] = atr_prev;
            continue;
        }
        atr_prev = (atr_prev * (window - 1) + trs[i]) / window;
        out[i] = atr_prev;
    }
    return out;
}
