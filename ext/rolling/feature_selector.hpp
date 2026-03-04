#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace feat_sel {

// Rank data with average tie-breaking (matches dynamic_selector._rankdata)
inline void rankdata(const double* arr, int n, double* ranks) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return arr[a] < arr[b];
    });

    // Assign ranks 1..N
    for (int i = 0; i < n; ++i)
        ranks[order[i]] = static_cast<double>(i + 1);

    // Average ties
    int i = 0;
    while (i < n) {
        int j = i + 1;
        while (j < n && arr[order[j]] == arr[order[i]])
            ++j;
        if (j > i + 1) {
            double avg = 0.0;
            for (int k = i; k < j; ++k)
                avg += ranks[order[k]];
            avg /= (j - i);
            for (int k = i; k < j; ++k)
                ranks[order[k]] = avg;
        }
        i = j;
    }
}

// Pearson correlation (population, matching numpy corrcoef)
inline double pearson_ic(const double* x, const double* y, int n) {
    if (n < 2) return 0.0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < n; ++i) {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; syy += y[i] * y[i];
        sxy += x[i] * y[i];
    }
    double mx = sx / n, my = sy / n;
    double vx = sxx / n - mx * mx;
    double vy = syy / n - my * my;
    double cov = sxy / n - mx * my;
    double den = std::sqrt(vx * vy);
    return (den < 1e-15) ? 0.0 : cov / den;
}

// Spearman IC = pearson of ranks
inline double spearman_ic(const double* x, const double* y, int n) {
    std::vector<double> rx(n), ry(n);
    rankdata(x, n, rx.data());
    rankdata(y, n, ry.data());
    return pearson_ic(rx.data(), ry.data(), n);
}

// Helper: compute IC for one feature column vs y, handling NaN mask
// Returns IC value; sets valid_count. If invalid, returns 0.0.
struct ICResult {
    double ic;
    bool valid;
};

inline ICResult compute_ic_masked(
    const double* X, int n_samples, int n_features, int col_j,
    const double* y, int start, int end,
    bool use_spearman,
    std::vector<double>& x_buf, std::vector<double>& y_buf
) {
    x_buf.clear();
    y_buf.clear();
    for (int i = start; i < end; ++i) {
        double xv = X[static_cast<size_t>(i) * n_features + col_j];
        double yv = y[i];
        if (std::isnan(xv) || std::isnan(yv)) continue;
        x_buf.push_back(xv);
        y_buf.push_back(yv);
    }
    int valid = static_cast<int>(x_buf.size());
    if (valid < 30) return {0.0, false};

    // Check std
    double sx = 0, sxx = 0, sy = 0, syy = 0;
    for (int i = 0; i < valid; ++i) {
        sx += x_buf[i]; sxx += x_buf[i] * x_buf[i];
        sy += y_buf[i]; syy += y_buf[i] * y_buf[i];
    }
    double vx = sxx / valid - (sx / valid) * (sx / valid);
    double vy = syy / valid - (sy / valid) * (sy / valid);
    if (vx < 1e-24 || vy < 1e-24) return {0.0, false};

    double ic;
    if (use_spearman) {
        ic = spearman_ic(x_buf.data(), y_buf.data(), valid);
    } else {
        ic = pearson_ic(x_buf.data(), y_buf.data(), valid);
    }
    return {ic, true};
}

// rolling_ic_select: Pearson IC on last ic_window bars, top_k by |IC|
inline std::vector<int> cpp_rolling_ic_select(
    const double* X, const double* y,
    int n_samples, int n_features,
    int top_k, int ic_window
) {
    if (n_samples < 50)
        return {};

    int start = std::max(0, n_samples - ic_window);
    int end = n_samples;

    std::vector<std::pair<double, int>> scores(n_features);
    std::vector<double> xbuf, ybuf;
    xbuf.reserve(ic_window);
    ybuf.reserve(ic_window);

    for (int j = 0; j < n_features; ++j) {
        auto r = compute_ic_masked(X, n_samples, n_features, j, y, start, end, false, xbuf, ybuf);
        scores[j] = {r.valid ? std::abs(r.ic) : 0.0, j};
    }

    int k = std::min(top_k, n_features);
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> result(k);
    for (int i = 0; i < k; ++i)
        result[i] = scores[i].second;
    return result;
}

// spearman_ic_select: Spearman IC on last ic_window bars, top_k by |IC|
inline std::vector<int> cpp_spearman_ic_select(
    const double* X, const double* y,
    int n_samples, int n_features,
    int top_k, int ic_window
) {
    if (n_samples < 50)
        return {};

    int start = std::max(0, n_samples - ic_window);
    int end = n_samples;

    std::vector<std::pair<double, int>> scores(n_features);
    std::vector<double> xbuf, ybuf;
    xbuf.reserve(ic_window);
    ybuf.reserve(ic_window);

    for (int j = 0; j < n_features; ++j) {
        auto r = compute_ic_masked(X, n_samples, n_features, j, y, start, end, true, xbuf, ybuf);
        scores[j] = {r.valid ? std::abs(r.ic) : 0.0, j};
    }

    int k = std::min(top_k, n_features);
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> result(k);
    for (int i = 0; i < k; ++i)
        result[i] = scores[i].second;
    return result;
}

// icir_select: ICIR = mean(|IC|) / std(IC, ddof=1) across n_windows
inline std::vector<int> cpp_icir_select(
    const double* X, const double* y,
    int n_samples, int n_features,
    int top_k, int ic_window, int n_windows,
    double min_icir, int max_consec_neg
) {
    int total_needed = ic_window * n_windows;
    if (n_samples < total_needed || n_samples < 50)
        return {};

    int start_offset = n_samples - total_needed;

    // Precompute window boundaries
    std::vector<std::pair<int, int>> windows(n_windows);
    for (int w = 0; w < n_windows; ++w) {
        int ws = start_offset + w * ic_window;
        windows[w] = {ws, ws + ic_window};
    }

    std::vector<std::pair<double, int>> results;
    results.reserve(n_features);

    std::vector<double> xbuf, ybuf;
    xbuf.reserve(ic_window);
    ybuf.reserve(ic_window);
    std::vector<double> ics(n_windows);

    for (int j = 0; j < n_features; ++j) {
        // Compute Spearman IC per window
        for (int w = 0; w < n_windows; ++w) {
            auto r = compute_ic_masked(X, n_samples, n_features, j, y,
                                        windows[w].first, windows[w].second,
                                        true, xbuf, ybuf);
            ics[w] = r.valid ? r.ic : 0.0;
        }

        // Check consecutive negative
        int max_neg = 0, cur_neg = 0;
        for (int w = 0; w < n_windows; ++w) {
            if (ics[w] < 0) {
                ++cur_neg;
                if (cur_neg > max_neg) max_neg = cur_neg;
            } else {
                cur_neg = 0;
            }
        }
        if (max_neg >= max_consec_neg) continue;

        // ICIR = mean(|IC|) / std(IC, ddof=1)
        double sum_abs = 0, sum = 0, sumsq = 0;
        for (int w = 0; w < n_windows; ++w) {
            sum_abs += std::abs(ics[w]);
            sum += ics[w];
            sumsq += ics[w] * ics[w];
        }
        double mean_abs = sum_abs / n_windows;
        double ic_std = 0.0;
        if (n_windows > 1) {
            double mean_ic = sum / n_windows;
            double var = (sumsq - n_windows * mean_ic * mean_ic) / (n_windows - 1);
            ic_std = (var > 0) ? std::sqrt(var) : 0.0;
        }

        double icir;
        if (ic_std < 1e-12) {
            icir = (mean_abs > 0) ? mean_abs * 100.0 : 0.0;
        } else {
            icir = mean_abs / ic_std;
        }

        if (icir < min_icir) continue;

        results.push_back({icir, j});
    }

    // Sort by descending ICIR
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    int k = std::min(top_k, static_cast<int>(results.size()));
    std::vector<int> out(k);
    for (int i = 0; i < k; ++i)
        out[i] = results[i].second;
    return out;
}

// stable_icir_select: Jackknife stability + sign consistency
inline std::vector<int> cpp_stable_icir_select(
    const double* X, const double* y,
    int n_samples, int n_features,
    int top_k, int ic_window, int n_windows,
    double min_icir, int min_stable_folds,
    double sign_consistency
) {
    int total_needed = ic_window * n_windows;
    if (n_samples < total_needed || n_samples < 50)
        return {};  // empty = Python fallback to greedy

    int start_offset = n_samples - total_needed;

    std::vector<std::pair<int, int>> windows(n_windows);
    for (int w = 0; w < n_windows; ++w) {
        int ws = start_offset + w * ic_window;
        windows[w] = {ws, ws + ic_window};
    }

    std::vector<std::pair<double, int>> candidates;
    candidates.reserve(n_features);

    std::vector<double> xbuf, ybuf;
    xbuf.reserve(ic_window);
    ybuf.reserve(ic_window);
    std::vector<double> ics(n_windows);

    for (int j = 0; j < n_features; ++j) {
        for (int w = 0; w < n_windows; ++w) {
            auto r = compute_ic_masked(X, n_samples, n_features, j, y,
                                        windows[w].first, windows[w].second,
                                        true, xbuf, ybuf);
            ics[w] = r.valid ? r.ic : 0.0;
        }

        // Jackknife: for fold i, compute std of OTHER folds
        int folds_above = 0;
        for (int iw = 0; iw < n_windows; ++iw) {
            // Compute std of others (ddof=1)
            double other_sum = 0, other_sumsq = 0;
            int n_other = n_windows - 1;
            for (int k = 0; k < n_windows; ++k) {
                if (k == iw) continue;
                other_sum += ics[k];
                other_sumsq += ics[k] * ics[k];
            }
            double std_other = 1e-12;
            if (n_other > 1) {
                double mean_other = other_sum / n_other;
                double var_other = (other_sumsq - n_other * mean_other * mean_other) / (n_other - 1);
                if (var_other > 0) std_other = std::sqrt(var_other);
                if (std_other < 1e-12) std_other = 1e-12;
            }
            double window_icir = std::abs(ics[iw]) / std_other;
            if (window_icir > min_icir)
                ++folds_above;
        }

        if (folds_above < min_stable_folds) continue;

        // Sign consistency
        int n_pos = 0, n_neg = 0;
        for (int w = 0; w < n_windows; ++w) {
            if (ics[w] > 0) ++n_pos;
            else if (ics[w] < 0) ++n_neg;
        }
        double dominant = static_cast<double>(std::max(n_pos, n_neg)) / std::max(n_windows, 1);
        if (dominant < sign_consistency) continue;

        // Overall ICIR
        double sum_abs = 0, sum = 0, sumsq = 0;
        for (int w = 0; w < n_windows; ++w) {
            sum_abs += std::abs(ics[w]);
            sum += ics[w];
            sumsq += ics[w] * ics[w];
        }
        double mean_abs = sum_abs / n_windows;
        double ic_std = 0.0;
        if (n_windows > 1) {
            double mean_ic = sum / n_windows;
            double var = (sumsq - n_windows * mean_ic * mean_ic) / (n_windows - 1);
            ic_std = (var > 0) ? std::sqrt(var) : 0.0;
        }
        double icir;
        if (ic_std < 1e-12) {
            icir = (mean_abs > 0) ? mean_abs * 100.0 : 0.0;
        } else {
            icir = mean_abs / ic_std;
        }

        candidates.push_back({icir, j});
    }

    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    int k = std::min(top_k, static_cast<int>(candidates.size()));

    // Return empty if < 5 pass (Python will fallback to greedy)
    if (k < 5) return {};

    std::vector<int> out(k);
    for (int i = 0; i < k; ++i)
        out[i] = candidates[i].second;
    return out;
}

// feature_icir_report: returns flat array (p x 5) with
// [mean_ic, std_ic, icir, max_consec_neg, pct_positive] per feature
inline std::vector<double> cpp_feature_icir_report(
    const double* X, const double* y,
    int n_samples, int n_features,
    int ic_window, int n_windows
) {
    int total_needed = ic_window * n_windows;
    std::vector<double> out(static_cast<size_t>(n_features) * 5, 0.0);
    if (n_samples < total_needed) return out;

    int start_offset = n_samples - total_needed;
    std::vector<std::pair<int, int>> windows(n_windows);
    for (int w = 0; w < n_windows; ++w) {
        int ws = start_offset + w * ic_window;
        windows[w] = {ws, ws + ic_window};
    }

    std::vector<double> xbuf, ybuf;
    xbuf.reserve(ic_window);
    ybuf.reserve(ic_window);
    std::vector<double> ics(n_windows);

    for (int j = 0; j < n_features; ++j) {
        for (int w = 0; w < n_windows; ++w) {
            auto r = compute_ic_masked(X, n_samples, n_features, j, y,
                                        windows[w].first, windows[w].second,
                                        true, xbuf, ybuf);
            ics[w] = r.valid ? r.ic : 0.0;
        }

        double sum = 0, sum_abs = 0, sumsq = 0;
        int n_pos = 0, max_neg = 0, cur_neg = 0;
        for (int w = 0; w < n_windows; ++w) {
            sum += ics[w];
            sum_abs += std::abs(ics[w]);
            sumsq += ics[w] * ics[w];
            if (ics[w] > 0) ++n_pos;
            if (ics[w] < 0) { ++cur_neg; if (cur_neg > max_neg) max_neg = cur_neg; }
            else cur_neg = 0;
        }

        double mean_ic = sum / n_windows;
        double mean_abs = sum_abs / n_windows;
        double ic_std = 0.0;
        if (n_windows > 1) {
            double var = (sumsq - n_windows * mean_ic * mean_ic) / (n_windows - 1);
            ic_std = (var > 0) ? std::sqrt(var) : 0.0;
        }
        double icir;
        if (ic_std < 1e-12) {
            icir = (mean_abs > 0) ? mean_abs * 100.0 : 0.0;
        } else {
            icir = mean_abs / ic_std;
        }

        size_t base = static_cast<size_t>(j) * 5;
        out[base + 0] = mean_ic;
        out[base + 1] = ic_std;
        out[base + 2] = icir;
        out[base + 3] = static_cast<double>(max_neg);
        out[base + 4] = static_cast<double>(n_pos) / n_windows;
    }

    return out;
}

} // namespace feat_sel
