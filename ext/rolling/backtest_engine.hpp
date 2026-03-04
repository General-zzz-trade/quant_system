// backtest_engine.hpp — C++ accelerated backtest pipeline
// Components: pred_to_signal, regime_switch, cost_model, trade_sim, metrics
// Replaces Python hot path in backtest_alpha_v8.py / walkforward_validate.py
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace backtest {

// ── Configuration ────────────────────────────────────────────

struct BacktestConfig {
    // Signal generation
    double deadzone = 0.5;
    int min_hold = 24;
    int zscore_window = 720;
    int zscore_warmup = 168;

    // Regime switch
    bool use_regime_switch = false;
    int ma_window = 480;
    // bear_thresholds: pairs of (prob_threshold, score), max 8
    double bear_thresh_prob[8] = {};
    double bear_thresh_score[8] = {};
    int n_bear_thresholds = 0;

    // Vol-adaptive sizing
    bool vol_adaptive = false;
    double vol_target = 0.0;

    // DD circuit breaker
    bool dd_breaker = false;
    double dd_limit = -0.15;
    int dd_cooldown = 48;

    // Monthly gate (simple: zero signal when close <= SMA)
    bool monthly_gate = false;

    // Long only
    bool long_only = false;

    // Cost model
    bool realistic_cost = false;
    double cost_per_trade = 6e-4;
    // Realistic cost params
    double maker_fee_bps = 2.0;
    double taker_fee_bps = 4.0;
    double taker_ratio = 0.7;
    double impact_eta = 0.5;
    double spread_multiplier = 0.05;
    double max_participation = 0.10;
    double capital = 10000.0;
};

// ── Simple JSON config parser (no external deps) ─────────────

namespace detail {

inline void skip_ws(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
}

inline bool match_str(const char*& p, const char* end, const char* target) {
    size_t len = std::strlen(target);
    if (p + len > end) return false;
    if (std::strncmp(p, target, len) == 0) { p += len; return true; }
    return false;
}

inline std::string parse_string(const char*& p, const char* end) {
    if (p >= end || *p != '"') return "";
    ++p;
    std::string s;
    while (p < end && *p != '"') { s += *p; ++p; }
    if (p < end) ++p; // skip closing "
    return s;
}

inline double parse_number(const char*& p, const char* end) {
    char* ep;
    double v = std::strtod(p, &ep);
    p = ep;
    return v;
}

inline bool parse_bool(const char*& p, const char* end) {
    skip_ws(p, end);
    if (match_str(p, end, "true")) return true;
    if (match_str(p, end, "false")) return false;
    // number: 0 = false, else true
    return parse_number(p, end) != 0.0;
}

// Parse array of [prob, score, prob, score, ...] pairs
inline int parse_threshold_array(const char*& p, const char* end,
                                  double* probs, double* scores, int max_pairs) {
    skip_ws(p, end);
    if (p >= end || *p != '[') return 0;
    ++p;
    int n = 0;
    while (p < end && *p != ']' && n < max_pairs) {
        skip_ws(p, end);
        if (*p == '[') ++p; // inner array start
        skip_ws(p, end);
        probs[n] = parse_number(p, end);
        skip_ws(p, end);
        if (p < end && *p == ',') ++p;
        skip_ws(p, end);
        scores[n] = parse_number(p, end);
        skip_ws(p, end);
        if (p < end && *p == ']') ++p; // inner array end
        skip_ws(p, end);
        if (p < end && *p == ',') ++p;
        ++n;
    }
    skip_ws(p, end);
    if (p < end && *p == ']') ++p;
    return n;
}

} // namespace detail

inline BacktestConfig parse_config(const std::string& json) {
    BacktestConfig cfg;
    if (json.empty()) return cfg;
    const char* p = json.c_str();
    const char* end = p + json.size();
    detail::skip_ws(p, end);
    if (p < end && *p == '{') ++p;

    while (p < end && *p != '}') {
        detail::skip_ws(p, end);
        if (p >= end || *p == '}') break;
        std::string key = detail::parse_string(p, end);
        detail::skip_ws(p, end);
        if (p < end && *p == ':') ++p;
        detail::skip_ws(p, end);

        if (key == "deadzone") cfg.deadzone = detail::parse_number(p, end);
        else if (key == "min_hold") cfg.min_hold = (int)detail::parse_number(p, end);
        else if (key == "zscore_window") cfg.zscore_window = (int)detail::parse_number(p, end);
        else if (key == "zscore_warmup") cfg.zscore_warmup = (int)detail::parse_number(p, end);
        else if (key == "use_regime_switch") cfg.use_regime_switch = detail::parse_bool(p, end);
        else if (key == "ma_window") cfg.ma_window = (int)detail::parse_number(p, end);
        else if (key == "bear_thresholds") {
            cfg.n_bear_thresholds = detail::parse_threshold_array(
                p, end, cfg.bear_thresh_prob, cfg.bear_thresh_score, 8);
        }
        else if (key == "vol_adaptive") cfg.vol_adaptive = detail::parse_bool(p, end);
        else if (key == "vol_target") cfg.vol_target = detail::parse_number(p, end);
        else if (key == "dd_breaker") cfg.dd_breaker = detail::parse_bool(p, end);
        else if (key == "dd_limit") cfg.dd_limit = detail::parse_number(p, end);
        else if (key == "dd_cooldown") cfg.dd_cooldown = (int)detail::parse_number(p, end);
        else if (key == "monthly_gate") cfg.monthly_gate = detail::parse_bool(p, end);
        else if (key == "long_only") cfg.long_only = detail::parse_bool(p, end);
        else if (key == "realistic_cost") cfg.realistic_cost = detail::parse_bool(p, end);
        else if (key == "cost_per_trade") cfg.cost_per_trade = detail::parse_number(p, end);
        else if (key == "maker_fee_bps") cfg.maker_fee_bps = detail::parse_number(p, end);
        else if (key == "taker_fee_bps") cfg.taker_fee_bps = detail::parse_number(p, end);
        else if (key == "taker_ratio") cfg.taker_ratio = detail::parse_number(p, end);
        else if (key == "impact_eta") cfg.impact_eta = detail::parse_number(p, end);
        else if (key == "spread_multiplier") cfg.spread_multiplier = detail::parse_number(p, end);
        else if (key == "max_participation") cfg.max_participation = detail::parse_number(p, end);
        else if (key == "capital") cfg.capital = detail::parse_number(p, end);
        else {
            // Skip unknown value
            if (p < end && *p == '"') detail::parse_string(p, end);
            else if (p < end && *p == '[') {
                int depth = 1; ++p;
                while (p < end && depth > 0) {
                    if (*p == '[') ++depth;
                    else if (*p == ']') --depth;
                    ++p;
                }
            } else detail::parse_number(p, end);
        }
        detail::skip_ws(p, end);
        if (p < end && *p == ',') ++p;
    }
    return cfg;
}

// ── Component 1: pred_to_signal ──────────────────────────────

inline void pred_to_signal(
    const double* y_pred, int n,
    double deadzone, int min_hold,
    int zscore_window, int zscore_warmup,
    double* out_signal
) {
    // Step 1: Rolling z-score → raw discrete signal
    std::vector<double> buf(zscore_window);
    int buf_idx = 0;
    int buf_count = 0;
    std::vector<double> raw(n, 0.0);

    int warmup = std::min(zscore_warmup, zscore_window);

    for (int i = 0; i < n; ++i) {
        buf[buf_idx] = y_pred[i];
        buf_idx = (buf_idx + 1) % zscore_window;
        buf_count = std::min(buf_count + 1, zscore_window);

        if (buf_count < warmup) continue;

        // Compute mean and std of buffer
        int cnt = buf_count;
        double sum = 0.0, sum2 = 0.0;
        for (int j = 0; j < cnt; ++j) {
            double v = buf[j];
            sum += v;
            sum2 += v * v;
        }
        double mu = sum / cnt;
        // Population std (matching np.std without ddof)
        double var = sum2 / cnt - mu * mu;
        if (var < 0.0) var = 0.0;
        double std_val = std::sqrt(var);

        if (std_val < 1e-12) continue;

        double z = (y_pred[i] - mu) / std_val;
        if (z > deadzone) raw[i] = 1.0;
        else if (z < -deadzone) raw[i] = -1.0;
    }

    // Step 2: Min-hold enforcement
    out_signal[0] = raw[0];
    int hold_count = 1;
    for (int i = 1; i < n; ++i) {
        if (hold_count < min_hold) {
            out_signal[i] = out_signal[i - 1];
            ++hold_count;
        } else {
            out_signal[i] = raw[i];
            if (raw[i] != out_signal[i - 1]) {
                hold_count = 1;
            } else {
                ++hold_count;
            }
        }
    }
}

// ── Component 2: compute_bear_mask (SMA via cumsum) ──────────

inline void compute_bear_mask(
    const double* closes, int n, int ma_window,
    char* out_mask
) {
    if (n < ma_window) {
        std::fill(out_mask, out_mask + n, (char)1);
        return;
    }

    // Cumsum
    std::vector<double> cs(n);
    cs[0] = closes[0];
    for (int i = 1; i < n; ++i) cs[i] = cs[i - 1] + closes[i];

    // First ma_window bars: bear (conservative)
    for (int i = 0; i < ma_window; ++i) out_mask[i] = 1;

    // SMA from cumsum
    for (int i = ma_window; i < n; ++i) {
        double ma = (cs[i] - cs[i - ma_window]) / ma_window;
        out_mask[i] = (closes[i] <= ma) ? 1 : 0;
    }
}

// ── Component 2b: prob_to_score ──────────────────────────────

inline double prob_to_score(
    double prob,
    const double* thresh_probs, const double* thresh_scores, int n_thresh
) {
    if (n_thresh == 0) {
        return (prob > 0.5) ? -1.0 : 0.0;
    }
    for (int j = 0; j < n_thresh; ++j) {
        if (prob > thresh_probs[j]) return thresh_scores[j];
    }
    return 0.0;
}

// ── Component 2c: apply_dd_breaker ───────────────────────────

inline void apply_dd_breaker(
    double* signal, const double* closes, int n,
    double dd_limit, int cooldown
) {
    int n_trade = std::min(n, n);  // signal length
    double equity = 1.0;
    double peak = 1.0;
    int cool_remaining = 0;

    for (int i = 0; i < n_trade; ++i) {
        if (cool_remaining > 0) {
            signal[i] = 0.0;
            --cool_remaining;
            continue;
        }

        if (i < n - 1) {
            double ret = (closes[i + 1] - closes[i]) / closes[i];
            equity *= (1.0 + signal[i] * ret);
        }
        if (equity > peak) peak = equity;
        double dd = (equity - peak) / peak;

        if (dd < dd_limit) {
            cool_remaining = cooldown;
            signal[i] = 0.0;
        }
    }
}

// ── Component 2d: apply_regime_switch (full) ─────────────────

inline void apply_regime_switch(
    double* signal, int n,
    const double* closes,
    const double* bear_probs,       // bear model predictions (may be nullptr)
    const double* vol_values,       // realized vol for vol-adaptive (may be nullptr)
    const BacktestConfig& cfg
) {
    // Compute bear mask (use char instead of bool for .data() compatibility)
    std::vector<char> bear_mask(n);
    compute_bear_mask(closes, n, cfg.ma_window, bear_mask.data());

    if (cfg.monthly_gate && bear_probs == nullptr) {
        // Simple monthly gate: zero signal in bear regime
        for (int i = 0; i < n; ++i) {
            if (bear_mask[i]) signal[i] = 0.0;
        }
    }

    if (bear_probs != nullptr) {
        // Regime switch: replace signal in bear regime with bear model score
        for (int i = 0; i < n; ++i) {
            if (bear_mask[i]) {
                signal[i] = prob_to_score(
                    bear_probs[i],
                    cfg.bear_thresh_prob, cfg.bear_thresh_score, cfg.n_bear_thresholds);
            }
        }
    }

    // Long only: clip negative signals
    if (cfg.long_only) {
        for (int i = 0; i < n; ++i) {
            if (signal[i] < 0.0) signal[i] = 0.0;
        }
    }

    // Vol-adaptive sizing
    if (cfg.vol_adaptive && vol_values != nullptr) {
        for (int i = 0; i < n; ++i) {
            if (signal[i] != 0.0 && !std::isnan(vol_values[i]) && vol_values[i] > 1e-8) {
                double scale = std::min(cfg.vol_target / vol_values[i], 1.0);
                signal[i] *= scale;
            }
        }
    }

    // Re-apply min_hold across regime switches (only when bear model is used)
    if (cfg.min_hold > 1 && bear_probs != nullptr) {
        std::vector<double> held(n);
        held[0] = signal[0];
        int hold_count = 1;
        for (int i = 1; i < n; ++i) {
            if (hold_count < cfg.min_hold) {
                held[i] = held[i - 1];
                ++hold_count;
            } else {
                held[i] = signal[i];
                if (signal[i] != held[i - 1]) {
                    hold_count = 1;
                } else {
                    ++hold_count;
                }
            }
        }
        std::memcpy(signal, held.data(), n * sizeof(double));
    }

    // DD circuit breaker
    if (cfg.dd_breaker) {
        apply_dd_breaker(signal, closes, n, cfg.dd_limit, cfg.dd_cooldown);
    }
}

// ── Component 3: compute_costs ───────────────────────────────

struct CostResult {
    std::vector<double> total_cost;
    std::vector<double> fee_cost;
    std::vector<double> impact_cost;
    std::vector<double> spread_cost;
    std::vector<double> clipped_signal;
};

inline CostResult compute_costs_flat(
    const double* signal, int n,
    double cost_per_trade
) {
    CostResult r;
    r.total_cost.resize(n);
    r.clipped_signal.assign(signal, signal + n);

    // turnover = |diff(signal, prepend=0)|
    double prev = 0.0;
    for (int i = 0; i < n; ++i) {
        double turnover = std::abs(signal[i] - prev);
        r.total_cost[i] = turnover * cost_per_trade;
        prev = signal[i];
    }
    return r;
}

inline CostResult compute_costs_realistic(
    const double* signal, int n,
    const double* closes,
    const double* volumes,       // bar volumes (base asset)
    const double* volatility,    // per-bar realized vol
    const BacktestConfig& cfg
) {
    CostResult r;
    r.fee_cost.resize(n, 0.0);
    r.impact_cost.resize(n, 0.0);
    r.spread_cost.resize(n, 0.0);
    r.total_cost.resize(n, 0.0);
    r.clipped_signal.resize(n);

    double notional = cfg.capital;

    // Step 1: Compute raw turnover and max position change
    std::vector<double> turnover_raw(n);
    std::vector<double> max_pos_change(n);
    double prev_sig = 0.0;
    bool has_excess = false;

    for (int i = 0; i < n; ++i) {
        turnover_raw[i] = std::abs(signal[i] - prev_sig);
        prev_sig = signal[i];

        if (closes[i] > 0.0) {
            max_pos_change[i] = (volumes[i] * cfg.max_participation * closes[i]) / notional;
        } else {
            max_pos_change[i] = 1e30;
        }
        if (turnover_raw[i] > max_pos_change[i]) has_excess = true;
    }

    // Step 2: Volume participation clipping
    if (has_excess) {
        r.clipped_signal[0] = std::max(-max_pos_change[0], std::min(signal[0], max_pos_change[0]));
        for (int i = 1; i < n; ++i) {
            double delta = signal[i] - r.clipped_signal[i - 1];
            double md = max_pos_change[i];
            double delta_clipped = std::max(-md, std::min(delta, md));
            r.clipped_signal[i] = r.clipped_signal[i - 1] + delta_clipped;
        }
    } else {
        std::memcpy(r.clipped_signal.data(), signal, n * sizeof(double));
    }

    // Recompute turnover from clipped signal
    std::vector<double> turnover(n);
    double prev_c = 0.0;
    for (int i = 0; i < n; ++i) {
        turnover[i] = std::abs(r.clipped_signal[i] - prev_c);
        prev_c = r.clipped_signal[i];
    }

    // Step 3: Trading fees
    double blended_fee = (cfg.taker_ratio * cfg.taker_fee_bps +
                          (1.0 - cfg.taker_ratio) * cfg.maker_fee_bps) / 10000.0;
    for (int i = 0; i < n; ++i) {
        r.fee_cost[i] = turnover[i] * blended_fee;
    }

    // Step 4: Market impact (Almgren-Chriss sqrt)
    for (int i = 0; i < n; ++i) {
        double safe_vol_notional = std::max(volumes[i] * closes[i], 1.0);
        double participation = (turnover[i] * notional) / safe_vol_notional;
        double vol_val = volatility[i];
        double sigma_daily = 0.0;
        if (!std::isnan(vol_val) && vol_val > 0.0) {
            sigma_daily = vol_val * std::sqrt(24.0);
        }
        r.impact_cost[i] = cfg.impact_eta * sigma_daily * std::sqrt(std::max(participation, 0.0));
    }

    // Step 5: Bid-ask spread
    for (int i = 0; i < n; ++i) {
        double vol_val = volatility[i];
        double spread_bps = std::isnan(vol_val) ? 0.0 : cfg.spread_multiplier * vol_val;
        r.spread_cost[i] = turnover[i] * spread_bps / 2.0;
    }

    // Total
    for (int i = 0; i < n; ++i) {
        r.total_cost[i] = r.fee_cost[i] + r.impact_cost[i] + r.spread_cost[i];
    }

    return r;
}

// ── Component 4: simulate_trades ─────────────────────────────

struct TradeResult {
    std::vector<double> gross_pnl;
    std::vector<double> net_pnl;
    std::vector<double> funding_cost;
    std::vector<double> equity;
};

inline TradeResult simulate_trades(
    const double* signal, int n_signal,
    const double* closes, int n_closes,
    const double* cost, // per-bar cost
    const double* funding_rates,    // may be nullptr
    const int64_t* funding_ts,      // may be nullptr
    int n_funding,
    const int64_t* bar_timestamps,  // may be nullptr
    double initial_capital
) {
    int n_trade = std::min(n_signal, n_closes - 1);
    TradeResult r;
    r.gross_pnl.resize(n_trade, 0.0);
    r.net_pnl.resize(n_trade, 0.0);
    r.funding_cost.resize(n_trade, 0.0);
    r.equity.resize(n_trade + 1);
    r.equity[0] = initial_capital;

    // Funding: forward-scan merge (ScheduleCursor pattern)
    int f_idx = 0;
    double current_rate = 0.0;

    for (int i = 0; i < n_trade; ++i) {
        // Update funding rate
        if (funding_rates != nullptr && funding_ts != nullptr && bar_timestamps != nullptr) {
            int64_t ts = bar_timestamps[i];
            while (f_idx < n_funding && funding_ts[f_idx] <= ts) {
                current_rate = funding_rates[f_idx];
                ++f_idx;
            }
            if (signal[i] != 0.0) {
                r.funding_cost[i] = signal[i] * current_rate / 8.0;
            }
        }

        double ret = (closes[i + 1] - closes[i]) / closes[i];
        r.gross_pnl[i] = signal[i] * ret;
        r.net_pnl[i] = r.gross_pnl[i] - cost[i] - r.funding_cost[i];
        r.equity[i + 1] = r.equity[i] * (1.0 + r.net_pnl[i]);
    }

    return r;
}

// ── Component 5: compute_metrics ─────────────────────────────

struct MonthlyStats {
    int year;
    int month;
    double total_return;
    double sharpe;
    double active_pct;
    int bars;
};

struct BacktestMetrics {
    double sharpe;
    double max_drawdown;
    double total_return;
    double annual_return;
    double win_rate;
    double profit_factor;
    int n_trades;
    double avg_holding;
    double total_turnover;
    double total_cost;
    int n_active;
    std::vector<MonthlyStats> monthly;
};

inline BacktestMetrics compute_metrics(
    const double* signal, int n_signal,
    const double* net_pnl, int n_pnl,
    const double* equity, int n_equity,
    const int64_t* timestamps, int n_ts   // millisecond timestamps
) {
    BacktestMetrics m;

    // Active bars
    int n_active = 0;
    for (int i = 0; i < n_signal; ++i) {
        if (signal[i] != 0.0) ++n_active;
    }
    m.n_active = n_active;

    // Sharpe (annualized, sqrt(8760), active bars only, ddof=1)
    m.sharpe = 0.0;
    if (n_active > 1) {
        double sum = 0.0, sum2 = 0.0;
        int cnt = 0;
        for (int i = 0; i < n_pnl; ++i) {
            if (i < n_signal && signal[i] != 0.0) {
                sum += net_pnl[i];
                sum2 += net_pnl[i] * net_pnl[i];
                ++cnt;
            }
        }
        if (cnt > 1) {
            double mean = sum / cnt;
            double var = (sum2 - sum * sum / cnt) / (cnt - 1); // ddof=1
            if (var > 0.0) {
                m.sharpe = mean / std::sqrt(var) * std::sqrt(8760.0);
            }
        }
    }

    // Max drawdown
    m.max_drawdown = 0.0;
    {
        double peak = equity[0];
        for (int i = 0; i < n_equity; ++i) {
            if (equity[i] > peak) peak = equity[i];
            double dd = (equity[i] - peak) / peak;
            if (dd < m.max_drawdown) m.max_drawdown = dd;
        }
    }

    // Total return
    m.total_return = (equity[n_equity - 1] / equity[0]) - 1.0;

    // Annual return
    int n_hours = n_pnl;
    if (n_hours > 0) {
        m.annual_return = std::pow(1.0 + m.total_return, 8760.0 / std::max(n_hours, 1)) - 1.0;
    } else {
        m.annual_return = 0.0;
    }

    // Win rate (bar-level, active bars)
    m.win_rate = 0.0;
    if (n_active > 0) {
        int wins = 0;
        for (int i = 0; i < n_pnl; ++i) {
            if (i < n_signal && signal[i] != 0.0 && net_pnl[i] > 0.0) ++wins;
        }
        m.win_rate = (double)wins / n_active;
    }

    // Profit factor
    double gross_wins = 0.0, gross_losses = 0.0;
    for (int i = 0; i < n_pnl; ++i) {
        if (net_pnl[i] > 0.0) gross_wins += net_pnl[i];
        else gross_losses += std::abs(net_pnl[i]);
    }
    m.profit_factor = (gross_losses > 0.0) ? (gross_wins / gross_losses) : 1e30;

    // Trade count (position changes) and turnover
    m.total_turnover = 0.0;
    m.n_trades = 0;
    double prev = 0.0;
    for (int i = 0; i < n_signal; ++i) {
        double tn = std::abs(signal[i] - prev);
        m.total_turnover += tn;
        if (i > 0 && signal[i] != signal[i - 1]) ++m.n_trades;
        prev = signal[i];
    }

    // Total cost (sum of net - gross doesn't work easily; pass via param or recompute)
    // We compute from equity and gross: total_cost = sum(gross_pnl - net_pnl)
    // But we don't have gross_pnl here. Caller sets this from cost array.
    m.total_cost = 0.0;

    // Average holding period
    if (m.n_trades > 0 && n_active > 0) {
        m.avg_holding = (double)n_active / std::max(m.n_trades, 1);
    } else {
        m.avg_holding = 0.0;
    }

    // Monthly breakdown
    if (timestamps != nullptr && n_ts >= n_pnl) {
        // Group by year-month using gmtime_r
        struct MonthKey {
            int year, month;
            bool operator==(const MonthKey& o) const { return year == o.year && month == o.month; }
        };

        std::vector<MonthKey> keys(n_pnl);
        for (int i = 0; i < n_pnl; ++i) {
            time_t sec = (time_t)(timestamps[i] / 1000);
            struct tm tm_buf;
            gmtime_r(&sec, &tm_buf);
            keys[i] = {tm_buf.tm_year + 1900, tm_buf.tm_mon + 1};
        }

        int start = 0;
        while (start < n_pnl) {
            MonthKey mk = keys[start];
            int end = start;
            while (end < n_pnl && keys[end] == mk) ++end;

            int bars = end - start;
            if (bars < 10) { start = end; continue; }

            double m_sum = 0.0;
            int m_active_count = 0;
            double m_pnl_sum = 0.0, m_pnl_sum2 = 0.0;
            int m_active_pnl_count = 0;

            for (int i = start; i < end; ++i) {
                m_sum += net_pnl[i];
                if (i < n_signal && signal[i] != 0.0) {
                    ++m_active_count;
                    m_pnl_sum += net_pnl[i];
                    m_pnl_sum2 += net_pnl[i] * net_pnl[i];
                    ++m_active_pnl_count;
                }
            }

            double m_sharpe = 0.0;
            if (m_active_pnl_count > 1) {
                double mean = m_pnl_sum / m_active_pnl_count;
                double var = (m_pnl_sum2 - m_pnl_sum * m_pnl_sum / m_active_pnl_count) / (m_active_pnl_count - 1);
                if (var > 0.0) {
                    m_sharpe = mean / std::sqrt(var) * std::sqrt(8760.0);
                }
            }

            double active_pct = (double)m_active_count / bars * 100.0;
            m.monthly.push_back({mk.year, mk.month, m_sum, m_sharpe, active_pct, bars});

            start = end;
        }
    }

    return m;
}

// ── Main entry: run_backtest ─────────────────────────────────

struct BacktestResult {
    std::vector<double> signal;
    std::vector<double> equity;
    std::vector<double> net_pnl;
    BacktestMetrics metrics;
};

inline BacktestResult run_backtest(
    const int64_t* timestamps, int n_bars,
    const double* closes,
    const double* volumes,        // may be nullptr for flat cost
    const double* vol_20,         // may be nullptr for flat cost
    const double* y_pred, int n_pred,
    const double* bear_probs,     // may be nullptr
    const double* vol_values,     // vol for vol-adaptive (may be nullptr)
    const double* funding_rates,  // may be nullptr
    const int64_t* funding_ts,    // may be nullptr
    int n_funding,
    const BacktestConfig& cfg
) {
    BacktestResult result;
    int n = n_pred;  // signal length = prediction length

    // Step 1: pred_to_signal
    result.signal.resize(n);
    pred_to_signal(y_pred, n, cfg.deadzone, cfg.min_hold,
                   cfg.zscore_window, cfg.zscore_warmup, result.signal.data());

    // Step 2: Regime switch / monthly gate / vol-adaptive / DD breaker
    if (cfg.use_regime_switch || cfg.monthly_gate || cfg.vol_adaptive || cfg.dd_breaker || cfg.long_only) {
        apply_regime_switch(result.signal.data(), n, closes, bear_probs, vol_values, cfg);
    }

    // Step 3: Cost computation
    CostResult cost_result;
    const double* trade_signal;
    if (cfg.realistic_cost && volumes != nullptr && vol_20 != nullptr) {
        cost_result = compute_costs_realistic(
            result.signal.data(), n, closes, volumes, vol_20, cfg);
        // Update signal with clipped version
        std::memcpy(result.signal.data(), cost_result.clipped_signal.data(), n * sizeof(double));
        trade_signal = result.signal.data();
    } else {
        cost_result = compute_costs_flat(result.signal.data(), n, cfg.cost_per_trade);
        trade_signal = result.signal.data();
    }

    // Step 4: Trade simulation
    TradeResult trade = simulate_trades(
        trade_signal, n,
        closes, n_bars,
        cost_result.total_cost.data(),
        funding_rates, funding_ts, n_funding,
        timestamps,
        cfg.capital
    );

    result.net_pnl = std::move(trade.net_pnl);
    result.equity = std::move(trade.equity);

    // Step 5: Metrics
    result.metrics = compute_metrics(
        result.signal.data(), n,
        result.net_pnl.data(), (int)result.net_pnl.size(),
        result.equity.data(), (int)result.equity.size(),
        timestamps, n_bars
    );

    // Set total_cost from cost_result
    double tc = 0.0;
    for (int i = 0; i < (int)cost_result.total_cost.size(); ++i) tc += cost_result.total_cost[i];
    result.metrics.total_cost = tc;

    return result;
}

// ── pybind11 entry points ────────────────────────────────────

using NpArr = py::array_t<double, py::array::c_style | py::array::forcecast>;
using NpArrI64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

inline py::dict cpp_run_backtest(
    NpArrI64 timestamps,
    NpArr closes,
    NpArr volumes,
    NpArr vol_20,
    NpArr y_pred,
    NpArr bear_probs,
    NpArr vol_values,
    NpArr funding_rates,
    NpArrI64 funding_ts,
    const std::string& config_json
) {
    auto ts_buf = timestamps.request();
    auto cl_buf = closes.request();
    auto vo_buf = volumes.request();
    auto v20_buf = vol_20.request();
    auto yp_buf = y_pred.request();
    auto bp_buf = bear_probs.request();
    auto vv_buf = vol_values.request();
    auto fr_buf = funding_rates.request();
    auto ft_buf = funding_ts.request();

    int n_bars = (int)cl_buf.shape[0];
    int n_pred = (int)yp_buf.shape[0];
    int n_funding = (int)fr_buf.shape[0];

    const int64_t* ts_ptr = static_cast<const int64_t*>(ts_buf.ptr);
    const double* cl_ptr = static_cast<const double*>(cl_buf.ptr);
    const double* vo_ptr = vo_buf.shape[0] > 0 ? static_cast<const double*>(vo_buf.ptr) : nullptr;
    const double* v20_ptr = v20_buf.shape[0] > 0 ? static_cast<const double*>(v20_buf.ptr) : nullptr;
    const double* yp_ptr = static_cast<const double*>(yp_buf.ptr);
    const double* bp_ptr = bp_buf.shape[0] > 0 ? static_cast<const double*>(bp_buf.ptr) : nullptr;
    const double* vv_ptr = vv_buf.shape[0] > 0 ? static_cast<const double*>(vv_buf.ptr) : nullptr;
    const double* fr_ptr = n_funding > 0 ? static_cast<const double*>(fr_buf.ptr) : nullptr;
    const int64_t* ft_ptr = n_funding > 0 ? static_cast<const int64_t*>(ft_buf.ptr) : nullptr;

    BacktestConfig cfg = parse_config(config_json);

    BacktestResult res = run_backtest(
        ts_ptr, n_bars, cl_ptr, vo_ptr, v20_ptr,
        yp_ptr, n_pred, bp_ptr, vv_ptr,
        fr_ptr, ft_ptr, n_funding, cfg
    );

    // Build result dict with numpy arrays
    int n_sig = (int)res.signal.size();
    int n_eq = (int)res.equity.size();
    int n_pnl = (int)res.net_pnl.size();

    auto sig_arr = py::array_t<double>(n_sig);
    auto eq_arr = py::array_t<double>(n_eq);
    auto pnl_arr = py::array_t<double>(n_pnl);

    std::memcpy(sig_arr.mutable_data(), res.signal.data(), n_sig * sizeof(double));
    std::memcpy(eq_arr.mutable_data(), res.equity.data(), n_eq * sizeof(double));
    std::memcpy(pnl_arr.mutable_data(), res.net_pnl.data(), n_pnl * sizeof(double));

    py::dict result;
    result["signal"] = sig_arr;
    result["equity"] = eq_arr;
    result["net_pnl"] = pnl_arr;
    result["sharpe"] = res.metrics.sharpe;
    result["max_drawdown"] = res.metrics.max_drawdown;
    result["total_return"] = res.metrics.total_return;
    result["annual_return"] = res.metrics.annual_return;
    result["win_rate"] = res.metrics.win_rate;
    result["profit_factor"] = res.metrics.profit_factor;
    result["n_trades"] = res.metrics.n_trades;
    result["avg_holding"] = res.metrics.avg_holding;
    result["total_turnover"] = res.metrics.total_turnover;
    result["total_cost"] = res.metrics.total_cost;
    result["n_active"] = res.metrics.n_active;

    // Monthly breakdown as list of dicts
    py::list monthly_list;
    for (const auto& ms : res.metrics.monthly) {
        py::dict md;
        char month_str[8];
        std::snprintf(month_str, sizeof(month_str), "%04d-%02d", ms.year, ms.month);
        md["month"] = std::string(month_str);
        md["return"] = ms.total_return;
        md["sharpe"] = ms.sharpe;
        md["active_pct"] = ms.active_pct;
        md["bars"] = ms.bars;
        monthly_list.append(md);
    }
    result["monthly"] = monthly_list;

    return result;
}

inline py::array_t<double> cpp_pred_to_signal(
    NpArr y_pred,
    double deadzone,
    int min_hold,
    int zscore_window,
    int zscore_warmup
) {
    auto buf = y_pred.request();
    int n = (int)buf.shape[0];
    const double* ptr = static_cast<const double*>(buf.ptr);

    auto result = py::array_t<double>(n);
    pred_to_signal(ptr, n, deadzone, min_hold, zscore_window, zscore_warmup,
                   result.mutable_data());
    return result;
}

} // namespace backtest
