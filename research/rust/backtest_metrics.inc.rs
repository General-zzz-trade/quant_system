// backtest_metrics.inc.rs — Metrics computation + monthly breakdown.
// Included by backtest.rs via include!() macro.

struct MonthlyStats {
    year: i32,
    month: i32,
    total_return: f64,
    sharpe: f64,
    active_pct: f64,
    bars: i32,
}

struct BacktestMetrics {
    sharpe: f64,
    max_drawdown: f64,
    total_return: f64,
    annual_return: f64,
    win_rate: f64,
    profit_factor: f64,
    n_trades: i32,
    avg_holding: f64,
    total_turnover: f64,
    total_cost: f64,
    n_active: i32,
    monthly: Vec<MonthlyStats>,
}

/// Convert millisecond timestamp to (year, month) via days-since-epoch calculation.
fn ts_to_year_month(ts_ms: i64) -> (i32, i32) {
    let secs = ts_ms / 1000;
    // Days since Unix epoch (1970-01-01)
    // Handle negative timestamps properly
    let mut days = if secs >= 0 {
        (secs / 86400) as i64
    } else {
        // For negative, floor division
        ((secs - 86399) / 86400) as i64
    };

    // Algorithm: civil_from_days (Howard Hinnant)
    // Shift epoch from 1970-01-01 to 0000-03-01
    days += 719468;
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = (days - era * 146097) as u64; // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    (year as i32, m as i32)
}

fn compute_metrics(
    signal: &[f64],
    net_pnl: &[f64],
    equity: &[f64],
    timestamps: Option<&[i64]>,
) -> BacktestMetrics {
    let n_signal = signal.len();
    let n_pnl = net_pnl.len();
    let n_equity = equity.len();

    // Active bars
    let mut n_active: i32 = 0;
    for i in 0..n_signal {
        if signal[i] != 0.0 {
            n_active += 1;
        }
    }

    // Sharpe (annualized, sqrt(8760), active bars only, ddof=1)
    let mut sharpe = 0.0_f64;
    if n_active > 1 {
        let mut sum = 0.0_f64;
        let mut sum2 = 0.0_f64;
        let mut cnt: i32 = 0;
        for i in 0..n_pnl {
            if i < n_signal && signal[i] != 0.0 {
                sum += net_pnl[i];
                sum2 += net_pnl[i] * net_pnl[i];
                cnt += 1;
            }
        }
        if cnt > 1 {
            let mean = sum / cnt as f64;
            let var = (sum2 - sum * sum / cnt as f64) / (cnt - 1) as f64; // ddof=1
            if var > 0.0 {
                sharpe = mean / var.sqrt() * 8760.0_f64.sqrt();
            }
        }
    }

    // Max drawdown
    let mut max_drawdown = 0.0_f64;
    {
        let mut peak = equity[0];
        for i in 0..n_equity {
            if equity[i] > peak {
                peak = equity[i];
            }
            let dd = (equity[i] - peak) / peak;
            if dd < max_drawdown {
                max_drawdown = dd;
            }
        }
    }

    // Total return
    let total_return = (equity[n_equity - 1] / equity[0]) - 1.0;

    // Annual return
    let n_hours = n_pnl as i32;
    let annual_return = if n_hours > 0 {
        (1.0 + total_return).powf(8760.0 / n_hours.max(1) as f64) - 1.0
    } else {
        0.0
    };

    // Win rate (bar-level, active bars)
    let mut win_rate = 0.0_f64;
    if n_active > 0 {
        let mut wins: i32 = 0;
        for i in 0..n_pnl {
            if i < n_signal && signal[i] != 0.0 && net_pnl[i] > 0.0 {
                wins += 1;
            }
        }
        win_rate = wins as f64 / n_active as f64;
    }

    // Profit factor
    let mut gross_wins = 0.0_f64;
    let mut gross_losses = 0.0_f64;
    for i in 0..n_pnl {
        if net_pnl[i] > 0.0 {
            gross_wins += net_pnl[i];
        } else {
            gross_losses += net_pnl[i].abs();
        }
    }
    let profit_factor = if gross_losses > 0.0 {
        gross_wins / gross_losses
    } else {
        1e30
    };

    // Trade count (position changes) and turnover
    let mut total_turnover = 0.0_f64;
    let mut n_trades: i32 = 0;
    let mut prev = 0.0_f64;
    for i in 0..n_signal {
        let tn = (signal[i] - prev).abs();
        total_turnover += tn;
        if i > 0 && signal[i] != signal[i - 1] {
            n_trades += 1;
        }
        prev = signal[i];
    }

    // Total cost — caller sets this from cost array
    let total_cost = 0.0_f64;

    // Average holding period
    let avg_holding = if n_trades > 0 && n_active > 0 {
        n_active as f64 / n_trades.max(1) as f64
    } else {
        0.0
    };

    // Monthly breakdown
    let mut monthly: Vec<MonthlyStats> = Vec::new();
    if let Some(ts) = timestamps {
        if ts.len() >= n_pnl {
            // Group by year-month
            let keys: Vec<(i32, i32)> = (0..n_pnl)
                .map(|i| ts_to_year_month(ts[i]))
                .collect();

            let mut start = 0;
            while start < n_pnl {
                let mk = keys[start];
                let mut end = start;
                while end < n_pnl && keys[end] == mk {
                    end += 1;
                }

                let bars = (end - start) as i32;
                if bars < 10 {
                    start = end;
                    continue;
                }

                let mut m_sum = 0.0_f64;
                let mut m_active_count: i32 = 0;
                let mut m_pnl_sum = 0.0_f64;
                let mut m_pnl_sum2 = 0.0_f64;
                let mut m_active_pnl_count: i32 = 0;

                for i in start..end {
                    m_sum += net_pnl[i];
                    if i < n_signal && signal[i] != 0.0 {
                        m_active_count += 1;
                        m_pnl_sum += net_pnl[i];
                        m_pnl_sum2 += net_pnl[i] * net_pnl[i];
                        m_active_pnl_count += 1;
                    }
                }

                let mut m_sharpe = 0.0_f64;
                if m_active_pnl_count > 1 {
                    let mean = m_pnl_sum / m_active_pnl_count as f64;
                    let var = (m_pnl_sum2 - m_pnl_sum * m_pnl_sum / m_active_pnl_count as f64)
                        / (m_active_pnl_count - 1) as f64;
                    if var > 0.0 {
                        m_sharpe = mean / var.sqrt() * 8760.0_f64.sqrt();
                    }
                }

                let active_pct = m_active_count as f64 / bars as f64 * 100.0;
                monthly.push(MonthlyStats {
                    year: mk.0,
                    month: mk.1,
                    total_return: m_sum,
                    sharpe: m_sharpe,
                    active_pct,
                    bars,
                });

                start = end;
            }
        }
    }

    BacktestMetrics {
        sharpe,
        max_drawdown,
        total_return,
        annual_return,
        win_rate,
        profit_factor,
        n_trades,
        avg_holding,
        total_turnover,
        total_cost,
        n_active,
        monthly,
    }
}
