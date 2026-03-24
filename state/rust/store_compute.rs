//! Free functions for portfolio/risk computation (reused by tick_processor and store).

use std::collections::HashMap;

use crate::state::fixed_decimal::Fd8;
use crate::state::market_state::RustMarketState;
use crate::state::position_state::RustPositionState;
use crate::state::account_state::RustAccountState;
use crate::state::portfolio_state::RustPortfolioState;
use crate::state::risk_state::{RustRiskLimits, RustRiskState};

pub fn compute_portfolio_from(
    markets: &HashMap<String, RustMarketState>,
    positions: &HashMap<String, RustPositionState>,
    account: &RustAccountState,
    last_ts: &Option<String>,
) -> RustPortfolioState {
    let balance = Fd8::from_raw(account.balance);
    let margin_used = Fd8::from_raw(account.margin_used);
    let margin_available = Fd8::from_raw(account.margin_available);
    let realized_pnl = Fd8::from_raw(account.realized_pnl);
    let fees_paid = Fd8::from_raw(account.fees_paid);

    let mut gross_exposure = Fd8::ZERO;
    let mut net_exposure = Fd8::ZERO;
    let mut total_unrealized = Fd8::ZERO;
    let mut symbols: Vec<String> = Vec::new();

    for (sym, pos) in positions {
        let qty = Fd8::from_raw(pos.qty);
        if qty.is_zero() {
            continue;
        }
        symbols.push(sym.clone());

        let mark = pos
            .last_price
            .map(Fd8::from_raw)
            .or_else(|| {
                markets
                    .get(sym)
                    .and_then(|m| m.last_price.map(Fd8::from_raw))
            })
            .or_else(|| pos.avg_price.map(Fd8::from_raw))
            .unwrap_or(Fd8::ZERO);

        let notional = qty.abs() * mark;
        gross_exposure = gross_exposure + notional;
        net_exposure = net_exposure + qty * mark;

        if let Some(avg_raw) = pos.avg_price {
            let avg = Fd8::from_raw(avg_raw);
            total_unrealized = total_unrealized + qty * (mark - avg);
        }
    }

    symbols.sort();

    let total_equity = balance + total_unrealized;
    let te_f = total_equity.to_f64();
    let ge_f = gross_exposure.to_f64();
    let mu_f = margin_used.to_f64();

    let leverage = if te_f > 0.0 && ge_f != 0.0 {
        Some(Fd8::from_f64(ge_f / te_f).to_string_stripped())
    } else if te_f > 0.0 {
        Some("0".to_string())
    } else {
        None
    };

    let margin_ratio = if mu_f > 0.0 && te_f > 0.0 {
        Some(Fd8::from_f64(te_f / mu_f).to_string_stripped())
    } else {
        None
    };

    RustPortfolioState {
        total_equity: total_equity.to_string_stripped(),
        cash_balance: balance.to_string_stripped(),
        realized_pnl: realized_pnl.to_string_stripped(),
        unrealized_pnl: total_unrealized.to_string_stripped(),
        fees_paid: fees_paid.to_string_stripped(),
        gross_exposure: gross_exposure.to_string_stripped(),
        net_exposure: net_exposure.to_string_stripped(),
        leverage,
        margin_used: margin_used.to_string_stripped(),
        margin_available: margin_available.to_string_stripped(),
        margin_ratio,
        symbols,
        last_ts: last_ts.clone().or_else(|| account.last_ts.clone()),
    }
}

pub fn compute_risk_from(
    portfolio: &RustPortfolioState,
    risk_limits: &RustRiskLimits,
    positions: &HashMap<String, RustPositionState>,
    prev_risk: &RustRiskState,
    last_ts: &Option<String>,
) -> RustRiskState {
    let equity: f64 = portfolio.total_equity.parse().unwrap_or(0.0);
    let max_leverage: f64 = risk_limits.max_leverage.parse().unwrap_or(5.0);
    let max_drawdown_pct: f64 = risk_limits.max_drawdown_pct.parse().unwrap_or(0.30);
    let max_position_notional: Option<f64> = risk_limits
        .max_position_notional
        .as_ref()
        .and_then(|s| s.parse().ok());

    let mut peak: f64 = prev_risk.equity_peak.parse().unwrap_or(0.0);
    if equity > peak {
        peak = equity;
    }

    let mut dd: f64 = 0.0;
    if peak > 0.0 {
        dd = (peak - equity) / peak;
        if dd < 0.0 {
            dd = 0.0;
        }
    }

    let halted = prev_risk.halted;
    let level = prev_risk.level.clone();
    let message = prev_risk.message.clone();
    let mut flags = prev_risk.flags.clone();

    if risk_limits.block_on_equity_le_zero && equity <= 0.0
        && !flags.iter().any(|f| f == "equity_le_zero")
    {
        flags.push("equity_le_zero".to_string());
    }

    let lev: Option<f64> = portfolio.leverage.as_ref().and_then(|s| s.parse().ok());
    if let Some(l) = lev {
        if l > max_leverage && !flags.iter().any(|f| f == "max_leverage") {
            flags.push("max_leverage".to_string());
        }
    }

    if let Some(cap) = max_position_notional {
        let mut found = None;
        for (_sym, pos) in positions {
            let qty = Fd8::from_raw(pos.qty);
            if qty.is_zero() {
                continue;
            }
            if let Some(lp_raw) = pos.last_price {
                let mark = Fd8::from_raw(lp_raw);
                let notional = (qty.abs() * mark).to_f64();
                if notional > cap {
                    found = Some(format!("max_position_notional:{}", pos.symbol));
                    break;
                }
            }
        }
        if let Some(flag) = found {
            if !flags.iter().any(|f| f == &flag) {
                flags.push(flag);
            }
        }
    }

    if dd > max_drawdown_pct && !flags.iter().any(|f| f == "max_drawdown") {
        flags.push("max_drawdown".to_string());
    }

    flags.sort();

    let blocked = halted
        || (!flags.is_empty()
            && (flags.iter().any(|f| f == "risk_block_event")
                || flags.iter().any(|f| f == "equity_le_zero")
                || flags.iter().any(|f| f == "max_drawdown")
                || flags.iter().any(|f| f == "max_leverage")
                || flags.iter().any(|f| f.starts_with("max_position_notional"))));

    RustRiskState {
        blocked,
        halted,
        level,
        message,
        flags,
        equity_peak: Fd8::from_f64(peak).to_string_stripped(),
        drawdown_pct: Fd8::from_f64(dd).to_string_stripped(),
        last_ts: last_ts.clone().or_else(|| prev_risk.last_ts.clone()),
    }
}
