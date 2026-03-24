use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashSet;

use crate::state::fixed_decimal::Fd8;
use crate::state::position_state::RustPositionState;
use crate::state::portfolio_state::RustPortfolioState;
use crate::state::risk_state::{RustRiskLimits, RustRiskState};
use crate::state::reducer_result::RustReducerResult;
use crate::state::reducer_helpers::{get_event_type, get_event_ts, ts_to_opt_string};

// ===========================================================================
// RustRiskReducer
// ===========================================================================

#[pyclass(name = "RustRiskReducer")]
pub struct RustRiskReducer {
    limits: RustRiskLimits,
    get_portfolio: PyObject,
    get_positions: PyObject,
}

#[pymethods]
impl RustRiskReducer {
    #[new]
    #[pyo3(signature = (*, limits, get_portfolio, get_positions))]
    fn new(limits: RustRiskLimits, get_portfolio: PyObject, get_positions: PyObject) -> Self {
        Self {
            limits,
            get_portfolio,
            get_positions,
        }
    }

    fn reduce(
        &self,
        py: Python<'_>,
        state: &RustRiskState,
        event: &Bound<'_, PyAny>,
    ) -> PyResult<RustReducerResult> {
        let et = get_event_type(event)?;
        let ts = get_event_ts(event)?;
        let ts_str = ts_to_opt_string(py, &ts)?;

        let portfolio_obj = self.get_portfolio.call0(py)?;
        let portfolio: RustPortfolioState = portfolio_obj.extract(py)?;
        let equity: f64 = portfolio.total_equity.parse().unwrap_or(0.0);

        let max_leverage: f64 = self.limits.max_leverage.parse().unwrap_or(5.0);
        let max_drawdown_pct: f64 = self.limits.max_drawdown_pct.parse().unwrap_or(0.30);
        let max_position_notional: Option<f64> = self
            .limits
            .max_position_notional
            .as_ref()
            .and_then(|s| s.parse().ok());

        let mut peak: f64 = state.equity_peak.parse().unwrap_or(0.0);
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

        let mut halted = state.halted;
        let mut level = state.level.clone();
        let mut message = state.message.clone();
        let mut flags: HashSet<String> = state.flags.iter().cloned().collect();

        if et == "risk" {
            if let Ok(lvl) = event.getattr("level") {
                if !lvl.is_none() {
                    let l = lvl.str()?.to_string().trim().to_lowercase();
                    level = Some(l.clone());
                    if l == "block" {
                        flags.insert("risk_block_event".to_string());
                    }
                }
            }
            if let Ok(msg) = event.getattr("message") {
                if !msg.is_none() {
                    message = Some(msg.str()?.to_string());
                }
            }
        } else if et == "control" {
            let cmd = event
                .getattr("command")
                .ok()
                .filter(|v| !v.is_none())
                .or_else(|| event.getattr("action").ok().filter(|v| !v.is_none()));

            if let Some(c) = cmd {
                let cmd_s = c.str()?.to_string().trim().to_lowercase();
                match cmd_s.as_str() {
                    "halt" | "pause" | "stop" => {
                        halted = true;
                        flags.insert("manual_halt".to_string());
                    }
                    "resume" | "unpause" | "start" => {
                        halted = false;
                        flags.remove("manual_halt");
                    }
                    _ => {}
                }
            }
        }

        if self.limits.block_on_equity_le_zero && equity <= 0.0 {
            flags.insert("equity_le_zero".to_string());
        }

        let lev: Option<f64> = portfolio.leverage.as_ref().and_then(|s| s.parse().ok());
        if let Some(l) = lev {
            if l > max_leverage {
                flags.insert("max_leverage".to_string());
            }
        }

        if let Some(cap) = max_position_notional {
            let positions_obj = self.get_positions.call0(py)?;
            let positions: &Bound<'_, PyDict> = positions_obj.downcast_bound(py)?;
            for (_key, val) in positions.iter() {
                let pos: RustPositionState = val.extract()?;
                let qty = Fd8::from_raw(pos.qty);
                if qty.is_zero() {
                    continue;
                }
                if let Some(lp_raw) = pos.last_price {
                    let mark = Fd8::from_raw(lp_raw);
                    let notional = (qty.abs() * mark).to_f64();
                    if notional > cap {
                        let sym = &pos.symbol;
                        flags.insert(format!("max_position_notional:{}", sym));
                        break;
                    }
                }
            }
        }

        if dd > max_drawdown_pct {
            flags.insert("max_drawdown".to_string());
        }

        let blocked = halted
            || (!flags.is_empty()
                && (flags.contains("risk_block_event")
                    || flags.contains("equity_le_zero")
                    || flags.contains("max_drawdown")
                    || flags.contains("max_leverage")
                    || flags.iter().any(|f| f.starts_with("max_position_notional"))));

        let mut new_flags: Vec<String> = flags.into_iter().collect();
        new_flags.sort();

        let fmt_peak = Fd8::from_f64(peak).to_string_stripped();
        let fmt_dd = Fd8::from_f64(dd).to_string_stripped();

        let new_state = RustRiskState {
            blocked,
            halted,
            level,
            message,
            flags: new_flags.clone(),
            equity_peak: fmt_peak,
            drawdown_pct: fmt_dd,
            last_ts: ts_str.or_else(|| state.last_ts.clone()),
        };

        let changed = new_state.blocked != state.blocked
            || new_state.halted != state.halted
            || new_state.level != state.level
            || new_state.message != state.message
            || new_state.flags != state.flags
            || new_state.equity_peak != state.equity_peak
            || new_state.drawdown_pct != state.drawdown_pct
            || new_state.last_ts != state.last_ts;

        Ok(RustReducerResult {
            state: new_state.into_pyobject(py)?.into_any().unbind(),
            changed,
            note: Some("risk_eval".to_string()),
        })
    }
}
