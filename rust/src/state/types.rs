use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

use crate::state::fixed_decimal::{Fd8, SCALE};

// ---------------------------------------------------------------------------
// Helper: compare two Option<String> for equality (used by Portfolio/Risk)
// ---------------------------------------------------------------------------
fn opt_str_eq(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        (None, None) => true,
        _ => false,
    }
}

fn opt_str_repr(v: &Option<String>) -> String {
    match v {
        Some(s) => format!("'{}'", s),
        None => "None".to_string(),
    }
}

fn opt_i64_repr(v: &Option<i64>) -> String {
    match v {
        Some(raw) => format!("'{}'", Fd8::from_raw(*raw).to_string_stripped()),
        None => "None".to_string(),
    }
}

fn i64_repr(raw: i64) -> String {
    Fd8::from_raw(raw).to_string_stripped()
}

// ===========================================================================
// MarketState — i64 fixed-point (×10^8)
// ===========================================================================
#[pyclass(name = "RustMarketState", frozen)]
#[derive(Clone)]
pub struct RustMarketState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub last_price: Option<i64>,
    #[pyo3(get)]
    pub open: Option<i64>,
    #[pyo3(get)]
    pub high: Option<i64>,
    #[pyo3(get)]
    pub low: Option<i64>,
    #[pyo3(get)]
    pub close: Option<i64>,
    #[pyo3(get)]
    pub volume: Option<i64>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustMarketState {
    #[new]
    #[pyo3(signature = (symbol, last_price=None, open=None, high=None, low=None, close=None, volume=None, last_ts=None))]
    fn new(
        symbol: String,
        last_price: Option<i64>,
        open: Option<i64>,
        high: Option<i64>,
        low: Option<i64>,
        close: Option<i64>,
        volume: Option<i64>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, last_price, open, high, low, close, volume, last_ts }
    }

    #[staticmethod]
    pub fn empty(symbol: String) -> Self {
        Self {
            symbol,
            last_price: None,
            open: None,
            high: None,
            low: None,
            close: None,
            volume: None,
            last_ts: None,
        }
    }

    #[pyo3(signature = (*, price, ts=None))]
    fn with_tick(&self, price: i64, ts: Option<String>) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(price),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            last_ts: ts,
        }
    }

    #[pyo3(signature = (*, o, h, l, c, v, ts=None))]
    fn with_bar(
        &self,
        o: i64,
        h: i64,
        l: i64,
        c: i64,
        v: i64,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(c),
            open: Some(o),
            high: Some(h),
            low: Some(l),
            close: Some(c),
            volume: Some(v),
            last_ts: ts,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustMarketState(symbol='{}', last_price={}, open={}, high={}, low={}, close={}, volume={}, last_ts={})",
            self.symbol,
            opt_i64_repr(&self.last_price),
            opt_i64_repr(&self.open),
            opt_i64_repr(&self.high),
            opt_i64_repr(&self.low),
            opt_i64_repr(&self.close),
            opt_i64_repr(&self.volume),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers (i64 ÷ SCALE → f64)
    #[getter]
    fn last_price_f(&self) -> Option<f64> {
        self.last_price.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn open_f(&self) -> Option<f64> {
        self.open.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn high_f(&self) -> Option<f64> {
        self.high.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn low_f(&self) -> Option<f64> {
        self.low.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn close_f(&self) -> Option<f64> {
        self.close.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn volume_f(&self) -> Option<f64> {
        self.volume.map(|v| v as f64 / SCALE as f64)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && self.last_price == other.last_price
            && self.open == other.open
            && self.high == other.high
            && self.low == other.low
            && self.close == other.close
            && self.volume == other.volume
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// PositionState — i64 fixed-point (×10^8)
// ===========================================================================
#[pyclass(name = "RustPositionState", frozen)]
#[derive(Clone)]
pub struct RustPositionState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub qty: i64,
    #[pyo3(get)]
    pub avg_price: Option<i64>,
    #[pyo3(get)]
    pub last_price: Option<i64>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustPositionState {
    #[new]
    #[pyo3(signature = (symbol, qty=0, avg_price=None, last_price=None, last_ts=None))]
    fn new(
        symbol: String,
        qty: i64,
        avg_price: Option<i64>,
        last_price: Option<i64>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, qty, avg_price, last_price, last_ts }
    }

    #[staticmethod]
    pub fn empty(symbol: String) -> Self {
        Self {
            symbol,
            qty: 0,
            avg_price: None,
            last_price: None,
            last_ts: None,
        }
    }

    #[getter]
    fn is_flat(&self) -> bool {
        self.qty == 0
    }

    #[pyo3(signature = (*, qty, avg_price, last_price, ts=None))]
    fn with_update(
        &self,
        qty: i64,
        avg_price: Option<i64>,
        last_price: Option<i64>,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol: self.symbol.clone(),
            qty,
            avg_price,
            last_price,
            last_ts: ts,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustPositionState(symbol='{}', qty='{}', avg_price={}, last_price={}, last_ts={})",
            self.symbol,
            i64_repr(self.qty),
            opt_i64_repr(&self.avg_price),
            opt_i64_repr(&self.last_price),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers
    #[getter]
    fn qty_f(&self) -> f64 {
        self.qty as f64 / SCALE as f64
    }

    #[getter]
    fn avg_price_f(&self) -> Option<f64> {
        self.avg_price.map(|v| v as f64 / SCALE as f64)
    }

    #[getter]
    fn last_price_f(&self) -> Option<f64> {
        self.last_price.map(|v| v as f64 / SCALE as f64)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && self.qty == other.qty
            && self.avg_price == other.avg_price
            && self.last_price == other.last_price
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// AccountState — i64 fixed-point (×10^8)
// ===========================================================================
#[pyclass(name = "RustAccountState", frozen)]
#[derive(Clone)]
pub struct RustAccountState {
    #[pyo3(get)]
    pub currency: String,
    #[pyo3(get)]
    pub balance: i64,
    #[pyo3(get)]
    pub margin_used: i64,
    #[pyo3(get)]
    pub margin_available: i64,
    #[pyo3(get)]
    pub realized_pnl: i64,
    #[pyo3(get)]
    pub unrealized_pnl: i64,
    #[pyo3(get)]
    pub fees_paid: i64,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustAccountState {
    #[new]
    #[pyo3(signature = (currency, balance, margin_used=0, margin_available=0, realized_pnl=0, unrealized_pnl=0, fees_paid=0, last_ts=None))]
    fn new(
        currency: String,
        balance: i64,
        margin_used: i64,
        margin_available: i64,
        realized_pnl: i64,
        unrealized_pnl: i64,
        fees_paid: i64,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            currency,
            balance,
            margin_used,
            margin_available,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            last_ts,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*, currency, balance))]
    fn initial(currency: String, balance: i64) -> Self {
        Self {
            currency,
            balance,
            margin_used: 0,
            margin_available: 0,
            realized_pnl: 0,
            unrealized_pnl: 0,
            fees_paid: 0,
            last_ts: None,
        }
    }

    #[pyo3(signature = (*, balance, margin_used, realized_pnl, unrealized_pnl, fees_paid, ts=None))]
    fn with_update(
        &self,
        balance: i64,
        margin_used: i64,
        realized_pnl: i64,
        unrealized_pnl: i64,
        fees_paid: i64,
        ts: Option<String>,
    ) -> Self {
        Self {
            currency: self.currency.clone(),
            balance,
            margin_used,
            margin_available: self.margin_available,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            last_ts: ts,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustAccountState(currency='{}', balance='{}', margin_used='{}', realized_pnl='{}', unrealized_pnl='{}', fees_paid='{}', last_ts={})",
            self.currency,
            i64_repr(self.balance),
            i64_repr(self.margin_used),
            i64_repr(self.realized_pnl),
            i64_repr(self.unrealized_pnl),
            i64_repr(self.fees_paid),
            opt_str_repr(&self.last_ts),
        )
    }

    // Float accessors for Python consumers
    #[getter]
    fn balance_f(&self) -> f64 {
        self.balance as f64 / SCALE as f64
    }

    #[getter]
    fn margin_used_f(&self) -> f64 {
        self.margin_used as f64 / SCALE as f64
    }

    #[getter]
    fn margin_available_f(&self) -> f64 {
        self.margin_available as f64 / SCALE as f64
    }

    #[getter]
    fn realized_pnl_f(&self) -> f64 {
        self.realized_pnl as f64 / SCALE as f64
    }

    #[getter]
    fn unrealized_pnl_f(&self) -> f64 {
        self.unrealized_pnl as f64 / SCALE as f64
    }

    #[getter]
    fn fees_paid_f(&self) -> f64 {
        self.fees_paid as f64 / SCALE as f64
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.currency == other.currency
            && self.balance == other.balance
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && self.realized_pnl == other.realized_pnl
            && self.unrealized_pnl == other.unrealized_pnl
            && self.fees_paid == other.fees_paid
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// PortfolioState (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustPortfolioState", frozen)]
#[derive(Clone)]
pub struct RustPortfolioState {
    #[pyo3(get)]
    pub total_equity: String,
    #[pyo3(get)]
    pub cash_balance: String,
    #[pyo3(get)]
    pub realized_pnl: String,
    #[pyo3(get)]
    pub unrealized_pnl: String,
    #[pyo3(get)]
    pub fees_paid: String,
    #[pyo3(get)]
    pub gross_exposure: String,
    #[pyo3(get)]
    pub net_exposure: String,
    #[pyo3(get)]
    pub leverage: Option<String>,
    #[pyo3(get)]
    pub margin_used: String,
    #[pyo3(get)]
    pub margin_available: String,
    #[pyo3(get)]
    pub margin_ratio: Option<String>,
    #[pyo3(get)]
    pub symbols: Vec<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustPortfolioState {
    #[new]
    #[pyo3(signature = (
        total_equity,
        cash_balance,
        realized_pnl,
        unrealized_pnl,
        fees_paid,
        gross_exposure,
        net_exposure,
        leverage,
        margin_used,
        margin_available,
        margin_ratio,
        symbols=vec![],
        last_ts=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        total_equity: String,
        cash_balance: String,
        realized_pnl: String,
        unrealized_pnl: String,
        fees_paid: String,
        gross_exposure: String,
        net_exposure: String,
        leverage: Option<String>,
        margin_used: String,
        margin_available: String,
        margin_ratio: Option<String>,
        symbols: Vec<String>,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            total_equity,
            cash_balance,
            realized_pnl,
            unrealized_pnl,
            fees_paid,
            gross_exposure,
            net_exposure,
            leverage,
            margin_used,
            margin_available,
            margin_ratio,
            symbols,
            last_ts,
        }
    }

    /// Compute a PortfolioState from account + positions + market data.
    /// Now reads i64 fields from core types.
    #[staticmethod]
    #[pyo3(signature = (account, positions, market))]
    fn compute(
        account: &RustAccountState,
        positions: &Bound<'_, PyDict>,
        market: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let balance = Fd8::from_raw(account.balance);
        let margin_used = Fd8::from_raw(account.margin_used);
        let margin_available = Fd8::from_raw(account.margin_available);
        let realized_pnl = Fd8::from_raw(account.realized_pnl);
        let fees_paid = Fd8::from_raw(account.fees_paid);

        let mut gross_exposure = Fd8::ZERO;
        let mut net_exposure = Fd8::ZERO;
        let mut total_unrealized = Fd8::ZERO;
        let mut symbols: Vec<String> = Vec::new();
        let mut latest_ts: Option<String> = account.last_ts.clone();

        // Build market price lookup
        let mut market_prices: HashMap<String, Fd8> = HashMap::new();
        for (key, val) in market.iter() {
            let sym: String = key.extract()?;
            let ms: RustMarketState = val.extract()?;
            if let Some(p) = ms.last_price {
                market_prices.insert(sym, Fd8::from_raw(p));
            }
        }

        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let pos: RustPositionState = val.extract()?;
            let qty = Fd8::from_raw(pos.qty);
            if qty.is_zero() {
                continue;
            }
            symbols.push(sym.clone());

            let mark = pos
                .last_price
                .map(Fd8::from_raw)
                .or_else(|| market_prices.get(&sym).copied())
                .or_else(|| pos.avg_price.map(Fd8::from_raw))
                .unwrap_or(Fd8::ZERO);

            let notional = qty.abs() * mark;
            gross_exposure = gross_exposure + notional;
            net_exposure = net_exposure + qty * mark;

            if let Some(avg_raw) = pos.avg_price {
                let avg = Fd8::from_raw(avg_raw);
                total_unrealized = total_unrealized + qty * (mark - avg);
            }

            if pos.last_ts.is_some() && (latest_ts.is_none() || pos.last_ts > latest_ts) {
                latest_ts = pos.last_ts.clone();
            }
        }

        symbols.sort();

        let total_equity = balance + total_unrealized;
        let te_f = total_equity.to_f64();
        let ge_f = gross_exposure.to_f64();
        let mu_f = margin_used.to_f64();

        let leverage = if te_f.abs() > 1e-12 {
            Some(format!("{}", ge_f / te_f))
        } else {
            None
        };
        let margin_ratio = if mu_f > 1e-12 && te_f.abs() > 1e-12 {
            Some(format!("{}", te_f / mu_f))
        } else {
            None
        };

        Ok(Self {
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
            last_ts: latest_ts,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RustPortfolioState(equity='{}', cash='{}', gross_exp='{}', net_exp='{}', leverage={}, symbols={:?})",
            self.total_equity,
            self.cash_balance,
            self.gross_exposure,
            self.net_exposure,
            opt_str_repr(&self.leverage),
            self.symbols,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.total_equity == other.total_equity
            && self.cash_balance == other.cash_balance
            && self.realized_pnl == other.realized_pnl
            && self.unrealized_pnl == other.unrealized_pnl
            && self.fees_paid == other.fees_paid
            && self.gross_exposure == other.gross_exposure
            && self.net_exposure == other.net_exposure
            && opt_str_eq(&self.leverage, &other.leverage)
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && opt_str_eq(&self.margin_ratio, &other.margin_ratio)
            && self.symbols == other.symbols
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// RiskLimits (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustRiskLimits", frozen)]
#[derive(Clone)]
pub struct RustRiskLimits {
    #[pyo3(get)]
    pub max_leverage: String,
    #[pyo3(get)]
    pub max_position_notional: Option<String>,
    #[pyo3(get)]
    pub max_drawdown_pct: String,
    #[pyo3(get)]
    pub block_on_equity_le_zero: bool,
}

#[pymethods]
impl RustRiskLimits {
    #[new]
    #[pyo3(signature = (max_leverage="5".to_string(), max_position_notional=None, max_drawdown_pct="0.30".to_string(), block_on_equity_le_zero=true))]
    fn new(
        max_leverage: String,
        max_position_notional: Option<String>,
        max_drawdown_pct: String,
        block_on_equity_le_zero: bool,
    ) -> Self {
        Self {
            max_leverage,
            max_position_notional,
            max_drawdown_pct,
            block_on_equity_le_zero,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustRiskLimits(max_leverage='{}', max_position_notional={}, max_drawdown_pct='{}', block_on_equity_le_zero={})",
            self.max_leverage,
            opt_str_repr(&self.max_position_notional),
            self.max_drawdown_pct,
            self.block_on_equity_le_zero,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.max_leverage == other.max_leverage
            && opt_str_eq(&self.max_position_notional, &other.max_position_notional)
            && self.max_drawdown_pct == other.max_drawdown_pct
            && self.block_on_equity_le_zero == other.block_on_equity_le_zero
    }
}

// ===========================================================================
// RiskState (keeps String — not on pipeline hot path)
// ===========================================================================
#[pyclass(name = "RustRiskState", frozen)]
#[derive(Clone)]
pub struct RustRiskState {
    #[pyo3(get)]
    pub blocked: bool,
    #[pyo3(get)]
    pub halted: bool,
    #[pyo3(get)]
    pub level: Option<String>,
    #[pyo3(get)]
    pub message: Option<String>,
    #[pyo3(get)]
    pub flags: Vec<String>,
    #[pyo3(get)]
    pub equity_peak: String,
    #[pyo3(get)]
    pub drawdown_pct: String,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustRiskState {
    #[new]
    #[pyo3(signature = (blocked=false, halted=false, level=None, message=None, flags=vec![], equity_peak="0".to_string(), drawdown_pct="0".to_string(), last_ts=None))]
    fn new(
        blocked: bool,
        halted: bool,
        level: Option<String>,
        message: Option<String>,
        flags: Vec<String>,
        equity_peak: String,
        drawdown_pct: String,
        last_ts: Option<String>,
    ) -> Self {
        Self {
            blocked,
            halted,
            level,
            message,
            flags,
            equity_peak,
            drawdown_pct,
            last_ts,
        }
    }

    #[pyo3(signature = (*, blocked, halted, level, message, flags, equity_peak, drawdown_pct, ts=None))]
    fn with_update(
        &self,
        blocked: bool,
        halted: bool,
        level: Option<String>,
        message: Option<String>,
        flags: Vec<String>,
        equity_peak: String,
        drawdown_pct: String,
        ts: Option<String>,
    ) -> Self {
        Self {
            blocked,
            halted,
            level,
            message,
            flags,
            equity_peak,
            drawdown_pct,
            last_ts: ts,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RustRiskState(blocked={}, halted={}, level={}, equity_peak='{}', drawdown_pct='{}', flags={:?})",
            self.blocked,
            self.halted,
            opt_str_repr(&self.level),
            self.equity_peak,
            self.drawdown_pct,
            self.flags,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.blocked == other.blocked
            && self.halted == other.halted
            && opt_str_eq(&self.level, &other.level)
            && opt_str_eq(&self.message, &other.message)
            && self.flags == other.flags
            && self.equity_peak == other.equity_peak
            && self.drawdown_pct == other.drawdown_pct
            && opt_str_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// ReducerResult
// ===========================================================================
#[pyclass(name = "RustReducerResult", frozen)]
pub struct RustReducerResult {
    #[pyo3(get)]
    pub state: PyObject,
    #[pyo3(get)]
    pub changed: bool,
    #[pyo3(get)]
    pub note: Option<String>,
}

impl Clone for RustReducerResult {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            state: self.state.clone_ref(py),
            changed: self.changed,
            note: self.note.clone(),
        })
    }
}

#[pymethods]
impl RustReducerResult {
    #[new]
    #[pyo3(signature = (state, changed, note=None))]
    fn new(state: PyObject, changed: bool, note: Option<String>) -> Self {
        Self { state, changed, note }
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let state_repr = self
            .state
            .bind(py)
            .repr()
            .map(|r| r.to_string())
            .unwrap_or_else(|_| "<?>".to_string());
        format!(
            "RustReducerResult(state={}, changed={}, note={})",
            state_repr,
            self.changed,
            opt_str_repr(&self.note),
        )
    }

    fn __eq__(&self, py: Python<'_>, other: &Self) -> PyResult<bool> {
        if self.changed != other.changed {
            return Ok(false);
        }
        if self.note != other.note {
            return Ok(false);
        }
        let eq = self.state.bind(py).eq(other.state.bind(py))?;
        Ok(eq)
    }
}
