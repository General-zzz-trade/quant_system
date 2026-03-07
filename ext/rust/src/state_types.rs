use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: compare two Option<String> for equality
// ---------------------------------------------------------------------------
fn opt_eq(a: &Option<String>, b: &Option<String>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a == b,
        (None, None) => true,
        _ => false,
    }
}

fn opt_repr(v: &Option<String>) -> String {
    match v {
        Some(s) => format!("'{}'", s),
        None => "None".to_string(),
    }
}

// ===========================================================================
// MarketState
// ===========================================================================
#[pyclass(name = "RustMarketState", frozen)]
#[derive(Clone)]
pub struct RustMarketState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub last_price: Option<String>,
    #[pyo3(get)]
    pub open: Option<String>,
    #[pyo3(get)]
    pub high: Option<String>,
    #[pyo3(get)]
    pub low: Option<String>,
    #[pyo3(get)]
    pub close: Option<String>,
    #[pyo3(get)]
    pub volume: Option<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustMarketState {
    #[new]
    #[pyo3(signature = (symbol, last_price=None, open=None, high=None, low=None, close=None, volume=None, last_ts=None))]
    fn new(
        symbol: String,
        last_price: Option<String>,
        open: Option<String>,
        high: Option<String>,
        low: Option<String>,
        close: Option<String>,
        volume: Option<String>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, last_price, open, high, low, close, volume, last_ts }
    }

    #[staticmethod]
    fn empty(symbol: String) -> Self {
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
    fn with_tick(&self, price: String, ts: Option<String>) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(price),
            open: self.open.clone(),
            high: self.high.clone(),
            low: self.low.clone(),
            close: self.close.clone(),
            volume: self.volume.clone(),
            last_ts: ts,
        }
    }

    #[pyo3(signature = (*, o, h, l, c, v, ts=None))]
    fn with_bar(
        &self,
        o: String,
        h: String,
        l: String,
        c: String,
        v: String,
        ts: Option<String>,
    ) -> Self {
        Self {
            symbol: self.symbol.clone(),
            last_price: Some(c.clone()),
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
            opt_repr(&self.last_price),
            opt_repr(&self.open),
            opt_repr(&self.high),
            opt_repr(&self.low),
            opt_repr(&self.close),
            opt_repr(&self.volume),
            opt_repr(&self.last_ts),
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && opt_eq(&self.last_price, &other.last_price)
            && opt_eq(&self.open, &other.open)
            && opt_eq(&self.high, &other.high)
            && opt_eq(&self.low, &other.low)
            && opt_eq(&self.close, &other.close)
            && opt_eq(&self.volume, &other.volume)
            && opt_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// PositionState
// ===========================================================================
#[pyclass(name = "RustPositionState", frozen)]
#[derive(Clone)]
pub struct RustPositionState {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub qty: String,
    #[pyo3(get)]
    pub avg_price: Option<String>,
    #[pyo3(get)]
    pub last_price: Option<String>,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustPositionState {
    #[new]
    #[pyo3(signature = (symbol, qty="0".to_string(), avg_price=None, last_price=None, last_ts=None))]
    fn new(
        symbol: String,
        qty: String,
        avg_price: Option<String>,
        last_price: Option<String>,
        last_ts: Option<String>,
    ) -> Self {
        Self { symbol, qty, avg_price, last_price, last_ts }
    }

    #[staticmethod]
    fn empty(symbol: String) -> Self {
        Self {
            symbol,
            qty: "0".to_string(),
            avg_price: None,
            last_price: None,
            last_ts: None,
        }
    }

    #[getter]
    fn is_flat(&self) -> bool {
        // "0", "0.0", "0.00", etc. all mean flat
        match self.qty.parse::<f64>() {
            Ok(v) => v == 0.0,
            Err(_) => self.qty == "0",
        }
    }

    #[pyo3(signature = (*, qty, avg_price, last_price, ts=None))]
    fn with_update(
        &self,
        qty: String,
        avg_price: Option<String>,
        last_price: Option<String>,
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
            self.qty,
            opt_repr(&self.avg_price),
            opt_repr(&self.last_price),
            opt_repr(&self.last_ts),
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.symbol == other.symbol
            && self.qty == other.qty
            && opt_eq(&self.avg_price, &other.avg_price)
            && opt_eq(&self.last_price, &other.last_price)
            && opt_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// AccountState
// ===========================================================================
#[pyclass(name = "RustAccountState", frozen)]
#[derive(Clone)]
pub struct RustAccountState {
    #[pyo3(get)]
    pub currency: String,
    #[pyo3(get)]
    pub balance: String,
    #[pyo3(get)]
    pub margin_used: String,
    #[pyo3(get)]
    pub margin_available: String,
    #[pyo3(get)]
    pub realized_pnl: String,
    #[pyo3(get)]
    pub unrealized_pnl: String,
    #[pyo3(get)]
    pub fees_paid: String,
    #[pyo3(get)]
    pub last_ts: Option<String>,
}

#[pymethods]
impl RustAccountState {
    #[new]
    #[pyo3(signature = (currency, balance, margin_used="0".to_string(), margin_available="0".to_string(), realized_pnl="0".to_string(), unrealized_pnl="0".to_string(), fees_paid="0".to_string(), last_ts=None))]
    fn new(
        currency: String,
        balance: String,
        margin_used: String,
        margin_available: String,
        realized_pnl: String,
        unrealized_pnl: String,
        fees_paid: String,
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
    fn initial(currency: String, balance: String) -> Self {
        Self {
            currency,
            balance,
            margin_used: "0".to_string(),
            margin_available: "0".to_string(),
            realized_pnl: "0".to_string(),
            unrealized_pnl: "0".to_string(),
            fees_paid: "0".to_string(),
            last_ts: None,
        }
    }

    #[pyo3(signature = (*, balance, margin_used, realized_pnl, unrealized_pnl, fees_paid, ts=None))]
    fn with_update(
        &self,
        balance: String,
        margin_used: String,
        realized_pnl: String,
        unrealized_pnl: String,
        fees_paid: String,
        ts: Option<String>,
    ) -> Self {
        Self {
            currency: self.currency.clone(),
            balance,
            margin_used,
            margin_available: self.margin_available.clone(),
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
            self.balance,
            self.margin_used,
            self.realized_pnl,
            self.unrealized_pnl,
            self.fees_paid,
            opt_repr(&self.last_ts),
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.currency == other.currency
            && self.balance == other.balance
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && self.realized_pnl == other.realized_pnl
            && self.unrealized_pnl == other.unrealized_pnl
            && self.fees_paid == other.fees_paid
            && opt_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// PortfolioState
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
    ///
    /// positions: dict[str, RustPositionState]
    /// market: dict[str, RustMarketState]
    ///
    /// All arithmetic is done in f64 then serialized back to String.
    /// Python side wraps with Decimal for downstream use.
    #[staticmethod]
    #[pyo3(signature = (account, positions, market))]
    fn compute(
        account: &RustAccountState,
        positions: &Bound<'_, PyDict>,
        market: &Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let balance: f64 = account.balance.parse().unwrap_or(0.0);
        let margin_used: f64 = account.margin_used.parse().unwrap_or(0.0);
        let margin_available: f64 = account.margin_available.parse().unwrap_or(0.0);
        let realized_pnl: f64 = account.realized_pnl.parse().unwrap_or(0.0);
        let fees_paid: f64 = account.fees_paid.parse().unwrap_or(0.0);

        let mut gross_exposure: f64 = 0.0;
        let mut net_exposure: f64 = 0.0;
        let mut total_unrealized: f64 = 0.0;
        let mut symbols: Vec<String> = Vec::new();
        let mut latest_ts: Option<String> = account.last_ts.clone();

        // Build market price lookup
        let mut market_prices: HashMap<String, f64> = HashMap::new();
        for (key, val) in market.iter() {
            let sym: String = key.extract()?;
            let ms: RustMarketState = val.extract()?;
            if let Some(ref p) = ms.last_price {
                if let Ok(px) = p.parse::<f64>() {
                    market_prices.insert(sym, px);
                }
            }
        }

        for (key, val) in positions.iter() {
            let sym: String = key.extract()?;
            let pos: RustPositionState = val.extract()?;
            let qty: f64 = pos.qty.parse().unwrap_or(0.0);
            if qty == 0.0 {
                continue;
            }
            symbols.push(sym.clone());

            // Determine mark price: position's last_price > market last_price > avg_price
            let mark = pos
                .last_price
                .as_ref()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| market_prices.get(&sym).copied())
                .or_else(|| {
                    pos.avg_price
                        .as_ref()
                        .and_then(|s| s.parse::<f64>().ok())
                })
                .unwrap_or(0.0);

            let notional = qty.abs() * mark;
            gross_exposure += notional;
            net_exposure += qty * mark;

            // Unrealized PnL
            if let Some(ref avg_str) = pos.avg_price {
                if let Ok(avg) = avg_str.parse::<f64>() {
                    total_unrealized += qty * (mark - avg);
                }
            }

            // Track latest timestamp
            if pos.last_ts.is_some() && (latest_ts.is_none() || pos.last_ts > latest_ts) {
                latest_ts = pos.last_ts.clone();
            }
        }

        symbols.sort();

        let total_equity = balance + total_unrealized;
        let leverage = if total_equity.abs() > 1e-12 {
            Some(format!("{}", gross_exposure / total_equity))
        } else {
            None
        };
        let margin_ratio = if total_equity.abs() > 1e-12 {
            Some(format!("{}", margin_used / total_equity))
        } else {
            None
        };

        Ok(Self {
            total_equity: format!("{}", total_equity),
            cash_balance: format!("{}", balance),
            realized_pnl: format!("{}", realized_pnl),
            unrealized_pnl: format!("{}", total_unrealized),
            fees_paid: format!("{}", fees_paid),
            gross_exposure: format!("{}", gross_exposure),
            net_exposure: format!("{}", net_exposure),
            leverage,
            margin_used: format!("{}", margin_used),
            margin_available: format!("{}", margin_available),
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
            opt_repr(&self.leverage),
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
            && opt_eq(&self.leverage, &other.leverage)
            && self.margin_used == other.margin_used
            && self.margin_available == other.margin_available
            && opt_eq(&self.margin_ratio, &other.margin_ratio)
            && self.symbols == other.symbols
            && opt_eq(&self.last_ts, &other.last_ts)
    }
}

// ===========================================================================
// RiskLimits
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
            opt_repr(&self.max_position_notional),
            self.max_drawdown_pct,
            self.block_on_equity_le_zero,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.max_leverage == other.max_leverage
            && opt_eq(&self.max_position_notional, &other.max_position_notional)
            && self.max_drawdown_pct == other.max_drawdown_pct
            && self.block_on_equity_le_zero == other.block_on_equity_le_zero
    }
}

// ===========================================================================
// RiskState
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
            opt_repr(&self.level),
            self.equity_peak,
            self.drawdown_pct,
            self.flags,
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.blocked == other.blocked
            && self.halted == other.halted
            && opt_eq(&self.level, &other.level)
            && opt_eq(&self.message, &other.message)
            && self.flags == other.flags
            && self.equity_peak == other.equity_peak
            && self.drawdown_pct == other.drawdown_pct
            && opt_eq(&self.last_ts, &other.last_ts)
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
            opt_repr(&self.note),
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
