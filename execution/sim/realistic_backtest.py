# execution/sim/realistic_backtest.py
"""Realistic backtest engine — fixes P0 issues from truth audit.

Addresses:
  P0-1: Stop-loss checked against bar high/low (intra-bar), not just close
  P0-2: Margin/liquidation simulation (equity < maintenance = forced close)
  P0-3: Position cap (max % of equity per trade, prevents exponential blowup)
  P1-1: Dynamic slippage (scales with position size / bar volume)
  P1-3: Bid/ask spread model (buy at ask, sell at bid)

Usage:
    from execution.sim.realistic_backtest import run_realistic_backtest
    result = run_realistic_backtest(
        closes, highs, lows, volumes, signal,
        leverage=3, stop_loss_pct=0.03, initial_equity=100,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from execution.sim.audit_log import BacktestAuditLog

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for realistic backtest."""
    initial_equity: float = 100.0
    leverage: float = 3.0
    stop_loss_pct: float = 0.03        # 3% hard stop (used when adaptive_stop=False)
    max_position_pct: float = 0.95     # max 95% of equity in one position
    fee_bps: float = 4.0              # 4 bps maker/taker average
    base_slippage_bps: float = 1.0    # base slippage
    volume_impact_factor: float = 0.1  # slippage scales with qty/volume
    spread_bps: float = 1.0           # half-spread cost
    funding_rates: Optional[np.ndarray] = None  # per-bar funding rates
    maintenance_margin: float = 0.005  # 0.5% maintenance margin ratio
    min_order_notional: float = 5.0    # $5 minimum order (Bybit/Binance)
    # Adaptive stop-loss (ATR-based trailing, matches run_bybit_alpha.py)
    adaptive_stop: bool = False        # enable ATR-based adaptive stop
    atr_stop_mult: float = 2.0         # initial stop = ATR × this
    atr_trail_trigger: float = 0.8     # trailing activates after profit >= ATR × this
    atr_trail_step: float = 0.3        # trail distance = ATR × this
    atr_breakeven_trigger: float = 1.0 # breakeven after profit >= ATR × this


@dataclass
class TradeRecord:
    """Record of a single completed trade."""
    entry_bar: int
    exit_bar: int
    side: int           # 1=long, -1=short
    entry_price: float
    exit_price: float
    size_usd: float     # notional size at entry
    pnl_gross: float    # before fees
    pnl_net: float      # after all costs
    fees: float
    slippage: float
    funding: float
    exit_reason: str    # signal, stop_loss, liquidation, max_hold


@dataclass
class BacktestResult:
    """Complete backtest result with audit trail."""
    equity_curve: np.ndarray
    trades: list[TradeRecord] = field(default_factory=list)
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    n_liquidations: int = 0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0


def _compute_slippage(notional: float, bar_volume_usd: float,
                      bar_volatility: float, cfg: BacktestConfig) -> float:
    """Dynamic slippage: base + volume impact + volatility component.

    Almgren-Chriss inspired: impact = base + η × σ × √(qty/ADV)
    """
    base = cfg.base_slippage_bps / 10000
    if bar_volume_usd > 0:
        participation = notional / bar_volume_usd
        volume_impact = cfg.volume_impact_factor * np.sqrt(participation)
    else:
        volume_impact = 0.01  # 1% if no volume data
    volatility_impact = bar_volatility * 0.05 if bar_volatility > 0 else 0
    return base + volume_impact + volatility_impact


def _check_intrabar_stoploss(
    side: int, entry_price: float, bar_high: float, bar_low: float,
    stop_loss_pct: float,
) -> tuple[bool, float]:
    """Check if fixed stop-loss was hit within the bar using high/low.

    Returns (triggered, stop_price).
    For longs: stop at entry × (1 - stop_loss_pct), triggered if low <= stop
    For shorts: stop at entry × (1 + stop_loss_pct), triggered if high >= stop
    """
    if side > 0:  # long
        stop_price = entry_price * (1 - stop_loss_pct)
        if bar_low <= stop_price:
            return True, stop_price
    elif side < 0:  # short
        stop_price = entry_price * (1 + stop_loss_pct)
        if bar_high >= stop_price:
            return True, stop_price
    return False, 0.0


def _compute_adaptive_stop(
    side: int, entry_price: float, peak_price: float,
    bar_high: float, bar_low: float, atr: float,
    cfg: BacktestConfig,
) -> tuple[bool, float, float]:
    """ATR-based adaptive stop with trailing — matches live runner logic.

    Three phases:
    1. Initial: entry ± ATR × atr_stop_mult (wide, let trade breathe)
    2. Breakeven: after profit >= ATR × breakeven_trigger, stop → entry
    3. Trailing: after profit >= ATR × trail_trigger, stop = peak - ATR × trail_step

    Returns (triggered, stop_price, updated_peak_price).
    """
    if atr < 1e-8:
        atr = 0.015  # fallback 1.5%

    # Update peak
    if side > 0:
        peak_price = max(peak_price, bar_high)
        profit_pct = (peak_price - entry_price) / entry_price
    else:
        peak_price = min(peak_price, bar_low)
        profit_pct = (entry_price - peak_price) / entry_price

    # Compute stop price based on phase
    if profit_pct >= atr * cfg.atr_trail_trigger:
        # Phase 3: trailing
        trail_dist = atr * cfg.atr_trail_step
        if side > 0:
            stop_price = peak_price * (1 - trail_dist)
        else:
            stop_price = peak_price * (1 + trail_dist)
    elif profit_pct >= atr * cfg.atr_breakeven_trigger:
        # Phase 2: breakeven
        buf = atr * 0.1
        if side > 0:
            stop_price = entry_price * (1 + buf)
        else:
            stop_price = entry_price * (1 - buf)
    else:
        # Phase 1: initial wide stop
        stop_dist = min(atr * cfg.atr_stop_mult, 0.05)  # cap at 5%
        stop_dist = max(stop_dist, 0.003)  # floor at 0.3%
        if side > 0:
            stop_price = entry_price * (1 - stop_dist)
        else:
            stop_price = entry_price * (1 + stop_dist)

    # Check trigger
    triggered = False
    if side > 0 and bar_low <= stop_price:
        triggered = True
    elif side < 0 and bar_high >= stop_price:
        triggered = True

    return triggered, stop_price, peak_price


def _check_liquidation(
    equity: float, position_notional: float, maintenance_margin: float,
) -> bool:
    """Check if position should be liquidated (margin call).

    Liquidation when equity < position_notional × maintenance_margin_ratio.
    """
    if position_notional <= 0:
        return False
    margin_ratio = equity / position_notional
    return margin_ratio < maintenance_margin


def run_realistic_backtest(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    signal: np.ndarray,
    cfg: BacktestConfig | None = None,
    funding_rates: np.ndarray | None = None,
    audit_log: "BacktestAuditLog | None" = None,
) -> BacktestResult:
    """Run bar-by-bar backtest with realistic execution simulation.

    Fixes all P0 issues:
    - Stop-loss checked against high/low (not just close)
    - Margin/liquidation model
    - Position size capped at max_position_pct × equity
    - Dynamic slippage
    - Spread costs (buy at ask, sell at bid)
    """
    if cfg is None:
        cfg = BacktestConfig()

    n = min(len(closes), len(signal))
    equity_curve = np.zeros(n + 1)
    equity_curve[0] = cfg.initial_equity

    equity = cfg.initial_equity
    position = 0        # current signal: -1, 0, 1
    entry_price = 0.0
    entry_bar = 0
    position_notional = 0.0
    cum_funding = 0.0
    peak_price = 0.0    # for adaptive trailing stop
    trades: list[TradeRecord] = []
    n_liquidations = 0
    total_fees = 0.0
    total_slippage = 0.0
    total_funding = 0.0

    # Compute per-bar volatility for slippage model
    rets = np.diff(np.log(np.maximum(closes, 1e-10)))
    rets = np.concatenate([[0], rets])
    vol_20 = np.zeros(n)
    for i in range(20, n):
        vol_20[i] = np.std(rets[i-20:i])

    # Compute ATR(14) for adaptive stop
    atr_14 = np.zeros(n)
    if cfg.adaptive_stop:
        for i in range(14, n):
            trs = []
            for j in range(i - 13, i + 1):
                tr = max(highs[j] - lows[j],
                         abs(highs[j] - closes[j - 1]),
                         abs(lows[j] - closes[j - 1]))
                trs.append(tr / closes[j] if closes[j] > 0 else 0)
            atr_14[i] = np.mean(trs)

    for i in range(1, n):
        price = closes[i]
        high = highs[i]
        low = lows[i]
        vol = volumes[i] * price if volumes[i] > 0 else 1e6  # USD volume
        sig = signal[i]

        # ── Check existing position ──
        if position != 0:
            # 1. Stop-loss check: adaptive (ATR trailing) or fixed
            if cfg.adaptive_stop:
                stopped, stop_px, peak_price = _compute_adaptive_stop(
                    position, entry_price, peak_price, high, low,
                    atr_14[i], cfg,
                )
                exit_reason_stop = "adaptive_stop"
            else:
                stopped, stop_px = _check_intrabar_stoploss(
                    position, entry_price, high, low, cfg.stop_loss_pct,
                )
                exit_reason_stop = "stop_loss"

            if stopped:
                # Exit at stop price (not close)
                exit_px = stop_px
                pnl_pct = position * (exit_px - entry_price) / entry_price
                pnl_gross = position_notional * pnl_pct
                fees = position_notional * cfg.fee_bps / 10000
                slip_cost = position_notional * _compute_slippage(
                    position_notional, vol, vol_20[i], cfg,
                )
                fund = cum_funding

                pnl_net = pnl_gross - fees - slip_cost - fund
                equity += pnl_net
                total_fees += fees
                total_slippage += slip_cost
                total_funding += fund

                trades.append(TradeRecord(
                    entry_bar=entry_bar, exit_bar=i, side=position,
                    entry_price=entry_price, exit_price=exit_px,
                    size_usd=position_notional,
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                    fees=fees, slippage=slip_cost, funding=fund,
                    exit_reason=exit_reason_stop,
                ))
                if audit_log:
                    audit_log.record_trade(trades[-1])
                    audit_log.record_event(i, exit_reason_stop, price=exit_px, equity=equity)
                position = 0
                entry_price = 0
                peak_price = 0
                position_notional = 0
                cum_funding = 0
                equity_curve[i] = equity
                continue

            # 2. Liquidation check (P0-2 fix)
            unrealized = position * (price - entry_price) / entry_price * position_notional
            current_equity = equity + unrealized
            if _check_liquidation(current_equity, position_notional, cfg.maintenance_margin):
                # Liquidated at close price
                pnl_gross = position * (price - entry_price) / entry_price * position_notional
                fees = position_notional * cfg.fee_bps / 10000
                pnl_net = pnl_gross - fees - cum_funding
                equity += pnl_net
                n_liquidations += 1

                trades.append(TradeRecord(
                    entry_bar=entry_bar, exit_bar=i, side=position,
                    entry_price=entry_price, exit_price=price,
                    size_usd=position_notional,
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                    fees=fees, slippage=0, funding=cum_funding,
                    exit_reason="liquidation",
                ))
                if audit_log:
                    audit_log.record_trade(trades[-1])
                    audit_log.record_event(i, "liquidation", price=price, equity=equity)
                position = 0
                entry_price = 0
                position_notional = 0
                cum_funding = 0
                total_fees += fees
                total_funding += cum_funding
                equity_curve[i] = equity
                continue

            # 3. Funding cost (accrues while position is open)
            if funding_rates is not None and i < len(funding_rates):
                bar_funding = position * funding_rates[i] * position_notional / 8.0
                cum_funding += abs(bar_funding)

        # ── Signal change → close old + open new ──
        if sig != position:
            # Close existing position
            if position != 0:
                pnl_pct = position * (price - entry_price) / entry_price
                pnl_gross = position_notional * pnl_pct
                fees = position_notional * cfg.fee_bps / 10000
                slip_cost = position_notional * _compute_slippage(
                    position_notional, vol, vol_20[i], cfg,
                )
                fund = cum_funding
                pnl_net = pnl_gross - fees - slip_cost - fund
                equity += pnl_net
                total_fees += fees
                total_slippage += slip_cost
                total_funding += fund

                trades.append(TradeRecord(
                    entry_bar=entry_bar, exit_bar=i, side=position,
                    entry_price=entry_price, exit_price=price,
                    size_usd=position_notional,
                    pnl_gross=pnl_gross, pnl_net=pnl_net,
                    fees=fees, slippage=slip_cost, funding=fund,
                    exit_reason="signal",
                ))
                if audit_log:
                    audit_log.record_trade(trades[-1])
                cum_funding = 0

            # Open new position
            if sig != 0 and equity > cfg.min_order_notional:
                # P0-3 fix: cap position at max_position_pct × equity × leverage
                max_notional = equity * cfg.max_position_pct * cfg.leverage
                position_notional = min(max_notional, equity * cfg.leverage)

                # Entry costs
                entry_fees = position_notional * cfg.fee_bps / 10000
                entry_slip = position_notional * _compute_slippage(
                    position_notional, vol, vol_20[i], cfg,
                )
                spread_cost = position_notional * cfg.spread_bps / 10000

                # Adjust entry price for spread
                if sig > 0:  # buy at ask
                    entry_price = price * (1 + cfg.spread_bps / 20000)
                else:  # sell at bid
                    entry_price = price * (1 - cfg.spread_bps / 20000)

                equity -= entry_fees + entry_slip + spread_cost
                total_fees += entry_fees
                total_slippage += entry_slip + spread_cost
                position = sig
                entry_bar = i
                peak_price = entry_price  # init trailing peak
            else:
                position = 0
                entry_price = 0
                peak_price = 0
                position_notional = 0

        equity_curve[i] = equity

        # Periodic audit snapshot
        if audit_log and i % 100 == 0:
            unreal = 0.0
            if position != 0 and entry_price > 0:
                unreal = position * (price - entry_price) / entry_price * position_notional
            audit_log.record_equity(i, equity, position, unreal)

    # Close final position at last price
    if position != 0:
        price = closes[n - 1]
        pnl_pct = position * (price - entry_price) / entry_price
        pnl_gross = position_notional * pnl_pct
        fees = position_notional * cfg.fee_bps / 10000
        pnl_net = pnl_gross - fees - cum_funding
        equity += pnl_net
        trades.append(TradeRecord(
            entry_bar=entry_bar, exit_bar=n-1, side=position,
            entry_price=entry_price, exit_price=price,
            size_usd=position_notional,
            pnl_gross=pnl_gross, pnl_net=pnl_net,
            fees=fees, slippage=0, funding=cum_funding,
            exit_reason="end_of_data",
        ))

    equity_curve[n] = equity

    # Compute metrics
    peak = np.maximum.accumulate(equity_curve[:n+1])
    drawdown = (peak - equity_curve[:n+1]) / np.maximum(peak, 1e-10)
    max_dd = float(np.max(drawdown))

    total_return = (equity / cfg.initial_equity - 1) * 100
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t.pnl_net > 0) / max(n_trades, 1)

    # Sharpe on trade returns
    if n_trades > 1:
        trade_rets = [t.pnl_net / max(t.size_usd, 1) for t in trades]
        sharpe = np.mean(trade_rets) / np.std(trade_rets) * np.sqrt(252 * 8) if np.std(trade_rets) > 0 else 0
    else:
        sharpe = 0.0

    return BacktestResult(
        equity_curve=equity_curve[:n+1],
        trades=trades,
        total_return_pct=total_return,
        max_drawdown_pct=max_dd * 100,
        sharpe=sharpe,
        win_rate=win_rate,
        n_trades=n_trades,
        n_liquidations=n_liquidations,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_funding=total_funding,
    )
