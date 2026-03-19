"""Market maker backtest engine using kline data.

Simulates A-S quoting on historical 5m bars with realistic fills:
  - Bid fills when bar low <= bid price
  - Ask fills when bar high >= ask price
  - Both sides can fill in same bar (round-trip)
  - Inventory tracking, PnL, risk limits, maker rebates
  - Intra-bar adverse selection modeled via high/low fill probability

Usage:
    from execution.market_maker.backtest import run_mm_backtest, MMBacktestConfig
    result = run_mm_backtest(df, MMBacktestConfig())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MarketMakerConfig
from .inventory_tracker import InventoryTracker
from .perp_quoter import PerpQuoter
from .vol_estimator import VolEstimator

log = logging.getLogger(__name__)


@dataclass
class MMBacktestConfig:
    """Backtest-specific configuration layered on top of MarketMakerConfig."""

    # Base MM config
    mm: MarketMakerConfig = field(default_factory=lambda: MarketMakerConfig(
        order_size_eth=0.01,
        max_inventory_notional=50.0,
        gamma=0.3,
        kappa=1.5,
        time_horizon_s=300.0,
        min_spread_bps=2.0,
        max_spread_bps=30.0,
        vpin_threshold=0.7,
        vpin_spread_mult=1.5,
        daily_loss_limit=10.0,
        circuit_breaker_losses=3,
        circuit_breaker_pause_s=120.0,
    ))

    # Backtest params
    initial_equity: float = 100.0
    leverage: float = 10.0
    maker_fee_bps: float = -1.0       # -1 bps = rebate on Binance
    taker_fee_bps: float = 4.0        # taker fee for flatten/stop
    slippage_bps: float = 0.5         # slippage on market orders only
    fill_probability: float = 0.7     # P(fill | price touched level)
    adverse_selection_bps: float = 1.0  # avg adverse move post-fill

    # Funding
    funding_rate_per_8h: float = 0.0001  # 1 bps default
    bars_per_8h: int = 96               # 5m bars per 8h

    # Vol params
    vol_lookback: int = 200
    vol_floor: float = 1e-6

    # Time horizon for A-S (rolling 5 min = 1 bar at 5m resolution)
    bars_per_horizon: int = 1


@dataclass
class MMTradeRecord:
    bar: int
    side: str
    qty: float
    price: float
    fee: float
    rpnl: float
    inventory_after: float
    equity_after: float


@dataclass
class MMBacktestResult:
    equity_curve: np.ndarray
    trades: list[MMTradeRecord] = field(default_factory=list)
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    total_pnl: float = 0.0
    maker_rebate: float = 0.0
    total_fees: float = 0.0
    n_fills: int = 0
    n_round_trips: int = 0
    avg_spread_bps: float = 0.0
    fill_rate: float = 0.0
    max_inventory: float = 0.0
    daily_pnl_series: np.ndarray = field(default_factory=lambda: np.array([]))
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_hold_bars: float = 0.0


def run_mm_backtest(
    df: pd.DataFrame,
    cfg: MMBacktestConfig | None = None,
    verbose: bool = False,
) -> MMBacktestResult:
    """Run market maker backtest on kline DataFrame.

    DataFrame must have columns: open, high, low, close, volume
    (and optionally: taker_buy_volume, trades, quote_volume)
    """
    if cfg is None:
        cfg = MMBacktestConfig()

    mm_cfg = cfg.mm
    n = len(df)

    # Extract arrays
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    # Taker buy ratio for VPIN proxy
    if "taker_buy_volume" in df.columns:
        taker_buy = df["taker_buy_volume"].values.astype(float)
    else:
        taker_buy = None

    # Components
    quoter = PerpQuoter(mm_cfg)
    inventory = InventoryTracker(
        max_notional=mm_cfg.max_inventory_notional,
        daily_loss_limit=mm_cfg.daily_loss_limit,
    )
    vol_est = VolEstimator(alpha=2.0 / (cfg.vol_lookback + 1), min_trades=20)

    # State
    equity = cfg.initial_equity
    equity_curve = np.zeros(n)
    trades: list[MMTradeRecord] = []
    spreads: list[float] = []
    total_fees = 0.0
    maker_rebate = 0.0
    fill_count = 0
    quote_count = 0
    max_inv = 0.0
    paused_until = 0

    # Daily tracking
    day_start_bar = 0
    bars_per_day = 288  # 5m bars per day

    rng = np.random.RandomState(42)

    for i in range(n):
        mid = (highs[i] + lows[i]) / 2.0
        close = closes[i]

        # Vol update (use close-to-close returns)
        vol = vol_est.on_trade(close)

        # Daily reset
        if i - day_start_bar >= bars_per_day:
            day_start_bar = i
            inventory.reset_daily()

        # Risk check: daily loss limit
        if inventory.hit_daily_limit:
            # Flatten at market
            if abs(inventory.net_qty) > 1e-10:
                _flatten(inventory, close, cfg, trades, i)
                flatten_fee = abs(inventory.net_qty) * close * cfg.taker_fee_bps * 1e-4
                total_fees += flatten_fee
            equity_curve[i] = equity + inventory.realised_pnl + inventory.unrealised_pnl
            continue

        # Circuit breaker
        if i < paused_until:
            equity_curve[i] = equity + inventory.realised_pnl
            continue
        if inventory.consecutive_losses >= mm_cfg.circuit_breaker_losses:
            paused_until = i + int(mm_cfg.circuit_breaker_pause_s / 300)  # convert seconds to bars

        # Skip if vol not ready
        if not vol_est.ready or vol < cfg.vol_floor:
            equity_curve[i] = equity + inventory.realised_pnl
            continue

        # VPIN proxy from taker buy ratio
        vpin = 0.0
        if taker_buy is not None and volumes[i] > 0:
            buy_ratio = taker_buy[i] / volumes[i]
            # VPIN ~ |buy_ratio - 0.5| * 2, scaled to [0, 1]
            vpin = abs(buy_ratio - 0.5) * 2.0

        # Funding rate (simplified: apply every bars_per_8h bars)
        funding_rate = cfg.funding_rate_per_8h
        if i % cfg.bars_per_8h == 0 and abs(inventory.net_qty) > 0:
            funding_cost = inventory.net_qty * close * funding_rate
            inventory.realised_pnl -= funding_cost
            inventory.daily_pnl -= funding_cost

        # Time remaining in horizon (cycles every bars_per_horizon bars)
        T = max(0.1, 1.0 - (i % max(cfg.bars_per_horizon, 1)) / max(cfg.bars_per_horizon, 1))

        # Compute quotes
        quote = quoter.compute_quotes(
            mid=mid,
            inventory=inventory.net_qty,
            vol=vol,
            time_remaining=T,
            funding_rate=funding_rate,
            vpin=vpin,
        )

        if quote is None:
            equity_curve[i] = equity + inventory.realised_pnl
            continue

        quote_count += 1
        spread_bps = quote.spread / mid * 10000
        spreads.append(spread_bps)

        # Determine which sides to quote
        bid_price = quote.bid if inventory.can_buy(mid) else None
        ask_price = quote.ask if inventory.can_sell(mid) else None
        order_size = mm_cfg.order_size_eth

        # ── Simulate fills ──────────────────────────────────
        # Realism constraints:
        # 1. Price must CROSS through level (strict inequality)
        # 2. At most ONE side fills per bar (no free round-trip)
        # 3. Adverse selection: fill price worse than quote
        # 4. Queue position modeled via fill_probability

        bid_touched = bid_price is not None and lows[i] < bid_price
        ask_touched = ask_price is not None and highs[i] > ask_price

        bid_filled = False
        ask_filled = False

        if bid_touched and ask_touched:
            # Both sides crossed — only fill the side price moved to first
            if closes[i] >= opens[i]:
                bid_filled = rng.random() < cfg.fill_probability
            else:
                ask_filled = rng.random() < cfg.fill_probability
        elif bid_touched:
            bid_filled = rng.random() < cfg.fill_probability
        elif ask_touched:
            ask_filled = rng.random() < cfg.fill_probability

        if bid_filled:
            adverse = cfg.adverse_selection_bps * 1e-4 * mid
            fill_price = bid_price - adverse  # worse fill due to adverse selection
            fee = order_size * fill_price * cfg.maker_fee_bps * 1e-4
            rpnl = inventory.on_fill("buy", order_size, fill_price)
            total_fees += fee
            if fee < 0:
                maker_rebate += abs(fee)
            fill_count += 1
            trades.append(MMTradeRecord(
                bar=i, side="buy", qty=order_size, price=fill_price,
                fee=fee, rpnl=rpnl,
                inventory_after=inventory.net_qty,
                equity_after=0.0,
            ))

        if ask_filled:
            adverse = cfg.adverse_selection_bps * 1e-4 * mid
            fill_price = ask_price + adverse  # worse fill
            fee = order_size * fill_price * cfg.maker_fee_bps * 1e-4
            rpnl = inventory.on_fill("sell", order_size, fill_price)
            total_fees += fee
            if fee < 0:
                maker_rebate += abs(fee)
            fill_count += 1
            trades.append(MMTradeRecord(
                bar=i, side="sell", qty=order_size, price=fill_price,
                fee=fee, rpnl=rpnl,
                inventory_after=inventory.net_qty,
                equity_after=0.0,
            ))

        # Track max inventory
        current_inv = abs(inventory.net_qty) * close
        if current_inv > max_inv:
            max_inv = current_inv

        # Update unrealised PnL
        inventory.update_unrealised(close)

        # Equity = initial + realised + unrealised - fees
        current_equity = cfg.initial_equity + inventory.realised_pnl + inventory.unrealised_pnl - total_fees
        equity_curve[i] = current_equity

        # Update trade equity
        if bid_filled and trades:
            trades[-1 if ask_filled else -1].equity_after = current_equity
        if ask_filled and trades:
            trades[-1].equity_after = current_equity

    # ── Final flatten ───────────────────────────────────────
    if abs(inventory.net_qty) > 1e-10:
        final_price = closes[-1]
        side = "sell" if inventory.net_qty > 0 else "buy"
        qty = abs(inventory.net_qty)
        fee = qty * final_price * cfg.taker_fee_bps * 1e-4
        rpnl = inventory.on_fill(side, qty, final_price)
        total_fees += fee
        trades.append(MMTradeRecord(
            bar=n - 1, side=side, qty=qty, price=final_price,
            fee=fee, rpnl=rpnl,
            inventory_after=0.0,
            equity_after=equity_curve[-1] if n > 0 else cfg.initial_equity,
        ))

    # ── Compute statistics ──────────────────────────────────
    equity_curve = np.maximum(equity_curve, 0.0)
    # Fix zero entries at start (before vol warmup)
    for idx in range(len(equity_curve)):
        if equity_curve[idx] > 0:
            break
        equity_curve[idx] = cfg.initial_equity

    final_equity = equity_curve[-1]
    total_pnl = final_equity - cfg.initial_equity
    total_return_pct = (final_equity / cfg.initial_equity - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.maximum(peak, 1e-10)
    max_dd_pct = abs(dd.min()) * 100

    # Sharpe (per-bar returns, annualised)
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-10)
    if len(returns) > 1 and np.std(returns) > 0:
        bars_per_year = 365.25 * 24 * 12  # 5m bars per year
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Round trips and win rate
    round_trips = 0
    wins = 0
    gross_profit = 0.0
    gross_loss = 0.0
    for t in trades:
        if t.rpnl != 0:
            round_trips += 1
            if t.rpnl > 0:
                wins += 1
                gross_profit += t.rpnl
            else:
                gross_loss += abs(t.rpnl)
    win_rate = wins / round_trips if round_trips > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Fill rate
    fill_rate = fill_count / max(quote_count, 1)

    # Average spread
    avg_spread = np.mean(spreads) if spreads else 0.0

    # Daily PnL series
    n_days = n // bars_per_day + 1
    daily_pnl = np.zeros(n_days)
    for d in range(n_days):
        start = d * bars_per_day
        end = min((d + 1) * bars_per_day, n) - 1
        if end >= start and end < n:
            daily_pnl[d] = equity_curve[end] - (equity_curve[start] if start > 0 else cfg.initial_equity)

    return MMBacktestResult(
        equity_curve=equity_curve,
        trades=trades,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_pct,
        sharpe=sharpe,
        total_pnl=total_pnl,
        maker_rebate=maker_rebate,
        total_fees=total_fees,
        n_fills=fill_count,
        n_round_trips=round_trips,
        avg_spread_bps=avg_spread,
        fill_rate=fill_rate,
        max_inventory=max_inv,
        daily_pnl_series=daily_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def _flatten(
    inventory: InventoryTracker,
    price: float,
    cfg: MMBacktestConfig,
    trades: list[MMTradeRecord],
    bar: int,
) -> None:
    """Flatten position at market."""
    if abs(inventory.net_qty) < 1e-10:
        return
    side = "sell" if inventory.net_qty > 0 else "buy"
    qty = abs(inventory.net_qty)
    # Taker fee + slippage for market flatten
    slip = price * cfg.slippage_bps * 1e-4
    fill_price = price - slip if side == "sell" else price + slip
    fee = qty * fill_price * cfg.taker_fee_bps * 1e-4
    rpnl = inventory.on_fill(side, qty, fill_price)
    trades.append(MMTradeRecord(
        bar=bar, side=side, qty=qty, price=fill_price,
        fee=fee, rpnl=rpnl,
        inventory_after=0.0,
        equity_after=0.0,
    ))
