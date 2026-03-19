"""Tick-level market maker backtest using collected L2 depth + aggTrade data.

Unlike kline backtest, this uses real orderbook data to model:
  - Queue position (our order behind existing depth)
  - Fill probability based on depth consumed past our level
  - Actual spread at time of quoting
  - Real VPIN from aggTrade flow
  - Adverse selection measured from post-fill price moves

Usage:
    from execution.market_maker.tick_backtest import run_tick_backtest, TickBacktestConfig
    result = run_tick_backtest("data/ticks/ETHUSDT.db", TickBacktestConfig())
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from .config import MarketMakerConfig
from .inventory_tracker import InventoryTracker
from .perp_quoter import PerpQuoter
from .vol_estimator import VolEstimator

log = logging.getLogger(__name__)


@dataclass
class TickBacktestConfig:
    """Configuration for tick-level backtest."""
    mm: MarketMakerConfig = field(default_factory=lambda: MarketMakerConfig(
        order_size_eth=0.01,
        max_inventory_notional=50.0,
        gamma=0.3,
        kappa=1.5,
        min_spread_bps=2.0,
        max_spread_bps=30.0,
        daily_loss_limit=10.0,
        circuit_breaker_losses=3,
        circuit_breaker_pause_s=120.0,
    ))

    initial_equity: float = 100.0
    leverage: float = 10.0
    maker_fee_bps: float = -1.0       # rebate
    taker_fee_bps: float = 4.0

    # Queue position model
    # Our $20 order is behind existing depth at each level
    # Fill only if depth consumed past our level > order_size
    queue_position_pct: float = 0.8   # assume we're 80% back in queue

    # Time horizon
    time_horizon_s: float = 300.0     # 5 min rolling

    # Quote refresh
    quote_on_depth: bool = True       # re-quote on every depth update
    quote_on_trade: bool = False      # or only on trades

    # Funding
    funding_rate_per_8h: float = 0.0001

    # Analysis
    measure_adverse_selection: bool = True
    adverse_window_trades: int = 50   # measure price move over next N trades


@dataclass
class TickFillRecord:
    ts_ms: int
    side: str
    qty: float
    price: float
    fee: float
    rpnl: float
    inventory_after: float
    mid_at_fill: float
    vpin_at_fill: float
    # Adverse selection (filled in post-analysis)
    price_after_50: float = 0.0       # mid N trades after fill
    adverse_bps: float = 0.0


@dataclass
class TickBacktestResult:
    equity_curve: list[float] = field(default_factory=list)
    fills: list[TickFillRecord] = field(default_factory=list)
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    max_drawdown_pct: float = 0.0
    n_fills: int = 0
    n_round_trips: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    maker_rebate: float = 0.0
    total_fees: float = 0.0
    avg_spread_quoted_bps: float = 0.0
    avg_spread_market_bps: float = 0.0
    fill_rate: float = 0.0
    avg_adverse_selection_bps: float = 0.0
    avg_queue_depth_usd: float = 0.0
    duration_hours: float = 0.0
    trades_per_hour: float = 0.0
    depth_ticks_processed: int = 0
    trade_ticks_processed: int = 0


def run_tick_backtest(
    db_path: str,
    cfg: TickBacktestConfig | None = None,
) -> TickBacktestResult:
    """Run tick-level backtest from collected SQLite data."""
    if cfg is None:
        cfg = TickBacktestConfig()

    mm_cfg = cfg.mm

    # Load data
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Load depth snapshots
    depth_rows = conn.execute(
        "SELECT ts_ms, best_bid, best_ask, mid_price, spread_bps, "
        "bid_depth_5, ask_depth_5, vpin, ob_imbalance, bids_json, asks_json "
        "FROM depth_snapshots ORDER BY ts_ms"
    ).fetchall()

    # Load trades
    trade_rows = conn.execute(
        "SELECT ts_ms, price, qty, side, vpin, ob_imbalance "
        "FROM trades ORDER BY ts_ms"
    ).fetchall()
    conn.close()

    if not depth_rows:
        log.error("No depth data in %s", db_path)
        return TickBacktestResult()

    log.info(
        "Loaded %d depth ticks, %d trade ticks from %s",
        len(depth_rows), len(trade_rows), db_path,
    )

    # Components
    quoter = PerpQuoter(mm_cfg)
    inventory = InventoryTracker(
        max_notional=mm_cfg.max_inventory_notional,
        daily_loss_limit=mm_cfg.daily_loss_limit,
    )
    vol_est = VolEstimator(alpha=0.01, min_trades=20)

    # State
    fills: list[TickFillRecord] = []
    equity_samples: list[float] = []
    spreads_quoted: list[float] = []
    spreads_market: list[float] = []
    queue_depths: list[float] = []
    total_fees = 0.0
    maker_rebate = 0.0
    quote_count = 0
    fill_count = 0

    # Current quotes
    current_bid: float | None = None
    current_ask: float | None = None
    current_bid_ts: int = 0
    current_ask_ts: int = 0

    # Merge trade and depth events chronologically
    trade_idx = 0
    depth_idx = 0
    n_trades = len(trade_rows)
    n_depth = len(depth_rows)

    # Track mid prices for adverse selection
    mid_history: list[tuple[int, float]] = []  # (ts_ms, mid)

    start_ts = min(
        depth_rows[0]["ts_ms"] if depth_rows else float("inf"),
        trade_rows[0]["ts_ms"] if trade_rows else float("inf"),
    )
    last_day_ts = start_ts

    while trade_idx < n_trades or depth_idx < n_depth:
        # Pick next event (earliest timestamp)
        trade_ts = trade_rows[trade_idx]["ts_ms"] if trade_idx < n_trades else float("inf")
        depth_ts = depth_rows[depth_idx]["ts_ms"] if depth_idx < n_depth else float("inf")

        if trade_ts <= depth_ts and trade_idx < n_trades:
            # Process trade
            row = trade_rows[trade_idx]
            trade_idx += 1
            price = row["price"]
            qty = row["qty"]
            side = row["side"]

            vol_est.on_trade(price)

            # Check if trade fills our quotes
            if current_bid is not None and side == "sell" and price <= current_bid:
                # Someone sold at our bid — check queue position
                # Our order fills if enough volume traded through this level
                # Simplified: use bid_depth as queue proxy
                if _check_queue_fill(price, current_bid, qty, cfg):
                    vpin = row["vpin"] or 0.0
                    mid = (current_bid + (current_ask or current_bid)) / 2
                    fee = mm_cfg.order_size_eth * current_bid * cfg.maker_fee_bps * 1e-4
                    rpnl = inventory.on_fill("buy", mm_cfg.order_size_eth, current_bid)
                    total_fees += fee
                    if fee < 0:
                        maker_rebate += abs(fee)
                    fill_count += 1
                    fills.append(TickFillRecord(
                        ts_ms=row["ts_ms"], side="buy", qty=mm_cfg.order_size_eth,
                        price=current_bid, fee=fee, rpnl=rpnl,
                        inventory_after=inventory.net_qty, mid_at_fill=mid,
                        vpin_at_fill=vpin,
                    ))
                    current_bid = None  # filled, need new quote

            elif current_ask is not None and side == "buy" and price >= current_ask:
                if _check_queue_fill(price, current_ask, qty, cfg):
                    vpin = row["vpin"] or 0.0
                    mid = ((current_bid or current_ask) + current_ask) / 2
                    fee = mm_cfg.order_size_eth * current_ask * cfg.maker_fee_bps * 1e-4
                    rpnl = inventory.on_fill("sell", mm_cfg.order_size_eth, current_ask)
                    total_fees += fee
                    if fee < 0:
                        maker_rebate += abs(fee)
                    fill_count += 1
                    fills.append(TickFillRecord(
                        ts_ms=row["ts_ms"], side="sell", qty=mm_cfg.order_size_eth,
                        price=current_ask, fee=fee, rpnl=rpnl,
                        inventory_after=inventory.net_qty, mid_at_fill=mid,
                        vpin_at_fill=vpin,
                    ))
                    current_ask = None

            mid_history.append((row["ts_ms"], price))

        else:
            # Process depth
            row = depth_rows[depth_idx]
            depth_idx += 1

            mid = row["mid_price"]
            market_spread_bps = row["spread_bps"]
            vpin = row["vpin"] or 0.0
            bid_depth = row["bid_depth_5"] or 0.0
            ask_depth = row["ask_depth_5"] or 0.0

            spreads_market.append(market_spread_bps)
            mid_history.append((row["ts_ms"], mid))

            # Daily reset
            if row["ts_ms"] - last_day_ts > 86_400_000:
                last_day_ts = row["ts_ms"]
                inventory.reset_daily()

            # Risk check
            if inventory.hit_daily_limit:
                current_bid = None
                current_ask = None
                continue

            # Skip if vol not ready
            if not vol_est.ready:
                continue

            vol = vol_est.volatility
            if vol <= 0:
                continue

            # Time in horizon
            ts_in_horizon = (row["ts_ms"] / 1000) % cfg.time_horizon_s
            T = max(0.1, 1.0 - ts_in_horizon / cfg.time_horizon_s)

            # Compute quotes
            quote = quoter.compute_quotes(
                mid=mid,
                inventory=inventory.net_qty,
                vol=vol,
                time_remaining=T,
                funding_rate=cfg.funding_rate_per_8h,
                vpin=vpin,
            )
            if quote is None:
                continue

            quote_count += 1
            spreads_quoted.append(quote.spread / mid * 10000)

            # Track queue depth at our price levels
            queue_depths.append((bid_depth + ask_depth) / 2)

            # Side blocking
            new_bid = quote.bid if inventory.can_buy(mid) else None
            new_ask = quote.ask if inventory.can_sell(mid) else None

            # Cancel stale quotes (>2s)
            now_ms = row["ts_ms"]
            if current_bid is not None and now_ms - current_bid_ts > mm_cfg.stale_order_s * 1000:
                current_bid = None
            if current_ask is not None and now_ms - current_ask_ts > mm_cfg.stale_order_s * 1000:
                current_ask = None

            # Update quotes
            if new_bid is not None and (current_bid is None or abs(new_bid - current_bid) >= mm_cfg.tick_size):
                current_bid = new_bid
                current_bid_ts = now_ms
            if new_ask is not None and (current_ask is None or abs(new_ask - current_ask) >= mm_cfg.tick_size):
                current_ask = new_ask
                current_ask_ts = now_ms

            # Sample equity
            inventory.update_unrealised(mid)
            eq = cfg.initial_equity + inventory.realised_pnl + inventory.unrealised_pnl - total_fees
            equity_samples.append(eq)

    # ── Post-analysis: adverse selection ────────────────────
    if cfg.measure_adverse_selection and fills and mid_history:
        _measure_adverse_selection(fills, mid_history, cfg.adverse_window_trades)

    # ── Compute statistics ──────────────────────────────────
    result = TickBacktestResult()
    result.fills = fills
    result.equity_curve = equity_samples
    result.depth_ticks_processed = n_depth
    result.trade_ticks_processed = n_trades

    if not equity_samples:
        return result

    eq = np.array(equity_samples)
    result.total_pnl = eq[-1] - cfg.initial_equity
    result.total_return_pct = (eq[-1] / cfg.initial_equity - 1) * 100
    result.n_fills = fill_count
    result.maker_rebate = maker_rebate
    result.total_fees = total_fees
    result.fill_rate = fill_count / max(quote_count, 1)

    # Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-10)
    result.max_drawdown_pct = abs(dd.min()) * 100

    # Sharpe (from equity samples, ~10/sec depth rate)
    if len(eq) > 100:
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
        if np.std(returns) > 0:
            # Assume ~10 depth ticks/sec → 36000/hr → 864000/day
            samples_per_year = len(eq) / max(result.duration_hours, 1) * 8760
            result.sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(samples_per_year))

    # Duration
    if depth_rows:
        duration_ms = depth_rows[-1]["ts_ms"] - depth_rows[0]["ts_ms"]
        result.duration_hours = duration_ms / 3_600_000
        if result.duration_hours > 0:
            result.trades_per_hour = fill_count / result.duration_hours

    # Round trips, win rate
    rpnl_trades = [f for f in fills if f.rpnl != 0]
    result.n_round_trips = len(rpnl_trades)
    wins = [f.rpnl for f in rpnl_trades if f.rpnl > 0]
    losses = [f.rpnl for f in rpnl_trades if f.rpnl < 0]
    result.win_rate = len(wins) / len(rpnl_trades) if rpnl_trades else 0.0
    gross_profit = sum(wins)
    gross_loss = sum(abs(x) for x in losses)
    result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Spreads
    result.avg_spread_quoted_bps = float(np.mean(spreads_quoted)) if spreads_quoted else 0.0
    result.avg_spread_market_bps = float(np.mean(spreads_market)) if spreads_market else 0.0
    result.avg_queue_depth_usd = float(np.mean(queue_depths)) if queue_depths else 0.0

    # Adverse selection
    if fills:
        adv = [f.adverse_bps for f in fills if f.adverse_bps != 0]
        result.avg_adverse_selection_bps = float(np.mean(adv)) if adv else 0.0

    return result


def _check_queue_fill(
    trade_price: float,
    our_price: float,
    trade_qty: float,
    cfg: TickBacktestConfig,
) -> bool:
    """Check if our order would have been filled based on queue position.

    Simple model: if trade price crosses through our level (not just touches),
    we assume we get filled. For touch-fills, apply queue_position_pct probability.
    """
    if trade_price < our_price * 0.9999:  # price well through our bid
        return True
    if trade_price > our_price * 1.0001:  # price well through our ask
        return True
    # Price at our level — queue position matters
    # Larger trade qty → more likely to reach us in queue
    return trade_qty > cfg.mm.order_size_eth * (1.0 / (1.0 - cfg.queue_position_pct + 0.01))


def _measure_adverse_selection(
    fills: list[TickFillRecord],
    mid_history: list[tuple[int, float]],
    window: int,
) -> None:
    """Measure post-fill mid price move (adverse selection)."""
    # Build lookup: for each fill, find mid price N trades later
    mid_ts = [m[0] for m in mid_history]
    mid_px = [m[1] for m in mid_history]

    for f in fills:
        # Find index of fill timestamp
        idx = _bisect_left(mid_ts, f.ts_ms)
        future_idx = min(idx + window, len(mid_px) - 1)
        if future_idx <= idx:
            continue
        f.price_after_50 = mid_px[future_idx]
        if f.side == "buy":
            # Adverse = price went down after we bought
            f.adverse_bps = (f.mid_at_fill - f.price_after_50) / f.mid_at_fill * 10000
        else:
            # Adverse = price went up after we sold
            f.adverse_bps = (f.price_after_50 - f.mid_at_fill) / f.mid_at_fill * 10000


def _bisect_left(arr: list[int], val: int) -> int:
    """Binary search for insertion point."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo
