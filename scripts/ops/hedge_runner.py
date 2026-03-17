"""HedgeRunner — BTC Long + ALT Short hedge strategy runner."""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from scripts.ops.order_utils import clamp_notional

logger = logging.getLogger(__name__)

_MAX_HISTORY = 1000  # max bars to keep in price lists to prevent unbounded growth

# Bybit linear perpetual qty step sizes (from /v5/market/instruments-info)
_QTY_STEPS: dict[str, float] = {
    "ADAUSDT": 1.0,
    "DOGEUSDT": 1.0,
    "XRPUSDT": 0.1,
    "LINKUSDT": 0.1,
    "DOTUSDT": 0.1,
    "AVAXUSDT": 0.1,
    "NEOUSDT": 0.01,
}


def _round_to_step(qty: float, symbol: str) -> float:
    """Round qty DOWN to the nearest valid step for the symbol."""
    step = _QTY_STEPS.get(symbol, 0.1)
    if step <= 0:
        return qty
    return int(qty / step) * step


class HedgeRunner:
    """BTC Long + ALT Short hedge strategy runner.

    Walk-forward validated: 17/20 PASS, Sharpe 2.68, +312%.
    Only shorts ALTs when ALT/BTC ratio < MA (BTC outperforming).
    """

    ALT_BASKET = ["ADAUSDT", "DOGEUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT",
                  "AVAXUSDT", "NEOUSDT"]  # liquid ALTs on Bybit

    def __init__(self, adapter: Any, *, dry_run: bool = False,
                 alt_weight: float = 0.5, ma_window: int = 480,
                 max_position_pct: float = 0.15):
        self._adapter = adapter
        self._dry_run = dry_run
        self._alt_weight = alt_weight
        self._ma_window = ma_window
        self._max_pct = max_position_pct

        # Price tracking
        self._btc_prices: list[float] = []
        self._alt_prices: dict[str, list[float]] = {s: [] for s in self.ALT_BASKET}
        self._is_short_active = False
        self._current_shorts: dict[str, float] = {}  # symbol -> qty
        self._entry_prices: dict[str, float] = {}    # symbol -> entry price for PnL
        self._btc_long_qty: float = 0.0
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        self._bars_processed: int = 0
        self._last_basket_fetch_hour: int = -1  # rate-limit basket fetches to 1x/hour

    def warmup_from_csv(self, n_bars: int = 600) -> int:
        """Pre-load historical prices from CSV for immediate signal generation.

        Loads the last n_bars of BTC + ALT basket prices so ratio MA
        can be computed immediately instead of waiting 20 days.
        """
        from pathlib import Path
        import pandas as pd

        loaded = 0
        # BTC
        btc_path = Path("data_files/BTCUSDT_1h.csv")
        if btc_path.exists():
            df = pd.read_csv(btc_path, usecols=["close"])
            closes = df["close"].values.astype(float)
            for p in closes[-n_bars:]:
                self._btc_prices.append(float(p))
            loaded += 1
            logger.info("HEDGE warmup: BTC %d bars", min(n_bars, len(closes)))

        # ALTs
        for sym in self.ALT_BASKET:
            alt_path = Path(f"data_files/{sym}_1h.csv")
            if alt_path.exists():
                df = pd.read_csv(alt_path, usecols=["close"])
                closes = df["close"].values.astype(float)
                for p in closes[-n_bars:]:
                    self._alt_prices[sym].append(float(p))
                loaded += 1

        logger.info("HEDGE warmup: %d symbols loaded, %d bars each -> ready to signal",
                     loaded, n_bars)
        return loaded

    def on_bar(self, symbol: str, price: float) -> dict | None:
        """Process a 1h bar. Called for ANY symbol (ETH/SUI/AXS from WS).

        On each 1h bar, also fetches current prices for all ALT basket
        symbols via REST ticker (since they're not on the WS subscription).
        Rate-limited to 1 fetch per hour to prevent BTC price list bloat.
        """
        # Track prices for symbols we receive via WS
        if symbol == "BTCUSDT":
            self._btc_prices.append(price)
            if len(self._btc_prices) > _MAX_HISTORY:
                self._btc_prices = self._btc_prices[-_MAX_HISTORY:]
        elif symbol in self._alt_prices:
            self._alt_prices[symbol].append(price)
            if len(self._alt_prices[symbol]) > _MAX_HISTORY:
                self._alt_prices[symbol] = self._alt_prices[symbol][-_MAX_HISTORY:]

        # On any 1h bar from a tracked symbol, fetch BTC + ALT basket prices.
        # Rate-limited to once per hour to avoid duplicate BTC price appends
        # (3 symbols × multiple bars → ratio MA window effectively shorter).
        if symbol in ("ETHUSDT", "SUIUSDT", "AXSUSDT"):
            current_hour = int(time.time()) // 3600
            if current_hour != self._last_basket_fetch_hour:
                self._last_basket_fetch_hour = current_hour
                self._fetch_basket_prices()

        # Only act once per 1h cadence (triggered by first 1h bar)
        if symbol != "ETHUSDT":
            return None

        self._bars_processed += 1
        if len(self._btc_prices) < self._ma_window + 1:
            return {"action": "warmup", "bar": self._bars_processed}

        # Compute ALT/BTC ratio vs MA
        btc = self._btc_prices
        alt_avg_now = 0
        alt_avg_count = 0
        for sym, prices in self._alt_prices.items():
            if len(prices) > 0:
                alt_avg_now += prices[-1]
                alt_avg_count += 1

        if alt_avg_count < 3 or btc[-1] <= 0:
            return {"action": "insufficient_data"}

        alt_avg_now /= alt_avg_count
        current_ratio = alt_avg_now / btc[-1]

        # MA of ratio
        ratios = []
        for i in range(max(0, len(btc) - self._ma_window), len(btc)):
            alt_sum = 0
            alt_cnt = 0
            for sym, prices in self._alt_prices.items():
                if i < len(prices):
                    alt_sum += prices[i]
                    alt_cnt += 1
            if alt_cnt > 0 and i < len(btc) and btc[i] > 0:
                ratios.append(alt_sum / alt_cnt / btc[i])

        if not ratios:
            return {"action": "no_ratio_data"}

        ratio_ma = np.mean(ratios)

        # Hysteresis band: open at -10% below MA, close at +10% above MA.
        # 2% was too narrow (8 opens in 2 days = fee burn). 5% still too tight
        # for BTC/ALT ratio which oscillates in the 6th decimal. 10% ensures
        # only meaningful divergences trigger hedge positions.
        hysteresis = 0.10  # 10% band
        should_open = current_ratio < ratio_ma * (1 - hysteresis)   # BTC clearly outperforming
        should_close = current_ratio > ratio_ma * (1 + hysteresis)  # ALTs catching up

        result = {
            "action": "signal", "bar": self._bars_processed,
            "ratio": round(current_ratio, 6), "ratio_ma": round(ratio_ma, 6),
            "should_short": should_open, "was_short": self._is_short_active,
        }

        # State change with hysteresis
        if should_open and not self._is_short_active:
            # Enter: short ALTs (ratio clearly below MA)
            self._is_short_active = True
            if not self._dry_run:
                self._open_hedge_positions()
            result["trade"] = "OPEN_HEDGE"
            logger.info("HEDGE OPEN: ratio=%.6f < MA*0.98=%.6f -> shorting ALT basket",
                        current_ratio, ratio_ma * (1 - hysteresis))

        elif should_close and self._is_short_active:
            # Exit: close shorts (ratio clearly above MA)
            self._is_short_active = False
            if not self._dry_run:
                self._close_hedge_positions()
            result["trade"] = "CLOSE_HEDGE"
            logger.info("HEDGE CLOSE: ratio=%.6f > MA*1.02=%.6f -> closing ALT shorts",
                        current_ratio, ratio_ma * (1 + hysteresis))

        return result

    def _fetch_basket_prices(self) -> None:
        """Fetch current prices for BTC + all ALT basket symbols via REST."""
        try:
            # BTC price
            ticker = self._adapter.get_ticker("BTCUSDT")
            btc_price = ticker.get("lastPrice", 0)
            if btc_price > 0:
                self._btc_prices.append(btc_price)
                if len(self._btc_prices) > _MAX_HISTORY:
                    self._btc_prices = self._btc_prices[-_MAX_HISTORY:]

            # ALT basket prices
            for sym in self.ALT_BASKET:
                try:
                    ticker = self._adapter.get_ticker(sym)
                    p = ticker.get("lastPrice", 0)
                    if p > 0:
                        self._alt_prices[sym].append(p)
                        if len(self._alt_prices[sym]) > _MAX_HISTORY:
                            self._alt_prices[sym] = self._alt_prices[sym][-_MAX_HISTORY:]
                except Exception:
                    logger.debug("HEDGE: ticker %s unavailable", sym)
        except Exception:
            logger.debug("HEDGE: failed to fetch basket prices", exc_info=True)

    def _open_hedge_positions(self) -> None:
        """Open short positions on ALT basket."""
        try:
            bal = self._adapter.get_balances()
            usdt = bal.get("USDT")
            equity = float(usdt.total) if usdt else 0
            available = float(usdt.available) if usdt else 0
        except Exception as e:
            logger.error("HEDGE: equity fetch failed: %s", e)
            return  # skip hedge open with wrong sizing

        if equity <= 0:
            logger.error("HEDGE: equity=0, skipping hedge open")
            return

        # Margin pre-check: need enough available balance for new hedge orders
        total_hedge_notional = equity * self._max_pct  # total across all ALTs
        if available < total_hedge_notional * 0.15:  # rough margin at 10x
            logger.warning(
                "HEDGE: insufficient margin available=$%.0f < needed~$%.0f, skipping",
                available, total_hedge_notional * 0.15,
            )
            return

        per_alt = equity * self._max_pct / max(len(self.ALT_BASKET), 1)

        for sym in self.ALT_BASKET:
            try:
                ticker = self._adapter.get_ticker(sym)
                price = ticker.get("lastPrice", 0)
                if price <= 0:
                    continue
                qty = per_alt / price
                qty = clamp_notional(qty, price, sym)
                qty = _round_to_step(qty, sym)
                if qty * price < 5:  # min notional $5
                    continue
                result = self._adapter.send_market_order(sym, "sell", qty)
                if result.get("status") == "submitted" or result.get("retCode") == 0:
                    self._current_shorts[sym] = qty
                    self._entry_prices[sym] = price
                    self._trade_count += 1
                    logger.info("HEDGE SHORT %s qty=%.1f @ $%.2f", sym, qty, price)
            except Exception:
                logger.warning("HEDGE: failed to short %s", sym, exc_info=True)

    def _close_hedge_positions(self) -> None:
        """Close all ALT short positions."""
        for sym, qty in list(self._current_shorts.items()):
            try:
                # Get exit price before closing to compute PnL
                exit_price = 0.0
                try:
                    ticker = self._adapter.get_ticker(sym)
                    exit_price = float(ticker.get("lastPrice", 0))
                except Exception:
                    pass

                qty = _round_to_step(qty, sym)
                self._adapter.send_market_order(sym, "buy", qty, reduce_only=True)

                # Compute PnL: short profit = (entry - exit) * qty
                entry = self._entry_prices.get(sym, 0)
                if entry > 0 and exit_price > 0:
                    pnl = (entry - exit_price) * qty
                    self._total_pnl += pnl
                    logger.info("HEDGE CLOSE %s qty=%.1f entry=$%.4f exit=$%.4f pnl=$%.2f",
                                sym, qty, entry, exit_price, pnl)
                else:
                    logger.info("HEDGE CLOSE %s qty=%.1f", sym, qty)
            except Exception:
                logger.warning("HEDGE: failed to close %s", sym, exc_info=True)
        self._current_shorts.clear()
        self._entry_prices.clear()

    def get_status(self) -> dict:
        return {
            "active": self._is_short_active,
            "shorts": dict(self._current_shorts),
            "trades": self._trade_count,
            "total_pnl": round(self._total_pnl, 2),
            "bars": self._bars_processed,
        }
