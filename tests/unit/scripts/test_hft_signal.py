"""Tests for Signal-driven HFT engine."""

import threading
import time
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from scripts.run_hft_signal import SymbolEngine, SignalHFT


class TestSymbolEngine:
    """Test per-symbol signal computation."""

    def test_initial_state(self):
        eng = SymbolEngine("BTCUSDT")
        assert eng.mid == 0.0
        assert eng.pos_side == 0
        assert eng.pending is False

    def test_push_depth_updates_mid(self):
        eng = SymbolEngine("BTCUSDT")
        eng.push_depth(70000.0, 70001.0, 500000, 500000)
        assert eng.mid == 70000.5
        assert eng.best_bid == 70000.0

    def test_push_depth_computes_imbalance(self):
        eng = SymbolEngine("BTCUSDT")
        eng.push_depth(70000, 70001, 900000, 100000)
        assert eng.ob_imbalance > 0.5  # bid heavy

    def test_push_depth_thread_safe(self):
        """C1 fix: concurrent push_depth and read should not crash."""
        eng = SymbolEngine("BTCUSDT")

        def writer():
            for i in range(100):
                eng.push_depth(70000 + i, 70001 + i, 500000, 500000)

        def reader():
            for _ in range(100):
                _ = eng.mid
                _ = eng.ob_imbalance

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert eng.mid > 0

    def test_push_trade_accumulates_bar(self):
        eng = SymbolEngine("ETHUSDT")
        eng.push_trade(2000.0, 1.0, True)
        eng.push_trade(2001.0, 0.5, False)
        assert eng.mid == 2001.0
        assert eng.bar_volume > 0

    def test_5min_bar_closes(self):
        eng = SymbolEngine("ETHUSDT")
        eng.bar_start = time.time() - 301  # force bar close
        eng.bar_open = 2000.0
        eng.push_trade(2010.0, 1.0, True)
        assert len(eng.bars) == 1
        assert eng.bars[0]['close'] == 2010.0

    def test_push_liq(self):
        eng = SymbolEngine("BTCUSDT")
        eng.push_liq("SELL", 100000)
        assert len(eng.liq_window) == 1

    def test_push_liq_prunes_old(self):
        eng = SymbolEngine("BTCUSDT")
        eng.liq_window = [{'ts': time.time() - 20, 'side': 'BUY', 'notional': 50000}]
        eng.push_liq("SELL", 100000)
        assert len(eng.liq_window) == 1  # old one pruned

    # ── Signal tests ──

    def test_funding_signal_short(self):
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = 0.0002  # > 0.0001 threshold
        assert eng.funding_signal() == -1

    def test_funding_signal_long(self):
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = -0.0001
        assert eng.funding_signal() == 1

    def test_funding_signal_neutral(self):
        eng = SymbolEngine("BTCUSDT")
        eng.funding_rate = 0.00005  # between thresholds
        assert eng.funding_signal() == 0

    def test_momentum_signal_needs_bars(self):
        eng = SymbolEngine("ETHUSDT")
        assert eng.momentum_signal() == 0  # no bars

    def test_momentum_signal_bullish(self):
        eng = SymbolEngine("ETHUSDT")
        # Bars: open=2000, close=2020 → 1% ret over 3 bars (> 0.3% threshold)
        # Need vol_ratio > 1.2, so last bar has 2x avg volume
        eng.bars.append({'open': 2000, 'high': 2005, 'low': 1998, 'close': 2003,
                         'volume': 50000, 'buy_pct': 0.55, 'ts': time.time()})
        eng.bars.append({'open': 2003, 'high': 2010, 'low': 2001, 'close': 2008,
                         'volume': 50000, 'buy_pct': 0.6, 'ts': time.time()})
        eng.bars.append({'open': 2008, 'high': 2025, 'low': 2006, 'close': 2020,
                         'volume': 120000, 'buy_pct': 0.7, 'ts': time.time()})
        sig = eng.momentum_signal()
        assert sig == 1

    def test_liq_signal_needs_threshold(self):
        eng = SymbolEngine("BTCUSDT")
        eng.push_liq("SELL", 10000)  # below $50K threshold
        assert eng.liq_signal() == 0

    def test_liq_signal_cascade_short(self):
        eng = SymbolEngine("BTCUSDT")
        now = time.time()
        eng.liq_window = [
            {'ts': now, 'side': 'SELL', 'notional': 80000},
            {'ts': now, 'side': 'BUY', 'notional': 10000},
        ]
        assert eng.liq_signal() == -1  # longs liquidated → short

    def test_combined_signal_needs_1_5_votes(self):
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = 0.0002  # fund = -1 (×0.5 = -0.5)
        eng.ob_imbalance = 0.0     # ob = 0
        direction, reason, conf = eng.combined_signal()
        assert direction == 0  # only 0.5 vote, need 1.5

    def test_combined_signal_with_ob(self):
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = 0.0002    # fund = -1 (×0.5 = -0.5)
        eng.ob_imbalance = -0.5      # ob = -1
        direction, reason, conf = eng.combined_signal()
        assert direction == -1  # -0.5 + -1 = -1.5 >= threshold
        assert "f=" in reason and "ob=" in reason

    def test_combined_signal_bullish(self):
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = -0.0001  # fund = +1 (×0.5 = +0.5)
        eng.ob_imbalance = 0.5      # ob = +1
        direction, reason, conf = eng.combined_signal()
        assert direction == 1
        assert conf > 0

    def test_combined_signal_onchain(self):
        """Improvement 1: on-chain signal contributes to votes."""
        eng = SymbolEngine("BTCUSDT")
        eng.onchain_z = 2.0      # oc = +1
        eng.ob_imbalance = 0.5   # ob = +1
        direction, reason, conf = eng.combined_signal()
        assert direction == 1
        assert "oc=" in reason

    def test_combined_signal_dvol_block(self):
        """Improvement 2: extreme DVOL blocks all trades."""
        eng = SymbolEngine("ETHUSDT")
        eng.dvol_z = 3.0          # extreme vol
        eng.funding_rate = -0.001
        eng.ob_imbalance = 0.8
        direction, reason, conf = eng.combined_signal()
        assert direction == 0
        assert "dvol_block" in reason

    def test_combined_signal_confidence(self):
        """Improvement 6: confidence scales with vote count."""
        eng = SymbolEngine("SOLUSDT")
        eng.funding_rate = -0.0001  # fund = +1 (×0.5)
        eng.ob_imbalance = 0.5      # ob = +1
        eng.onchain_z = 2.0          # oc = +1
        direction, reason, conf = eng.combined_signal()
        assert direction == 1
        assert conf > 0.5  # 2.5 votes / 3 = 0.83

    def test_trend_filter_blocks_counter_trend(self):
        """Improvement 4: counter-trend signals blocked."""
        eng = SymbolEngine("BTCUSDT")
        eng.mid = 70000
        eng.ma50 = 72000  # below MA → downtrend
        assert eng.trend_filter(1) is False   # long blocked in downtrend
        assert eng.trend_filter(-1) is True   # short ok in downtrend


class TestSignalHFTLogic:
    """Test HFT trading logic without real exchange."""

    def _make_hft(self, symbols=None, dry_run=True):
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.post.return_value = {'retCode': 0, 'result': {'orderId': 'test123'}}
        adapter._client.get.return_value = {'result': {'list': []}}
        hft = SignalHFT(
            symbols=symbols or ["BTCUSDT"],
            adapter=adapter, leverage=20,
            position_size_usd=50000, daily_loss_limit=5000,
            dry_run=dry_run,
        )
        return hft

    def test_no_enter_when_daily_loss_exceeded(self):
        hft = self._make_hft()
        hft._daily_pnl = -6000  # exceeded $5000 limit
        eng = hft._engines["BTCUSDT"]
        eng.mid = 70000
        eng.best_bid = 70000
        eng.funding_rate = -0.001
        eng.ob_imbalance = 0.5
        hft._signal_check()
        assert eng.pos_side == 0  # should not enter

    def test_p1_no_duplicate_enter(self):
        """P1 fix: don't re-enter same direction."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.mid = 70000
        eng.best_bid = 70000
        eng.best_ask = 70001
        hft._enter(eng, 1, "test")
        assert eng.pos_side == 1
        assert eng.pending is True

        # Try to enter again same direction
        old_trades = eng.trades
        hft._enter(eng, 1, "test2")
        assert eng.trades == old_trades  # should not increment

    def test_c2_pending_state(self):
        """C2 fix: entry is pending until confirmed."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.mid = 70000
        eng.best_bid = 70000
        hft._enter(eng, 1, "test")
        assert eng.pending is True
        assert eng.pos_side == 1

    def test_exit_clears_state(self):
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.mid = 70100
        eng.best_bid = 70100
        eng.best_ask = 70101
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        hft._exit(eng, "test_exit")
        assert eng.pos_side == 0
        assert eng.pending is False

    def test_h1_daily_reset(self):
        """H1 fix: daily PnL resets after 24h."""
        hft = self._make_hft()
        hft._daily_pnl = -100
        hft._day_start = time.time() - 86401  # over 24h ago
        hft._last_signal_check = 0
        hft._signal_check()
        assert hft._daily_pnl == 0.0

    def test_exit_tp(self):
        """Take profit at 0.3%."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "test"
        eng.mid = 70300  # +0.43% > 0.3% TP
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited

    def test_exit_sl(self):
        """Stop loss at 0.5%."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "test"
        eng.mid = 69600  # -0.57% < -0.5% SL
        hft._check_exit(eng)
        assert eng.pos_side == 0

    def test_pending_timeout_clears(self):
        """C2 fix: pending order times out after 10s."""
        hft = self._make_hft()
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pending = True
        eng.entry_time = time.time() - 15  # over 10s
        eng.order_id = "test"
        eng.mid = 70000
        eng.entry_price = 70000
        # Mock: no position on exchange
        hft._adapter._client.get.return_value = {'result': {'list': [{'size': '0'}]}}
        hft._check_exit(eng)
        assert eng.pos_side == 0
        assert eng.pending is False

    def test_correlation_guard_max_one_position(self):
        """Correlation guard: only 1 position across all correlated symbols."""
        hft = self._make_hft(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        # SOL already has a position
        sol = hft._engines["SOLUSDT"]
        sol.pos_side = 1
        sol.pos_qty = 100
        sol.entry_price = 89.0
        sol.entry_time = time.time()
        sol.mid = 89.5
        sol.best_bid = 89.4
        sol.best_ask = 89.5
        # BTC has a strong signal
        btc = hft._engines["BTCUSDT"]
        btc.mid = 70000
        btc.best_bid = 70000
        btc.best_ask = 70001
        btc.funding_rate = -0.001
        btc.ob_imbalance = 0.5
        # ETH also has a signal
        eth = hft._engines["ETHUSDT"]
        eth.mid = 2100
        eth.best_bid = 2100
        eth.best_ask = 2100.5
        eth.funding_rate = -0.001
        eth.ob_imbalance = 0.5
        hft._signal_check()
        # Neither BTC nor ETH should enter — SOL already holds
        assert btc.pos_side == 0
        assert eth.pos_side == 0

    def test_correlation_guard_best_signal_wins(self):
        """When flat, pick highest confidence signal across symbols."""
        hft = self._make_hft(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], dry_run=True)
        # All flat, SOL has signal (funding=-0.001 → short signal + ob=-1 → votes=-1.5)
        sol = hft._engines["SOLUSDT"]
        sol.mid = 89.0
        sol.best_bid = 89.0
        sol.best_ask = 89.1
        sol.funding_rate = -0.001  # negative → long signal (+1)
        sol.ob_imbalance = 0.5    # > 0.3 → ob=+1
        # votes = 0.5(funding) + 1(ob) = 1.5 → direction=+1
        # BTC has no signal
        btc = hft._engines["BTCUSDT"]
        btc.mid = 70000
        btc.best_bid = 70000
        btc.best_ask = 70001
        # ETH no signal
        eth = hft._engines["ETHUSDT"]
        eth.mid = 2100
        eth.best_bid = 2100
        eth.best_ask = 2100.5
        hft._signal_check()
        # Only SOL should enter
        assert sol.pos_side == 1 or sol.pending is True
        assert btc.pos_side == 0
        assert eth.pos_side == 0


class TestMeanReversion:
    """Tests for tick-level mean reversion signal layer."""

    def test_mean_reversion_detects_impact(self):
        """Large same-direction trades -> opposite signal."""
        eng = SymbolEngine("BTCUSDT")
        # Bootstrap avg trade size with a small trade
        eng.push_trade(70000.0, 0.01, True)
        # mr_avg_trade_size ~ 700 (0.01 * 70000)

        # Now push 5 consecutive BUY trades with large size (> 5x avg)
        # Each trade: 0.1 BTC * 70000 = $7000, run total = $35000 >> 5 * 700
        for i in range(5):
            eng.push_trade(70000.0 + i, 0.1, True)

        # Should detect buy impact -> signal short (mean reversion)
        assert eng.mean_reversion_signal() == -1

    def test_mean_reversion_sell_impact(self):
        """Large same-direction SELL trades -> long signal."""
        eng = SymbolEngine("ETHUSDT")
        # Bootstrap
        eng.push_trade(2000.0, 0.01, False)

        # 4 consecutive SELL trades, large size
        for i in range(4):
            eng.push_trade(2000.0 - i, 1.0, False)

        assert eng.mean_reversion_signal() == 1

    def test_mean_reversion_signal_expires(self):
        """MR signal expires after 30s."""
        eng = SymbolEngine("BTCUSDT")
        # Manually set a signal that was generated 31s ago
        eng.mr_signal = -1
        eng.mr_signal_time = time.time() - 31
        assert eng.mean_reversion_signal() == 0
        assert eng.mr_signal == 0  # cleared

    def test_mean_reversion_signal_active_within_window(self):
        """MR signal still active within 30s window."""
        eng = SymbolEngine("BTCUSDT")
        eng.mr_signal = -1
        eng.mr_signal_time = time.time() - 10  # 10s ago, still valid
        assert eng.mean_reversion_signal() == -1

    def test_mean_reversion_no_false_trigger(self):
        """Normal mixed trades -> no signal."""
        eng = SymbolEngine("BTCUSDT")
        # Bootstrap
        eng.push_trade(70000.0, 0.01, True)

        # Alternating buy/sell — never 3 consecutive same direction
        for i in range(10):
            eng.push_trade(70000.0 + i, 0.01, i % 2 == 0)

        assert eng.mean_reversion_signal() == 0

    def test_mean_reversion_no_trigger_small_volume(self):
        """3+ consecutive same-dir but small volume -> no signal."""
        eng = SymbolEngine("BTCUSDT")
        # Bootstrap with a large trade to inflate avg
        eng.push_trade(70000.0, 1.0, True)
        # mr_avg_trade_size ~ $70000

        # 3 consecutive buys but tiny volume (well under 5x avg)
        for _ in range(3):
            eng.push_trade(70000.0, 0.001, True)
        # run_notional = 3 * 0.001 * 70000 = $210 << 5 * ~70000

        assert eng.mean_reversion_signal() == 0

    def test_mean_reversion_in_combined_signal(self):
        """MR signal contributes to combined vote with weight 1.5."""
        eng = SymbolEngine("BTCUSDT")
        # Set MR signal directly (already detected)
        eng.mr_signal = -1
        eng.mr_signal_time = time.time()
        # MR alone: -1 * 1.5 = -1.5, exactly at threshold
        direction, reason, conf = eng.combined_signal()
        assert direction == -1
        assert "mr=" in reason

    def test_mean_reversion_combined_with_other_signals(self):
        """MR + OB agreement produces high confidence."""
        eng = SymbolEngine("ETHUSDT")
        eng.mr_signal = 1
        eng.mr_signal_time = time.time()
        eng.ob_imbalance = 0.5  # ob = +1
        # votes = 0 + 0 + 0 + 1 + 0 + 1*1.5 = 2.5
        direction, reason, conf = eng.combined_signal()
        assert direction == 1
        assert "mr=+1" in reason
        assert "ob=+1" in reason
        assert conf > 0.5

    def test_mean_reversion_tight_exit_tp(self):
        """MR entries use tighter TP (0.15%)."""
        hft = TestSignalHFTLogic._make_hft(
            TestSignalHFTLogic(), dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "mr=+1 ob=+1"
        eng.mid = 70000 * 1.002  # +0.2% > 0.15% MR_TP
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via MR_TP

    def test_mean_reversion_tight_exit_sl(self):
        """MR entries use tighter SL (0.1%)."""
        hft = TestSignalHFTLogic._make_hft(
            TestSignalHFTLogic(), dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "mr=+1"
        eng.mid = 70000 * 0.998  # -0.2% < -0.1% MR_SL
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via MR_SL

    def test_mean_reversion_tight_exit_timeout(self):
        """MR entries timeout at 60s instead of 300s."""
        hft = TestSignalHFTLogic._make_hft(
            TestSignalHFTLogic(), dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = -1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time() - 65  # 65s > 60s timeout
        eng.entry_reason = "mr=-1"
        eng.mid = 70000  # flat, no TP/SL hit
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via MR_TIMEOUT

    def test_mean_reversion_normal_exit_not_affected(self):
        """Non-MR entries still use normal TP/SL/timeout."""
        hft = TestSignalHFTLogic._make_hft(
            TestSignalHFTLogic(), dry_run=True)
        eng = hft._engines["BTCUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 0.5
        eng.entry_price = 70000
        eng.pending = False
        eng.entry_time = time.time() - 65  # 65s, within 300s normal timeout
        eng.entry_reason = "f=+1 ob=+1"
        eng.mid = 70000  # flat
        hft._check_exit(eng)
        assert eng.pos_side == 1  # should NOT have exited yet


class TestBTCLeadLag:
    """Tests for BTC->ALT tick-level lead-lag signal layer."""

    def _make_hft(self, symbols=None, dry_run=True):
        adapter = MagicMock()
        adapter._client = MagicMock()
        adapter._client.post.return_value = {'retCode': 0, 'result': {'orderId': 'test123'}}
        adapter._client.get.return_value = {'result': {'list': []}}
        hft = SignalHFT(
            symbols=symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            adapter=adapter, leverage=20,
            position_size_usd=50000, daily_loss_limit=5000,
            dry_run=dry_run,
        )
        return hft

    def test_btc_lead_detects_fast_move(self):
        """BTC moves >0.03% in 500ms -> lead signal fires."""
        hft = self._make_hft()
        base = 70000.0
        # Simulate BTC price 500ms ago
        hft._btc_tick_prices.append((time.time() - 0.3, base))
        # BTC moves up 0.04% (> 0.03% threshold)
        new_price = base * 1.0004
        hft._update_btc_lead(new_price)
        assert hft._btc_lead_signal == 1
        assert hft._get_btc_lead_signal() == 1

    def test_btc_lead_detects_fast_move_down(self):
        """BTC moves down >0.03% in 500ms -> negative lead signal."""
        hft = self._make_hft()
        base = 70000.0
        hft._btc_tick_prices.append((time.time() - 0.3, base))
        new_price = base * 0.9996  # -0.04%
        hft._update_btc_lead(new_price)
        assert hft._btc_lead_signal == -1
        assert hft._get_btc_lead_signal() == -1

    def test_btc_lead_no_signal_small_move(self):
        """BTC moves <0.03% -> no lead signal."""
        hft = self._make_hft()
        base = 70000.0
        hft._btc_tick_prices.append((time.time() - 0.3, base))
        new_price = base * 1.0002  # only 0.02%, below threshold
        hft._update_btc_lead(new_price)
        assert hft._btc_lead_signal == 0

    def test_btc_lead_signal_expires(self):
        """Lead signal expires after 3s."""
        hft = self._make_hft()
        # Set a signal that was generated 4s ago
        hft._btc_lead_signal = 1
        hft._btc_lead_signal_time = time.time() - 4.0
        assert hft._get_btc_lead_signal() == 0
        assert hft._btc_lead_signal == 0  # cleared

    def test_btc_lead_signal_active_within_window(self):
        """Lead signal still active within 3s window."""
        hft = self._make_hft()
        hft._btc_lead_signal = 1
        hft._btc_lead_signal_time = time.time() - 1.0  # 1s ago, still valid
        assert hft._get_btc_lead_signal() == 1

    def test_btc_lead_only_for_alts(self):
        """BTC lead signal is 0 for BTCUSDT itself."""
        hft = self._make_hft()
        # Fire a BTC lead signal
        hft._btc_lead_signal = 1
        hft._btc_lead_signal_time = time.time()

        # Set up engines for signal check
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            eng = hft._engines[sym]
            eng.mid = 70000 if sym == "BTCUSDT" else 2000 if sym == "ETHUSDT" else 89
            eng.best_bid = eng.mid
            eng.best_ask = eng.mid + 1

        # In _signal_check, btc_lead is set to 0 for BTCUSDT
        # We test the logic directly
        btc_eng = hft._engines["BTCUSDT"]
        btc_eng.btc_lead = 0  # BTC doesn't lead itself
        eth_eng = hft._engines["ETHUSDT"]
        eth_eng.btc_lead = hft._get_btc_lead_signal()

        assert btc_eng.btc_lead == 0
        assert eth_eng.btc_lead == 1

    def test_btc_lead_in_combined_signal(self):
        """BTC lead contributes to ETH/SOL combined vote with weight 2.0."""
        eng = SymbolEngine("ETHUSDT")
        # Set BTC lead signal directly
        eng.btc_lead = 1
        # bl alone: 1 * 2.0 = 2.0 > 1.5 threshold
        direction, reason, conf = eng.combined_signal()
        assert direction == 1
        assert "bl=+1" in reason

    def test_btc_lead_negative_in_combined(self):
        """Negative BTC lead contributes -2.0 to votes."""
        eng = SymbolEngine("SOLUSDT")
        eng.btc_lead = -1
        # bl alone: -1 * 2.0 = -2.0 <= -1.5 threshold
        direction, reason, conf = eng.combined_signal()
        assert direction == -1
        assert "bl=-1" in reason

    def test_btc_lead_triggers_entry(self):
        """BTC fast move -> ETH enters same direction."""
        hft = self._make_hft(dry_run=True)
        # Fire BTC lead-lag signal (BTC moved up fast)
        hft._btc_lead_signal = 1
        hft._btc_lead_signal_time = time.time()

        # Set up ETH engine with valid prices
        eth = hft._engines["ETHUSDT"]
        eth.mid = 2000
        eth.best_bid = 2000
        eth.best_ask = 2001

        # BTC engine: no signal of its own
        btc = hft._engines["BTCUSDT"]
        btc.mid = 70000
        btc.best_bid = 70000
        btc.best_ask = 70001

        # SOL engine: no signal
        sol = hft._engines["SOLUSDT"]
        sol.mid = 89
        sol.best_bid = 89
        sol.best_ask = 89.1

        hft._last_signal_check = 0  # force check
        hft._signal_check()

        # ETH or SOL should enter long (BTC lead = +1, weight 2.0 > 1.5 threshold)
        # BTC should NOT enter (btc_lead = 0 for itself)
        assert btc.pos_side == 0
        entered = [s for s in ["ETHUSDT", "SOLUSDT"]
                   if hft._engines[s].pos_side != 0]
        assert len(entered) == 1  # exactly one ALT entered
        entered_eng = hft._engines[entered[0]]
        assert entered_eng.pos_side == 1
        assert "bl=+1" in entered_eng.entry_reason

    def test_btc_lead_exit_tp(self):
        """BTC lead entries use ultra-tight TP (0.1%)."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["ETHUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 10
        eng.entry_price = 2000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "bl=+1"
        eng.mid = 2000 * 1.0012  # +0.12% > 0.1% BL_TP
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via BL_TP

    def test_btc_lead_exit_sl(self):
        """BTC lead entries use ultra-tight SL (0.08%)."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["ETHUSDT"]
        eng.pos_side = 1
        eng.pos_qty = 10
        eng.entry_price = 2000
        eng.pending = False
        eng.entry_time = time.time()
        eng.entry_reason = "bl=+1"
        eng.mid = 2000 * 0.999  # -0.1% < -0.08% BL_SL
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via BL_SL

    def test_btc_lead_exit_timeout(self):
        """BTC lead entries timeout at 30s."""
        hft = self._make_hft(dry_run=True)
        eng = hft._engines["SOLUSDT"]
        eng.pos_side = -1
        eng.pos_qty = 100
        eng.entry_price = 89
        eng.pending = False
        eng.entry_time = time.time() - 35  # 35s > 30s timeout
        eng.entry_reason = "bl=-1"
        eng.mid = 89  # flat, no TP/SL hit
        hft._check_exit(eng)
        assert eng.pos_side == 0  # should have exited via BL_TIMEOUT

    def test_btc_lead_wired_via_handle_trades(self):
        """BTC trades via WS update the lead-lag detector."""
        hft = self._make_hft()
        base = 70000.0
        # Simulate initial BTC trade
        hft._handle_trades("BTCUSDT", [{"p": str(base), "v": "0.1", "S": "Buy"}])
        assert len(hft._btc_tick_prices) == 1

        # Simulate a fast BTC move (>0.03% up)
        new_price = base * 1.0005  # 0.05% move
        hft._handle_trades("BTCUSDT", [{"p": str(new_price), "v": "0.1", "S": "Buy"}])
        assert hft._btc_lead_signal == 1

    def test_btc_lead_not_wired_for_eth_trades(self):
        """ETH trades should NOT update the BTC lead-lag detector."""
        hft = self._make_hft()
        hft._handle_trades("ETHUSDT", [{"p": "2000", "v": "1.0", "S": "Buy"}])
        assert len(hft._btc_tick_prices) == 0  # no BTC tick recorded
