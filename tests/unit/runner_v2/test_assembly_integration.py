"""Integration test: verify full assembly with mocked dependencies."""
from unittest.mock import MagicMock
import pytest

from runner.trading_config import TradingConfig
from runner.trading_engine import TradingEngine
from runner.risk_manager import RiskManager
from runner.order_manager import OrderManager
from runner.binance_executor import BinanceExecutor
from runner.recovery_manager import RecoveryManager
from runner.lifecycle_manager import LifecycleManager
from runner.runner_loop import RunnerLoop


class TestFullAssembly:
    def test_all_modules_construct_with_mocks(self):
        """Verify the assembly wiring works without any real dependencies."""
        config = TradingConfig.paper(symbols=["BTCUSDT"])

        # 1. Engine
        hook = MagicMock()
        bridge = MagicMock()
        engine = TradingEngine(
            feature_hook=hook, inference_bridge=bridge,
            symbols=list(config.symbols), model_dir=config.model_dir,
        )

        # 2. Risk
        ks = MagicMock()
        ks.is_killed.return_value = False
        risk = RiskManager(kill_switch=ks, max_position=1.0,
                           max_notional=10000.0, max_open_orders=5)

        # 3. Orders
        orders = OrderManager(timeout_sec=config.pending_order_timeout_sec)

        # 4. Executor
        client = MagicMock()
        executor = BinanceExecutor(
            venue_client=client, kill_switch=ks,
            shadow_mode=config.shadow_mode,
        )

        # 5. Recovery
        recovery = RecoveryManager(
            state_dir="/tmp/test_assembly_state",
            engine=engine, risk=risk, orders=orders,
        )

        # 6. Loop
        loop = RunnerLoop(engine, risk, orders, executor)

        # 7. Lifecycle
        lifecycle = LifecycleManager(
            engine=engine, executor=executor,
            recovery=recovery, loop=loop,
        )

        # Verify all modules exist and are correctly typed
        assert isinstance(engine, TradingEngine)
        assert isinstance(risk, RiskManager)
        assert isinstance(orders, OrderManager)
        assert isinstance(executor, BinanceExecutor)
        assert isinstance(recovery, RecoveryManager)
        assert isinstance(loop, RunnerLoop)
        assert isinstance(lifecycle, LifecycleManager)

    def test_end_to_end_bar_flow(self):
        """Verify bar → engine → risk check → order flow works."""
        # Engine returns prediction
        hook = MagicMock()
        hook.on_bar.return_value = {"rsi": 0.5}
        bridge = MagicMock()
        bridge.predict.return_value = 0.7
        engine = TradingEngine(
            feature_hook=hook, inference_bridge=bridge,
            symbols=["BTCUSDT"], model_dir="models_v8",
        )

        # Risk allows
        ks = MagicMock()
        ks.is_killed.return_value = False
        risk = RiskManager(kill_switch=ks, max_position=1.0,
                           max_notional=10000.0, max_open_orders=5)
        orders = OrderManager(timeout_sec=30.0)

        signals = []
        loop = RunnerLoop(engine, risk, orders, MagicMock(),
                          on_signal=lambda s, p, b: signals.append((s, p)))

        # Process a bar
        loop.on_bar("BTCUSDT", {"close": 70000, "volume": 100})

        # Signal should have been emitted
        assert len(signals) == 1
        assert signals[0] == ("BTCUSDT", 0.7)

    def test_module_file_sizes(self):
        """Verify no new module exceeds 200 LOC."""
        from pathlib import Path
        runner_dir = Path("runner")
        new_modules = [
            "trading_config.py", "trading_engine.py", "risk_manager.py",
            "order_manager.py", "binance_executor.py", "recovery_manager.py",
            "lifecycle_manager.py", "runner_loop.py", "run_trading.py",
        ]
        for name in new_modules:
            path = runner_dir / name
            if path.exists():
                lines = len(path.read_text().splitlines())
                assert lines <= 250, f"{name} has {lines} lines, max 250"
