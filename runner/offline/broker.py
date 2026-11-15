"""Old import path of the research broker, which moved to ``trading.adapters.simulated.bar_broker`` (S10).

Only the broker's unit tests (tests/unit/runner/test_broker_*.py) still import this path."""
from trading.adapters.simulated.bar_broker import BarBroker, BrokerFill, ReplayConfig

__all__ = ["BarBroker", "BrokerFill", "ReplayConfig"]
