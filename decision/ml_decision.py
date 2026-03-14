# decision/ml_decision.py
"""ML-based decision module — sizes positions as % of equity based on ml_score.

NOTE: This module (RustMLDecisionModule / make_ml_decision) is not currently used
in the default production path. The production decision flow uses
MLSignalDecisionModule (backtest_module.py) via LiveInferenceBridge. This wrapper
is kept for potential future Rust-native decision integration and is exercised by
testnet_validation.py and parity tests.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Optional

from types import SimpleNamespace as _SimpleNamespace

from _quant_hotpath import RustMLDecision as _RustMLDecision

from event.header import EventHeader as _EventHeader
from event.types import EventType as _EventType

def _get_close(market: Any) -> Optional[float]:
    """Get close price as float from Rust (close_f) or Python (close) market state."""
    cf = getattr(market, "close_f", None)
    if cf is not None:
        return cf
    c = getattr(market, "close", None)
    return float(c) if c is not None else None


def _get_balance(account: Any) -> float:
    """Get balance as float from Rust (balance_f) or Python (balance) account state."""
    bf = getattr(account, "balance_f", None)
    if bf is not None:
        return bf
    return float(Decimal(str(getattr(account, "balance", 0))))


def _get_qty(pos: Any) -> float:
    """Get qty as float from Rust (qty_f) or Python (qty) position state."""
    qf = getattr(pos, "qty_f", None)
    if qf is not None:
        return qf
    return float(Decimal(str(getattr(pos, "qty", 0))))


class RustMLDecisionModule:
    """Rust-accelerated ML decision module with same API as MLDecisionModule.

    Extracts flat values from snapshot, delegates to Rust state machine,
    wraps results back into event objects.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.symbol = kwargs.get("symbol", "").upper()
        self._rust = _RustMLDecision(**kwargs)

    def decide(self, snapshot: Any) -> Iterable[Any]:
        if isinstance(snapshot, dict):
            market = snapshot.get("market")
            if market is None:
                markets = snapshot.get("markets") or {}
                market = next(iter(markets.values()), None)
            positions = snapshot.get("positions") or {}
            features = snapshot.get("features") or {}
            account = snapshot.get("account")
        else:
            market = getattr(snapshot, "market", None)
            positions = getattr(snapshot, "positions", {})
            features = getattr(snapshot, "features", {})
            account = getattr(snapshot, "account", None)

        if market is None:
            return ()

        close_f = _get_close(market)
        if close_f is None or close_f <= 0:
            return ()

        ml_score = features.get("ml_score")
        if ml_score is None:
            return ()

        pos = positions.get(self.symbol)
        current_qty = _get_qty(pos) if pos else 0.0

        balance = 10000.0
        if account is not None:
            balance = _get_balance(account)

        atr_norm = features.get("atr_norm_14")

        intents = self._rust.decide(close_f, ml_score, current_qty, balance, atr_norm)
        return self._wrap_intents(intents, close_f)

    def _wrap_intents(self, intents: list, price: Any) -> list:
        result = []
        for intent in intents:
            h = _EventHeader.new_root(event_type=_EventType.ORDER, version=1, source="ml_decision")
            result.append(_SimpleNamespace(
                header=h,
                event_type=_EventType.ORDER,
                symbol=self.symbol,
                side=intent.side,
                qty=Decimal(str(intent.qty)),
                price=price,
                order_type="MARKET",
                origin="ml_lgbm",
                reason=intent.reason,
            ))
        return result


def make_ml_decision(**kwargs: Any) -> RustMLDecisionModule:
    """Factory: returns Rust-backed decision module."""
    return RustMLDecisionModule(**kwargs)
