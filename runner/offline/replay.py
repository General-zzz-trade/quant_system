"""Closed bars -> production features/AlphaDecisionModule/risk -> simulated fills."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

from _quant_hotpath import RustInferenceBridge
from data.quality.causal_bars import HOUR, aggregate_closed, dataset_manifest, fingerprint, validate_bars
from decision.modules.alpha import AlphaDecisionModule
from decision.modules.alpha_orders import make_close_order, make_open_order
from decision.signals.alpha_signal import SignalDiscretizer
from decision.sizing.adaptive import AdaptivePositionSizer
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from event.header import EventHeader
from event.types import EventType, MarketEvent
from execution.safety.limits import OrderLimiter, OrderLimitsConfig
from features.enriched_computer import EnrichedFeatureComputer
from shared.drawdown_guard import DrawdownGuard
from shared.entry_risk import CompositeRiskGate

from .broker import BarBroker, ReplayConfig
from .models import FrozenPredictor, compute_bar, validate_bundle


class FeatureHook:
    def __init__(self, symbol="BTCUSDT", context=None):
        self.computer = EnrichedFeatureComputer()
        self.features = {}
        self.bar = None
        self.symbol, self.context = symbol, context

    def on_event(self, event):
        if event.event_type == EventType.MARKET:
            self.features = compute_bar(self.computer, self.bar, self.symbol, self.context)
        return self.features


class HistoricalRisk:
    def __init__(self, broker, guard):
        self.broker, self.guard = broker, guard

    def check(self):
        return self.guard.check(now=self.broker.now / 1000)

    def check_would_exceed(self, *, proposed_notional_usd, **kwargs):
        broker = self.broker
        pending = sum((D(r["remaining"]) * D(r["limit"]) for r in broker.pending if not r["reduce_only"]), D(0))
        gross = abs(broker.position.qty) * broker.mark + pending + D(str(proposed_notional_usd))
        allowed = broker.equity > 0 and gross <= broker.equity * D(broker.config.max_exposure)
        return allowed, "exposure cap including pending orders" if not allowed else ""


class SnapshotModule:
    """Use the reconciled simulation ledger for sizing; keep production decisions."""
    def __init__(self, module, broker, start_ms, end_ms):
        self.module, self.broker = module, broker
        self.start_ms, self.end_ms = start_ms, end_ms

    def decide(self, snapshot):
        at = self.broker.now
        # Warm features, regimes and z-scores, but never place a training-period order.
        self.module._execution_ready = self.start_ms < at <= self.end_ms
        return self.module.decide(replace(snapshot, ts=datetime.fromtimestamp(at / 1000, timezone.utc),
                                          account=NS(balance=self.broker.equity,
                                                               balance_f=float(self.broker.equity))))

    def on_fill(self, event):
        self.module.on_fill(event)


class RecordingGate:
    def __init__(self, gate, broker):
        self.gate, self.broker = gate, broker

    def check(self, event):
        result = self.gate.check(event)
        if not result.allowed:
            self.broker.register(event)["rejection"] = result.reason
        return result


class Baseline:
    """Single-position buy/hold or 50-bar trend, sharing broker/risk/cost rules."""
    def __init__(self, broker, mode, start_ms, symbol="BTCUSDT"):
        self.broker, self.mode, self.start_ms = broker, mode, start_ms
        self.symbol = symbol
        self.closes = []

    def decide(self, snapshot):
        price = self.broker.mark
        self.closes.append(price)
        if self.broker.now <= self.start_ms or self.broker.pending:
            return []
        direction = 1 if self.mode == "buy_hold" else (
            1 if price > sum(self.closes[-50:]) / len(self.closes[-50:]) else -1)
        qty = self.broker.position.qty
        if qty * direction < 0:
            return make_close_order(self.symbol, "baseline", float(price), 1 if qty > 0 else -1,
                                    "baseline_flip", qty, D(self.broker.config.qty_step))
        if qty == 0:
            size = self.broker.round_qty(max(D(0), self.broker.equity) * D("0.9") *
                                         D(self.broker.config.max_exposure) / price)
            if size:
                return make_open_order(self.symbol, "baseline", float(price), direction, size)
        return []


def market_event(bar, interval, symbol="BTCUSDT"):
    ts = datetime.fromtimestamp(bar.available_at / 1000, timezone.utc)
    header = EventHeader(f"offline-{interval}-{bar.open_time}", EventType.MARKET, 1,
                         bar.available_at * 1_000_000, "offline")
    return MarketEvent(header=header, ts=ts, symbol=symbol, open=bar.open, high=bar.high,
                       low=bar.low, close=bar.close, volume=bar.volume)


def run_replay(bars, bundle, funding, *, start_ms, config=None, mode="strategy", stop_after=None, synthetic=False,
               symbol="BTCUSDT", context=None):
    validate_bars(bars)
    config = config or ReplayConfig()
    validate_bundle(bundle, start_ms, symbol=symbol)
    if mode not in {"strategy", "buy_hold", "trend"}:
        raise ValueError("invalid replay mode")
    if not bars[0].open_time <= start_ms < bars[-1].available_at:
        raise ValueError("evaluation start outside dataset")
    if stop_after is not None and not 1 <= stop_after <= len(bars):
        raise ValueError("invalid checkpoint bar count")
    end_ms = bars[-1].available_at
    # Explicit 8-hour settlement contract. Missing real data never silently means zero funding.
    required = {b.open_time for b in bars if start_ms <= b.open_time and b.open_time % (8 * HOUR) == 0}
    if not required.issubset(funding):
        raise ValueError("missing 8-hour funding observations")
    broker = BarBroker(config)
    consensus, coords, hooks, modules = {}, {}, {}, {}
    four = {b.available_at: b for b in aggregate_closed(bars)}
    curve, signals, risks = [], [], []
    with TemporaryDirectory(prefix="quant-offline-") as temp:
        guard = DrawdownGuard(Path(temp) / "risk.sqlite", "offline", daily_pct=config.daily_loss_pct,
                              weekly_pct=config.weekly_loss_pct, max_age_seconds=1)
        risk = HistoricalRisk(broker, guard)
        gate = CompositeRiskGate(OrderLimiter(OrderLimitsConfig()), portfolio_reader=risk,
                                 drawdown_guard=risk, venue="offline")
        gate = RecordingGate(gate, broker)
        for interval in ("4h", "1h"):
            hooks[interval] = FeatureHook(symbol, context)
            key = f"{symbol}_4h" if interval == "4h" else symbol
            module = AlphaDecisionModule(symbol=symbol, runner_key=key,
                predictor=FrozenPredictor(bundle["models"][interval]),
                discretizer=SignalDiscretizer(RustInferenceBridge(config.zscore_window, config.zscore_warmup),
                    symbol=symbol, deadzone=config.deadzone, min_hold=config.min_hold, max_hold=config.max_hold),
                sizer=AdaptivePositionSizer(key, step_size=float(config.qty_step), min_size=float(config.qty_step)),
                leverage=1.0, signal_only=interval == "4h", venue="offline", offline=True)
            module.set_consensus(consensus)
            modules[interval] = module
            coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default=symbol, symbols=(symbol,),
                starting_balance=float(config.starting_balance), feature_hook=hooks[interval]))
            coords[interval] = coord

            def emit(event, *, coordinator=coord, tf=interval):
                if event.event_type == EventType.ORDER:
                    broker.register(event)
                elif event.event_type == EventType.SIGNAL:
                    signals.append({"at": broker.now, "tf": tf, "side": event.side, "strength": str(event.strength)})
                elif event.event_type == EventType.RISK:
                    risks.append({"at": broker.now, "tf": tf, "reason": event.rule_id})
                coordinator.emit(event)

            decision = SnapshotModule(module, broker, start_ms, end_ms)
            if interval == "1h" and mode != "strategy":
                decision = Baseline(broker, mode, start_ms, symbol)
            coord.attach_decision_bridge(DecisionBridge(emit, [decision]))
            if interval == "1h":
                def status(oid, state):
                    broker.status_update(oid, state)
                coord.attach_execution_bridge(ExecutionBridge(broker, coord.emit, risk_gate=gate, max_retries=0,
                                                               on_order_status=status))
                broker.status = module.on_order_status if mode == "strategy" else lambda *_: None
                broker.emit = coord.emit
            coord.start()
        try:
            selected = bars[:stop_after] if stop_after else bars
            for bar in selected:
                broker.now, broker.mark = bar.open_time, bar.open
                if bar.open_time >= start_ms and bar.open_time in funding:
                    rate, mark = funding[bar.open_time]
                    broker.settle_funding(bar.open_time, rate, mark)
                broker.process_bar(bar)
                if bar.available_at > start_ms:
                    guard.observe(float(broker.equity), now=broker.now / 1000)
                # At a common close, publish the completed 4h signal before the 1h decision.
                if bar.available_at in four:
                    hooks["4h"].bar = four[bar.available_at]
                    coords["4h"].emit(market_event(four[bar.available_at], "4h", symbol))
                hooks["1h"].bar = bar
                coords["1h"].emit(market_event(bar, "1h", symbol))
                if bar.available_at > start_ms:
                    curve.append(broker.snapshot())
                if mode == "strategy" and modules["1h"]._confirmed_position.qty != broker.position.qty:
                    raise RuntimeError("strategy/broker confirmed positions diverged")
                engine_qty = D(coords["1h"]._store.get_position(symbol).qty) / D(100_000_000)
                if abs(engine_qty - broker.position.qty) > D("0.00000001"):
                    raise RuntimeError("engine/broker confirmed positions diverged")
        finally:
            for coord in coords.values():
                coord.stop()
    result = {"schema": 1, "mode": mode, "model_sha256": bundle["sha256"], "config": config.record(),
              "evaluation_start_ms": start_ms, "processed_bars": len(selected),
              "dataset": dataset_manifest(selected, synthetic=synthetic),
              "orders": broker.order_records(), "fills": broker.fills,
              "funding": broker.funding_log, "equity": curve, "signals": signals, "risk_events": risks,
              "final": broker.snapshot(), "pending_orders": len(broker.pending)}
    result["result_sha256"] = fingerprint(result)
    return result
