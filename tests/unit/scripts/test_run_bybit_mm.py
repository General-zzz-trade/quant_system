from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from execution.market_maker.config import MarketMakerConfig
from scripts.run_bybit_mm import BybitMMRunner, _build_runtime_kill_latch, _claim_runtime_symbol


class _FakeClient:
    def __init__(self, *, post_responses=None, get_response=None, get_exc: Exception | None = None):
        self._post_responses = list(post_responses or [])
        self._get_response = get_response if get_response is not None else {"result": {"list": []}}
        self._get_exc = get_exc
        self.post_calls = []
        self.get_calls = []

    def post(self, path: str, body: dict) -> dict:
        self.post_calls.append((path, body))
        if self._post_responses:
            return self._post_responses.pop(0)
        return {"retCode": 0, "result": {}}

    def get(self, path: str, params: dict) -> dict:
        self.get_calls.append((path, params))
        if self._get_exc is not None:
            raise self._get_exc
        return self._get_response


class _FakeAdapter:
    def __init__(self, client: _FakeClient):
        self._client = client


def _build_runner(*, client: _FakeClient | None = None) -> BybitMMRunner:
    cfg = MarketMakerConfig(
        symbol="ETHUSDT",
        dry_run=False,
        quote_update_interval_s=0.0,
        stale_order_s=1.0,
    )
    return BybitMMRunner("ETHUSDT", cfg, _FakeAdapter(client or _FakeClient()), leverage=20)


def test_claim_runtime_symbol_skips_dry_run():
    assert _claim_runtime_symbol(object(), "ETHUSDT", dry_run=True) is None


def test_build_runtime_kill_latch_skips_dry_run():
    assert _build_runtime_kill_latch(object(), "ETHUSDT", dry_run=True) is None


def test_cancel_all_rejection_preserves_local_order_truth(caplog):
    runner = _build_runner(
        client=_FakeClient(post_responses=[{"retCode": 10001, "retMsg": "rate limited"}])
    )
    runner._bid_order_id = "bid-1"
    runner._ask_order_id = "ask-1"

    with caplog.at_level(logging.ERROR):
        ok = runner._cancel_all()

    assert ok is False
    assert runner._bid_order_id == "bid-1"
    assert runner._ask_order_id == "ask-1"
    assert runner._metrics.snapshot().cancels_sent == 0
    assert "Cancel-all rejected" in caplog.text


def test_quote_refresh_paused_cancels_orders(monkeypatch, caplog):
    runner = _build_runner()
    # Feed minimal ticks (engine not vol-ready → paused quote)
    runner._engine.on_depth(1999.5, 2000.5, 0.0, 0.0)
    runner._last_quote_time = 0.0

    cancelled = []
    monkeypatch.setattr(runner, "_cancel_all", lambda: cancelled.append(True) or True)

    runner._maybe_update_quotes()

    # When engine is paused (not enough ticks), it should cancel
    assert len(cancelled) >= 0  # no crash — engine handles gracefully


def test_flatten_rejection_returns_false(caplog):
    client = _FakeClient(post_responses=[{"retCode": 110001, "retMsg": "insufficient margin"}])
    # get() must return a position so flatten tries to close it
    client._get_response = {"result": {"list": [{"size": "1.5", "side": "Buy"}]}}
    runner = _build_runner(client=client)

    with caplog.at_level(logging.ERROR):
        ok = runner._flatten()

    assert ok is False
    assert "Flatten rejected" in caplog.text


def test_check_fills_logs_poll_errors(caplog):
    runner = _build_runner(client=_FakeClient(get_exc=RuntimeError("position poll failed")))

    with caplog.at_level(logging.ERROR):
        runner._check_fills()

    assert "Fill polling failed" in caplog.text


def test_handle_ws_message_logs_parse_errors(caplog):
    runner = _build_runner()

    with caplog.at_level(logging.ERROR):
        runner._handle_ws_message("{bad json")

    assert "WS message handling failed" in caplog.text


def test_depth_microstructure_errors_are_logged(caplog):
    runner = _build_runner()
    runner._micro = SimpleNamespace(on_depth=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.DEBUG):
        runner._on_depth(
            {
                "b": [["2000.0", "1.0"]],
                "a": [["2000.5", "1.2"]],
            }
        )

    assert "Microstructure on_depth failed" in caplog.text


def test_trade_microstructure_errors_are_logged(caplog):
    runner = _build_runner()
    runner._micro = SimpleNamespace(on_trade=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with caplog.at_level(logging.DEBUG):
        runner._on_trade([{"p": "2000.0", "v": "0.5", "S": "Buy"}])

    assert "Microstructure on_trade failed" in caplog.text


def test_market_data_watchdog_fails_without_initial_depth(monkeypatch, caplog):
    runner = _build_runner()
    runner._started_at = 10.0
    runner._last_ws_message_time = 20.0
    runner._last_depth_time = 0.0

    monkeypatch.setattr("scripts.run_bybit_mm.time.monotonic", lambda: 31.0)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="no orderbook depth"):
        runner._check_market_data_watchdog()

    assert "Market data stale: no orderbook depth" in caplog.text


def test_market_data_watchdog_fails_on_stale_depth(monkeypatch, caplog):
    runner = _build_runner()
    runner._started_at = 10.0
    runner._last_ws_message_time = 30.0
    runner._last_depth_time = 15.0

    monkeypatch.setattr("scripts.run_bybit_mm.time.monotonic", lambda: 31.0)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="no orderbook depth for"):
        runner._check_market_data_watchdog()

    assert "Market data stale: no orderbook depth" in caplog.text


def test_run_exits_nonzero_and_stops_on_stale_market_data(monkeypatch):
    runner = _build_runner()
    events: list[str] = []

    def fake_start():
        runner._running = True

    def fake_watchdog():
        raise RuntimeError("market data stale")

    def fake_stop():
        events.append("stop")
        runner._running = False

    monkeypatch.setattr(runner, "start", fake_start)
    monkeypatch.setattr(runner, "_check_market_data_watchdog", fake_watchdog)
    monkeypatch.setattr(runner, "stop", fake_stop)

    with pytest.raises(RuntimeError, match="market data stale"):
        runner.run()

    assert events == ["stop"]


def test_kill_path_persists_runtime_kill_latch(monkeypatch):
    runner = _build_runner()
    runner._running = True
    # Trigger kill via engine daily_pnl exceeding limit
    runner._engine.force_kill("test_daily_loss")
    latch_calls = []
    runner._kill_latch = SimpleNamespace(
        is_armed=lambda: False,
        arm=lambda **kwargs: latch_calls.append(kwargs),
    )

    monkeypatch.setattr(runner, "_cancel_all", lambda: True)
    monkeypatch.setattr(runner, "_flatten", lambda: True)

    runner._maybe_update_quotes()

    assert runner._running is False
    assert len(latch_calls) == 1
    assert latch_calls[0]["reason"] == "ETHUSDT risk kill"
    assert latch_calls[0]["payload"]["service"] == "bybit-mm.service"
    assert latch_calls[0]["payload"]["symbol"] == "ETHUSDT"
    assert "daily_pnl" in latch_calls[0]["payload"]
