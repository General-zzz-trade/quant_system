from __future__ import annotations

from datetime import datetime


def test_parse_log_timestamp_supports_milliseconds():
    from scripts.ops.runtime_health_check import parse_log_timestamp

    ts = parse_log_timestamp("2026-03-19 12:34:56,789 INFO bybit_mm FILL buy 1.0")
    assert ts == datetime(2026, 3, 19, 12, 34, 56)


def test_venue_symbols_collapses_timeframe_aliases():
    from scripts.ops.runtime_health_check import venue_symbols

    assert venue_symbols(["ETHUSDT", "ETHUSDT_15m", "BTCUSDT"]) == (
        "ETHUSDT",
        "BTCUSDT",
    )


def test_merge_runtime_env_prefers_process_env(monkeypatch, tmp_path):
    from scripts.ops.runtime_health_check import merge_runtime_env

    env_file = tmp_path / ".env"
    env_file.write_text("BYBIT_API_KEY=file_key\nBYBIT_API_SECRET=file_secret\n")
    monkeypatch.setenv("BYBIT_API_KEY", "process_key")

    merged = merge_runtime_env(str(env_file))

    assert merged["BYBIT_API_KEY"] == "process_key"
    assert merged["BYBIT_API_SECRET"] == "file_secret"


def test_summarize_log_health_counts_recent_markers(tmp_path):
    from scripts.ops.runtime_health_check import ALPHA_SPEC, summarize_log_health

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text(
        "\n".join(
            [
                "2026-03-19 19:58:00,000 INFO scripts.ops.run_bybit_alpha: WS HEARTBEAT sigs={'ETHUSDT': 0}",
                "2026-03-19 19:59:30,000 INFO scripts.ops.portfolio_combiner: COMBO OPEN buy 0.23 @ $2177.2",
            ]
        )
    )

    result = summarize_log_health(
        spec=ALPHA_SPEC,
        log_path=str(log),
        now=now,
        log_max_age_s=300,
        activity_window_s=300,
    )

    assert result["fresh"] is True
    assert result["marker_counts"]["WS HEARTBEAT"] == 1
    assert result["marker_counts"]["COMBO OPEN"] == 1


def test_evaluate_alpha_health_passes_on_fresh_heartbeat(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text(
        "2026-03-19 19:59:00,000 INFO scripts.ops.run_bybit_alpha: WS HEARTBEAT sigs={'ETHUSDT': 0}\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {"status": "skipped", "reason": "missing_credentials"},
    )

    result = rhc.evaluate_service_health(
        spec=rhc.ALPHA_SPEC,
        log_path=str(log),
        now=now,
    )

    assert result["ok"] is True
    assert result["problems"] == []


def test_evaluate_alpha_health_fails_when_portfolio_is_killed(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text(
        "2026-03-19 19:59:00,000 INFO scripts.ops.run_bybit_alpha: "
        "WS HEARTBEAT sigs={'ETHUSDT': 0} pm={'positions': {}, 'killed': True}\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {"status": "ok", "reason": "", "armed": False, "path": None, "record": None},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {
            "status": "ok",
            "reason": "",
            "positions": 1,
            "open_orders": 2,
            "recent_fills": 5,
            "usdt_total": "49531.77",
        },
    )

    result = rhc.evaluate_service_health(
        spec=rhc.ALPHA_SPEC,
        log_path=str(log),
        now=now,
    )

    assert result["ok"] is False
    assert "portfolio_killed" in result["problems"]
    assert result["log"]["kill_active"] is True


def test_evaluate_alpha_health_does_not_trust_account_only_activity(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text("2026-03-19 19:59:00,000 INFO scripts.ops.run_bybit_alpha: Boot complete\n")
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {"status": "ok", "reason": "", "armed": False, "path": None, "record": None},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {
            "status": "ok",
            "reason": "",
            "positions": 1,
            "open_orders": 2,
            "recent_fills": 5,
            "usdt_total": "49531.77",
        },
    )

    result = rhc.evaluate_service_health(
        spec=rhc.ALPHA_SPEC,
        log_path=str(log),
        now=now,
    )

    assert result["ok"] is False
    assert "no_recent_runtime_evidence" in result["problems"]


def test_evaluate_alpha_health_fails_when_log_is_stale(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text(
        "2026-03-19 19:40:00,000 INFO scripts.ops.run_bybit_alpha: WS HEARTBEAT sigs={'ETHUSDT': 0}\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {"status": "ok", "reason": "", "armed": False, "path": None, "record": None},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {"status": "skipped", "reason": "missing_credentials"},
    )

    result = rhc.evaluate_service_health(
        spec=rhc.ALPHA_SPEC,
        log_path=str(log),
        now=now,
    )

    assert result["ok"] is False
    assert "log_stale" in result["problems"]
    assert "no_recent_runtime_evidence" in result["problems"]


def test_evaluate_mm_health_allows_account_truth_to_prove_activity(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "mm.log"
    log.write_text(
        "2026-03-19 19:59:20,000 INFO bybit_mm Boot complete\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {"status": "ok", "reason": "", "armed": False, "path": None, "record": None},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {
            "status": "ok",
            "reason": "",
            "positions": 0,
            "open_orders": 2,
            "recent_fills": 0,
            "usdt_total": "49531.77",
        },
    )

    result = rhc.evaluate_service_health(
        spec=rhc.MM_SPEC,
        log_path=str(log),
        symbols=("ETHUSDT",),
        now=now,
    )

    assert result["ok"] is True
    assert result["problems"] == []


def test_require_account_turns_skip_into_failure(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "mm.log"
    log.write_text(
        "2026-03-19 19:59:50,000 INFO bybit_mm FILL buy 1.0 @ 2170.0 rpnl=0.1\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "active")
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {"status": "ok", "reason": "", "armed": False, "path": None, "record": None},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {"status": "skipped", "reason": "missing_credentials"},
    )

    result = rhc.evaluate_service_health(
        spec=rhc.MM_SPEC,
        log_path=str(log),
        symbols=("ETHUSDT",),
        now=now,
        require_account=True,
    )

    assert result["ok"] is False
    assert "account_skipped" in result["problems"]


def test_evaluate_alpha_health_fails_when_persistent_kill_latch_is_armed(monkeypatch, tmp_path):
    from runner import runtime_health_check as rhc

    now = datetime(2026, 3, 19, 20, 0, 0)
    log = tmp_path / "alpha.log"
    log.write_text(
        "2026-03-19 19:59:00,000 INFO scripts.ops.run_bybit_alpha: WS HEARTBEAT sigs={'ETHUSDT': 0}\n"
    )
    monkeypatch.setattr(rhc, "service_state", lambda _service: "inactive")
    monkeypatch.setattr(
        rhc,
        "summarize_account_truth",
        lambda **_kw: {"status": "skipped", "reason": "missing_credentials"},
    )
    monkeypatch.setattr(
        rhc,
        "summarize_kill_latch",
        lambda **_kw: {
            "status": "ok",
            "reason": "",
            "armed": True,
            "path": "data/runtime/kills/bybit-alpha.service_demo_portfolio.json",
            "record": {"reason": "PM drawdown 24.4%"},
        },
    )

    result = rhc.evaluate_service_health(
        spec=rhc.ALPHA_SPEC,
        log_path=str(log),
        now=now,
    )

    assert result["ok"] is False
    assert "persistent_kill_latched" in result["problems"]


def test_account_truth_filters_old_fills():
    from decimal import Decimal
    from types import SimpleNamespace

    from execution.models.balances import BalanceSnapshot, CanonicalBalance
    from execution.models.fills import CanonicalFill
    from execution.models.positions import VenuePosition
    from scripts.ops.runtime_health_check import summarize_account_truth

    class StubAdapter:
        def get_balances(self):
            return BalanceSnapshot(
                venue="bybit",
                balances=(
                    CanonicalBalance.from_free_locked(
                        venue="bybit",
                        asset="USDT",
                        free=Decimal("100"),
                        locked=Decimal("0"),
                    ),
                ),
            )

        def get_positions(self):
            return (
                VenuePosition(venue="bybit", symbol="ETHUSDT", qty=Decimal("0.5")),
            )

        def get_open_orders(self, *, symbol=None):
            return (SimpleNamespace(symbol=symbol),)

        def get_recent_fills(self, *, symbol=None):
            now_ms = int(datetime(2026, 3, 19, 20, 0, 0).timestamp() * 1000)
            return (
                CanonicalFill(
                    venue="bybit",
                    symbol=symbol or "ETHUSDT",
                    order_id="1",
                    trade_id="t1",
                    fill_id="f1",
                    side="buy",
                    qty=Decimal("1"),
                    price=Decimal("2000"),
                    ts_ms=now_ms - 30_000,
                ),
                CanonicalFill(
                    venue="bybit",
                    symbol=symbol or "ETHUSDT",
                    order_id="2",
                    trade_id="t2",
                    fill_id="f2",
                    side="sell",
                    qty=Decimal("1"),
                    price=Decimal("2001"),
                    ts_ms=now_ms - 10_000_000,
                ),
            )

    result = summarize_account_truth(
        symbols=("ETHUSDT", "ETHUSDT_15m"),
        activity_window_s=300,
        adapter=StubAdapter(),
        now=datetime(2026, 3, 19, 20, 0, 0),
    )

    assert result["status"] == "ok"
    assert result["positions"] == 1
    assert result["open_orders"] == 1
    assert result["recent_fills"] == 1


def test_main_returns_nonzero_on_failure(monkeypatch, capsys):
    from runner import runtime_health_check as rhc

    monkeypatch.setattr(
        rhc,
        "collect_runtime_health",
        lambda _args: [
            {
                "service": "bybit-alpha.service",
                "runtime": "directional alpha",
                "service_state": "failed",
                "log": {
                    "path": "logs/bybit_alpha.log",
                    "fresh": False,
                    "last_ts": None,
                    "last_age_s": None,
                    "marker_counts": {},
                },
                "account": {"status": "skipped"},
                "ok": False,
                "problems": ["service_state=failed"],
            }
        ],
    )

    exit_code = rhc.main(["--service", "alpha"])

    assert exit_code == 1
    assert "RUNTIME HEALTH CHECK" in capsys.readouterr().out
