"""Tests for runner/recovery.py — recovery infrastructure."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock


from runner.recovery import (
    EventRecorder,
    PeriodicCheckpointer,
    reconcile_and_heal,
    restore_feature_hook_state,
    restore_inference_bridge_state,
    restore_kill_switch_state,
    save_feature_hook_state,
    save_inference_bridge_state,
    save_kill_switch_state,
)


# ============================================================
# Kill Switch Persistence
# ============================================================

class TestKillSwitchPersistence:
    def test_save_and_restore_roundtrip(self, tmp_path: Path):
        from risk.kill_switch import KillMode, KillScope, KillSwitch

        ks = KillSwitch()
        ks.trigger(
            scope=KillScope.GLOBAL, key="*",
            mode=KillMode.HARD_KILL, reason="test_crash",
            source="test",
        )
        ks.trigger(
            scope=KillScope.SYMBOL, key="BTCUSDT",
            mode=KillMode.REDUCE_ONLY, reason="drawdown",
            source="risk",
        )

        save_kill_switch_state(ks, data_dir=str(tmp_path))

        # Verify file exists
        assert (tmp_path / "kill_switch_state.json").exists()

        # Restore into fresh kill switch
        ks2 = KillSwitch()
        restored = restore_kill_switch_state(ks2, data_dir=str(tmp_path))

        assert restored == 2
        # Global kill should be active
        rec = ks2.is_killed(symbol="ETHUSDT")
        assert rec is not None
        assert rec.mode == KillMode.HARD_KILL
        # Symbol kill should be active
        rec_sym = ks2.is_killed(symbol="BTCUSDT")
        assert rec_sym is not None

    def test_save_empty_removes_file(self, tmp_path: Path):
        from risk.kill_switch import KillSwitch

        ks = KillSwitch()  # No active kills
        # Create a dummy file first
        (tmp_path / "kill_switch_state.json").write_text("{}")
        save_kill_switch_state(ks, data_dir=str(tmp_path))
        assert not (tmp_path / "kill_switch_state.json").exists()

    def test_restore_nonexistent_file(self, tmp_path: Path):
        from risk.kill_switch import KillSwitch

        ks = KillSwitch()
        restored = restore_kill_switch_state(ks, data_dir=str(tmp_path))
        assert restored == 0

    def test_expired_ttl_not_restored(self, tmp_path: Path):
        from risk.kill_switch import KillMode, KillScope, KillSwitch

        ks = KillSwitch()
        # Trigger with 1-second TTL in the past
        ks.trigger(
            scope=KillScope.GLOBAL, key="*",
            mode=KillMode.HARD_KILL, reason="old",
            ttl_seconds=1, now_ts=time.time() - 100,
        )
        save_kill_switch_state(ks, data_dir=str(tmp_path))

        ks2 = KillSwitch()
        restored = restore_kill_switch_state(ks2, data_dir=str(tmp_path))
        assert restored == 0  # Expired, not restored


# ============================================================
# Inference Bridge Recovery
# ============================================================

class TestInferenceBridgeRecovery:
    def test_save_and_restore_single_bridge(self, tmp_path: Path):
        bridge = MagicMock()
        bridge.checkpoint.return_value = {"zscore_state": [1.0, 2.0], "hold_count": {"BTCUSDT": 5}}

        save_inference_bridge_state(bridge, data_dir=str(tmp_path))
        assert (tmp_path / "inference_bridge_checkpoint.json").exists()

        bridge2 = MagicMock()
        result = restore_inference_bridge_state(bridge2, data_dir=str(tmp_path))
        assert result is True
        bridge2.restore.assert_called_once()
        restored_data = bridge2.restore.call_args[0][0]
        assert restored_data["hold_count"]["BTCUSDT"] == 5

    def test_save_and_restore_dict_bridges(self, tmp_path: Path):
        btc_bridge = MagicMock()
        btc_bridge.checkpoint.return_value = {"sym": "BTC", "state": [1]}
        eth_bridge = MagicMock()
        eth_bridge.checkpoint.return_value = {"sym": "ETH", "state": [2]}

        bridges = {"BTCUSDT": btc_bridge, "ETHUSDT": eth_bridge}

        save_inference_bridge_state(bridges, data_dir=str(tmp_path))

        btc2 = MagicMock()
        eth2 = MagicMock()
        bridges2 = {"BTCUSDT": btc2, "ETHUSDT": eth2}
        result = restore_inference_bridge_state(bridges2, data_dir=str(tmp_path))
        assert result is True
        btc2.restore.assert_called_once()
        eth2.restore.assert_called_once()

    def test_restore_no_file(self, tmp_path: Path):
        bridge = MagicMock()
        result = restore_inference_bridge_state(bridge, data_dir=str(tmp_path))
        assert result is False

    def test_save_none_bridge(self, tmp_path: Path):
        save_inference_bridge_state(None, data_dir=str(tmp_path))
        assert not (tmp_path / "inference_bridge_checkpoint.json").exists()


# ============================================================
# Feature Hook Recovery
# ============================================================

class TestFeatureHookRecovery:
    def test_save_and_restore_bar_count(self, tmp_path: Path):
        hook = MagicMock()
        hook._bar_count = {"BTCUSDT": 120, "ETHUSDT": 85}

        save_feature_hook_state(hook, data_dir=str(tmp_path))
        assert (tmp_path / "feature_hook_state.json").exists()

        hook2 = MagicMock()
        hook2._bar_count = {}
        result = restore_feature_hook_state(hook2, data_dir=str(tmp_path))
        assert result is True
        assert hook2._bar_count == {"BTCUSDT": 120, "ETHUSDT": 85}

    def test_restore_no_file(self, tmp_path: Path):
        hook = MagicMock()
        hook._bar_count = {}
        result = restore_feature_hook_state(hook, data_dir=str(tmp_path))
        assert result is False

    def test_save_empty_bar_count(self, tmp_path: Path):
        hook = MagicMock()
        hook._bar_count = {}
        save_feature_hook_state(hook, data_dir=str(tmp_path))
        # Empty bar_count shouldn't create file
        assert not (tmp_path / "feature_hook_state.json").exists()


# ============================================================
# Event Recorder
# ============================================================

class TestEventRecorder:
    def _make_event_log(self):
        from execution.store.event_log import InMemoryEventLog
        return InMemoryEventLog()

    def test_record_market_event(self):
        log = self._make_event_log()
        recorder = EventRecorder(log)

        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.value = "market"
        event.symbol = "BTCUSDT"
        event.open = 100
        event.high = 101
        event.low = 99
        event.close = 100
        event.volume = 10
        event.ts = "2026-01-01T00:00:00"

        recorder.record_market(event)

        assert recorder.count == 1
        rows = log.list_recent(event_type="market")
        assert len(rows) == 1
        assert rows[0]["payload"]["symbol"] == "BTCUSDT"
        assert rows[0]["payload"]["close"] == "100"

    def test_record_fill_event(self):
        log = self._make_event_log()
        recorder = EventRecorder(log)

        fill = MagicMock()
        fill.event_type = MagicMock()
        fill.event_type.value = "fill"
        fill.symbol = "BTCUSDT"
        fill.side = "BUY"
        fill.qty = 0.01
        fill.price = 101.5
        fill.fill_id = "f1"
        fill.order_id = "o1"
        fill.ts = None

        recorder.record_fill(fill)

        assert recorder.count == 1
        rows = log.list_recent(event_type="fill")
        assert len(rows) == 1
        assert rows[0]["payload"]["side"] == "BUY"
        assert rows[0]["payload"]["qty"] == "0.01"

    def test_on_pipeline_output_records_market(self):
        log = self._make_event_log()
        recorder = EventRecorder(log)

        out = MagicMock()
        event = MagicMock()
        event.event_type = MagicMock()
        event.event_type.value = "market"
        event.symbol = "BTCUSDT"
        event.close = 100
        event.open = 100
        event.high = 101
        event.low = 99
        event.volume = 10
        event.ts = None
        out.event = event

        recorder.on_pipeline_output(out)
        assert recorder.count == 1


# ============================================================
# Periodic Checkpointer
# ============================================================

class TestPeriodicCheckpointer:
    def test_save_once(self, tmp_path: Path):
        store = MagicMock()
        snapshot = MagicMock()

        checkpointer = PeriodicCheckpointer(
            state_store=store,
            get_snapshot=lambda: snapshot,
            interval_sec=1.0,
        )
        checkpointer._save_once()

        store.save.assert_called_once_with(snapshot)

    def test_save_none_snapshot(self):
        store = MagicMock()
        checkpointer = PeriodicCheckpointer(
            state_store=store,
            get_snapshot=lambda: None,
            interval_sec=1.0,
        )
        checkpointer._save_once()
        store.save.assert_not_called()


# ============================================================
# Reconcile and Heal
# ============================================================

class TestReconcileAndHeal:
    def test_no_mismatch(self):
        coordinator = MagicMock()
        pos = MagicMock()
        pos.qty = 0.5
        coordinator.get_state_view.return_value = {
            "positions": {"BTCUSDT": pos},
            "account": MagicMock(balance=1000.0),
        }

        venue_state = {
            "positions": {"BTCUSDT": {"qty": 0.5}},
            "balance": 1000.0,
        }

        actions = reconcile_and_heal(coordinator, venue_state, ("BTCUSDT",))
        assert len(actions) == 0

    def test_position_mismatch_logged(self):
        coordinator = MagicMock()
        pos = MagicMock()
        pos.qty = 0.5
        store = MagicMock(spec=[])  # No set_position_qty
        coordinator._store = store
        coordinator.get_state_view.return_value = {
            "positions": {"BTCUSDT": pos},
            "account": MagicMock(balance=1000.0),
        }

        venue_state = {
            "positions": {"BTCUSDT": {"qty": 0.3}},
            "balance": 1000.0,
        }

        actions = reconcile_and_heal(coordinator, venue_state, ("BTCUSDT",))
        assert len(actions) == 1
        assert "BTCUSDT" in actions[0]
        assert "no heal API" in actions[0]

    def test_position_healed_when_api_available(self):
        coordinator = MagicMock()
        pos = MagicMock()
        pos.qty = 0.5
        store = MagicMock()
        store.set_position_qty = MagicMock()
        coordinator._store = store
        coordinator.get_state_view.return_value = {
            "positions": {"BTCUSDT": pos},
            "account": MagicMock(balance=1000.0),
        }

        venue_state = {
            "positions": {"BTCUSDT": {"qty": 0.3}},
            "balance": 1000.0,
        }

        actions = reconcile_and_heal(coordinator, venue_state, ("BTCUSDT",))
        assert len(actions) == 1
        assert "healed" in actions[0]
        store.set_position_qty.assert_called_once_with("BTCUSDT", 0.3)

    def test_balance_mismatch_info_only(self):
        coordinator = MagicMock()
        coordinator.get_state_view.return_value = {
            "positions": {},
            "account": MagicMock(balance=1000.0),
        }

        venue_state = {
            "positions": {},
            "balance": 900.0,
        }

        actions = reconcile_and_heal(coordinator, venue_state, ())
        assert len(actions) == 1
        assert "Balance" in actions[0]
        assert "info only" in actions[0]
