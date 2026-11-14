"""Research infrastructure: purged walk-forward, trial registry, forward-window looks."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from lab import forward, trials
from lab.metrics import deflated_sharpe_probability, deflated_sharpe_probability_k
from lab.strategy import StrategySpec
from lab.validation import walk_forward

DAY = 86_400_000
HOUR = 3_600_000


# -- purged walk-forward -------------------------------------------------------------------------
def test_walk_forward_purges_overlapping_labels_and_respects_the_limit():
    t = np.arange(20) * DAY
    end = t + 3 * DAY                                  # 3-day labels
    folds = walk_forward(t, end, first_test=10 * DAY, block=5 * DAY, limit=18 * DAY)
    assert [(f.test_start, f.test_end) for f in folds] == [(10 * DAY, 15 * DAY), (15 * DAY, 18 * DAY)]
    first, second = folds
    assert first.test.tolist() == [10, 11, 12, 13, 14]
    assert first.train.tolist() == list(range(0, 7))   # label of sample 7 ends at day 10: purged
    assert second.test.tolist() == [15]                # 16, 17 have labels ending after the limit
    assert second.train.max() == 11
    assert all(end[f.train].max() < f.test_start for f in folds)


def test_walk_forward_embargo_and_validation():
    t = np.arange(10) * DAY
    folds = walk_forward(t, t + DAY, first_test=5 * DAY, block=5 * DAY, limit=11 * DAY, embargo=2 * DAY)
    assert folds[0].train.tolist() == [0, 1]
    with pytest.raises(ValueError):
        walk_forward(t, t, first_test=5 * DAY, block=DAY, limit=9 * DAY)
    with pytest.raises(ValueError):
        walk_forward(t[::-1], t[::-1] + DAY, first_test=5 * DAY, block=DAY, limit=9 * DAY)


# -- trial registry ------------------------------------------------------------------------------
def row(study, trial, sharpe, family="f", count=1):
    return {"family": family, "study": study, "trial": trial, "config": {"x": trial}, "window": [0, 1],
            "sharpe": sharpe, "count": count, "note": ""}


def test_registry_is_append_only_and_counts_families(tmp_path):
    path = tmp_path / "trials.jsonl"
    trials.record([row("a", "1", 0.5), row("a", "2", 1.0), row("b", "1", None, count=17)], path=path)
    trials.record([row("c", "1", 0.2, family="other")], path=path)
    with pytest.raises(ValueError, match="append-only"):
        trials.record([row("a", "1", 0.7)], path=path)
    with pytest.raises(ValueError, match="missing"):
        trials.record([{"family": "f"}], path=path)
    with pytest.raises(ValueError, match="finite"):
        trials.record([row("d", "1", float("nan"))], path=path)
    stats = trials.family_stats("f", path=path)
    assert stats["n_trials"] == 19 and stats["n_with_sharpe"] == 2
    assert stats["sharpe_std"] == pytest.approx(np.std([0.5, 1.0], ddof=1))


def test_more_trials_deflate_more(tmp_path):
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.002, 0.02, 1000))
    path = tmp_path / "trials.jsonl"
    trials.record([row("a", str(i), s) for i, s in enumerate([0.1, 0.4, 0.9])], path=path)
    few = trials.deflated(returns, "f", path=path)["probability"]
    trials.record([row("b", "rest", None, count=50)], path=path)
    many = trials.deflated(returns, "f", path=path)["probability"]
    alone = trials.deflated(returns, "empty", path=path)["probability"]
    assert alone > few > many


def test_k_form_matches_the_list_form():
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(0.001, 0.02, 800))
    sharpes = [0.2, 0.5, 0.9, 1.3]
    expected = deflated_sharpe_probability(returns, sharpes)
    got = deflated_sharpe_probability_k(returns, float(np.std(sharpes, ddof=1)), len(sharpes))
    assert got == pytest.approx(expected, rel=1e-9)


# -- forward looks -------------------------------------------------------------------------------
def test_forward_start_is_2026_11_13():
    assert forward.FORWARD_START == int(datetime(2026, 11, 13, tzinfo=timezone.utc).timestamp() * 1000)


def write_looks(path, **kw):
    look = {"study": "s", "spec_sha256": "abc", "end": "2027-02-13", "registered": "2026-11-14", **kw}
    path.write_text(json.dumps(look) + "\n")


def test_unregistered_premature_or_late_registered_looks_are_refused(tmp_path):
    path = tmp_path / "looks.jsonl"
    after = forward.day_ms("2027-02-15")
    write_looks(path)
    assert forward.registered_look("s", "abc", "2027-02-13", path=path, now_ms=after)["registered"] == "2026-11-14"
    with pytest.raises(PermissionError, match="no registered"):
        forward.registered_look("s", "changed-spec", "2027-02-13", path=path, now_ms=after)
    with pytest.raises(PermissionError, match="not fully passed"):
        forward.registered_look("s", "abc", "2027-02-13", path=path, now_ms=forward.day_ms("2027-02-13") + HOUR)
    write_looks(path, registered="2027-02-13")
    with pytest.raises(PermissionError, match="on or after"):
        forward.registered_look("s", "abc", "2027-02-13", path=path, now_ms=after)


def write_symbol(directory, symbol, closes, start):
    with (directory / f"{symbol}_1h.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        prev = closes[0]
        for i, c in enumerate(closes):
            w.writerow([start + i * HOUR, f"{prev:.4f}", f"{max(prev, c) * 1.001:.4f}",
                        f"{min(prev, c) * 0.999:.4f}", f"{c:.4f}", "100000"])
            prev = c
    with (directory / f"{symbol}_funding.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["funding_time", "rate", "mark_price"])
        for i, c in enumerate(closes):
            if (start + i * HOUR) % (8 * HOUR) == 0:
                w.writerow([start + i * HOUR, "0.0001", f"{c:.4f}"])


def test_forward_evaluation_ignores_data_from_the_end_on(tmp_path):
    start = forward.FORWARD_START - 40 * DAY
    n = 60 * 24
    rng = np.random.default_rng(4)
    spec = StrategySpec(lookbacks=(3, 6), vol_lookback=5, target_vol=0.3, execution="market")
    frozen = {"kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT", "ETHUSDT"], "spec": spec.record()}
    spec_path = tmp_path / "final_spec.json"
    spec_path.write_text(json.dumps(frozen))
    end = forward.FORWARD_START + 15 * DAY
    results = []
    for name, wild in (("a", 1.0), ("b", 3.0)):
        d = tmp_path / name
        d.mkdir()
        for symbol, px in (("BTCUSDT", 60000.0), ("ETHUSDT", 3000.0)):
            closes = list(px * np.exp(np.cumsum(rng.normal(0, 0.006, n))))
            cut = (end - start) // HOUR
            closes = closes[:cut] + [c * wild for c in closes[cut:]]
            write_symbol(d, symbol, closes, start)
        rng = np.random.default_rng(4)
        results.append(forward.evaluate_target_spec(spec_path, d, end))
    assert json.dumps(results[0], default=float) == json.dumps(results[1], default=float)
    assert results[0]["window"]["days"] == 15 and math.isfinite(results[0]["strategy"]["total_return"])


def test_every_registered_look_points_at_an_unchanged_spec_and_was_registered_before_its_end():
    """The committed look registry: a frozen spec that is edited or missing fails here, not months later."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    looks = forward.load_looks(root / "research/forward_looks.jsonl")
    assert len(looks) >= 8
    for look in looks:
        spec = root / look["spec"]
        assert spec.is_file(), look["spec"]
        assert forward.spec_digest(spec) == look["spec_sha256"], look["spec"]
        assert forward.day_ms(look["registered"]) < forward.day_ms(look["end"])
        assert forward.day_ms(look["end"]) > forward.FORWARD_START
        frozen = json.loads(spec.read_text())
        assert frozen.get("passed") is False                   # no spec in the registry may open mainnet
        if frozen.get("kind") != "btc_eth_multitimeframe_target":
            assert frozen.get("registered") is True            # a candidate closed by its floor has no look
    studies = {look["study"] for look in looks}
    assert {f"forward_candidates_2026-11-14:{name}" for name in
            ("pair_ridge_pd_7d", "dir_ridge_7d", "flush_reversal_72h")} <= studies
