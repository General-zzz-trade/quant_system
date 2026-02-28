"""Tests for decision/graph.py, decision/registry.py, and decision/replay.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from decision.graph import DecisionGraph, Node
from decision.registry import Registry
from decision.replay import DecisionReplayer


# ── DecisionGraph ────────────────────────────────────────────────────

class TestDecisionGraph:
    def test_empty_graph_topo(self):
        g = DecisionGraph(nodes={})
        assert g.topo() == []

    def test_single_node(self):
        g = DecisionGraph(nodes={"a": Node(name="a")})
        assert g.topo() == ["a"]

    def test_linear_chain(self):
        g = DecisionGraph(nodes={
            "a": Node(name="a"),
            "b": Node(name="b", deps=("a",)),
            "c": Node(name="c", deps=("b",)),
        })
        order = g.topo()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_deps(self):
        g = DecisionGraph(nodes={
            "a": Node(name="a"),
            "b": Node(name="b", deps=("a",)),
            "c": Node(name="c", deps=("a",)),
            "d": Node(name="d", deps=("b", "c")),
        })
        order = g.topo()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_missing_dep_refs_skipped(self):
        """Deps referencing nodes not in the graph are silently skipped."""
        g = DecisionGraph(nodes={
            "a": Node(name="a", deps=("missing",)),
            "b": Node(name="b", deps=("a",)),
        })
        order = g.topo()
        assert order.index("a") < order.index("b")
        assert "missing" not in order

    def test_node_with_fn(self):
        fn = lambda: 42
        n = Node(name="x", fn=fn)
        assert n.fn is fn
        assert n.fn() == 42

    def test_parallel_nodes(self):
        g = DecisionGraph(nodes={
            "a": Node(name="a"),
            "b": Node(name="b"),
        })
        order = g.topo()
        assert set(order) == {"a", "b"}
        assert len(order) == 2


# ── Registry ─────────────────────────────────────────────────────────

class TestRegistry:
    def test_register_and_build(self):
        reg = Registry()
        reg.register("sig_a", lambda: "instance_a", category="signal")
        result = reg.build("sig_a")
        assert result == "instance_a"

    def test_duplicate_raises(self):
        reg = Registry()
        reg.register("x", lambda: 1)
        with pytest.raises(KeyError, match="Already registered"):
            reg.register("x", lambda: 2)

    def test_overwrite_true(self):
        reg = Registry()
        reg.register("x", lambda: 1)
        reg.register("x", lambda: 2, overwrite=True)
        assert reg.build("x") == 2

    def test_build_unknown_raises(self):
        reg = Registry()
        reg.register("a", lambda: 1)
        with pytest.raises(KeyError, match="Unknown component"):
            reg.build("z")

    def test_list_names_all(self):
        reg = Registry()
        reg.register("b", lambda: 2)
        reg.register("a", lambda: 1)
        assert reg.list_names() == ["a", "b"]

    def test_list_names_with_category(self):
        reg = Registry()
        reg.register("s1", lambda: 1, category="signal")
        reg.register("a1", lambda: 2, category="allocator")
        reg.register("s2", lambda: 3, category="signal")
        assert reg.list_names(category="signal") == ["s1", "s2"]
        assert reg.list_names(category="allocator") == ["a1"]
        assert reg.list_names(category="nonexistent") == []

    def test_has(self):
        reg = Registry()
        reg.register("x", lambda: 1)
        assert reg.has("x") is True
        assert reg.has("y") is False

    def test_len(self):
        reg = Registry()
        assert len(reg) == 0
        reg.register("a", lambda: 1)
        reg.register("b", lambda: 2)
        assert len(reg) == 2

    def test_get_returns_factory(self):
        reg = Registry()
        factory = lambda: 42
        reg.register("x", factory)
        assert reg.get("x") is factory

    def test_get_missing_returns_none(self):
        reg = Registry()
        assert reg.get("nope") is None


# ── DecisionReplayer ─────────────────────────────────────────────────

class TestDecisionReplayer:
    def test_iter_outputs_delegates_to_store(self):
        mock_store = MagicMock()
        records = [{"action": "buy"}, {"action": "sell"}]
        mock_store.iter_records.return_value = iter(records)

        replayer = DecisionReplayer(store=mock_store)
        results = list(replayer.iter_outputs())
        assert results == records
        mock_store.iter_records.assert_called_once()

    def test_iter_outputs_empty(self):
        mock_store = MagicMock()
        mock_store.iter_records.return_value = iter([])
        replayer = DecisionReplayer(store=mock_store)
        assert list(replayer.iter_outputs()) == []
