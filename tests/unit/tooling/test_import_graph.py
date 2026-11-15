"""The deletion-inventory import graph (scripts/dev/import_graph.py) sees relative, lazy and parent imports."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("import_graph", ROOT / "scripts" / "dev" / "import_graph.py")
ig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ig)

FILES = {
    "pkg/__init__.py": "",
    "pkg/a.py": "from . import b\n\n\ndef f():\n    import pkg.c\n",
    "pkg/b.py": "from .sub.d import X\n",
    "pkg/c.py": "",
    "pkg/sub/__init__.py": "",
    "pkg/sub/d.py": "from .. import c\nX = 1\n",
    "other.py": "import pkg.a\nimport json\n",
    "tool.py": "from . import other\n",
}


def _graph(tmp_path):
    for name, text in FILES.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return ig.build_graph(tmp_path, list(FILES))


def test_edges_include_relative_lazy_and_parent_package_imports(tmp_path):
    graph = _graph(tmp_path)
    assert graph["pkg.a"] == {"pkg", "pkg.b", "pkg.c"}                 # relative, lazy, parent package
    assert graph["pkg.b"] == {"pkg", "pkg.sub.d"}
    assert graph["pkg.sub.d"] == {"pkg", "pkg.sub", "pkg.c"}           # two levels up
    assert graph["other"] == {"pkg.a"}                                 # json is not a repository module
    assert graph["tool"] == {"other"}                                  # relative import in a top-level module


def test_reachability_and_cut_edges(tmp_path):
    graph = _graph(tmp_path)
    assert ig.reachable(graph, ["other"]) == {"other", "pkg", "pkg.a", "pkg.b", "pkg.c", "pkg.sub", "pkg.sub.d"}
    without_lazy = ig.reachable(graph, ["other"], {("pkg.a", "pkg.c"), ("pkg.sub.d", "pkg.c")})
    assert "pkg.c" not in without_lazy and "pkg.sub.d" in without_lazy


def test_chain_shows_how_a_module_is_reached(tmp_path):
    graph = _graph(tmp_path)
    assert ig.chain(graph, "other", ["pkg.sub"]) == ["other", "pkg.a", "pkg.b", "pkg.sub.d"]
    assert ig.chain(graph, "pkg.c", ["other"]) is None
