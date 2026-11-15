"""Static import graph of the tracked Python files, for deletion inventories.

    python3 scripts/dev/import_graph.py unreachable --keep "runner/binance_target.py,trading/*,lab/*" \
        [--cut lab.orders:event.header ...]
    python3 scripts/dev/import_graph.py why --roots "lab/*" --to decision,engine
    python3 scripts/dev/import_graph.py importers trading.domain.orders

Imports inside functions count (they run when the function runs), and importing ``a.b`` counts as
importing the package ``a``. ``--cut src:dst`` drops an edge, to see what a planned move would free.
A file can still be launched by name from CI, a Makefile, a Dockerfile or a systemd unit; check those
with ``git grep`` (not the shell's ``grep -r``, which skips ignored but tracked paths).
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import subprocess
import sys
from collections import deque
from pathlib import Path


def module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def build_graph(root: Path, files: list[str]) -> dict[str, set[str]]:
    """Module -> repository modules it imports (top level and inside functions)."""
    modules = {module_name(f): f for f in files}

    def resolve(name: str) -> str | None:
        parts = name.split(".")
        for k in range(len(parts), 0, -1):
            if ".".join(parts[:k]) in modules:
                return ".".join(parts[:k])
        return None

    graph: dict[str, set[str]] = {}
    for f in files:
        mod = module_name(f)
        package = mod if f.endswith("__init__.py") else mod.rpartition(".")[0]
        deps: set[str] = set()
        for node in ast.walk(ast.parse((root / f).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                deps.update(r for a in node.names if (r := resolve(a.name)))
            elif isinstance(node, ast.ImportFrom):
                here = [p for p in package.split(".") if p]
                base = here[: len(here) - node.level + 1] if node.level else []
                prefix = ".".join([*base, node.module] if node.module else base)
                for alias in node.names:
                    full = f"{prefix}.{alias.name}" if prefix else alias.name
                    r = resolve(full) or (resolve(prefix) if prefix else None)
                    if r:
                        deps.add(r)
        parts = mod.split(".")
        deps.update(p for k in range(1, len(parts)) if (p := ".".join(parts[:k])) in modules)
        deps.discard(mod)
        graph[mod] = deps
    return graph


def reachable(graph: dict[str, set[str]], roots: list[str], cut: set[tuple[str, str]] = frozenset()) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        mod = stack.pop()
        if mod in seen or mod not in graph:
            continue
        seen.add(mod)
        stack.extend(n for n in graph[mod] if (mod, n) not in cut)
    return seen


def chain(graph: dict[str, set[str]], src: str, targets: list[str]) -> list[str] | None:
    prev: dict[str, str | None] = {src: None}
    queue = deque([src])
    while queue:
        mod = queue.popleft()
        if any(mod == t or mod.startswith(t + ".") for t in targets):
            path = []
            node: str | None = mod
            while node is not None:
                path.append(node)
                node = prev[node]
            return path[::-1]
        for nxt in sorted(graph.get(mod, ())):
            if nxt not in prev:
                prev[nxt] = mod
                queue.append(nxt)
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    un = sub.add_parser("unreachable")
    un.add_argument("--keep", required=True)
    un.add_argument("--cut", nargs="*", default=[])
    why = sub.add_parser("why")
    why.add_argument("--roots", required=True)
    why.add_argument("--to", required=True)
    imp = sub.add_parser("importers")
    imp.add_argument("module")
    args = parser.parse_args(argv[1:])

    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    files = subprocess.check_output(["git", "-C", str(root), "ls-files", "*.py"], text=True).split()
    graph = build_graph(root, files)
    path_of = {module_name(f): f for f in files}

    def matching(globs: str) -> list[str]:
        return [module_name(f) for f in files if any(fnmatch.fnmatch(f, g) for g in globs.split(","))]

    if args.cmd == "unreachable":
        cut = {tuple(c.split(":", 1)) for c in args.cut}
        keep = reachable(graph, matching(args.keep), cut)
        rest = sorted(path_of[m] for m in graph
                      if m not in keep and not path_of[m].startswith(("tests/", ".")))   # .claude/ is tooling
        lines = {f: sum(1 for _ in open(root / f, encoding="utf-8")) for f in rest}
        for f in rest:
            print(f"{lines[f]:6d}  {f}")
        print(f"{len(rest)} files, {sum(lines.values())} lines not reachable from the kept roots (tests excluded)")
    elif args.cmd == "why":
        for src in sorted(matching(args.roots)):
            found = chain(graph, src, args.to.split(","))
            if found:
                print(" -> ".join(found))
    else:
        for mod in sorted(m for m, deps in graph.items() if args.module in deps):
            print(path_of[mod])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
