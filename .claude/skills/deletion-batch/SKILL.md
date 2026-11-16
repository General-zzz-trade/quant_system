---
name: deletion-batch
description: How to delete a batch of code in this repository safely - inventory with the import graph, reference scan with git grep, retiring systemd units, config touchpoints, tests, checks and the execution-log record. Use for any slimming batch, the frozen-line deletion (M4) or removing compatibility shells.
---

# A deletion batch

Read `CLAUDE.md` first. Approval: ADR-0008 sub-decisions 2b (unreachable code) and 3b (frozen lines: ML
runtime, Bybit, OKX, Polymarket and their units); the tech lead allowed batches unrelated to the runtime to
be interleaved with other work (2026-11-14, `docs/arch/execution-log.md`, batch A). Tools a person might run
by hand, and anything the code cannot prove unused, need the tech lead's item-by-item approval: ask,
recommended option first.

## 1. Inventory

```bash
python3 scripts/dev/import_graph.py unreachable --keep "<kept roots, comma separated globs>" [--cut a:b ...]
python3 scripts/dev/import_graph.py why --roots "<globs>" --to <module,...>
python3 scripts/dev/import_graph.py importers <module>
```

- The graph includes imports inside functions. `--cut` shows what a planned move would free.
- Kept roots when this was written: `runner/binance_target.py`, `trading/*`, `lab/*`,
  `data/quality/causal_bars.py`, `data/downloads/binance_public*.py`, the CI scripts
  (`scripts/check_local_safety.py`, `scripts/ci_shard.py`, `scripts/check_ci_security.py`),
  `scripts/publish_strategy_spec.py`, `infra/model_signing.py`, `scripts/dev/*`. Re-derive them; they change.
- Group the candidates by which entry points reach them (frozen runtime, venue entry points, systemd
  units, CI, tests only, nothing). A file reached by a kept runtime is not deletable, whatever it is named.

## 2. Reference scan

For every file: its module name and path in all tracked files, with `git grep -n -F -e <module> -e <path>`.
Never the shell's `grep -r`: it skips paths that are ignored but tracked (`git ls-files -ci
--exclude-standard` lists them: mostly `research/results/`, and Python files under `data/`). Check code, tests,
`.github/workflows/`, `Makefile`, `infra/systemd/`, `.importlinter`, `pyproject.toml`, `.env.example`.
Also check what reads a tool's output files (a runtime gate reading a monitor's JSON keeps the monitor
until the runtime goes).

## 3. Delete and fix the touchpoints

- `git rm` the files; check what remains with `git status`.
- systemd: delete the unit and timer files and add them to `RETIRED_UNITS` in `infra/sync_systemd.sh`
  with a one-line reason; check `bash -n infra/sync_systemd.sh`. Hosts stop them only when a person runs
  `sudo bash infra/sync_systemd.sh`; say so in the record.
- Config: `.importlinter` lists, `pyproject.toml` package globs, CI ruff paths and test lists, Makefile
  gate lists, docstrings, `.env.example`.
- Keep `.gitignore` rules that protect secret files (a leftover key file must not become committable).
- Do not change dependency lock files in a deletion batch; list what could go.
- Tests: delete tests of deleted code. A test that also covers kept code gets its imports changed, never
  its assertions; log every such change.
- Remove leftover untracked `__pycache__`-only directories of deleted packages.
- Docs: remove or correct lines that point at deleted code (`README.md`, `docs/deploy_truth.md`,
  `docs/binance_runbook.md`, `CLAUDE.md`); delete a doc whose whole subject is gone. Never read `docs/_archive/`.

## 4. Check

```bash
ruff check --select E,W,F .
lint-imports-isolated
python3 -m pytest tests/architecture <tests of touched kept code> -q -p no:cacheprovider
python3 scripts/ci_shard.py 1 4 >/dev/null   # the shard list still builds
```

The whole suite runs in CI; every batch must be green there before merge.

## 5. Record

Add a section to `docs/arch/execution-log.md`: approval, method (graph plus `git grep`), the list
(grouped), exact counts (production files and lines, test files and lines, other files and lines, from
`git show master:<file> | wc -l`), what was kept and why, the touchpoints changed, test setup changes,
local checks. Then follow the PR steps of the migration-step skill.
