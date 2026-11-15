.PHONY: all clean test test-py test-exec test-all lint lint-arch

all: test

clean:  ## Build output of the Rust extension deleted on 2026-11-15
	rm -rf target

# ── Test targets (aligned with CI: .github/workflows/ci.yml) ───────────

test: test-py test-exec lint lint-arch  ## Core local gate; CI also runs the whole suite (make test-all) as shards

CORE_TESTS = tests/unit/infra/ \
	tests/unit/data/test_binance_public_download.py tests/unit/lab/ tests/unit/binance/ tests/architecture/ \
	tests/characterization/ tests/offline/ tests/contract/ tests/unit/trading/
test-py:  ## Money-path and core tests (CI runs them within the whole-suite shards)
	python3 -m pytest $(CORE_TESTS) -x -q --tb=short

test-exec:  ## Execution subsystem tests (adapters, durable execution, Binance runtime)
	python3 -m pytest tests/unit/execution/ tests/unit/binance/ -x -q --tb=short

test-all:  ## Entire suite; must pass (see docs/known_test_failures.md for the 2026-11-13 cleanup)
	python3 -m pytest tests/ -q --tb=line --ignore=tests/performance

lint:  ## Lint (matches CI 'Ruff check' step)
	ruff check --select E,W,F .

# import-linter and mypy are dev tools only: pip install import-linter==2.16 mypy==1.18.2
LINT_IMPORTS ?= lint-imports
MYPY ?= mypy

lint-arch:  ## Architecture contracts in .importlinter and strict types for trading/ (matches CI steps)
	$(LINT_IMPORTS)
	$(MYPY) trading/
