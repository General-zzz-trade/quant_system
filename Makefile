.PHONY: all rust clean test test-py test-exec test-all test-rust lint

all: rust

rust:
	RUSTFLAGS="-C target-cpu=skylake-avx512" PATH="$(HOME)/.cargo/bin:$(PATH)" maturin build --release --features python && pip install --break-system-packages --force-reinstall target/wheels/*.whl

clean:
	rm -rf target

# ── Test targets (aligned with CI: .github/workflows/ci.yml) ───────────

test: test-py test-exec test-rust lint  ## Core local gate (same suites as CI); full suite: make test-all

CORE_TESTS = tests/unit/decision/ tests/unit/state/ tests/unit/event/ tests/unit/engine/ tests/unit/bybit/ \
	tests/unit/execution/adapters/okx/ tests/unit/runner/test_offline_research.py \
	tests/unit/runner/test_offline_candidate_research.py tests/unit/runner/test_data_readiness.py \
	tests/unit/runner/test_paper_check.py tests/unit/runner/test_shadow_main.py tests/unit/infra/ \
	tests/unit/monitoring/test_pause_state_fail_closed.py tests/unit/data/test_binance_public_download.py \
	tests/unit/features/test_daily_alignment_causal.py

test-py:  ## Python core tests (matches CI 'Run core unit tests')
	python3 -m pytest $(CORE_TESTS) -x -q --tb=short

test-exec:  ## Execution subsystem tests (adapters, durable execution, Binance runtime)
	python3 -m pytest tests/unit/execution/ tests/unit/binance/ -x -q --tb=short

test-all:  ## Entire suite; pre-existing legacy failures are listed in docs/known_test_failures.md
	python3 -m pytest tests/ -q --tb=line --ignore=tests/performance

test-rust:  ## Rust unit tests (matches CI 'Run Rust tests' step)
	cargo test

lint:  ## Lint (matches CI 'Ruff check' step)
	ruff check --select E,W,F .
