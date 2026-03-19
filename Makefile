.PHONY: all rust clean test test-py test-exec test-rust lint

all: rust

rust:
	cd ext/rust && RUSTFLAGS="-C target-cpu=skylake-avx512" PATH="$(HOME)/.cargo/bin:$(PATH)" maturin build --release --features python && pip install --break-system-packages --force-reinstall target/wheels/*.whl

clean:
	rm -rf ext/rust/target

# ── Test targets (aligned with CI: .github/workflows/ci.yml) ───────────

test: test-py test-exec test-rust lint  ## Core local test gate; CI also runs security, model-check, compose smoke, and a dedicated framework integration step

test-py:  ## Python core tests (matches CI 'Run Python tests' step)
	python3 -m pytest tests/ -x -q --tb=short --ignore=tests/performance

test-exec:  ## Execution subsystem tests (matches CI 'Run execution tests' step)
	python3 -m pytest execution/tests/ -x -q --tb=short

test-rust:  ## Rust unit tests (matches CI 'Run Rust tests' step)
	cd ext/rust && cargo test

lint:  ## Lint (matches CI 'Ruff check' step)
	ruff check --select E,W,F .
