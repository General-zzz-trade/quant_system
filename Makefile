.PHONY: all rust clean

all: rust

rust:
	cd ext/rust && PATH="$(HOME)/.cargo/bin:$(PATH)" maturin build --release && pip install --break-system-packages --force-reinstall target/wheels/*.whl

clean:
	rm -rf ext/rust/target
