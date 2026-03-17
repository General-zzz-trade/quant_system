"""Test AdapterRegistry thread safety."""
import threading
from unittest.mock import MagicMock


class TestAdapterRegistryThreadSafety:
    def test_concurrent_register_and_get(self):
        from execution.adapters.registry import AdapterRegistry
        registry = AdapterRegistry()
        errors = []
        def register_adapters(start):
            try:
                for i in range(start, start + 50):
                    mock = MagicMock()
                    registry.register(f"venue_{i}", mock)
            except Exception as e:
                errors.append(e)
        def read_adapters():
            try:
                for _ in range(100):
                    _ = registry.venues
                    _ = len(registry)
            except Exception as e:
                errors.append(e)
        threads = [
            threading.Thread(target=register_adapters, args=(0,)),
            threading.Thread(target=register_adapters, args=(50,)),
            threading.Thread(target=read_adapters),
            threading.Thread(target=read_adapters),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(registry) == 100

    def test_register_and_get_basic(self):
        from execution.adapters.registry import AdapterRegistry, AdapterNotFoundError
        import pytest
        registry = AdapterRegistry()
        mock = MagicMock()
        registry.register("bybit", mock)
        assert registry.get("bybit") is mock
        assert registry.get("BYBIT") is mock
        assert "bybit" in registry
        assert len(registry) == 1
        with pytest.raises(AdapterNotFoundError):
            registry.get("nonexistent")
