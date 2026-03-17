"""Test TracingInterceptor _active dict cleanup."""
import time
from types import SimpleNamespace


class TestTracingInterceptorLeak:
    def test_orphaned_entries_cleaned_after_ttl(self):
        from core.observability import TracingInterceptor
        tracer = TracingInterceptor(max_spans=100, active_ttl_seconds=0.1)
        envelope = SimpleNamespace(event_id="evt-1")
        tracer.before_reduce(envelope, state=None)
        assert len(tracer._active) == 1
        time.sleep(0.15)
        tracer.cleanup_stale()
        assert len(tracer._active) == 0

    def test_normal_flow_not_affected(self):
        from core.observability import TracingInterceptor
        tracer = TracingInterceptor(max_spans=100, active_ttl_seconds=60.0)
        # Provide sufficiently complete mock for after_reduce
        trace = SimpleNamespace(trace_id="t1", span_id="s1", parent_span_id=None)
        metadata = SimpleNamespace(trace=trace, source="test")
        kind = SimpleNamespace(name="MARKET")
        envelope = SimpleNamespace(
            event_id="evt-2",
            trace_id="t1",
            metadata=metadata,
            kind=kind,
        )
        tracer.before_reduce(envelope, state=None)
        tracer.after_reduce(envelope, old_state=None, new_state=None)
        assert len(tracer._active) == 0
        assert len(tracer._spans) == 1
