from __future__ import annotations

from _quant_hotpath import RustSequenceBuffer

from execution.safety.out_of_order_guard import OutOfOrderGuard, SequencedMessage


def _push_py(guard: OutOfOrderGuard, key: str, seq: int, payload: object) -> list[object]:
    return list(guard.process(SequencedMessage(key=key, seq=seq, payload=payload)))


def _push_rust(buf: RustSequenceBuffer, key: str, seq: int, payload: object) -> list[object]:
    return list(buf.push(key, seq, payload))


def test_out_of_order_guard_matches_rust_sequence_buffer_for_reordering() -> None:
    py = OutOfOrderGuard(max_buffer_per_key=4)
    rs = RustSequenceBuffer(max_buffer_size=4)

    ops = [
        ("ord-1", 0, "A"),
        ("ord-1", 2, "C"),
        ("ord-1", 1, "B"),
        ("ord-2", 1, "Y"),
        ("ord-2", 0, "X"),
        ("ord-1", 4, "E"),
        ("ord-1", 3, "D"),
        ("ord-2", 0, "X-dup"),
    ]

    for key, seq, payload in ops:
        assert _push_py(py, key, seq, payload) == _push_rust(rs, key, seq, payload)
        assert py.pending_count(key) == rs.pending_count(key)


def test_out_of_order_guard_matches_rust_sequence_buffer_for_flush_reset_and_capacity() -> None:
    py = OutOfOrderGuard(max_buffer_per_key=2)
    rs = RustSequenceBuffer(max_buffer_size=2)

    # Fill the gap buffer up to capacity. The third out-of-order item is dropped.
    assert _push_py(py, "ord-1", 2, "C") == _push_rust(rs, "ord-1", 2, "C") == []
    assert _push_py(py, "ord-1", 3, "D") == _push_rust(rs, "ord-1", 3, "D") == []
    assert _push_py(py, "ord-1", 4, "E") == _push_rust(rs, "ord-1", 4, "E") == []
    assert py.pending_count("ord-1") == rs.pending_count("ord-1") == 2

    assert list(py.flush("ord-1")) == list(rs.flush("ord-1")) == ["C", "D"]
    assert py.pending_count("ord-1") == rs.pending_count("ord-1") == 0

    # Reset semantics should also line up after a fresh buffered gap.
    assert _push_py(py, "ord-2", 1, "B") == _push_rust(rs, "ord-2", 1, "B") == []
    py.reset()
    rs.clear()

    assert py.pending_count() == rs.pending_count(None) == 0
    assert _push_py(py, "ord-2", 0, "A") == _push_rust(rs, "ord-2", 0, "A") == ["A"]
