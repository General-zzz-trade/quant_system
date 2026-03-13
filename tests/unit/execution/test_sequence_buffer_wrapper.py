from __future__ import annotations

from execution.ingress.sequence_buffer import SequenceBuffer
from execution.safety.out_of_order_guard import OutOfOrderGuard, SequencedMessage


def _push_py(guard: OutOfOrderGuard, key: str, seq: int, payload: object) -> list[object]:
    return list(guard.process(SequencedMessage(key=key, seq=seq, payload=payload)))


def _push_seq(buf: SequenceBuffer, key: str, seq: int, payload: object) -> list[object]:
    return list(buf.push(key=key, seq=seq, payload=payload))


def test_sequence_buffer_wrapper_matches_out_of_order_guard_for_reordering() -> None:
    py = OutOfOrderGuard(max_buffer_per_key=4)
    seq = SequenceBuffer(max_buffer_size=4)

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

    for key, seq_no, payload in ops:
        assert _push_py(py, key, seq_no, payload) == _push_seq(seq, key, seq_no, payload)
        assert py.pending_count(key) == seq.pending_count(key)
        assert py.pending_count() == seq.pending_count()


def test_sequence_buffer_wrapper_exposes_flush_reset_and_pending_count() -> None:
    py = OutOfOrderGuard(max_buffer_per_key=2)
    seq = SequenceBuffer(max_buffer_size=2)

    assert _push_py(py, "ord-1", 2, "C") == _push_seq(seq, "ord-1", 2, "C") == []
    assert _push_py(py, "ord-1", 3, "D") == _push_seq(seq, "ord-1", 3, "D") == []
    assert _push_py(py, "ord-1", 4, "E") == _push_seq(seq, "ord-1", 4, "E") == []

    assert py.pending_count("ord-1") == seq.pending_count("ord-1") == 2
    assert list(py.flush("ord-1")) == list(seq.flush("ord-1")) == ["C", "D"]
    assert py.pending_count("ord-1") == seq.pending_count("ord-1") == 0

    assert _push_py(py, "ord-2", 1, "B") == _push_seq(seq, "ord-2", 1, "B") == []
    py.reset()
    seq.reset("ord-2")

    assert py.pending_count() == seq.pending_count() == 0
    assert _push_py(py, "ord-2", 0, "A") == _push_seq(seq, "ord-2", 0, "A") == ["A"]
