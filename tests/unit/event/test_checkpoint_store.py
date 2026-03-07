from __future__ import annotations

from event.checkpoint import Checkpoint, InMemoryCheckpointStore, StreamCursor


def _checkpoint(*, checkpoint_id: str, version: int, created_at: str) -> Checkpoint:
    return Checkpoint(
        run_id='run-1',
        name='replay-max',
        checkpoint_id=checkpoint_id,
        version=version,
        created_at=created_at,
        stream_cursors=(StreamCursor(stream_id='bars', cursor={'index': 10}),),
        last_event_time='2026-01-01T00:00:00+00:00',
        fingerprint='abc',
    )


class TestCheckpointContentHash:
    def test_hash_excludes_metadata_fields(self) -> None:
        cp1 = _checkpoint(checkpoint_id='a', version=1, created_at='2026-01-01T00:00:00+00:00')
        cp2 = _checkpoint(checkpoint_id='b', version=99, created_at='2026-02-01T00:00:00+00:00')

        assert cp1.content_hash() == cp2.content_hash()


class TestInMemoryCheckpointStore:
    def test_save_and_load_latest(self) -> None:
        store = InMemoryCheckpointStore()
        cp = _checkpoint(checkpoint_id='a', version=1, created_at='2026-01-01T00:00:00+00:00')

        saved = store.save(cp)
        loaded = store.load_latest(run_id='run-1', name='replay-max')

        assert saved == cp
        assert loaded == cp
