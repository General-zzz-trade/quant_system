"""Tests for alpha.nn_utils — sliding window and target alignment."""
import numpy as np
import pytest

from alpha.nn_utils import make_sliding_windows, align_target


class TestMakeSlidingWindows:
    def test_basic_shape(self):
        X = np.arange(50).reshape(10, 5).astype(float)
        out = make_sliding_windows(X, seq_len=3)
        assert out.shape == (8, 3, 5)

    def test_values_correct(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        out = make_sliding_windows(X, seq_len=2)
        # Window 0: rows 0,1; Window 1: rows 1,2; Window 2: rows 2,3
        np.testing.assert_array_equal(out[0], X[0:2])
        np.testing.assert_array_equal(out[1], X[1:3])
        np.testing.assert_array_equal(out[2], X[2:4])

    def test_seq_len_1(self):
        X = np.ones((5, 3))
        out = make_sliding_windows(X, seq_len=1)
        assert out.shape == (5, 1, 3)

    def test_seq_len_equals_n(self):
        X = np.ones((4, 2))
        out = make_sliding_windows(X, seq_len=4)
        assert out.shape == (1, 4, 2)

    def test_seq_len_too_large(self):
        X = np.ones((3, 2))
        with pytest.raises(ValueError, match="seq_len.*> N"):
            make_sliding_windows(X, seq_len=5)

    def test_seq_len_zero(self):
        X = np.ones((3, 2))
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            make_sliding_windows(X, seq_len=0)

    def test_preserves_dtype(self):
        X = np.ones((10, 3), dtype=np.float32)
        out = make_sliding_windows(X, seq_len=3)
        assert out.dtype == np.float32


class TestAlignTarget:
    def test_basic(self):
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        out = align_target(y, seq_len=3)
        np.testing.assert_array_equal(out, [2.0, 3.0, 4.0])

    def test_seq_len_1(self):
        y = np.array([10, 20, 30])
        out = align_target(y, seq_len=1)
        np.testing.assert_array_equal(out, y)

    def test_matches_window_length(self):
        N, F = 100, 5
        X = np.random.randn(N, F)
        y = np.random.randn(N)
        seq_len = 20
        windows = make_sliding_windows(X, seq_len)
        y_aligned = align_target(y, seq_len)
        assert len(windows) == len(y_aligned)

    def test_seq_len_zero(self):
        y = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            align_target(y, seq_len=0)
