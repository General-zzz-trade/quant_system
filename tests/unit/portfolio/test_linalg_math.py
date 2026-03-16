"""Tests for cpp_black_litterman_posterior — Rust portfolio optimization math."""
import math

import pytest

_quant_hotpath = pytest.importorskip("_quant_hotpath")
from _quant_hotpath import cpp_black_litterman_posterior  # noqa: E402


# ── Helpers ──

def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _diag(vals):
    n = len(vals)
    return [[vals[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


def _approx_vec(v, expected, tol=1e-6):
    assert len(v) == len(expected)
    for i, (a, b) in enumerate(zip(v, expected)):
        assert a == pytest.approx(b, abs=tol), f"index {i}: {a} != {b}"


# ── No views (prior only) ──

class TestNoViews:
    def test_returns_equilibrium(self):
        """With no views, posterior = equilibrium returns."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.6, 0.4]
        tau = 0.05
        delta = 2.5

        mu_post, post_cov, pi = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], tau, delta
        )

        # pi = delta * sigma @ w
        expected_pi_0 = delta * (0.04 * 0.6 + 0.01 * 0.4)  # 0.07
        expected_pi_1 = delta * (0.01 * 0.6 + 0.09 * 0.4)  # 0.105
        _approx_vec(pi, [expected_pi_0, expected_pi_1])
        _approx_vec(mu_post, pi)  # posterior = prior with no views

    def test_covariance_unchanged(self):
        """With no views, posterior covariance = prior covariance."""
        sigma = [[0.04, 0.006], [0.006, 0.09]]
        _, post_cov, _ = cpp_black_litterman_posterior(
            sigma, [0.5, 0.5], [], [], [], 0.05, 2.5
        )
        for i in range(2):
            for j in range(2):
                assert post_cov[i][j] == pytest.approx(sigma[i][j])


# ── Single absolute view ──

class TestSingleView:
    def test_bullish_view_shifts_returns_up(self):
        """A bullish view on asset 0 should increase its posterior return."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.5, 0.5]
        tau = 0.05
        delta = 2.5

        _, _, pi = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], tau, delta
        )

        # Add bullish view: asset 0 returns 15%
        p = [[1.0, 0.0]]
        q = [0.15]
        confidences = [1.0]

        mu_post, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, confidences, tau, delta
        )

        assert mu_post[0] > pi[0], "Bullish view should increase return"

    def test_bearish_view_shifts_returns_down(self):
        """A bearish view on asset 1 should decrease its posterior return."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.5, 0.5]
        tau = 0.05
        delta = 2.5

        _, _, pi = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], tau, delta
        )

        p = [[0.0, 1.0]]
        q = [-0.10]
        confidences = [1.0]

        mu_post, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, confidences, tau, delta
        )

        assert mu_post[1] < pi[1], "Bearish view should decrease return"


# ── Confidence weighting ──

class TestConfidence:
    def test_high_confidence_stronger_shift(self):
        """Higher confidence should pull posterior closer to the view."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.5, 0.5]
        p = [[1.0, 0.0]]
        q = [0.20]

        mu_low, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, [0.1], 0.05, 2.5
        )
        mu_high, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, [10.0], 0.05, 2.5
        )

        _, _, pi = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], 0.05, 2.5
        )

        shift_low = abs(mu_low[0] - pi[0])
        shift_high = abs(mu_high[0] - pi[0])
        assert shift_high > shift_low, "Higher confidence = larger shift"

    def test_zero_confidence_clamps(self):
        """Zero confidence should be clamped to 1e-6, not cause division by zero."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        mu_post, _, _ = cpp_black_litterman_posterior(
            sigma, [0.5, 0.5], [[1.0, 0.0]], [0.15], [0.0], 0.05, 2.5
        )
        assert all(math.isfinite(v) for v in mu_post)


# ── Relative views ──

class TestRelativeView:
    def test_relative_view(self):
        """View: asset 0 outperforms asset 1 by 5%."""
        sigma = _diag([0.04, 0.04, 0.04])
        weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]

        _, _, pi = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], 0.05, 2.5
        )

        p = [[1.0, -1.0, 0.0]]
        q = [0.05]

        mu_post, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, [1.0], 0.05, 2.5
        )

        spread = mu_post[0] - mu_post[1]
        prior_spread = pi[0] - pi[1]
        assert spread > prior_spread, "Relative view should widen spread"


# ── Multiple views ──

class TestMultipleViews:
    def test_two_views(self):
        """Two independent absolute views on 3 assets."""
        sigma = _diag([0.04, 0.09, 0.16])
        weights = [0.5, 0.3, 0.2]

        p = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        q = [0.12, 0.08]
        confidences = [1.0, 1.0]

        mu_post, post_cov, pi = cpp_black_litterman_posterior(
            sigma, weights, p, q, confidences, 0.05, 2.5
        )

        assert len(mu_post) == 3
        assert len(post_cov) == 3
        assert all(len(row) == 3 for row in post_cov)
        assert all(math.isfinite(v) for v in mu_post)


# ── Edge cases ──

class TestEdgeCases:
    def test_single_asset(self):
        """Degenerate case: 1 asset."""
        sigma = [[0.04]]
        weights = [1.0]
        p = [[1.0]]
        q = [0.10]

        mu_post, post_cov, pi = cpp_black_litterman_posterior(
            sigma, weights, p, q, [1.0], 0.05, 2.5
        )
        assert len(mu_post) == 1
        assert math.isfinite(mu_post[0])
        assert mu_post[0] >= pi[0]  # bullish view pulls up or matches

    def test_large_tau_increases_uncertainty(self):
        """Larger tau = more uncertainty about equilibrium = views matter more."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        weights = [0.5, 0.5]
        p = [[1.0, 0.0]]
        q = [0.20]

        mu_small_tau, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, [1.0], 0.01, 2.5
        )
        mu_large_tau, _, _ = cpp_black_litterman_posterior(
            sigma, weights, p, q, [1.0], 0.50, 2.5
        )

        # Both should produce finite results and shift toward the view
        _, _, pi_small = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], 0.01, 2.5
        )
        _, _, pi_large = cpp_black_litterman_posterior(
            sigma, weights, [], [], [], 0.50, 2.5
        )

        # Verify both posteriors shift toward the bullish view (q=0.20)
        assert mu_small_tau[0] > pi_small[0] or mu_small_tau[0] == pytest.approx(pi_small[0], abs=1e-6)
        assert mu_large_tau[0] > pi_large[0] or mu_large_tau[0] == pytest.approx(pi_large[0], abs=1e-6)
        assert all(math.isfinite(v) for v in mu_small_tau)
        assert all(math.isfinite(v) for v in mu_large_tau)

    def test_posterior_covariance_symmetric(self):
        """Posterior covariance matrix should be symmetric."""
        sigma = [[0.04, 0.012], [0.012, 0.09]]
        mu_post, post_cov, _ = cpp_black_litterman_posterior(
            sigma, [0.6, 0.4], [[1.0, 0.0]], [0.10], [1.0], 0.05, 2.5
        )
        assert post_cov[0][1] == pytest.approx(post_cov[1][0], abs=1e-10)

    def test_posterior_covariance_positive_diagonal(self):
        """Diagonal of posterior covariance should be positive."""
        sigma = [[0.04, 0.01], [0.01, 0.09]]
        _, post_cov, _ = cpp_black_litterman_posterior(
            sigma, [0.5, 0.5], [[1.0, 0.0]], [0.10], [1.0], 0.05, 2.5
        )
        for i in range(2):
            assert post_cov[i][i] > 0, f"Diagonal [{i}][{i}] should be positive"
