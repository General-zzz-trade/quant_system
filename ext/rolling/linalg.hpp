#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

/**
 * Black-Litterman posterior computation (all-in-one).
 *
 * Computes posterior returns and covariance from:
 * - sigma: N x N asset covariance matrix
 * - market_weights: N vector
 * - P: K x N view pick matrix
 * - Q: K vector of view expected returns
 * - confidences: K vector of view confidence levels
 * - tau: uncertainty scaling
 * - risk_aversion: delta parameter
 *
 * Returns: {posterior_returns (N), posterior_covariance (NxN), equilibrium_returns (N)}
 */

namespace linalg {

using Vec = std::vector<double>;
using Mat = std::vector<std::vector<double>>;

inline Mat mat_zeros(size_t n, size_t m) {
    return Mat(n, Vec(m, 0.0));
}

inline Mat mat_transpose(const Mat& a) {
    if (a.empty()) return {};
    size_t r = a.size(), c = a[0].size();
    Mat result(c, Vec(r));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            result[j][i] = a[i][j];
    return result;
}

inline Mat mat_multiply(const Mat& a, const Mat& b) {
    size_t ra = a.size(), ca = a[0].size(), cb = b[0].size();
    Mat result(ra, Vec(cb, 0.0));
    for (size_t i = 0; i < ra; ++i)
        for (size_t k = 0; k < ca; ++k) {
            double aik = a[i][k];
            for (size_t j = 0; j < cb; ++j)
                result[i][j] += aik * b[k][j];
        }
    return result;
}

inline Mat mat_scale(const Mat& a, double s) {
    size_t r = a.size(), c = a[0].size();
    Mat result(r, Vec(c));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            result[i][j] = a[i][j] * s;
    return result;
}

inline Mat mat_add(const Mat& a, const Mat& b) {
    size_t r = a.size(), c = a[0].size();
    Mat result(r, Vec(c));
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            result[i][j] = a[i][j] + b[i][j];
    return result;
}

inline Vec mat_vec_multiply(const Mat& m, const Vec& v) {
    size_t r = m.size(), c = v.size();
    Vec result(r, 0.0);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            result[i] += m[i][j] * v[j];
    return result;
}

inline Mat mat_inverse(const Mat& mat) {
    size_t n = mat.size();
    // Augmented matrix [A | I]
    Mat aug(n, Vec(2 * n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j)
            aug[i][j] = mat[i][j];
        aug[i][n + i] = 1.0;
    }

    for (size_t col = 0; col < n; ++col) {
        // Partial pivoting
        size_t max_row = col;
        double max_val = std::abs(aug[col][col]);
        for (size_t row = col + 1; row < n; ++row) {
            double v = std::abs(aug[row][col]);
            if (v > max_val) {
                max_val = v;
                max_row = row;
            }
        }
        if (max_val < 1e-15)
            throw std::runtime_error("Singular matrix");
        if (max_row != col)
            std::swap(aug[col], aug[max_row]);

        double pivot = aug[col][col];
        double inv_pivot = 1.0 / pivot;
        for (size_t j = 0; j < 2 * n; ++j)
            aug[col][j] *= inv_pivot;

        for (size_t row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (size_t j = 0; j < 2 * n; ++j)
                aug[row][j] -= factor * aug[col][j];
        }
    }

    Mat inv(n, Vec(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            inv[i][j] = aug[i][n + j];
    return inv;
}

struct BLResult {
    Vec posterior_returns;
    Mat posterior_covariance;
    Vec equilibrium_returns;
};

inline BLResult black_litterman_posterior(
    const Mat& sigma,
    const Vec& market_weights,
    const Mat& P,
    const Vec& Q,
    const Vec& confidences,
    double tau,
    double risk_aversion
) {
    size_t N = sigma.size();
    size_t K = P.size();

    // pi = delta * Sigma @ w
    Vec sigma_w = mat_vec_multiply(sigma, market_weights);
    Vec pi(N);
    for (size_t i = 0; i < N; ++i)
        pi[i] = risk_aversion * sigma_w[i];

    if (K == 0) {
        return BLResult{pi, sigma, pi};
    }

    Mat tau_sigma = mat_scale(sigma, tau);

    // Omega (KxK diagonal)
    Mat omega = mat_zeros(K, K);
    for (size_t v = 0; v < K; ++v) {
        Vec tau_sigma_p = mat_vec_multiply(tau_sigma, P[v]);
        double view_var = 0.0;
        for (size_t j = 0; j < N; ++j)
            view_var += P[v][j] * tau_sigma_p[j];
        double conf = confidences[v];
        if (conf <= 0.0) conf = 1e-6;
        omega[v][v] = view_var / conf;
    }

    Mat tau_sigma_inv = mat_inverse(tau_sigma);
    Mat omega_inv = mat_inverse(omega);

    Mat Pt = mat_transpose(P);
    Mat Pt_omega_inv = mat_multiply(Pt, omega_inv);
    Mat Pt_omega_inv_P = mat_multiply(Pt_omega_inv, P);

    Mat M_inv = mat_add(tau_sigma_inv, Pt_omega_inv_P);
    Mat M = mat_inverse(M_inv);

    Vec tsi_pi = mat_vec_multiply(tau_sigma_inv, pi);
    Vec poi_Q = mat_vec_multiply(Pt_omega_inv, Q);

    Vec combined(N);
    for (size_t i = 0; i < N; ++i)
        combined[i] = tsi_pi[i] + poi_Q[i];

    Vec mu_post = mat_vec_multiply(M, combined);
    Mat post_cov = mat_add(sigma, M);

    return BLResult{mu_post, post_cov, pi};
}

} // namespace linalg

inline std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<double>>
cpp_black_litterman_posterior(
    const std::vector<std::vector<double>>& sigma,
    const std::vector<double>& market_weights,
    const std::vector<std::vector<double>>& P,
    const std::vector<double>& Q,
    const std::vector<double>& confidences,
    double tau,
    double risk_aversion
) {
    auto res = linalg::black_litterman_posterior(sigma, market_weights, P, Q, confidences, tau, risk_aversion);
    return {std::move(res.posterior_returns), std::move(res.posterior_covariance), std::move(res.equilibrium_returns)};
}
