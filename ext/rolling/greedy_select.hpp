#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Greedy forward IC selection with OLS residualization.
// Uses precomputed Gram matrix to avoid O(n) work per candidate per step.
// Returns indices of selected features (0-based).
inline std::vector<int> cpp_greedy_ic_select(
    const double* X,     // row-major (n_samples x n_features), NaN→0 already
    const double* y,     // (n_samples,), NaN→0 already
    int n_samples,
    int n_features,
    int top_k = 20
) {
    if (n_samples < 50 || n_features == 0)
        return {};

    constexpr double EPS = 1e-12;
    const int n = n_samples;
    const int p = n_features;
    const int k_max = std::min(top_k, p);

    // 1. Precompute column sums, sum-of-squares, and X'y
    std::vector<double> col_sum(p, 0.0);
    std::vector<double> col_sumsq(p, 0.0);
    std::vector<double> Xty(p, 0.0);
    double y_sum = 0.0, y_sumsq = 0.0;

    for (int i = 0; i < n; ++i) {
        const double* row = X + static_cast<size_t>(i) * p;
        double yi = y[i];
        y_sum += yi;
        y_sumsq += yi * yi;
        for (int j = 0; j < p; ++j) {
            col_sum[j] += row[j];
            col_sumsq[j] += row[j] * row[j];
            Xty[j] += row[j] * yi;
        }
    }

    double y_mean = y_sum / n;
    double y_var = y_sumsq / n - y_mean * y_mean;  // population var
    if (y_var < EPS) return {};

    // Column means and stds
    std::vector<double> col_mean(p);
    std::vector<double> col_var(p);  // population variance
    for (int j = 0; j < p; ++j) {
        col_mean[j] = col_sum[j] / n;
        col_var[j] = col_sumsq[j] / n - col_mean[j] * col_mean[j];
    }

    // 2. Precompute Gram matrix G = X'X (p x p) — only upper triangle + diagonal
    //    and cross-sums X_i'X_j for Pearson correlation needs
    std::vector<double> G(static_cast<size_t>(p) * p, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* row = X + static_cast<size_t>(i) * p;
        for (int a = 0; a < p; ++a) {
            double ra = row[a];
            // Only compute upper triangle including diagonal
            for (int b = a; b < p; ++b) {
                G[static_cast<size_t>(a) * p + b] += ra * row[b];
            }
        }
    }
    // Mirror
    for (int a = 0; a < p; ++a)
        for (int b = a + 1; b < p; ++b)
            G[static_cast<size_t>(b) * p + a] = G[static_cast<size_t>(a) * p + b];

    // Pearson correlation helper using precomputed sums
    // corr(a, b) = (sum(a*b)/n - mean_a*mean_b) / sqrt(var_a * var_b)
    // For column j vs y:
    auto corr_col_y = [&](int j) -> double {
        double cov = Xty[j] / n - col_mean[j] * y_mean;
        double den = std::sqrt(col_var[j] * y_var);
        return (den < 1e-15) ? 0.0 : cov / den;
    };

    // 3. Greedy selection
    std::vector<int> selected;
    selected.reserve(k_max);
    std::vector<bool> used(p, false);

    // Buffers for the normal equations at each step
    // When we have k selected features, we need to solve (k x k) systems
    // G_sel = G[selected, selected], and for each candidate j:
    //   v = G[selected, j]
    //   coef = G_sel^{-1} v
    //   residual stats can be computed from Gram matrix:
    //     var(residual) = var(col_j) - v' @ coef / n + adjustment for means
    //   But it's simpler to track centered quantities.

    // We'll maintain a Cholesky factor L of the centered Gram matrix of selected features.
    // For simplicity with small k (<=20), just re-solve each step.

    // Precompute centered X'y: cXty[j] = sum((X_j - mean_j)*(y - mean_y))
    std::vector<double> cXty(p);
    for (int j = 0; j < p; ++j) {
        cXty[j] = Xty[j] - n * col_mean[j] * y_mean;
    }

    // Precompute centered Gram: cG[a][b] = sum((X_a - mean_a)(X_b - mean_b))
    //   = G[a][b] - n * mean_a * mean_b
    std::vector<double> cG(static_cast<size_t>(p) * p);
    for (int a = 0; a < p; ++a)
        for (int b = 0; b < p; ++b)
            cG[static_cast<size_t>(a) * p + b] = G[static_cast<size_t>(a) * p + b]
                                                  - n * col_mean[a] * col_mean[b];

    for (int step = 0; step < k_max; ++step) {
        double best_ic = -1.0;
        int best_idx = -1;

        if (selected.empty()) {
            // First step: just pick highest |corr(col_j, y)|
            for (int j = 0; j < p; ++j) {
                if (col_var[j] < EPS) continue;
                double ic = std::abs(corr_col_y(j));
                if (ic > best_ic) { best_ic = ic; best_idx = j; }
            }
        } else {
            int k = static_cast<int>(selected.size());

            // Build centered Gram sub-matrix for selected features: cG_sel (k x k)
            std::vector<double> cG_sel(k * k);
            for (int a = 0; a < k; ++a)
                for (int b = 0; b < k; ++b)
                    cG_sel[a * k + b] = cG[static_cast<size_t>(selected[a]) * p + selected[b]];

            // For each candidate j, compute |corr(residual_j, y)|
            // residual_j = col_j - X_sel @ coef, where coef = (cG_sel)^{-1} @ cG[selected, j]
            //
            // var(residual) = cG[j,j] - v' @ coef  (all divided by n for variance)
            // cov(residual, y) = cXty[j] - v_y' @ coef
            //   where v_y[a] = cG[selected[a], :] @ (y centered) = cXty[selected[a]]
            //
            // Actually: cov(residual, y) = cXty[j] - sum_a coef[a] * cXty[selected[a]]

            // Precompute LU factorization of cG_sel (k x k, k <= 20)
            // Using augmented matrix approach: solve for multiple RHS
            // But we need different RHS per candidate, so factorize once, back-sub per candidate.

            // LU with partial pivot
            std::vector<double> LU(cG_sel);
            std::vector<int> piv(k);
            for (int i = 0; i < k; ++i) piv[i] = i;

            for (int col = 0; col < k; ++col) {
                // Partial pivot
                int max_row = col;
                double max_val = std::abs(LU[col * k + col]);
                for (int r = col + 1; r < k; ++r) {
                    double v = std::abs(LU[r * k + col]);
                    if (v > max_val) { max_val = v; max_row = r; }
                }
                if (max_val < 1e-14) {
                    // Singular — skip to Python-like fallback behavior
                    break;
                }
                if (max_row != col) {
                    std::swap(piv[col], piv[max_row]);
                    for (int c = 0; c < k; ++c)
                        std::swap(LU[col * k + c], LU[max_row * k + c]);
                }
                for (int r = col + 1; r < k; ++r) {
                    double factor = LU[r * k + col] / LU[col * k + col];
                    LU[r * k + col] = factor;  // store L factor
                    for (int c = col + 1; c < k; ++c)
                        LU[r * k + c] -= factor * LU[col * k + c];
                }
            }

            // Solve function using LU
            auto lu_solve = [&](const double* rhs, double* out) {
                // Apply pivot
                std::vector<double> b(k);
                for (int i = 0; i < k; ++i) b[i] = rhs[piv[i]];
                // Forward sub (L)
                for (int i = 1; i < k; ++i)
                    for (int j = 0; j < i; ++j)
                        b[i] -= LU[i * k + j] * b[j];
                // Back sub (U)
                for (int i = k - 1; i >= 0; --i) {
                    for (int j = i + 1; j < k; ++j)
                        b[i] -= LU[i * k + j] * b[j];
                    b[i] /= LU[i * k + i];
                }
                for (int i = 0; i < k; ++i) out[i] = b[i];
            };

            std::vector<double> v(k);
            std::vector<double> coef(k);

            for (int j = 0; j < p; ++j) {
                if (used[j]) continue;

                // v = cG[selected, j]
                for (int a = 0; a < k; ++a)
                    v[a] = cG[static_cast<size_t>(selected[a]) * p + j];

                // coef = cG_sel^{-1} @ v
                lu_solve(v.data(), coef.data());

                // var(residual) * n = cG[j,j] - v' @ coef
                double res_var_n = cG[static_cast<size_t>(j) * p + j];
                for (int a = 0; a < k; ++a)
                    res_var_n -= v[a] * coef[a];

                if (res_var_n / n < EPS) continue;

                // cov(residual, y) * n = cXty[j] - sum_a coef[a] * cXty[selected[a]]
                double res_cov_y_n = cXty[j];
                for (int a = 0; a < k; ++a)
                    res_cov_y_n -= coef[a] * cXty[selected[a]];

                // corr = cov / sqrt(var_res * var_y)
                double ic = std::abs(res_cov_y_n) / std::sqrt(res_var_n * y_var * n);
                if (ic > best_ic) { best_ic = ic; best_idx = j; }
            }
        }

        if (best_idx < 0) break;
        selected.push_back(best_idx);
        used[best_idx] = true;
    }

    return selected;
}

// Overload accepting std::vector (for backward compatibility with list-based binding)
inline std::vector<int> cpp_greedy_ic_select(
    const std::vector<double>& X_flat,
    const std::vector<double>& y,
    int n_samples,
    int n_features,
    int top_k = 20
) {
    return cpp_greedy_ic_select(X_flat.data(), y.data(), n_samples, n_features, top_k);
}
