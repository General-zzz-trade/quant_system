#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

/**
 * Batch factor exposure (beta) computation.
 *
 * For each asset-factor pair: beta = cov(asset, factor) / var(factor).
 * Precomputes factor means/demeaned data once for efficiency.
 *
 * asset_returns: N_assets x T matrix
 * factor_returns: N_factors x T matrix
 * Returns: N_assets x N_factors exposure matrix
 */
inline std::vector<std::vector<double>> cpp_compute_exposures(
    const std::vector<std::vector<double>>& asset_returns,
    const std::vector<std::vector<double>>& factor_returns
) {
    size_t N = asset_returns.size();
    size_t F = factor_returns.size();
    if (N == 0 || F == 0)
        return std::vector<std::vector<double>>(N, std::vector<double>(F, 0.0));

    // Find minimum observation count across all series
    size_t T = asset_returns[0].size();
    for (size_t i = 1; i < N; ++i)
        T = std::min(T, asset_returns[i].size());
    for (size_t f = 0; f < F; ++f)
        T = std::min(T, factor_returns[f].size());

    std::vector<std::vector<double>> result(N, std::vector<double>(F, 0.0));
    if (T < 2) return result;

    double inv_t = 1.0 / static_cast<double>(T);
    double inv_t1 = 1.0 / static_cast<double>(T - 1);

    // Precompute factor means and demeaned data
    std::vector<double> f_means(F);
    std::vector<std::vector<double>> f_dm(F, std::vector<double>(T));
    std::vector<double> f_var(F);

    for (size_t f = 0; f < F; ++f) {
        double s = 0.0;
        for (size_t t = 0; t < T; ++t)
            s += factor_returns[f][t];
        f_means[f] = s * inv_t;

        double var_sum = 0.0;
        for (size_t t = 0; t < T; ++t) {
            double d = factor_returns[f][t] - f_means[f];
            f_dm[f][t] = d;
            var_sum += d * d;
        }
        f_var[f] = var_sum * inv_t1;
    }

    // For each asset, compute mean, then beta against each factor
    for (size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (size_t t = 0; t < T; ++t)
            s += asset_returns[i][t];
        double a_mean = s * inv_t;

        for (size_t f = 0; f < F; ++f) {
            if (f_var[f] < 1e-12) continue;

            double cov = 0.0;
            for (size_t t = 0; t < T; ++t)
                cov += (asset_returns[i][t] - a_mean) * f_dm[f][t];
            cov *= inv_t1;

            result[i][f] = cov / f_var[f];
        }
    }

    return result;
}

/**
 * Factor model covariance: Sigma = B * F * B' + D.
 *
 * Optimized: precompute BF = B * F, then Sigma = BF * B' + D.
 * This is O(N*F^2 + N^2*F) instead of O(N^2*F^2).
 *
 * exposures: N_assets x N_factors (B matrix)
 * factor_cov: N_factors x N_factors (F matrix)
 * specific_risk: N_assets vector (diagonal D)
 * Returns: N_assets x N_assets covariance matrix
 */
inline std::vector<std::vector<double>> cpp_factor_model_covariance(
    const std::vector<std::vector<double>>& exposures,
    const std::vector<std::vector<double>>& factor_cov,
    const std::vector<double>& specific_risk
) {
    size_t N = exposures.size();
    size_t F = factor_cov.size();

    std::vector<std::vector<double>> result(N, std::vector<double>(N, 0.0));
    if (N == 0 || F == 0) {
        // Only diagonal specific risk
        for (size_t i = 0; i < N; ++i)
            if (i < specific_risk.size())
                result[i][i] = specific_risk[i];
        return result;
    }

    // Step 1: Compute BF = B * F  (N x F)
    std::vector<std::vector<double>> BF(N, std::vector<double>(F, 0.0));
    for (size_t i = 0; i < N; ++i) {
        for (size_t f2 = 0; f2 < F; ++f2) {
            double s = 0.0;
            for (size_t f1 = 0; f1 < F; ++f1) {
                double b = (f1 < exposures[i].size()) ? exposures[i][f1] : 0.0;
                double fc = (f2 < factor_cov[f1].size()) ? factor_cov[f1][f2] : 0.0;
                s += b * fc;
            }
            BF[i][f2] = s;
        }
    }

    // Step 2: Sigma = BF * B' + D
    // Note: result is symmetric only if factor_cov is symmetric.
    // Compute all N×N entries for correctness with any input.
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double cov = 0.0;
            for (size_t f = 0; f < F; ++f) {
                double bj = (f < exposures[j].size()) ? exposures[j][f] : 0.0;
                cov += BF[i][f] * bj;
            }
            if (i == j && i < specific_risk.size())
                cov += specific_risk[i];
            result[i][j] = cov;
        }
    }

    return result;
}

/**
 * Estimate specific (idiosyncratic) risk for each asset.
 *
 * residual = asset_return - sum(beta * factor_return)
 * specific_risk = var(residuals)
 *
 * asset_returns: N_assets x T
 * factor_returns: N_factors x T
 * exposures: N_assets x N_factors
 * Returns: N_assets vector of residual variances
 */
inline std::vector<double> cpp_estimate_specific_risk(
    const std::vector<std::vector<double>>& asset_returns,
    const std::vector<std::vector<double>>& factor_returns,
    const std::vector<std::vector<double>>& exposures
) {
    size_t N = asset_returns.size();
    size_t F = factor_returns.size();

    std::vector<double> result(N, 0.0);
    if (N == 0) return result;

    // Find min observation count
    size_t T = asset_returns[0].size();
    for (size_t i = 1; i < N; ++i)
        T = std::min(T, asset_returns[i].size());
    for (size_t f = 0; f < F; ++f)
        T = std::min(T, factor_returns[f].size());

    if (T < 2) return result;

    double inv_t1 = 1.0 / static_cast<double>(T - 1);

    for (size_t i = 0; i < N; ++i) {
        // Compute residuals and running sum for mean
        double sum_r = 0.0;
        std::vector<double> residuals(T);

        for (size_t t = 0; t < T; ++t) {
            double predicted = 0.0;
            for (size_t f = 0; f < F; ++f) {
                double beta = (f < exposures[i].size()) ? exposures[i][f] : 0.0;
                predicted += beta * factor_returns[f][t];
            }
            residuals[t] = asset_returns[i][t] - predicted;
            sum_r += residuals[t];
        }

        double mean_r = sum_r / static_cast<double>(T);
        double var_sum = 0.0;
        for (size_t t = 0; t < T; ++t) {
            double d = residuals[t] - mean_r;
            var_sum += d * d;
        }
        result[i] = var_sum * inv_t1;
    }

    return result;
}
