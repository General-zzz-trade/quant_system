#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "rolling_window.hpp"
#include "vwap_window.hpp"
#include "technical.hpp"
#include "ols.hpp"
#include "cross_sectional.hpp"
#include "portfolio_math.hpp"
#include "factor_math.hpp"
#include "feature_selection.hpp"
#include "linalg.hpp"
#include "bootstrap.hpp"
#include "monte_carlo.hpp"
#include "target.hpp"
#include "greedy_select.hpp"
#include "feature_engine.hpp"
#include "backtest_engine.hpp"
#include "feature_selector.hpp"
#include "multi_timeframe.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_quant_rolling, m) {
    m.doc() = "C++ accelerated rolling window, technical indicators, VWAP, and OLS";

    py::class_<RollingWindow>(m, "RollingWindow")
        .def(py::init<int>(), py::arg("size"))
        .def("push", &RollingWindow::push, py::arg("x"))
        .def_property_readonly("full", &RollingWindow::full)
        .def_property_readonly("n", &RollingWindow::n)
        .def_property_readonly("size", &RollingWindow::get_size)
        .def_property_readonly("mean", &RollingWindow::mean)
        .def_property_readonly("variance", &RollingWindow::variance)
        .def_property_readonly("std", &RollingWindow::std_dev);

    py::class_<VWAPWindow>(m, "VWAPWindow")
        .def(py::init<int>(), py::arg("size"))
        .def("push", &VWAPWindow::push, py::arg("price"), py::arg("volume"))
        .def_property_readonly("full", &VWAPWindow::full)
        .def_property_readonly("n", &VWAPWindow::n)
        .def_property_readonly("size", &VWAPWindow::get_size)
        .def_property_readonly("vwap", &VWAPWindow::vwap)
        .def_property_readonly("sum_pv", &VWAPWindow::sum_pv)
        .def_property_readonly("sum_v", &VWAPWindow::sum_v);

    // Existing technical indicators
    m.def("cpp_sma", &cpp_sma, py::arg("values"), py::arg("window"));
    m.def("cpp_ema", &cpp_ema, py::arg("values"), py::arg("window"));
    m.def("cpp_returns", &cpp_returns, py::arg("values"), py::arg("log_ret") = false);
    m.def("cpp_volatility", &cpp_volatility, py::arg("rets"), py::arg("window"));
    m.def("cpp_rsi", &cpp_rsi, py::arg("values"), py::arg("window") = 14);
    m.def("cpp_macd", &cpp_macd,
          py::arg("values"), py::arg("fast") = 12,
          py::arg("slow") = 26, py::arg("signal") = 9);
    m.def("cpp_bollinger_bands", &cpp_bollinger_bands,
          py::arg("values"), py::arg("window") = 20,
          py::arg("num_std") = 2.0);
    m.def("cpp_atr", &cpp_atr,
          py::arg("highs"), py::arg("lows"),
          py::arg("closes"), py::arg("window") = 14);

    // New: VWAP and microstructure batch functions
    m.def("cpp_vwap", &cpp_vwap,
          py::arg("closes"), py::arg("volumes"), py::arg("window"));
    m.def("cpp_order_flow_imbalance", &cpp_order_flow_imbalance,
          py::arg("opens"), py::arg("closes"), py::arg("volumes"), py::arg("window"));
    m.def("cpp_rolling_volatility", &cpp_rolling_volatility,
          py::arg("rets"), py::arg("window"));
    m.def("cpp_price_impact", &cpp_price_impact,
          py::arg("closes"), py::arg("volumes"), py::arg("window"));

    // New: OLS regression kernel
    m.def("cpp_ols", &cpp_ols, py::arg("x"), py::arg("y"));

    // Cross-sectional features
    m.def("cpp_momentum_rank", &cpp_momentum_rank,
          py::arg("returns_matrix"), py::arg("lookback"));
    m.def("cpp_rolling_beta", &cpp_rolling_beta,
          py::arg("asset_returns"), py::arg("market_returns"), py::arg("window"));
    m.def("cpp_relative_strength", &cpp_relative_strength,
          py::arg("target_returns"), py::arg("benchmark_returns"), py::arg("window"));

    // Portfolio math: covariance, correlation, variance
    m.def("cpp_sample_covariance", &cpp_sample_covariance,
          py::arg("returns_matrix"));
    m.def("cpp_ewma_covariance", &cpp_ewma_covariance,
          py::arg("returns_matrix"), py::arg("alpha"));
    m.def("cpp_rolling_correlation", &cpp_rolling_correlation,
          py::arg("returns_matrix"), py::arg("window"));
    m.def("cpp_portfolio_variance", &cpp_portfolio_variance,
          py::arg("weights"), py::arg("cov"));

    // Factor model math
    m.def("cpp_compute_exposures", &cpp_compute_exposures,
          py::arg("asset_returns"), py::arg("factor_returns"));
    m.def("cpp_factor_model_covariance", &cpp_factor_model_covariance,
          py::arg("exposures"), py::arg("factor_cov"), py::arg("specific_risk"));
    m.def("cpp_estimate_specific_risk", &cpp_estimate_specific_risk,
          py::arg("asset_returns"), py::arg("factor_returns"), py::arg("exposures"));

    // Feature selection
    m.def("cpp_correlation_select", &cpp_correlation_select,
          py::arg("features"), py::arg("target"));
    m.def("cpp_mutual_info_select", &cpp_mutual_info_select,
          py::arg("features"), py::arg("target"), py::arg("n_bins"));

    // Black-Litterman
    m.def("cpp_black_litterman_posterior", &cpp_black_litterman_posterior,
          py::arg("sigma"), py::arg("market_weights"),
          py::arg("P"), py::arg("Q"), py::arg("confidences"),
          py::arg("tau"), py::arg("risk_aversion"));

    // Bootstrap Sharpe CI
    py::class_<BootstrapResult>(m, "BootstrapResult")
        .def_readonly("sharpe_mean", &BootstrapResult::sharpe_mean)
        .def_readonly("sharpe_95ci_lo", &BootstrapResult::sharpe_95ci_lo)
        .def_readonly("sharpe_95ci_hi", &BootstrapResult::sharpe_95ci_hi)
        .def_readonly("p_sharpe_gt_0", &BootstrapResult::p_sharpe_gt_0)
        .def_readonly("p_sharpe_gt_05", &BootstrapResult::p_sharpe_gt_05);
    m.def("cpp_bootstrap_sharpe_ci", &cpp_bootstrap_sharpe_ci,
          py::arg("returns"),
          py::arg("n_bootstrap") = 10000,
          py::arg("block_size") = 5,
          py::arg("seed") = 42);

    // Monte Carlo simulation
    py::class_<MCResult>(m, "MCResult")
        .def_readonly("paths", &MCResult::paths)
        .def_readonly("mean_final", &MCResult::mean_final)
        .def_readonly("median_final", &MCResult::median_final)
        .def_readonly("percentile_5", &MCResult::percentile_5)
        .def_readonly("percentile_95", &MCResult::percentile_95)
        .def_readonly("prob_loss", &MCResult::prob_loss)
        .def_readonly("prob_target", &MCResult::prob_target)
        .def_readonly("max_drawdown_mean", &MCResult::max_drawdown_mean)
        .def_readonly("max_drawdown_95", &MCResult::max_drawdown_95);
    m.def("cpp_simulate_paths", &cpp_simulate_paths,
          py::arg("returns"),
          py::arg("n_paths") = 1000,
          py::arg("horizon") = 252,
          py::arg("parametric") = false,
          py::arg("target_return") = 0.0,
          py::arg("block_size") = 5,
          py::arg("seed") = 42);

    // Vol-normalized target
    m.def("cpp_vol_normalized_target", &cpp_vol_normalized_target,
          py::arg("closes"),
          py::arg("horizon") = 5,
          py::arg("vol_window") = 20);

    // Greedy IC-based feature selection (raw vector version)
    m.def("cpp_greedy_ic_select",
          static_cast<std::vector<int>(*)(const std::vector<double>&, const std::vector<double>&, int, int, int)>(&cpp_greedy_ic_select),
          py::arg("X_flat"),
          py::arg("y"),
          py::arg("n_samples"),
          py::arg("n_features"),
          py::arg("top_k") = 20);

    // Greedy IC-based feature selection (numpy version — zero-copy)
    m.def("cpp_greedy_ic_select_np",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int top_k) -> std::vector<int> {
              auto xbuf = X.request();
              auto ybuf = y.request();
              if (xbuf.ndim != 2 || ybuf.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              int n_samples = static_cast<int>(xbuf.shape[0]);
              int n_features = static_cast<int>(xbuf.shape[1]);
              const double* xptr = static_cast<const double*>(xbuf.ptr);
              const double* yptr = static_cast<const double*>(ybuf.ptr);
              return cpp_greedy_ic_select(xptr, yptr, n_samples, n_features, top_k);
          },
          py::arg("X"), py::arg("y"), py::arg("top_k") = 20);

    // Batch feature engine
    m.def("cpp_compute_all_features", &cpp_compute_all_features,
          py::arg("timestamps"), py::arg("opens"), py::arg("highs"),
          py::arg("lows"), py::arg("closes"), py::arg("volumes"),
          py::arg("trades"), py::arg("tbv"), py::arg("qv"), py::arg("tbqv"),
          py::arg("funding_sched"), py::arg("oi_sched"), py::arg("ls_sched"),
          py::arg("spot_sched"), py::arg("fgi_sched"), py::arg("iv_sched"),
          py::arg("pcr_sched"), py::arg("onchain_sched"));
    m.def("cpp_feature_names", &cpp_feature_names);

    // Backtest engine
    m.def("cpp_run_backtest", &backtest::cpp_run_backtest,
          py::arg("timestamps"), py::arg("closes"), py::arg("volumes"),
          py::arg("vol_20"), py::arg("y_pred"), py::arg("bear_probs"),
          py::arg("vol_values"), py::arg("funding_rates"),
          py::arg("funding_ts"), py::arg("config_json"));
    m.def("cpp_pred_to_signal", &backtest::cpp_pred_to_signal,
          py::arg("y_pred"), py::arg("deadzone") = 0.5,
          py::arg("min_hold") = 24, py::arg("zscore_window") = 720,
          py::arg("zscore_warmup") = 168);

    // Feature selector (IC-based)
    m.def("cpp_rolling_ic_select",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int top_k, int ic_window) -> std::vector<int> {
              auto xb = X.request(); auto yb = y.request();
              if (xb.ndim != 2 || yb.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              return feat_sel::cpp_rolling_ic_select(
                  static_cast<const double*>(xb.ptr), static_cast<const double*>(yb.ptr),
                  static_cast<int>(xb.shape[0]), static_cast<int>(xb.shape[1]),
                  top_k, ic_window);
          },
          py::arg("X"), py::arg("y"), py::arg("top_k") = 20, py::arg("ic_window") = 500);

    m.def("cpp_spearman_ic_select",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int top_k, int ic_window) -> std::vector<int> {
              auto xb = X.request(); auto yb = y.request();
              if (xb.ndim != 2 || yb.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              return feat_sel::cpp_spearman_ic_select(
                  static_cast<const double*>(xb.ptr), static_cast<const double*>(yb.ptr),
                  static_cast<int>(xb.shape[0]), static_cast<int>(xb.shape[1]),
                  top_k, ic_window);
          },
          py::arg("X"), py::arg("y"), py::arg("top_k") = 20, py::arg("ic_window") = 500);

    m.def("cpp_icir_select",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int top_k, int ic_window, int n_windows,
             double min_icir, int max_consec_neg) -> std::vector<int> {
              auto xb = X.request(); auto yb = y.request();
              if (xb.ndim != 2 || yb.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              return feat_sel::cpp_icir_select(
                  static_cast<const double*>(xb.ptr), static_cast<const double*>(yb.ptr),
                  static_cast<int>(xb.shape[0]), static_cast<int>(xb.shape[1]),
                  top_k, ic_window, n_windows, min_icir, max_consec_neg);
          },
          py::arg("X"), py::arg("y"), py::arg("top_k") = 20,
          py::arg("ic_window") = 200, py::arg("n_windows") = 5,
          py::arg("min_icir") = 0.3, py::arg("max_consec_neg") = 3);

    m.def("cpp_stable_icir_select",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int top_k, int ic_window, int n_windows,
             double min_icir, int min_stable_folds,
             double sign_consistency) -> std::vector<int> {
              auto xb = X.request(); auto yb = y.request();
              if (xb.ndim != 2 || yb.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              return feat_sel::cpp_stable_icir_select(
                  static_cast<const double*>(xb.ptr), static_cast<const double*>(yb.ptr),
                  static_cast<int>(xb.shape[0]), static_cast<int>(xb.shape[1]),
                  top_k, ic_window, n_windows, min_icir, min_stable_folds, sign_consistency);
          },
          py::arg("X"), py::arg("y"), py::arg("top_k") = 20,
          py::arg("ic_window") = 200, py::arg("n_windows") = 5,
          py::arg("min_icir") = 0.3, py::arg("min_stable_folds") = 4,
          py::arg("sign_consistency") = 0.8);

    m.def("cpp_feature_icir_report",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> X,
             py::array_t<double, py::array::c_style | py::array::forcecast> y,
             int ic_window, int n_windows) -> py::array_t<double> {
              auto xb = X.request(); auto yb = y.request();
              if (xb.ndim != 2 || yb.ndim != 1)
                  throw std::runtime_error("X must be 2D, y must be 1D");
              int p = static_cast<int>(xb.shape[1]);
              auto flat = feat_sel::cpp_feature_icir_report(
                  static_cast<const double*>(xb.ptr), static_cast<const double*>(yb.ptr),
                  static_cast<int>(xb.shape[0]), p, ic_window, n_windows);
              auto result = py::array_t<double>({p, 5});
              std::memcpy(result.mutable_data(), flat.data(), flat.size() * sizeof(double));
              return result;
          },
          py::arg("X"), py::arg("y"), py::arg("ic_window") = 200, py::arg("n_windows") = 5);

    // Multi-timeframe 4h features
    m.def("cpp_compute_4h_features", &mtf::cpp_compute_4h_features,
          py::arg("ts"), py::arg("opens"), py::arg("highs"),
          py::arg("lows"), py::arg("closes"), py::arg("volumes"));
    m.def("cpp_4h_feature_names", &mtf::cpp_4h_feature_names);
}
