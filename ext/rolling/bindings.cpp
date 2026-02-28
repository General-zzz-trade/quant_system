#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rolling_window.hpp"
#include "vwap_window.hpp"
#include "technical.hpp"
#include "ols.hpp"
#include "cross_sectional.hpp"
#include "portfolio_math.hpp"
#include "factor_math.hpp"
#include "feature_selection.hpp"
#include "linalg.hpp"

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
}
