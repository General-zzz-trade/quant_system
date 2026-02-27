#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rolling_window.hpp"
#include "technical.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_quant_rolling, m) {
    m.doc() = "C++ accelerated rolling window and technical indicators";

    py::class_<RollingWindow>(m, "RollingWindow")
        .def(py::init<int>(), py::arg("size"))
        .def("push", &RollingWindow::push, py::arg("x"))
        .def_property_readonly("full", &RollingWindow::full)
        .def_property_readonly("n", &RollingWindow::n)
        .def_property_readonly("size", &RollingWindow::get_size)
        .def_property_readonly("mean", &RollingWindow::mean)
        .def_property_readonly("variance", &RollingWindow::variance)
        .def_property_readonly("std", &RollingWindow::std_dev);

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
}
