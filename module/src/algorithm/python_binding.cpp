#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algorithm/algorithm.h"

namespace py = pybind11;

PYBIND11_MODULE(algorithm, m) {
    m.doc() = "CUDA algorithm library with Python bindings";
    
    py::class_<std::vector<float>>(m, "FloatVector")
        .def(py::init<>())
        .def("__len__", [](const std::vector<float>& v) { return v.size(); })
        .def("__getitem__", [](const std::vector<float>& v, size_t i) {
            if (i >= v.size()) {
                throw py::index_error();
            }
            return v[i];
        })
        .def("__setitem__", [](std::vector<float>& v, size_t i, float value) {
            if (i >= v.size()) {
                throw py::index_error();
            }
            v[i] = value;
        })
        .def("append", [](std::vector<float>& v, float value) {
            v.push_back(value);
        });
    
    m.def("add_vectors", &algorithm::add_vectors,
          "Add two vectors using CUDA",
          py::arg("a"), py::arg("b"));
}

