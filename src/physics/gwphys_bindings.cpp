#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> cosine_scores_gpu(py::array_t<float, py::array::c_style | py::array::forcecast> A,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> B);
py::str cuda_info();

PYBIND11_MODULE(gwphys_cuda, m) {
    m.doc() = "GW physics CUDA kernels";
    m.def("cosine_scores_gpu", &cosine_scores_gpu,
          "Retorna matriz [B,K] de similaridade coseno entre janelas e templates",
          py::arg("windows"), py::arg("templates"));
    m.def("info", &cuda_info, "Info rápida do runtime CUDA");
}