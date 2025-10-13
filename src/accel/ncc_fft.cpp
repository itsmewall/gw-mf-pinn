// clang-format off
/*
<%
setup_pybind11(cfg)
import sys
# OpenMP flags por plataforma
if sys.platform.startswith("win"):
    cfg["compiler_args"] += ["/O2", "/openmp"]
    cfg["linker_args"] += []
else:
    cfg["compiler_args"] += ["-O3", "-fopenmp", "-march=native"]
    cfg["linker_args"] += ["-fopenmp"]
%>
*/
// clang-format on
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <vector>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace py = pybind11;

// Normaliza um vetor: (x - mean)/std  (retorna mean e std para debug/checagem se quiser)
static void zscore_inplace(double* x, ssize_t n, double &mean, double &stdv) {
    double s = 0.0, s2 = 0.0;
    for (ssize_t i = 0; i < n; ++i) { s += x[i]; }
    mean = s / double(n);
    for (ssize_t i = 0; i < n; ++i) { double d = x[i] - mean; s2 += d*d; }
    stdv = std::sqrt(s2 / double(n));
    if (stdv < 1e-12) stdv = 1e-12;
    for (ssize_t i = 0; i < n; ++i) { x[i] = (x[i] - mean)/stdv; }
}

// Correlação normalizada para todos os lags em [-max_shift, +max_shift].
// a e b com mesmo tamanho N, ambos 1D.
py::array_t<double> correlate_all_lags(py::array_t<double, py::array::c_style | py::array::forcecast> a_in,
                                       py::array_t<double, py::array::c_style | py::array::forcecast> b_in,
                                       int max_shift) {
    if (max_shift < 0) throw std::runtime_error("max_shift must be >= 0");

    auto a_buf = a_in.request();
    auto b_buf = b_in.request();
    if (a_buf.ndim != 1 || b_buf.ndim != 1) throw std::runtime_error("Inputs must be 1D");
    if (a_buf.size != b_buf.size) throw std::runtime_error("Inputs must have same size");

    const ssize_t n = a_buf.size;
    double* a = static_cast<double*>(a_buf.ptr);
    double* b = static_cast<double*>(b_buf.ptr);

    // Copia para buffers temporários (não alteramos entrada do Python)
    std::vector<double> va(a, a + n);
    std::vector<double> vb(b, b + n);

    double am=0.0, as=1.0, bm=0.0, bs=1.0;
    zscore_inplace(va.data(), n, am, as);
    zscore_inplace(vb.data(), n, bm, bs);

    const int L = 2*max_shift + 1;
    py::array_t<double> out(L);
    auto out_buf = out.request();
    double* outp = static_cast<double*>(out_buf.ptr);

    // Paraleliza por lag; cada lag computa um dot product parcial
    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < L; ++idx) {
        int k = idx - max_shift; // lag em [-max_shift..+max_shift]
        double acc = 0.0;
        ssize_t lo = 0, hi = 0;
        if (k >= 0) {
            lo = 0; hi = n - k;
            for (ssize_t i = lo; i < hi; ++i) {
                acc += va[i] * vb[i + k];
            }
        } else {
            int k2 = -k;
            lo = 0; hi = n - k2;
            for (ssize_t i = lo; i < hi; ++i) {
                acc += va[i + k2] * vb[i];
            }
        }
        double denom = double(hi - lo);
        outp[idx] = (denom > 0.0) ? (acc / denom) : 0.0;
    }

    return out;
}

PYBIND11_MODULE(ncc_fft, m) {
    m.doc() = "Normalized cross-correlation for multi-lag (OpenMP)";
    m.def("correlate_all_lags", &correlate_all_lags,
          py::arg("a"), py::arg("b"), py::arg("max_shift"),
          "Compute normalized cross-correlation over integer lags in [-max_shift, +max_shift].");
}