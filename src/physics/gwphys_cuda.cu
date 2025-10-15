// src/physics/gwphys_cuda.cu
// Build: CMakeLists.txt shown earlier
// Exports: info(), cosine_scores_gpu(A[m,k], B[n,k]) -> C[m,n] with cosine similarities

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdio>

namespace py = pybind11;

// -----------------------------
// CUDA helpers
// -----------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            std::ostringstream _oss;                                              \
            _oss << "CUDA error " << cudaGetErrorName(_err)                       \
                 << " (" << (int)_err << "): " << cudaGetErrorString(_err)        \
                 << " at " << __FILE__ << ":" << __LINE__;                        \
            throw std::runtime_error(_oss.str());                                 \
        }                                                                         \
    } while (0)
#endif

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

__device__ inline float l2_sqrt_safe(float x) {
    return sqrtf(fmaxf(x, 0.0f));
}

// -----------------------------
// Kernel: compute row-wise L2 norms
// X is row-major [rows, cols]; norms[rows]
// One block per row, reduction across threads
// -----------------------------
template<int TPB>
__global__ void row_l2norms_kernel(const float* __restrict__ X,
                                   float* __restrict__ norms,
                                   int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float sh[TPB];
    float acc = 0.0f;

    // grid-stride over columns within the row
    for (int c = threadIdx.x; c < cols; c += TPB) {
        float v = X[row * cols + c];
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();

    // reduction
    for (int s = TPB >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norms[row] = l2_sqrt_safe(sh[0]);
    }
}

// -----------------------------
// Kernel: compute cosine matrix
// A [M,K], B [N,K], norms_A[M], norms_B[N], C [M,N]
// One thread per output (i,j)
// -----------------------------
__global__ void cosine_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              const float* __restrict__ norms_A,
                              const float* __restrict__ norms_B,
                              float* __restrict__ C,
                              int M, int N, int K,
                              float eps)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = M * N;
    if (tid >= size) return;

    int i = tid / N;
    int j = tid % N;

    const float* a = A + i * K;
    const float* b = B + j * K;

    float dot = 0.0f;

    // unroll by 4 for a small boost
    int k = 0;
#pragma unroll 4
    for (; k + 3 < K; k += 4) {
        dot += a[k + 0] * b[k + 0];
        dot += a[k + 1] * b[k + 1];
        dot += a[k + 2] * b[k + 2];
        dot += a[k + 3] * b[k + 3];
    }
    for (; k < K; ++k) {
        dot += a[k] * b[k];
    }

    float denom = fmaxf(norms_A[i] * norms_B[j], eps);
    C[i * N + j] = dot / denom;
}

// -----------------------------
// Host wrappers
// -----------------------------
static std::string info_impl() {
    int device = 0;
    cudaDeviceProp prop{};
    cudaError_t st = cudaGetDeviceProperties(&prop, device);
    if (st != cudaSuccess) {
        return "GPU=? cc=? driver_fail";
    }
    std::ostringstream oss;
    oss << "GPU=" << prop.name
        << " cc=" << prop.major << prop.minor
        << " driver_ok";
    return oss.str();
}

py::array_t<float> cosine_scores_gpu(py::array_t<float, py::array::c_style | py::array::forcecast> A_in,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> B_in,
                                     float eps = 1e-8f)
{
    // shapes
    if (A_in.ndim() != 2 || B_in.ndim() != 2) {
        throw std::invalid_argument("A and B must be 2D float32 arrays");
    }
    ssize_t M = A_in.shape(0);
    ssize_t K = A_in.shape(1);
    ssize_t N = B_in.shape(0);
    if (B_in.shape(1) != K) {
        throw std::invalid_argument("A.shape[1] must equal B.shape[1]");
    }
    if (M == 0 || N == 0 || K == 0) {
        return py::array_t<float>({M, N});
    }

    // request buffers
    auto rA = A_in.unchecked<2>();
    auto rB = B_in.unchecked<2>();

    // allocate device buffers
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dNA = nullptr, *dNB = nullptr;
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
    size_t bytesNA = static_cast<size_t>(M) * sizeof(float);
    size_t bytesNB = static_cast<size_t>(N) * sizeof(float);

    // output
    py::array_t<float> C_out({M, N});
    auto rC = C_out.mutable_unchecked<2>();

    // release GIL during CUDA work
    py::gil_scoped_release release;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dNA, bytesNA));
    CUDA_CHECK(cudaMalloc(&dNB, bytesNB));

    // host to device copies
    // pack host data into contiguous buffers if needed
    // they already are c_style due to forcecast flag, so we can copy row-wise in one go
    CUDA_CHECK(cudaMemcpy(dA, A_in.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_in.data(), bytesB, cudaMemcpyHostToDevice));

    // norms
    const int TPB = 256;
    row_l2norms_kernel<TPB><<<static_cast<unsigned>(M), TPB>>>(dA, dNA, static_cast<int>(M), static_cast<int>(K));
    row_l2norms_kernel<TPB><<<static_cast<unsigned>(N), TPB>>>(dB, dNB, static_cast<int>(N), static_cast<int>(K));
    CUDA_CHECK(cudaGetLastError());

    // cosine matrix
    int threads = 256;
    int blocks  = div_up(static_cast<int>(M * N), threads);
    cosine_kernel<<<blocks, threads>>>(dA, dB, dNA, dNB, dC,
                                       static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                                       eps);
    CUDA_CHECK(cudaGetLastError());

    // device to host
    CUDA_CHECK(cudaMemcpy((void*)C_out.mutable_data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // free
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dNA);
    cudaFree(dNB);

    // GIL auto re-acquired here
    return C_out;
}

py::dict info() {
    py::dict d;
    d["text"] = info_impl();
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        d["name"] = prop.name;
        d["cc_major"] = prop.major;
        d["cc_minor"] = prop.minor;
        d["multiProcessorCount"] = prop.multiProcessorCount;
        d["totalGlobalMem"] = static_cast<long long>(prop.totalGlobalMem);
    } else {
        d["name"] = py::none();
        d["cc_major"] = py::none();
        d["cc_minor"] = py::none();
    }
    return d;
}

// -----------------------------
// Pybind
// -----------------------------
PYBIND11_MODULE(gwphys_cuda, m) {
    m.doc() = "CUDA kernels for fast similarity on GPU";
    m.def("cosine_scores_gpu", &cosine_scores_gpu,
          py::arg("A"), py::arg("B"), py::arg("eps") = 1e-8f,
          "Compute cosine similarity matrix C = A @ B.T / (||A|| * ||B||)");
    m.def("info", &info, "GPU info dict");
}
