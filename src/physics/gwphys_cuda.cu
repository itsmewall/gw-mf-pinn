// src/physics/gwphys_cuda.cu
// Build: use your existing CMakeLists for pybind11 + CUDA
// Exports:
//   - info() -> dict
//   - cosine_scores_gpu(A[m,k], B[n,k]) -> C[m,n]
//   - make_lf_stack(x[L], n_lf, ks[int32]) -> Y[n_lf, L]
//       * avg-pool por fator k (k∈ks), repete para L, aplica ganho [0.85,1.15],
//         ruído gaussiano ~ N(0, 0.02), e normaliza por janela

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace py = pybind11;

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
__device__ inline float l2_sqrt_safe(float x) { return sqrtf(fmaxf(x, 0.0f)); }

// ============================================================================
// RNG leve (xorshift32) + utilitários
// ============================================================================
__device__ inline uint32_t xorshift32(uint32_t& s) {
    // estado não pode ser 0
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
__device__ inline float u01(uint32_t& s) {
    // uniform ~ [0,1)
    return (xorshift32(s) & 0x00FFFFFF) / 16777216.0f;
}
__device__ inline float gauss01(uint32_t& s) {
    // Aproximação de normal padrão via soma de 12 uniformes (Irwin–Hall)
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; ++i) acc += u01(s);
    return acc - 6.0f;
}

// ============================================================================
// Normas por linha (reaproveitado no cosine_scores)
// ============================================================================
template<int TPB>
__global__ void row_l2norms_kernel(const float* __restrict__ X,
                                   float* __restrict__ norms,
                                   int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float sh[TPB];
    float acc = 0.0f;

    for (int c = threadIdx.x; c < cols; c += TPB) {
        float v = X[row * cols + c];
        acc += v * v;
    }
    sh[threadIdx.x] = acc;
    __syncthreads();

    for (int s = TPB >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[row] = l2_sqrt_safe(sh[0]);
}

// ============================================================================
// Cosine(A[M,K], B[N,K]) -> C[M,N]
// ============================================================================
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

py::array_t<float> cosine_scores_gpu(py::array_t<float, py::array::c_style | py::array::forcecast> A_in,
                                     py::array_t<float, py::array::c_style | py::array::forcecast> B_in,
                                     float eps = 1e-8f)
{
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

    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dNA = nullptr, *dNB = nullptr;
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(N) * K * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
    size_t bytesNA = static_cast<size_t>(M) * sizeof(float);
    size_t bytesNB = static_cast<size_t>(N) * sizeof(float);

    py::array_t<float> C_out({M, N});

    // Release GIL durante CUDA
    py::gil_scoped_release release;

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));
    CUDA_CHECK(cudaMalloc(&dNA, bytesNA));
    CUDA_CHECK(cudaMalloc(&dNB, bytesNB));

    CUDA_CHECK(cudaMemcpy(dA, A_in.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B_in.data(), bytesB, cudaMemcpyHostToDevice));

    const int TPB = 256;
    row_l2norms_kernel<TPB><<<static_cast<unsigned>(M), TPB>>>(dA, dNA, (int)M, (int)K);
    row_l2norms_kernel<TPB><<<static_cast<unsigned>(N), TPB>>>(dB, dNB, (int)N, (int)K);
    CUDA_CHECK(cudaGetLastError());

    int threads = 256;
    int blocks  = div_up((int)(M * N), threads);
    cosine_kernel<<<blocks, threads>>>(dA, dB, dNA, dNB, dC, (int)M, (int)N, (int)K, eps);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy((void*)C_out.mutable_data(), dC, bytesC, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dNA); cudaFree(dNB);
    return C_out;
}

// ============================================================================
// MF2: make_lf_stack(x[L], n_lf, ks[?]) -> [n_lf, L]
//   - Para cada saída i, pega k = ks[i % len(ks)]
//   - Faz avg-pool por blocos de tamanho k (alinhado no início) e repete ao longo de L
//   - Aplica ganho e ruído gaussiano fracos
//   - Normaliza (z-score) por saída
// ============================================================================
__global__ void lf_pool_kernel(const float* __restrict__ x,
                               float* __restrict__ Y,  // [n_lf, L]
                               const int* __restrict__ ks,
                               int L, int n_lf, int ks_len,
                               float gain_lo, float gain_hi, float noise_std,
                               uint32_t seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_lf * (long long)L;
    if ((long long)tid >= total) return;

    int i = tid / L;  // linha (saída)
    int t = tid % L;  // coluna (amostra)

    int k = (ks_len > 0 ? ks[i % ks_len] : 1);
    if (k < 1) k = 1;

    // ganho fixo por saída (determinístico)
    uint32_t st_gain = seed ^ (uint32_t)(i * 747796405u + 2891336453u);
    float g = gain_lo + (gain_hi - gain_lo) * u01(st_gain);

    // ruído por amostra
    uint32_t st_noise = seed ^ (uint32_t)(i * 916191u + t * 2147483647u);
    float n = noise_std * gauss01(st_noise);

    // média por bloco de tamanho k
    int m = L / k; // número de blocos completos
    float avg;
    if (m <= 0) {
        avg = x[t];
    } else {
        int j = t / k;
        if (j >= m) j = m - 1; // garante faixa válida
        int base = j * k;
        float s = 0.0f;
#pragma unroll
        for (int r = 0; r < 32; ++r) { // k max ~ 32 seguro; extra iterações são cortadas
            if (r >= k) break;
            s += x[base + r];
        }
        avg = s / (float)k;
    }
    float y = g * avg + n;
    Y[(size_t)i * (size_t)L + (size_t)t] = y;
}

template<int TPB>
__global__ void row_mean_var_kernel(const float* __restrict__ Y,
                                    float* __restrict__ means,
                                    float* __restrict__ stds,
                                    int n_rows, int L)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;

    __shared__ float shm_sum[TPB];
    __shared__ float shm_sq[TPB];

    float acc = 0.0f, acc2 = 0.0f;
    const float* y = Y + (size_t)row * (size_t)L;

    for (int c = threadIdx.x; c < L; c += TPB) {
        float v = y[c];
        acc  += v;
        acc2 += v * v;
    }
    shm_sum[threadIdx.x] = acc;
    shm_sq[threadIdx.x]  = acc2;
    __syncthreads();

    for (int s = TPB >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shm_sum[threadIdx.x] += shm_sum[threadIdx.x + s];
            shm_sq[threadIdx.x]  += shm_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = shm_sum[0] / fmaxf((float)L, 1.0f);
        float var  = shm_sq[0] / fmaxf((float)L, 1.0f) - mean * mean;
        var = fmaxf(var, 1e-12f);
        means[row] = mean;
        stds[row]  = sqrtf(var);
    }
}

__global__ void row_norm_kernel(float* __restrict__ Y,
                                const float* __restrict__ means,
                                const float* __restrict__ stds,
                                int n_rows, int L)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_rows * (long long)L;
    if ((long long)tid >= total) return;
    int i = tid / L;
    int t = tid % L;
    float mean = means[i];
    float stdv = stds[i];
    float* y = Y + (size_t)i * (size_t)L;
    y[t] = (y[t] - mean) / stdv;
}

py::array_t<float> make_lf_stack(py::array_t<float, py::array::c_style | py::array::forcecast> x_in,
                                 int n_lf,
                                 py::array_t<int,   py::array::c_style | py::array::forcecast> ks_in,
                                 float gain_lo = 0.85f,
                                 float gain_hi = 1.15f,
                                 float noise_std = 0.02f,
                                 uint32_t seed = 0xC0FFEEu)
{
    if (x_in.ndim() != 1) {
        throw std::invalid_argument("x must be 1D float32");
    }
    if (n_lf <= 0) {
        return py::array_t<float>({0, (py::ssize_t)x_in.shape(0)});
    }
    int L = (int)x_in.shape(0);

    int ks_len = (int)(ks_in.ndim() == 1 ? ks_in.shape(0) : 0);
    if (ks_len == 0) {
        // padrão seguro
        py::array_t<int> ks_def(1);
        auto r = ks_def.mutable_unchecked<1>();
        r(0) = 1;
        ks_in = ks_def;
        ks_len = 1;
    }

    // aloca saídas
    py::array_t<float> Y_out({(py::ssize_t)n_lf, (py::ssize_t)L});

    // buffers device
    float *dX = nullptr, *dY = nullptr, *dMean = nullptr, *dStd = nullptr;
    int   *dK = nullptr;

    size_t bytesX = (size_t)L * sizeof(float);
    size_t bytesY = (size_t)n_lf * (size_t)L * sizeof(float);
    size_t bytesK = (size_t)ks_len * sizeof(int);
    size_t bytesR = (size_t)n_lf * sizeof(float);

    // Release GIL
    py::gil_scoped_release release;

    CUDA_CHECK(cudaMalloc(&dX, bytesX));
    CUDA_CHECK(cudaMalloc(&dY, bytesY));
    CUDA_CHECK(cudaMalloc(&dK, bytesK));
    CUDA_CHECK(cudaMalloc(&dMean, bytesR));
    CUDA_CHECK(cudaMalloc(&dStd,  bytesR));

    CUDA_CHECK(cudaMemcpy(dX, x_in.data(), bytesX, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, ks_in.data(), bytesK, cudaMemcpyHostToDevice));

    // 1) pool + ganho + ruído
    {
        int threads = 256;
        long long total = (long long)n_lf * (long long)L;
        int blocks = (int)div_up((int)((total > INT32_MAX) ? INT32_MAX : (int)total), threads);
        // se total > INT32_MAX, fazemos em laços (improvável para nosso L e n_lf)
        lf_pool_kernel<<<blocks, threads>>>(dX, dY, dK, L, n_lf, ks_len, gain_lo, gain_hi, noise_std, seed);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) stats por linha
    {
        const int TPB = 256;
        row_mean_var_kernel<TPB><<<n_lf, TPB>>>(dY, dMean, dStd, n_lf, L);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3) normalização por linha
    {
        int threads = 256;
        long long total = (long long)n_lf * (long long)L;
        int blocks = (int)div_up((int)((total > INT32_MAX) ? INT32_MAX : (int)total), threads);
        row_norm_kernel<<<blocks, threads>>>(dY, dMean, dStd, n_lf, L);
        CUDA_CHECK(cudaGetLastError());
    }

    // H->D
    CUDA_CHECK(cudaMemcpy((void*)Y_out.mutable_data(), dY, bytesY, cudaMemcpyDeviceToHost));

    cudaFree(dX); cudaFree(dY); cudaFree(dK); cudaFree(dMean); cudaFree(dStd);
    return Y_out;
}

// ============================================================================
// info()
// ============================================================================
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
        d["totalGlobalMem"] = (long long)prop.totalGlobalMem;
    } else {
        d["name"] = py::none();
        d["cc_major"] = py::none();
        d["cc_minor"] = py::none();
    }
    return d;
}

// ============================================================================
// PyBind
// ============================================================================
PYBIND11_MODULE(gwphys_cuda, m) {
    m.doc() = "CUDA kernels for MF pipeline (cosine + LF stack)";
    m.def("cosine_scores_gpu", &cosine_scores_gpu,
          py::arg("A"), py::arg("B"), py::arg("eps") = 1e-8f,
          "Compute cosine similarity matrix C = A @ B.T / (||A|| * ||B||)");
    m.def("make_lf_stack", &make_lf_stack,
          py::arg("x"), py::arg("n_lf"), py::arg("ks"),
          py::arg("gain_lo") = 0.85f, py::arg("gain_hi") = 1.15f,
          py::arg("noise_std") = 0.02f, py::arg("seed") = 0xC0FFEEu,
          "Build low-fidelity stack: avg-pool by k, repeat to L, add noise/gain, normalize per row");
    m.def("info", &info, "GPU info dict");
}
