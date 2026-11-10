// SPDX-License-Identifier: GPL-3.0-or-later
// Device context of the compressible-flow GPU code: one stream, a cuBLAS handle
// (true FP32), a cuFFT plan cache and stream-ordered buffers.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bspf::cuda::flow {

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
inline void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(s)));
}
inline void check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw std::runtime_error(std::string(what) + ": cuFFT status " + std::to_string(int(r)));
}

struct Context {
    std::recursive_mutex mutex;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::map<std::tuple<int, long, bool>, std::pair<cufftHandle, cufftHandle>> plans;  // (n, batch, fp64)

    Context() {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count == 0) throw std::runtime_error("no CUDA device");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cublasCreate(&blas), "cublasCreate");
        check(cublasSetStream(blas, stream), "cublasSetStream");
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaMemPool_t pool;
        check(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
        uint64_t threshold = UINT64_MAX;
        check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold), "cudaMemPoolSetAttribute");
    }

    std::pair<cufftHandle, cufftHandle> plan(int n, long batch, bool fp64 = false) {
        auto it = plans.find({n, batch, fp64});
        if (it != plans.end()) return it->second;
        if (plans.size() >= 128) {
            check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            for (auto& e : plans) {
                cufftDestroy(e.second.first);
                cufftDestroy(e.second.second);
            }
            plans.clear();
        }
        const int h = n / 2 + 1;
        int dims[1] = {n};
        cufftHandle fwd, inv;
        check(cufftPlanMany(&fwd, 1, dims, nullptr, 1, n, nullptr, 1, h, fp64 ? CUFFT_D2Z : CUFFT_R2C,
                            static_cast<int>(batch)),
              "cufftPlanMany forward");
        check(cufftPlanMany(&inv, 1, dims, nullptr, 1, h, nullptr, 1, n, fp64 ? CUFFT_Z2D : CUFFT_C2R,
                            static_cast<int>(batch)),
              "cufftPlanMany inverse");
        check(cufftSetStream(fwd, stream), "cufftSetStream");
        check(cufftSetStream(inv, stream), "cufftSetStream");
        return plans.emplace(std::make_tuple(n, batch, fp64), std::make_pair(fwd, inv)).first->second;
    }
};

// Never destroyed (Python may release objects after CUDA teardown).
inline Context& context() {
    static Context* c = new Context();
    return *c;
}

// Stream-ordered device buffer.
template <class T>
struct Buffer {
    T* ptr = nullptr;
    size_t count = 0;
    Buffer() = default;
    explicit Buffer(size_t n) { resize(n); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr, o.count = 0; }
    Buffer& operator=(Buffer&& o) noexcept {
        std::swap(ptr, o.ptr);
        std::swap(count, o.count);
        return *this;
    }
    ~Buffer() {
        if (ptr) cudaFreeAsync(ptr, context().stream);
    }
    void resize(size_t n) {
        if (n <= count && ptr) return;
        if (ptr) cudaFreeAsync(ptr, context().stream);
        ptr = nullptr;
        check(cudaMallocAsync(reinterpret_cast<void**>(&ptr), sizeof(T) * std::max<size_t>(n, 1), context().stream),
              "cudaMallocAsync");
        count = n;
    }
    void upload(const std::vector<T>& host) {
        resize(host.size());
        check(cudaMemcpyAsync(ptr, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice, context().stream),
              "cudaMemcpyAsync H2D");
    }
    static std::vector<T> download_from(const T* src, size_t n) {
        std::vector<T> host(n);
        check(cudaMemcpyAsync(host.data(), src, sizeof(T) * n, cudaMemcpyDeviceToHost, context().stream),
              "cudaMemcpyAsync D2H");
        check(cudaStreamSynchronize(context().stream), "cudaStreamSynchronize");
        return host;
    }
    std::vector<T> download(size_t n) const {
        std::vector<T> host(n);
        check(cudaMemcpyAsync(host.data(), ptr, sizeof(T) * n, cudaMemcpyDeviceToHost, context().stream),
              "cudaMemcpyAsync D2H");
        check(cudaStreamSynchronize(context().stream), "cudaStreamSynchronize");
        return host;
    }
};

// C = alpha op(A) op(B) + beta C (column-major).
inline void gemm(bool ta, bool tb, long m, long n, long k, float alpha, const float* A, long lda, const float* B,
                 long ldb, float beta, float* C, long ldc) {
    if (m == 0 || n == 0) return;
    check(cublasSgemm(context().blas, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, int(m), int(n),
                      int(k), &alpha, A, int(lda), B, int(ldb), &beta, C, int(ldc)),
          "cublasSgemm");
}

inline int blocks(long n, int threads = 256) { return static_cast<int>((n + threads - 1) / threads); }

}  // namespace bspf::cuda::flow
