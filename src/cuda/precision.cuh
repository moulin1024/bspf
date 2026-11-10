// SPDX-License-Identifier: GPL-3.0-or-later
// Precision traits of the compressible-flow device code: T = float (FP32, multigrid
// preconditioner levels) or double (FP64, the default solver). cuBLAS S/D routines, cuFFT
// R2C/C2R or D2Z/Z2D, and the matching complex type.
#pragma once

#include <cublas_v2.h>
#include <cufft.h>

#include <vector>

#include "flow_context.cuh"

namespace bspf::cuda::flow {

template <class T>
struct Prec;

template <>
struct Prec<float> {
    using C = cufftComplex;
    static constexpr bool fp64 = false;
    static cufftResult forward(cufftHandle p, float* in, C* out) { return cufftExecR2C(p, in, out); }
    static cufftResult inverse(cufftHandle p, C* in, float* out) { return cufftExecC2R(p, in, out); }
    static cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                               const float* al, const float* A, int lda, const float* B, int ldb, const float* be,
                               float* Cm, int ldc) {
        return cublasSgemm(h, a, b, m, n, k, al, A, lda, B, ldb, be, Cm, ldc);
    }
    static cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                                       const float* al, const float* A, int lda, long sa, const float* B, int ldb,
                                       long sb, const float* be, float* Cm, int ldc, long sc, int batch) {
        return cublasSgemmStridedBatched(h, a, b, m, n, k, al, A, lda, sa, B, ldb, sb, be, Cm, ldc, sc, batch);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const float* a, const float* x, float* y) {
        return cublasSaxpy(h, n, a, x, 1, y, 1);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, float* r) { return cublasSnrm2(h, n, x, 1, r); }
    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t t, int m, int n, const float* al, const float* A,
                               int lda, const float* x, const float* be, float* y) {
        return cublasSgemv(h, t, m, n, al, A, lda, x, 1, be, y, 1);
    }
    static cublasStatus_t getrf(cublasHandle_t h, int n, float* const* A, int lda, int* piv, int* info, int batch) {
        return cublasSgetrfBatched(h, n, A, lda, piv, info, batch);
    }
    static cublasStatus_t getrs(cublasHandle_t h, int n, const float* const* A, int lda, const int* piv,
                                float* const* B, int ldb, int* info, int batch) {
        return cublasSgetrsBatched(h, CUBLAS_OP_N, n, 1, A, lda, piv, B, ldb, info, batch);
    }
};

template <>
struct Prec<double> {
    using C = cufftDoubleComplex;
    static constexpr bool fp64 = true;
    static cufftResult forward(cufftHandle p, double* in, C* out) { return cufftExecD2Z(p, in, out); }
    static cufftResult inverse(cufftHandle p, C* in, double* out) { return cufftExecZ2D(p, in, out); }
    static cublasStatus_t gemm(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                               const double* al, const double* A, int lda, const double* B, int ldb, const double* be,
                               double* Cm, int ldc) {
        return cublasDgemm(h, a, b, m, n, k, al, A, lda, B, ldb, be, Cm, ldc);
    }
    static cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k,
                                       const double* al, const double* A, int lda, long sa, const double* B, int ldb,
                                       long sb, const double* be, double* Cm, int ldc, long sc, int batch) {
        return cublasDgemmStridedBatched(h, a, b, m, n, k, al, A, lda, sa, B, ldb, sb, be, Cm, ldc, sc, batch);
    }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const double* a, const double* x, double* y) {
        return cublasDaxpy(h, n, a, x, 1, y, 1);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, double* r) { return cublasDnrm2(h, n, x, 1, r); }
    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t t, int m, int n, const double* al, const double* A,
                               int lda, const double* x, const double* be, double* y) {
        return cublasDgemv(h, t, m, n, al, A, lda, x, 1, be, y, 1);
    }
    static cublasStatus_t getrf(cublasHandle_t h, int n, double* const* A, int lda, int* piv, int* info, int batch) {
        return cublasDgetrfBatched(h, n, A, lda, piv, info, batch);
    }
    static cublasStatus_t getrs(cublasHandle_t h, int n, const double* const* A, int lda, const int* piv,
                                double* const* B, int ldb, int* info, int batch) {
        return cublasDgetrsBatched(h, CUBLAS_OP_N, n, 1, A, lda, piv, B, ldb, info, batch);
    }
};

template <class Cx>
__device__ inline Cx cmul(Cx a, Cx b) {
    Cx r;
    r.x = a.x * b.x - a.y * b.y;
    r.y = a.x * b.y + a.y * b.x;
    return r;
}
template <class Cx>
__device__ inline Cx cconj_mul(Cx a, Cx b) {  // conj(a) b
    Cx r;
    r.x = a.x * b.x + a.y * b.y;
    r.y = a.x * b.y - a.y * b.x;
    return r;
}

// C = alpha op(A) op(B) + beta C (column-major), in T.
template <class T>
inline void gemm_t(bool ta, bool tb, long m, long n, long k, T alpha, const T* A, long lda, const T* B, long ldb,
                   T beta, T* Cm, long ldc) {
    if (m == 0 || n == 0) return;
    check(Prec<T>::gemm(context().blas, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, int(m), int(n),
                        int(k), &alpha, A, int(lda), B, int(ldb), &beta, Cm, int(ldc)),
          "cublas gemm");
}

// cuFFT plans for T: (n, batch, precision) -> (r2c/d2z, c2r/z2d)
template <class T>
inline std::pair<cufftHandle, cufftHandle> plan_t(int n, long batch) {
    return context().plan(n, batch, Prec<T>::fp64);
}

template <class T>
inline std::vector<T> cast(const std::vector<double>& v) {
    return std::vector<T>(v.begin(), v.end());
}

}  // namespace bspf::cuda::flow
