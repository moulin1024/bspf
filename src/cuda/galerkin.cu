// SPDX-License-Identifier: GPL-3.0-or-later
// GalerkinLine's fast evaluation/projection on the device, in FP32 or FP64.
#include "flow_context.cuh"
#include "galerkin_device.hpp"
#include "precision.cuh"

namespace bspf::cuda::flow {

namespace {

// spec_out[(col Q + q) h + f] = spec[col h + f] * shift[q h + f]; DC / even-N Nyquist kept real.
template <class Cx>
__global__ void shift_kernel(const Cx* spec, const Cx* shift, long m, int Q, int h, bool nyquist, Cx* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= m * Q * h) return;
    const int f = int(idx % h);
    const long cq = idx / h;
    const int q = int(cq % Q);
    const long col = cq / Q;
    Cx v = cmul(spec[col * h + f], shift[q * h + f]);
    if (f == 0 || (nyquist && f == h - 1)) v.y = 0;
    out[idx] = v;
}

// values[col P + rows[q n + j]] = V[(col Q + q) n + j]
template <class T>
__global__ void scatter_kernel(const T* V, const int* rows, long m, int Q, int n, long P, T* values) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= m * Q * n) return;
    const int j = int(idx % n);
    const long cq = idx / n;
    const int q = int(cq % Q);
    const long col = cq / Q;
    const int r = rows[q * n + j];
    if (r >= 0) values[col * P + r] = V[idx];
}

// V[(col Q + q) n + j] = w[q n + j] values[col P + rows[q n + j]] (0 where no row)
template <class T>
__global__ void gather_kernel(const T* values, const int* rows, const T* w, long m, int Q, int n, long P, T* V) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= m * Q * n) return;
    const int j = int(idx % n);
    const long cq = idx / n;
    const int q = int(cq % Q);
    const long col = cq / Q;
    const int r = rows[q * n + j];
    V[idx] = r >= 0 ? w[q * n + j] * values[col * P + r] : T(0);
}

// A[col h + f] = sum_q conj(shift[q h + f]) S[(col Q + q) h + f]; DC / even-N Nyquist kept real.
template <class Cx>
__global__ void adjoint_kernel(const Cx* S, const Cx* shift, long m, int Q, int h, bool nyquist, Cx* A) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= m * h) return;
    const int f = int(idx % h);
    const long col = idx / h;
    Cx sum;
    sum.x = 0;
    sum.y = 0;
    for (int q = 0; q < Q; ++q) {
        const Cx t = cconj_mul(shift[q * h + f], S[(col * Q + q) * h + f]);
        sum.x += t.x;
        sum.y += t.y;
    }
    if (f == 0 || (nyquist && f == h - 1)) sum.y = 0;
    A[idx] = sum;
}

// out[col ne + e] = w[e] values[col P + P - ne + e]
template <class T>
__global__ void extra_weights_kernel(const T* values, const T* w, long m, int ne, long P, T* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= m * ne) return;
    const int e = int(idx % ne);
    const long col = idx / ne;
    out[idx] = w[e] * values[col * P + P - ne + e];
}

template <class T>
std::vector<T> nonempty(const std::vector<double>& v) {
    return v.empty() ? std::vector<T>(1, T(0)) : cast<T>(v);
}

}  // namespace

template <class T>
struct AxisDevice {
    using Cx = typename Prec<T>::C;
    int n = 0, nb = 0, Q = 0, h = 0, ne = 0;
    long P = 0;
    Buffer<T> F, B;             // (nb, n), (n, nb)
    Buffer<T> S[2];             // stacked (Q n, nb): rows (q, j)
    Buffer<Cx> shift[2];        // (Q, h)
    Buffer<int> rows;           // (Q, n)
    Buffer<T> w;                // (Q, n)
    Buffer<T> E[2], G[2], we;   // extras (ne, n), (ne, nb), (ne)
};

template <class T>
std::shared_ptr<AxisDevice<T>> make_axis(const AxisTables& t) {
    using Cx = typename Prec<T>::C;
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    auto a = std::make_shared<AxisDevice<T>>();
    AxisDevice<T>& d = *a;
    d.n = t.n;
    d.nb = t.nb;
    d.Q = t.Q;
    d.h = t.n / 2 + 1;
    d.P = t.P;
    d.ne = t.ne;
    d.F.upload(cast<T>(t.F));
    d.B.upload(cast<T>(t.B));
    for (int k = 0; k < 2; ++k) {
        d.S[k].upload(cast<T>(t.S[k]));
        std::vector<Cx> sh(size_t(d.Q) * d.h);
        for (size_t i = 0; i < sh.size(); ++i) {
            sh[i].x = T(t.shift[k][2 * i]);
            sh[i].y = T(t.shift[k][2 * i + 1]);
        }
        d.shift[k].upload(sh);
        d.E[k].upload(nonempty<T>(t.E[k]));
        d.G[k].upload(nonempty<T>(t.G[k]));
    }
    d.rows.upload(t.rows);
    d.w.upload(cast<T>(t.w));
    d.we.upload(nonempty<T>(t.we));
    check(cudaStreamSynchronize(context().stream), "cudaStreamSynchronize");
    return a;
}

template <class T>
void axis_evaluate(const AxisDevice<T>& d, const T* X, long m, int k, T* values) {
    using Cx = typename Prec<T>::C;
    Context& ctx = context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    const int n = d.n, nb = d.nb, Q = d.Q, h = d.h;
    Buffer<T> C(size_t(nb) * m), R(size_t(n) * m), V(size_t(n) * Q * m);
    Buffer<Cx> spec(size_t(h) * m), shifted(size_t(h) * Q * m);
    gemm_t<T>(false, false, nb, m, n, 1, d.F.ptr, nb, X, n, 0, C.ptr, nb);                     // c = F u
    check(cudaMemcpyAsync(R.ptr, X, sizeof(T) * n * m, cudaMemcpyDeviceToDevice, ctx.stream), "copy");
    gemm_t<T>(false, false, n, m, nb, -1, d.B.ptr, n, C.ptr, nb, 1, R.ptr, n);                 // r = u - B c
    check(Prec<T>::forward(plan_t<T>(n, m).first, R.ptr, spec.ptr), "cufft forward");
    shift_kernel<<<blocks(long(h) * Q * m), 256, 0, ctx.stream>>>(spec.ptr, d.shift[k].ptr, m, Q, h, n % 2 == 0,
                                                                  shifted.ptr);
    check(Prec<T>::inverse(plan_t<T>(n, long(Q) * m).second, shifted.ptr, V.ptr), "cufft inverse");
    const T one = 1;  // V_q += S_kq c for every offset: V_q is (n, m) with leading dimension Q n
    check(Prec<T>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, int(m), nb, &one, d.S[k].ptr, Q * n, long(n),
                                C.ptr, nb, 0, &one, V.ptr, Q * n, long(n), Q),
          "gemm strided batched");
    scatter_kernel<<<blocks(long(n) * Q * m), 256, 0, ctx.stream>>>(V.ptr, d.rows.ptr, m, Q, n, d.P, values);
    if (d.ne > 0) {
        T* tail = values + (d.P - d.ne);
        gemm_t<T>(false, false, d.ne, m, n, 1, d.E[k].ptr, d.ne, R.ptr, n, 0, tail, d.P);
        gemm_t<T>(false, false, d.ne, m, nb, 1, d.G[k].ptr, d.ne, C.ptr, nb, 1, tail, d.P);
    }
    check(cudaGetLastError(), "evaluate kernels");
}

template <class T>
void axis_project(const AxisDevice<T>& d, const T* values, long m, int k, T* X) {
    using Cx = typename Prec<T>::C;
    Context& ctx = context();
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);
    const int n = d.n, nb = d.nb, Q = d.Q, h = d.h;
    Buffer<T> V(size_t(n) * Q * m), Cs(size_t(nb) * m), t(size_t(n) * m);
    Buffer<Cx> spec(size_t(h) * Q * m), adj(size_t(h) * m);
    gather_kernel<<<blocks(long(n) * Q * m), 256, 0, ctx.stream>>>(values, d.rows.ptr, d.w.ptr, m, Q, n, d.P, V.ptr);
    gemm_t<T>(true, false, nb, m, long(Q) * n, 1, d.S[k].ptr, long(Q) * n, V.ptr, long(Q) * n, 0, Cs.ptr, nb);
    check(Prec<T>::forward(plan_t<T>(n, long(Q) * m).first, V.ptr, spec.ptr), "cufft forward");
    adjoint_kernel<<<blocks(long(h) * m), 256, 0, ctx.stream>>>(spec.ptr, d.shift[k].ptr, m, Q, h, n % 2 == 0, adj.ptr);
    check(Prec<T>::inverse(plan_t<T>(n, m).second, adj.ptr, t.ptr), "cufft inverse");
    if (d.ne > 0) {
        Buffer<T> ve(size_t(d.ne) * m);
        extra_weights_kernel<<<blocks(long(d.ne) * m), 256, 0, ctx.stream>>>(values, d.we.ptr, m, d.ne, d.P, ve.ptr);
        gemm_t<T>(true, false, n, m, d.ne, 1, d.E[k].ptr, d.ne, ve.ptr, d.ne, 1, t.ptr, n);
        gemm_t<T>(true, false, nb, m, d.ne, 1, d.G[k].ptr, d.ne, ve.ptr, d.ne, 1, Cs.ptr, nb);
    }
    gemm_t<T>(true, false, nb, m, n, -1, d.B.ptr, n, t.ptr, n, 1, Cs.ptr, nb);   // spline -= B^T t
    check(cudaMemcpyAsync(X, t.ptr, sizeof(T) * n * m, cudaMemcpyDeviceToDevice, ctx.stream), "copy");
    gemm_t<T>(true, false, n, m, nb, 1, d.F.ptr, nb, Cs.ptr, nb, 1, X, n);        // + F^T spline
    check(cudaGetLastError(), "project kernels");
}

template <class T>
std::vector<double> axis_evaluate_host(const AxisDevice<T>& a, const std::vector<double>& X, long m, int k) {
    Buffer<T> dx, dv{size_t(a.P) * size_t(m)};
    dx.upload(cast<T>(X));
    axis_evaluate(a, dx.ptr, m, k, dv.ptr);
    const std::vector<T> v = dv.download(size_t(a.P) * m);
    return std::vector<double>(v.begin(), v.end());
}

template <class T>
std::vector<double> axis_project_host(const AxisDevice<T>& a, const std::vector<double>& values, long m, int k) {
    Buffer<T> dv, dx{size_t(a.n) * size_t(m)};
    dv.upload(cast<T>(values));
    axis_project(a, dv.ptr, m, k, dx.ptr);
    const std::vector<T> x = dx.download(size_t(a.n) * m);
    return std::vector<double>(x.begin(), x.end());
}

void* flow_stream() { return context().stream; }

#define BSPF_AXIS(T)                                                                                      \
    template struct AxisDevice<T>;                                                                        \
    template std::shared_ptr<AxisDevice<T>> make_axis<T>(const AxisTables&);                              \
    template void axis_evaluate<T>(const AxisDevice<T>&, const T*, long, int, T*);                        \
    template void axis_project<T>(const AxisDevice<T>&, const T*, long, int, T*);                         \
    template std::vector<double> axis_evaluate_host<T>(const AxisDevice<T>&, const std::vector<double>&, long, int); \
    template std::vector<double> axis_project_host<T>(const AxisDevice<T>&, const std::vector<double>&, long, int);
BSPF_AXIS(float)
BSPF_AXIS(double)
#undef BSPF_AXIS

}  // namespace bspf::cuda::flow
