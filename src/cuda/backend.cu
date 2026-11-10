// SPDX-License-Identifier: GPL-3.0-or-later
// Device side of the FP32 CUDA backend: one stream, cuBLAS SGEMMs, batched
// cuFFT, a stream-ordered memory pool and call-reused workspaces.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "bspf/cuda.hpp"
#include "tables.hpp"

namespace bspf::cuda {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void check(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(s)));
}
void check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) throw std::runtime_error(std::string(what) + ": cuFFT status " + std::to_string(int(r)));
}

// Samples per line block: bounds every workspace to ~64 MB
// (BSPF_CUDA_CHUNK overrides it, e.g. to test the chunked paths).
long chunk_samples() {
    static const long value = [] {
        const char* env = std::getenv("BSPF_CUDA_CHUNK");
        return env ? std::max(1L, std::atol(env)) : 1L << 24;
    }();
    return value;
}

// Growable device buffers, one per role, reused across calls (stream order
// makes reuse after an async free safe).
enum Slot { kX, kY, kR, kSpec, kJets, kC, kVals, kSlots };

struct Context {
    std::recursive_mutex mutex;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    std::map<std::pair<int, long>, std::pair<cufftHandle, cufftHandle>> plans;  // (n, batch) -> (r2c, c2r)
    void* slots[kSlots] = {};
    size_t capacity[kSlots] = {};
    float* pinned = nullptr;  // host staging for uploads/downloads (page-locked)
    size_t pinned_capacity = 0;

    float* staging(size_t count) {
        if (count > pinned_capacity) {
            check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            if (pinned) cudaFreeHost(pinned);
            pinned = nullptr;
            check(cudaMallocHost(&pinned, sizeof(float) * count), "cudaMallocHost");
            pinned_capacity = count;
        }
        return pinned;
    }

    Context() {
        int count = 0;
        check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count == 0) throw std::runtime_error("no CUDA device");
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        check(cublasCreate(&blas), "cublasCreate");
        check(cublasSetStream(blas, stream), "cublasSetStream");
        check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");  // true FP32, no TF32
        // Keep freed memory in the pool instead of returning it at each sync.
        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaMemPool_t pool;
        check(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
        uint64_t threshold = UINT64_MAX;
        check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold), "cudaMemPoolSetAttribute");
    }

    void* slot(Slot s, size_t bytes) {
        if (bytes > capacity[s]) {
            if (slots[s]) cudaFreeAsync(slots[s], stream);
            const size_t grown = std::max(bytes, capacity[s] + capacity[s] / 2);
            check(cudaMallocAsync(&slots[s], grown, stream), "cudaMallocAsync (workspace)");
            capacity[s] = grown;
        }
        return slots[s];
    }
    float* floats(Slot s, long count) { return static_cast<float*>(slot(s, sizeof(float) * std::max(count, 1L))); }

    std::pair<cufftHandle, cufftHandle> plan(int n, long batch) {
        auto it = plans.find({n, batch});
        if (it != plans.end()) return it->second;
        if (plans.size() >= 64) {  // bounded: drop every plan (after pending work)
            check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            for (auto& e : plans) { cufftDestroy(e.second.first); cufftDestroy(e.second.second); }
            plans.clear();
        }
        const int h = n / 2 + 1;
        int dims[1] = {n};
        cufftHandle r2c, c2r;
        check(cufftPlanMany(&r2c, 1, dims, nullptr, 1, n, nullptr, 1, h, CUFFT_R2C, static_cast<int>(batch)), "cufftPlanMany r2c");
        check(cufftPlanMany(&c2r, 1, dims, nullptr, 1, h, nullptr, 1, n, CUFFT_C2R, static_cast<int>(batch)), "cufftPlanMany c2r");
        check(cufftSetStream(r2c, stream), "cufftSetStream");
        check(cufftSetStream(c2r, stream), "cufftSetStream");
        return plans.emplace(std::make_pair(n, batch), std::make_pair(r2c, c2r)).first->second;
    }
};

// Never destroyed: Python may release Arrays after CUDA's own teardown.
Context& context() {
    static Context* c = new Context();
    return *c;
}

using Lock = std::lock_guard<std::recursive_mutex>;

// C = alpha A B + beta C, all column-major (cuBLAS convention).
void gemm(long m, long n, long k, float alpha, const float* A, long lda, const float* B, long ldb, float beta,
          float* C, long ldc) {
    if (m == 0 || n == 0) return;
    check(cublasSgemm(context().blas, CUBLAS_OP_N, CUBLAS_OP_N, int(m), int(n), int(k), &alpha, A, int(lda), B,
                      int(ldb), &beta, C, int(ldc)),
          "cublasSgemm");
}

void axpy(long count, const float* x, float* y) {
    const float one = 1.0f;
    check(cublasSaxpy(context().blas, int(count), &one, x, 1, y, 1), "cublasSaxpy");
}

// Device copy of a host double matrix, rounded to float.
struct DeviceMatrix {
    long rows = 0, cols = 0;
    std::shared_ptr<float> data;
    const float* get() const { return data.get(); }
};

std::shared_ptr<float> device_floats(const std::vector<float>& host) {
    Context& cx = context();
    void* p = nullptr;
    check(cudaMallocAsync(&p, sizeof(float) * std::max<size_t>(host.size(), 1), cx.stream), "cudaMallocAsync");
    std::shared_ptr<float> out(static_cast<float*>(p), [](float* q) { cudaFreeAsync(q, context().stream); });
    if (!host.empty())
        check(cudaMemcpyAsync(p, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice, cx.stream), "upload");
    check(cudaStreamSynchronize(cx.stream), "cudaStreamSynchronize");  // host staging is freed next
    return out;
}

DeviceMatrix upload(const detail::HostMatrix& m) {
    std::vector<float> f(m.data.begin(), m.data.end());
    return {m.rows, m.cols, device_floats(f)};
}

constexpr int kTile = 32, kRows = 8;

// Lines (l in [l0, l0+nl), t in [t0, t0+nt)) of src (before, n, after) into
// contiguous lines dst[m * n + i], m = (l - l0) nt + (t - t0).
__global__ void gather(const float* __restrict__ src, float* __restrict__ dst, long n, long after, long l0, long t0,
                       long nt, long nl) {
    __shared__ float tile[kTile][kTile + 1];
    const long tb = long(blockIdx.x) * kTile, ib = long(blockIdx.y) * kTile;
    for (long z = blockIdx.z; z < nl; z += gridDim.z) {
        const float* s = src + (l0 + z) * n * after + t0;
        for (int k = threadIdx.y; k < kTile; k += kRows) {
            const long i = ib + k, t = tb + threadIdx.x;
            if (i < n && t < nt) tile[k][threadIdx.x] = s[i * after + t];
        }
        __syncthreads();
        float* d = dst + z * nt * n;
        for (int k = threadIdx.y; k < kTile; k += kRows) {
            const long t = tb + k, i = ib + threadIdx.x;
            if (i < n && t < nt) d[t * n + i] = tile[threadIdx.x][k];
        }
        __syncthreads();
    }
}

// Inverse of gather for P values per line, assigning or accumulating.
__global__ void scatter(const float* __restrict__ src, float* __restrict__ dst, long P, long after, long l0, long t0,
                        long nt, long nl, bool accumulate) {
    __shared__ float tile[kTile][kTile + 1];
    const long tb = long(blockIdx.x) * kTile, ib = long(blockIdx.y) * kTile;
    for (long z = blockIdx.z; z < nl; z += gridDim.z) {
        const float* s = src + z * nt * P;
        for (int k = threadIdx.y; k < kTile; k += kRows) {
            const long t = tb + k, i = ib + threadIdx.x;
            if (i < P && t < nt) tile[threadIdx.x][k] = s[t * P + i];
        }
        __syncthreads();
        float* d = dst + (l0 + z) * P * after + t0;
        for (int k = threadIdx.y; k < kTile; k += kRows) {
            const long i = ib + k, t = tb + threadIdx.x;
            if (i < P && t < nt) {
                const float v = tile[k][threadIdx.x];
                d[i * after + t] = accumulate ? d[i * after + t] + v : v;
            }
        }
        __syncthreads();
    }
}

// spec *= m on every line; the zero and (even n) Nyquist bins keep only their
// real part, as FFTW's c2r (and numpy's irfft) do.
__global__ void multiply(float2* __restrict__ spec, const float2* __restrict__ m, long h, long total, bool even) {
    for (long idx = long(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += long(gridDim.x) * blockDim.x) {
        const long j = idx % h;
        const float2 a = spec[idx], b = m[j];
        float2 out{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
        if (j == 0 || (even && j == h - 1)) out.y = 0.0f;
        spec[idx] = out;
    }
}

dim3 transpose_grid(long rows, long nt, long nl) {
    return dim3(unsigned((nt + kTile - 1) / kTile), unsigned((rows + kTile - 1) / kTile),
                unsigned(std::min<long>(nl, 65535)));
}

}  // namespace

// ---------------------------------------------------------------- the op

struct DeviceLine {
    std::shared_ptr<const void> keepalive;
    int n = 0, nb = 0, q = 0, p = 0;
    DeviceMatrix fit, fit_total, fit_jets, basis, jets_left, jets_right;
};

namespace {
// Lines are shared by every op compiled from the same host Line.
std::shared_ptr<const DeviceLine> device_line(const detail::LineTables& t) {
    static std::map<const void*, std::weak_ptr<const DeviceLine>> cache;
    auto it = cache.find(t.key);
    if (it != cache.end())
        if (auto line = it->second.lock()) return line;
    auto line = std::make_shared<DeviceLine>();
    line->keepalive = t.keepalive;
    line->n = t.n;
    line->nb = t.nb;
    line->q = t.q;
    line->p = t.p;
    line->fit = upload(t.fit);
    line->fit_total = upload(t.fit_total);
    line->fit_jets = upload(t.fit_jets);
    line->basis = upload(t.basis);
    line->jets_left = upload(t.jets_left);
    line->jets_right = upload(t.jets_right);
    cache[t.key] = line;
    return line;
}
}  // namespace

struct Op::Impl {
    std::shared_ptr<const DeviceLine> line;
    long rows = 0, cols = 0, modes = 0;  // modes: rows of E (k)
    bool jets_only = false, correction = true, pass_remainder = false;
    DeviceMatrix spline, multiplier, mode_rows, lift;  // multiplier: h complex as 2h floats

    // Y (rows, M) for M contiguous lines X (n, M) and optional jets (2q, M).
    // With x_writable, X is reused as the remainder buffer.
    void run(const float* X, bool x_writable, float* Y, long M, const float* jets) const {
        Context& cx = context();
        const DeviceLine& L = *line;
        const long n = L.n, nb = L.nb, q = L.q, p = L.p, J = 2 * q, h = n / 2 + 1;
        if (jets_only) {  // g = J f from the endpoint windows, or the given jets
            if (q && jets) {
                check(cudaMemcpyAsync(Y, jets, sizeof(float) * J * M, cudaMemcpyDeviceToDevice, cx.stream), "copy jets");
            } else if (q) {
                gemm(q, M, p, 1.0f, L.jets_left.get(), q, X, n, 0.0f, Y, J);
                gemm(q, M, p, 1.0f, L.jets_right.get(), q, X + (n - p), n, 0.0f, Y + q, J);
            }
            return;
        }
        float* c = cx.floats(kC, nb * M);
        if (jets && J) {  // c = F f + G g
            gemm(nb, M, n, 1.0f, L.fit.get(), nb, X, n, 0.0f, c, nb);
            gemm(nb, M, J, 1.0f, L.fit_jets.get(), nb, jets, J, 1.0f, c, nb);
        } else {          // c = (F + G J) f
            gemm(nb, M, n, 1.0f, L.fit_total.get(), nb, X, n, 0.0f, c, nb);
        }
        gemm(rows, M, nb, 1.0f, spline.get(), rows, c, nb, 0.0f, Y, rows);
        if (!correction) return;
        float* r = const_cast<float*>(X);
        if (!x_writable) {
            r = cx.floats(kR, n * M);
            check(cudaMemcpyAsync(r, X, sizeof(float) * n * M, cudaMemcpyDeviceToDevice, cx.stream), "copy samples");
        }
        gemm(n, M, nb, -1.0f, L.basis.get(), n, c, nb, 1.0f, r, n);  // r = f - B c
        if (pass_remainder) {
            axpy(n * M, r, Y);
            return;
        }
        const auto plans = cx.plan(int(n), M);
        float* spec = cx.floats(kSpec, 2 * h * M);
        check(cufftExecR2C(plans.first, r, reinterpret_cast<cufftComplex*>(spec)), "cufftExecR2C");
        if (modes) {
            if (lift.data) {
                float* values = cx.floats(kVals, modes * M);
                gemm(modes, M, 2 * h, 1.0f, mode_rows.get(), modes, spec, 2 * h, 0.0f, values, modes);
                gemm(rows, M, modes, 1.0f, lift.get(), rows, values, modes, 1.0f, Y, rows);
            } else {
                gemm(modes, M, 2 * h, 1.0f, mode_rows.get(), modes, spec, 2 * h, 1.0f, Y, rows);
            }
        }
        if (multiplier.data) {
            const long total = h * M;
            const int threads = 256;
            const int blocks = int(std::min<long>((total + threads - 1) / threads, 65535L * 8));
            multiply<<<blocks, threads, 0, cx.stream>>>(reinterpret_cast<float2*>(spec),
                                                        reinterpret_cast<const float2*>(multiplier.get()), h, total,
                                                        n % 2 == 0);
            check(cudaGetLastError(), "multiply kernel");
            check(cufftExecC2R(plans.second, reinterpret_cast<cufftComplex*>(spec), r), "cufftExecC2R");
            axpy(n * M, r, Y);
        }
    }
};

namespace detail {

std::shared_ptr<const Op::Impl> make_op(const OpTables& t) {
    Lock lock(context().mutex);
    auto op = std::make_shared<Op::Impl>();
    op->line = device_line(t.line);
    op->rows = t.rows;
    op->cols = t.cols;
    op->jets_only = t.jets_only;
    op->correction = t.correction;
    op->pass_remainder = t.pass_remainder;
    if (!t.jets_only) op->spline = upload(t.spline);
    if (!t.multiplier.empty()) {
        std::vector<float> m;
        for (const auto& z : t.multiplier) { m.push_back(float(z.real())); m.push_back(float(z.imag())); }
        op->multiplier = {long(t.multiplier.size()), 1, device_floats(m)};
    }
    if (t.modes.rows) {
        op->modes = t.modes.rows;
        op->mode_rows = upload(t.modes);
    }
    if (t.lift.rows) op->lift = upload(t.lift);
    return op;
}

}  // namespace detail

std::ptrdiff_t Op::rows() const { return impl_->rows; }
std::ptrdiff_t Op::cols() const { return impl_->cols; }
std::ptrdiff_t Op::jet_count() const { return 2 * impl_->line->q; }

void Op::apply_along(const Array& in, Array& out, int axis, const Array* jets, bool accumulate) const {
    const Impl& op = *impl_;
    const int ndim = int(in.shape().size());
    if (axis < 0 || axis >= ndim) throw std::invalid_argument("axis out of range");
    const long n = op.cols, P = op.rows, J = jet_count();
    if (in.shape()[axis] != n)
        throw std::invalid_argument("expected " + std::to_string(n) + " samples along axis " + std::to_string(axis) +
                                    ", got " + std::to_string(in.shape()[axis]));
    Shape expected = in.shape();
    expected[axis] = P;
    if (out.shape() != expected) throw std::invalid_argument("output has the wrong shape");
    if (jets) {
        Shape js = in.shape();
        js[axis] = J;
        if (jets->shape() != js) throw std::invalid_argument("jets must have the field's shape with the axis of length 2q");
    }
    if (in.size() && out.data() == in.data()) throw std::invalid_argument("output must not alias the input");
    long before = 1, after = 1;
    for (int i = 0; i < axis; ++i) before *= in.shape()[i];
    for (int i = axis + 1; i < ndim; ++i) after *= in.shape()[i];
    if (before == 0 || after == 0) return;

    Lock lock(context().mutex);
    Context& cx = context();
    const long width = std::max({n, P, J, 1L});
    const long cap = std::max(1L, chunk_samples() / width);  // lines per block

    if (after == 1) {  // innermost axis: lines are already contiguous
        for (long m0 = 0; m0 < before; m0 += cap) {
            const long M = std::min(cap, before - m0);
            float* dest = out.data() + m0 * P;
            float* Y = accumulate ? cx.floats(kY, P * M) : dest;
            op.run(in.data() + m0 * n, false, Y, M, jets ? jets->data() + m0 * J : nullptr);
            if (accumulate) axpy(P * M, Y, dest);
        }
        return;
    }
    // Interleaved lines: gather blocks of (slabs x t-range) by tiled transpose.
    auto block = [&](long l0, long nl, long t0, long nt) {
        const long M = nl * nt;
        float* X = cx.floats(kX, n * M);
        gather<<<transpose_grid(n, nt, nl), dim3(kTile, kRows), 0, cx.stream>>>(in.data(), X, n, after, l0, t0, nt, nl);
        check(cudaGetLastError(), "gather kernel");
        const float* G = nullptr;
        if (jets && J) {
            float* g = cx.floats(kJets, J * M);
            gather<<<transpose_grid(J, nt, nl), dim3(kTile, kRows), 0, cx.stream>>>(jets->data(), g, J, after, l0, t0, nt, nl);
            check(cudaGetLastError(), "gather kernel");
            G = g;
        }
        float* Y = cx.floats(kY, P * M);
        op.run(X, true, Y, M, G);
        scatter<<<transpose_grid(P, nt, nl), dim3(kTile, kRows), 0, cx.stream>>>(Y, out.data(), P, after, l0, t0, nt, nl,
                                                                                 accumulate);
        check(cudaGetLastError(), "scatter kernel");
    };
    if (after <= cap) {
        const long slabs = cap / after;
        for (long l0 = 0; l0 < before; l0 += slabs) block(l0, std::min(slabs, before - l0), 0, after);
    } else {
        for (long l0 = 0; l0 < before; ++l0)
            for (long t0 = 0; t0 < after; t0 += cap) block(l0, 1, t0, std::min(cap, after - t0));
    }
}

// ---------------------------------------------------------------- arrays

bool available() {
    try {
        context();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string device_name() {
    context();
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    return prop.name;
}

void synchronize() {
    Context& cx = context();
    check(cudaStreamSynchronize(cx.stream), "cudaStreamSynchronize");
}

std::ptrdiff_t Array::size() const {
    std::ptrdiff_t s = 1;
    for (auto d : shape_) s *= d;
    return s;
}

Array::Array(Shape shape) : shape_(std::move(shape)) {
    for (auto d : shape_)
        if (d < 0) throw std::invalid_argument("negative dimension");
    Context& cx = context();
    void* p = nullptr;
    check(cudaMallocAsync(&p, sizeof(real) * std::max<std::ptrdiff_t>(size(), 1), cx.stream), "cudaMallocAsync");
    data_.reset(static_cast<real*>(p), [](real* q) { cudaFreeAsync(q, context().stream); });
}

Array Array::upload(const double* data, Shape shape) {
    Array a(std::move(shape));
    Lock lock(context().mutex);
    Context& cx = context();
    const size_t count = static_cast<size_t>(a.size());
    float* host = cx.staging(count);
    std::copy(data, data + count, host);  // rounds to float32
    check(cudaMemcpyAsync(a.data(), host, sizeof(float) * count, cudaMemcpyHostToDevice, cx.stream), "upload");
    check(cudaStreamSynchronize(cx.stream), "cudaStreamSynchronize");  // staging is reused next
    return a;
}

void Array::download(double* out) const {
    Lock lock(context().mutex);
    Context& cx = context();
    const size_t count = static_cast<size_t>(size());
    float* host = cx.staging(count);
    check(cudaMemcpyAsync(host, data(), sizeof(float) * count, cudaMemcpyDeviceToHost, cx.stream), "download");
    check(cudaStreamSynchronize(cx.stream), "cudaStreamSynchronize");
    std::copy(host, host + count, out);
}

Array Array::clone() const {
    Array a(shape_);
    check(cudaMemcpyAsync(a.data(), data(), sizeof(real) * size(), cudaMemcpyDeviceToDevice, context().stream), "clone");
    return a;
}

Array Array::reshaped(Shape shape) const {
    Array a;
    a.shape_ = std::move(shape);
    if (a.size() != size()) throw std::invalid_argument("reshape changes the size");
    a.data_ = data_;
    return a;
}

}  // namespace bspf::cuda
