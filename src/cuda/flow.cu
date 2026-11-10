// SPDX-License-Identifier: GPL-3.0-or-later
// Device compressible solver (FP32): Galerkin BSPF weak form with the fast
// spline + cuFFT evaluation/projection of galerkin.cu, weak face fluxes, free-
// stream correction, per-axis mass solve and SVV (cuBLAS), local-time-step RK4
// and Anderson acceleration. Mirrors CompressibleSolver2D (compressible.cpp).
#include <algorithm>
#include <map>
#include <chrono>
#include <cmath>

#include "flow_context.cuh"
#include "flow_device.hpp"
#include "precision.cuh"

namespace bspf::cuda::flow {

namespace {

template <class T>
struct DevFace {
    int kind = kSlipWall, P = 0;
    T p0 = 0, T0 = 0, back = 0;
    bool has_back = false;
    Buffer<T> nx, ny, area, ext, ext_nodes;
};

template <class T>
__device__ inline bool primitive(const T* q, T g, T& u, T& v, T& p, T& c) {
    u = q[1] / q[0];
    v = q[2] / q[0];
    p = (g - 1) * (q[3] - T(0.5) * (q[1] * u + q[2] * v));
    if (!(q[0] > 0) || !(p > 0)) return false;
    c = sqrt(g * p / q[0]);
    return true;
}

// normal flux F . n (n need not be unit) and |u.n| + c |n|
template <class T>
__device__ inline T normal_flux(const T* q, T nx, T ny, T g, T* F, bool& ok) {
    T u, v, p, c;
    ok = primitive(q, g, u, v, p, c) && ok;
    const T un = u * nx + v * ny;
    F[0] = q[0] * un;
    F[1] = q[1] * un + p * nx;
    F[2] = q[2] * un + p * ny;
    F[3] = (q[3] + p) * un;
    return fabs(un) + c * sqrt(nx * nx + ny * ny);
}

template <class T>
__device__ inline void conservative(T rho, T u, T v, T p, T g, T* q) {
    q[0] = rho;
    q[1] = rho * u;
    q[2] = rho * v;
    q[3] = p / (g - 1) + T(0.5) * rho * (u * u + v * v);
}

// batch of column-major (R, C) matrices -> (C, R)
template <class T>
__global__ void transpose_kernel(const T* in, int R, int C, int batch, T* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= long(R) * C * batch) return;
    const int r = int(idx % R);
    const long t = idx / R;
    const int c = int(t % C);
    const int b = int(t / C);
    out[(long(b) * R + r) * C + c] = in[idx];
}

// contravariant Euler fluxes at the Gauss points; Uq, Fh, Gh planar [c][b][a]
template <class T>
__global__ void flux_kernel(const T* Uq, const T* x_xi, const T* x_eta, const T* y_xi,
                            const T* y_eta, long PQ, T g, T* Fh, T* Gh, int* flag) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= PQ) return;
    T q[4] = {Uq[m], Uq[PQ + m], Uq[2 * PQ + m], Uq[3 * PQ + m]};
    T u, v, p, c;
    if (!primitive(q, g, u, v, p, c)) {
        atomicExch(flag, 1);
        return;
    }
    const T Ux = y_eta[m] * u - x_eta[m] * v, Ue = -y_xi[m] * u + x_xi[m] * v, H = q[3] + p;
    Fh[m] = q[0] * Ux;
    Fh[PQ + m] = q[1] * Ux + y_eta[m] * p;
    Fh[2 * PQ + m] = q[2] * Ux - x_eta[m] * p;
    Fh[3 * PQ + m] = H * Ux;
    Gh[m] = q[0] * Ue;
    Gh[PQ + m] = q[1] * Ue - y_xi[m] * p;
    Gh[2 * PQ + m] = q[2] * Ue + x_xi[m] * p;
    Gh[3 * PQ + m] = H * Ue;
}

// face traces into (P, 4) column-major: from src with element (p, c) at src[c * cstride + off + p]
template <class T>
__global__ void trace_kernel(const T* src, long cstride, long off, int P, T* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * P) return;
    const int c = idx / P, p = idx % P;
    out[idx] = src[c * cstride + off + p];
}

// eta-face nodal lines: L[(f 4 + c) n0 + i] = U[(c n0 + i) n1 + (f ? n1 - 1 : 0)]
template <class T>
__global__ void eta_lines_kernel(const T* U, int n0, int n1, T* L) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 8 * n0) return;
    const int i = idx % n0, fc = idx / n0, c = fc % 4, f = fc / 4;
    L[idx] = U[(long(c) * n0 + i) * n1 + (f ? n1 - 1 : 0)];
}

// outlet: sum of pressure and count over supersonic points (one block)
template <class T>
__global__ void outlet_kernel(const T* Uf, const T* nx, const T* ny, int P, T g, T* out) {
    __shared__ T sp[256];
    __shared__ T sc[256];
    T s = 0, n = 0;
    for (int p = threadIdx.x; p < P; p += blockDim.x) {
        const T q[4] = {Uf[p], Uf[P + p], Uf[2 * P + p], Uf[3 * P + p]};
        T u, v, pr, c;
        if (primitive(q, g, u, v, pr, c) && u * nx[p] + v * ny[p] >= c) {
            s += pr;
            n += 1;
        }
    }
    sp[threadIdx.x] = s;
    sc[threadIdx.x] = n;
    __syncthreads();
    for (int w = blockDim.x / 2; w > 0; w /= 2) {
        if (threadIdx.x < w) {
            sp[threadIdx.x] += sp[threadIdx.x + w];
            sc[threadIdx.x] += sc[threadIdx.x + w];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = sp[0];
        out[1] = sc[0];
    }
}

// numerical flux minus nothing: star (P, 4) = area * F*(exterior, interior)
template <class T>
__global__ void star_kernel(const T* Uf, const T* nx, const T* ny, const T* area, const T* ext,
                            int P, int kind, int low, T p0, T T0, T back, int has_back,
                            const T* outlet, T g, T* star, int* flag) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;
    const T in[4] = {Uf[p], Uf[P + p], Uf[2 * P + p], Uf[3 * P + p]};
    const T n_x = nx[p], n_y = ny[p];
    T u, v, pr, c;
    if (!primitive(in, g, u, v, pr, c)) {
        atomicExch(flag, 1);
        return;
    }
    T e[4] = {in[0], in[1], in[2], in[3]};
    bool interior = false;
    if (kind == kInlet) {
        const T Rm = u * n_x + v * n_y - 2 * c / (g - 1);
        const T A = (g + 1) / (g - 1), B = 2 * Rm, C = T(0.5) * (g - 1) * Rm * Rm - T0;
        const T cc = (-B + sqrt(fmax(B * B - 4 * A * C, T(0.0)))) / (2 * A);
        const T Tc = cc * cc, un = Rm + 2 * cc / (g - 1);
        const T pe = p0 * pow(Tc / T0, g / (g - 1));
        conservative(g * pe / Tc, un * n_x, un * n_y, pe, g, e);
    } else if (kind == kOutlet) {
        if (u * n_x + v * n_y >= c) {
            interior = true;
        } else {
            T pb = back;
            if (outlet[1] > 0) pb = outlet[0] / outlet[1];
            else if (!has_back) atomicExch(flag, 2);
            conservative(in[0], u, v, pb, g, e);
        }
    } else if (kind == kSlipWall) {
        const T mn = in[1] * n_x + in[2] * n_y;
        e[1] = in[1] - 2 * mn * n_x;
        e[2] = in[2] - 2 * mn * n_y;
    } else if (kind == kNoSlipWall) {
        e[1] = -in[1];
        e[2] = -in[2];
    } else {
        for (int k = 0; k < 4; ++k) e[k] = ext[k * P + p];
    }
    T F[4];
    bool ok = true;
    if (interior) {
        normal_flux(in, n_x, n_y, g, F, ok);
    } else {
        T FL[4], FR[4];
        const T* L = low ? e : in;
        const T* R = low ? in : e;
        const T a = fmax(normal_flux(L, n_x, n_y, g, FL, ok), normal_flux(R, n_x, n_y, g, FR, ok));
        for (int k = 0; k < 4; ++k) F[k] = T(0.5) * (FL[k] + FR[k]) - T(0.5) * a * (R[k] - L[k]);
    }
    if (!ok) atomicExch(flag, 1);
    for (int k = 0; k < 4; ++k) star[k * P + p] = area[p] * F[k];
}

// R at the face nodes += sign * proj (count, 4) column-major
template <class T>
__global__ void face_add_kernel(const T* proj, int face, int n0, int n1, T sign, T* R) {
    const int count = face < 2 ? n1 : n0;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 4 * count) return;
    const int c = idx / count, t = idx % count;
    long node;
    if (face == 0) node = t;
    else if (face == 1) node = long(n0 - 1) * n1 + t;
    else if (face == 2) node = long(t) * n1;
    else node = long(t) * n1 + n1 - 1;
    R[long(c) * n0 * n1 + node] += sign * proj[idx];
}

// R -= gcl0 F + gcl1 G with the nodal physical fluxes
template <class T>
__global__ void gcl_kernel(const T* U, const T* gcl0, const T* gcl1, long n, T g, T* R, int* flag) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const T q[4] = {U[m], U[n + m], U[2 * n + m], U[3 * n + m]};
    T u, v, p, c;
    if (!primitive(q, g, u, v, p, c)) {
        atomicExch(flag, 1);
        return;
    }
    const T F[4] = {q[1], q[1] * u + p, q[1] * v, u * (q[3] + p)};
    const T G[4] = {q[2], q[2] * u, q[2] * v + p, v * (q[3] + p)};
    for (int k = 0; k < 4; ++k) R[k * n + m] -= gcl0[m] * F[k] + gcl1[m] * G[k];
}

__device__ inline void atomic_max_positive(float* address, float value) {
    atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));  // valid for non-negative floats
}
__device__ inline void atomic_max_positive(double* address, double value) {
    atomicMax(reinterpret_cast<long long*>(address), __double_as_longlong(value));  // non-negative doubles
}

// wave-rate maxima along xi and eta -> rates[0], rates[1]
template <class T>
__global__ void rates_kernel(const T* U, const T* xi_x, const T* xi_y, const T* eta_x,
                             const T* eta_y, const T* grad_xi, const T* grad_eta, long n, T g,
                             T* rates) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const T q[4] = {U[m], U[n + m], U[2 * n + m], U[3 * n + m]};
    T u, v, p, c;
    if (!primitive(q, g, u, v, p, c)) return;
    atomic_max_positive(&rates[0], fabs(u * xi_x[m] + v * xi_y[m]) + c * grad_xi[m]);
    atomic_max_positive(&rates[1], fabs(u * eta_x[m] + v * eta_y[m]) + c * grad_eta[m]);
}

// out = out / J + source - sx QaU - sy QbU, with sx = svv radius_a rates[0], sy likewise
template <class T>
__global__ void finish_kernel(T* out, const T* J, const T* source, const T* QaU, const T* QbU,
                              const T* rates, T fa, T fb, long n) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= 4 * n) return;
    const long m = idx % n;
    T r = out[idx] / J[m];
    if (source) r += source[idx];
    if (QaU) r -= fa * rates[0] * QaU[idx] + fb * rates[1] * QbU[idx];
    out[idx] = r;
}

template <class T>
__global__ void dt_kernel(const T* U, const T* xi_x, const T* xi_y, const T* eta_x, const T* eta_y,
                          const T* grad_xi, const T* grad_eta, long n, T g, T ra, T rb,
                          const T* rates, T fa, T fb, T cfl, T eta_scale, T* dt) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const T q[4] = {U[m], U[n + m], U[2 * n + m], U[3 * n + m]};
    T u, v, p, c;
    if (!primitive(q, g, u, v, p, c)) {
        dt[m] = 0;
        return;
    }
    const T lx = fabs(u * xi_x[m] + v * xi_y[m]) + c * grad_xi[m];
    const T le = fabs(u * eta_x[m] + v * eta_y[m]) + c * grad_eta[m];
    T rate = (ra * lx + eta_scale * rb * le) / T(2.83);
    if (rates) rate += (fa * rates[0] + eta_scale * fb * rates[1]) / T(2.78);
    dt[m] = cfl / rate;
}

// W = U + c dt k (dt per node)
template <class T>
__global__ void stage_kernel(const T* U, const T* k, const T* dt, T c, long n, T* W) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= 4 * n) return;
    W[idx] = U[idx] + c * dt[idx % n] * k[idx];
}

template <class T>
__global__ void rk4_kernel(const T* U, const T* k1, const T* k2, const T* k3, const T* k4,
                           const T* dt, long n, T* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= 4 * n) return;
    out[idx] = U[idx] + dt[idx % n] / T(6.0) * (k1[idx] + 2 * k2[idx] + 2 * k3[idx] + k4[idx]);
}

template <class T>
__global__ void positive_kernel(const T* U, long n, T g, int* flag) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const T q[4] = {U[m], U[n + m], U[2 * n + m], U[3 * n + m]};
    T u, v, p, c;
    if (!primitive(q, g, u, v, p, c)) atomicExch(flag, 1);
}

// ---- line-implicit preconditioner along eta lines
// flux Jacobians dGh/dU at the nodes (central differences in T), Bj[m 16 + r 4 + c]
template <class T>
__global__ void jacobian_kernel(const T* U, const T* J, const T* eta_x, const T* eta_y, long n,
                                T g, T* Bj) {
    const long m = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (m >= n) return;
    const T q[4] = {U[m], U[n + m], U[2 * n + m], U[3 * n + m]};
    const T nx = J[m] * eta_x[m], ny = J[m] * eta_y[m];  // Gh = -y_xi F + x_xi G
    for (int c = 0; c < 4; ++c) {
        T a[4] = {q[0], q[1], q[2], q[3]}, b[4] = {q[0], q[1], q[2], q[3]};
        const T h = T(1e-3) * fmax(T(1.0), fabs(q[c]));
        a[c] += h;
        b[c] -= h;
        T Fa[4], Fb[4];
        bool ok = true;
        normal_flux(a, nx, ny, g, Fa, ok);
        normal_flux(b, nx, ny, g, Fb, ok);
        for (int r = 0; r < 4; ++r) Bj[m * 16 + r * 4 + c] = (Fa[r] - Fb[r]) / (2 * h);
    }
}

template <class T>
__device__ inline void wall_jump(const T* in, T nx, T ny, T area, int noslip, int low, T g,
                                 const T* fixed, T* out) {
    T e[4] = {in[0], in[1], in[2], in[3]};
    if (fixed) {
        for (int k = 0; k < 4; ++k) e[k] = fixed[k];
    } else if (noslip) {
        e[1] = -in[1];
        e[2] = -in[2];
    } else {
        const T mn = in[1] * nx + in[2] * ny;
        e[1] = in[1] - 2 * mn * nx;
        e[2] = in[2] - 2 * mn * ny;
    }
    bool ok = true;
    T FL[4], FR[4];
    const T* L = low ? e : in;
    const T* R = low ? in : e;
    const T a = fmax(normal_flux(L, nx, ny, g, FL, ok), normal_flux(R, nx, ny, g, FR, ok));
    for (int k = 0; k < 4; ++k) out[k] = area * (T(0.5) * (FL[k] + FR[k]) - T(0.5) * a * (R[k] - L[k]));  // star only
}

// wall-penalty Jacobians d(own - star)/dU at the two wall nodes of every eta line, Jw[(i 2 + f) 16 + r 4 + c]
template <class T>
__global__ void wall_jacobian_kernel(const T* U, const T* J, const T* eta_x, const T* eta_y, int n0,
                                     int n1, T g, int kind_low, int kind_high, const T* ext_low,
                                     const T* ext_high, T* Jw) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 2 * n0) return;
    const int i = idx / 2, f = idx % 2;
    const int kind = f ? kind_high : kind_low;
    T* out = Jw + size_t(idx) * 16;
    T fixed[4];
    const T* fx = nullptr;
    if (kind == kExterior) {  // fixed exterior state at this face node, (n0, 4) column-major
        const T* e = f ? ext_high : ext_low;
        for (int k = 0; k < 4; ++k) fixed[k] = e[k * n0 + i];
        fx = fixed;
    }
    if (kind != kSlipWall && kind != kNoSlipWall && kind != kExterior) {
        for (int k = 0; k < 16; ++k) out[k] = 0;
        return;
    }
    const long nn = long(n0) * n1, m = long(i) * n1 + (f ? n1 - 1 : 0);
    const T q[4] = {U[m], U[nn + m], U[2 * nn + m], U[3 * nn + m]};
    T nx = J[m] * eta_x[m], ny = J[m] * eta_y[m];
    const T area = sqrt(nx * nx + ny * ny);
    nx /= area;
    ny /= area;
    for (int c = 0; c < 4; ++c) {
        T a[4] = {q[0], q[1], q[2], q[3]}, b[4] = {q[0], q[1], q[2], q[3]};
        const T h = T(1e-3) * fmax(T(1.0), fabs(q[c]));
        a[c] += h;
        b[c] -= h;
        T ja[4], jb[4];
        wall_jump(a, nx, ny, area, kind == kNoSlipWall, !f, g, fx, ja);
        wall_jump(b, nx, ny, area, kind == kNoSlipWall, !f, g, fx, jb);
        for (int r = 0; r < 4; ++r) out[r * 4 + c] = (ja[r] - jb[r]) / (2 * h);
    }
}

// w_b dGh/dU at the eta Gauss points of every line from Y1 ([c][i][b], Pb per line):
// coef[(i 16 + r 4 + c) Pb + b]
template <class T>
__global__ void line_coef_kernel(const T* Y1, const T* nxq, const T* nyq, const T* w, int n0, int Pb,
                                 T g, T* coef) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= long(n0) * Pb) return;
    const int i = int(idx / Pb), b = int(idx % Pb);
    const long cs = long(n0) * Pb;  // component stride of Y1
    const T q[4] = {Y1[long(i) * Pb + b], Y1[cs + long(i) * Pb + b], Y1[2 * cs + long(i) * Pb + b],
                        Y1[3 * cs + long(i) * Pb + b]};
    const T nx = nxq[idx], ny = nyq[idx];
    for (int c = 0; c < 4; ++c) {
        T a[4] = {q[0], q[1], q[2], q[3]}, bq[4] = {q[0], q[1], q[2], q[3]};
        const T h = T(1e-3) * fmax(T(1.0), fabs(q[c]));
        a[c] += h;
        bq[c] -= h;
        T Fa[4], Fb[4];
        bool ok = true;
        normal_flux(a, nx, ny, g, Fa, ok);
        normal_flux(bq, nx, ny, g, Fb, ok);
        for (int r = 0; r < 4; ++r) coef[(long(i) * 16 + r * 4 + c) * Pb + b] = w[b] * (Fa[r] - Fb[r]) / (2 * h);
    }
}

// P_i = I - diag(dt) A_i with the weak-form eta operator MK[(i 16 + rc)][j + l n1] = (Mb^{-1} K_rc)(j, l):
// A = (1/J) MK - nu_svv Q x I4 + face blocks (+ Mb^{-1} e_0 dstar_0/dU, - Mb^{-1} e_N dstar_N/dU) / J
template <class T>
__global__ void weak_assemble_kernel(const T* MK, const T* Minv, const T* Q, const T* Jw,
                                     const T* J, const T* dt, const T* rates, T fb, int n0, int n1,
                                     T* P) {
    const int nb = 4 * n1;
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= long(n0) * nb * nb) return;
    const int col = int(idx % nb);
    const long t = idx / nb;
    const int row = int(t % nb), i = int(t / nb);
    const int j = row / 4, r = row % 4, l = col / 4, c = col % 4;
    const long m = long(i) * n1 + j;
    const T invJ = T(1.0) / J[m];
    const long nn1 = long(n1) * n1;
    T a = invJ * MK[(long(i) * 16 + r * 4 + c) * nn1 + j + long(l) * n1];
    if (r == c && rates) a -= fb * rates[1] * Q[j + l * n1];
    if (l == 0) a += Minv[j] * invJ * Jw[(size_t(i) * 2 + 0) * 16 + r * 4 + c];
    if (l == n1 - 1) a -= Minv[j + long(n1 - 1) * n1] * invJ * Jw[(size_t(i) * 2 + 1) * 16 + r * 4 + c];
    P[size_t(i) * nb * nb + row + size_t(col) * nb] = (row == col ? T(1.0) : T(0.0)) - dt[m] * a;
}

// P_i = I - diag(dt) A_i, A_i = -diag(1/J)(D x I4) diag(B) - nu_svv Q x I4 + wall blocks; column-major (nb, nb)
template <class T>
__global__ void assemble_kernel(const T* D, const T* Q, const T* s0, const T* s1, const T* Bj,
                                const T* Jw, const T* J, const T* dt, const T* rates, T fb,
                                int n0, int n1, T* P) {
    const int nb = 4 * n1;
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= long(n0) * nb * nb) return;
    const int col = int(idx % nb);
    const long t = idx / nb;
    const int row = int(t % nb), i = int(t / nb);
    const int j = row / 4, r = row % 4, l = col / 4, c = col % 4;
    const long m = long(i) * n1 + j, ml = long(i) * n1 + l;
    const T invJ = T(1.0) / J[m];
    T a = -invJ * D[j + l * n1] * Bj[ml * 16 + r * 4 + c];
    if (r == c && rates) a -= fb * rates[1] * Q[j + l * n1];
    if (l == 0) a -= s0[j] * invJ * Jw[(size_t(i) * 2 + 0) * 16 + r * 4 + c];
    if (l == n1 - 1) a += s1[j] * invJ * Jw[(size_t(i) * 2 + 1) * 16 + r * 4 + c];
    P[size_t(i) * nb * nb + row + size_t(col) * nb] = (row == col ? T(1.0) : T(0.0)) - dt[m] * a;
}

// planar k [c][i][j] <-> line vectors [i][j][c]
template <class T>
__global__ void to_lines_kernel(const T* k, long nn, int n1, T* lines) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= 4 * nn) return;
    const long m = idx / 4;
    const int c = int(idx % 4);
    lines[idx] = k[c * nn + m];
}
template <class T>
__global__ void from_lines_kernel(const T* lines, long nn, int n1, T* k) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (idx >= 4 * nn) return;
    const long m = idx / 4;
    const int c = int(idx % 4);
    k[c * nn + m] = lines[idx];
}

template <class T>
__global__ void pointer_kernel(T* base, long stride, int count, T** out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = base + size_t(i) * stride;
}

}  // namespace

template <class T>
struct FlowDevice {
    int n0 = 0, n1 = 0;
    long nn = 0;
    T gamma = T(1.4), svv = 0, cfl = T(0.6);
    std::shared_ptr<AxisDevice<T>> ax[2];
    long P[2] = {0, 0};
    Buffer<T> Minv[2], Q[2];
    T radius[2] = {0, 0};
    Buffer<T> J, xi_x, xi_y, eta_x, eta_y, grad_xi, grad_eta;
    Buffer<T> qx_xi, qx_eta, qy_xi, qy_eta, gcl0, gcl1, source;
    bool has_source = false;
    const T* external_source = nullptr;  // device forcing set by the multigrid cycle (overrides `source`)
    DevFace<T> face[4];
    Buffer<int> flag;
    Buffer<T> scalars;  // rates[2], outlet sums per face [2 + 2 k]
    // workspaces
    Buffer<T> Y1, Y1T, Uq, Fh, Gh, Z1, Z2, Z1T, Z2T, R2, Leta, Yeta, Uf, star, proj, Tw, QaU, QbU;

    void reset_flag() { check(cudaMemsetAsync(flag.ptr, 0, sizeof(int), context().stream), "memset"); }
    int read_flag() {
        int h = 0;
        check(cudaMemcpyAsync(&h, flag.ptr, sizeof(int), cudaMemcpyDeviceToHost, context().stream), "flag");
        check(cudaStreamSynchronize(context().stream), "sync");
        return h;
    }

    // dU/dt (planar) of planar U, into out; returns the error flag (0 ok, 1 nonpositive, 2 back pressure)
    void rhs(const T* U, T* out) {
        Context& ctx = context();
        cudaStream_t st = ctx.stream;
        const int Pa = int(P[0]), Pb = int(P[1]);
        const long PQ = long(Pa) * Pb;
        // 1-3: U -> Gauss points: eta lines, transpose, xi lines
        axis_evaluate(*ax[1], U, 4L * n0, 0, Y1.ptr);                                   // (Pb, 4 n0) [c][i][b]
        transpose_kernel<T><<<blocks(4L * Pb * n0), 256, 0, st>>>(Y1.ptr, Pb, n0, 4, Y1T.ptr);  // (n0, 4 Pb) [c][b][i]
        axis_evaluate(*ax[0], Y1T.ptr, 4L * Pb, 0, Uq.ptr);                             // (Pa, 4 Pb) [c][b][a]
        // 4: fluxes
        flux_kernel<T><<<blocks(PQ), 256, 0, st>>>(Uq.ptr, qx_xi.ptr, qx_eta.ptr, qy_xi.ptr, qy_eta.ptr, PQ, gamma,
                                                Fh.ptr, Gh.ptr, flag.ptr);
        // 5-7: projection int d_xi phi Fh + d_eta phi Gh
        axis_project(*ax[0], Fh.ptr, 4L * Pb, 1, Z1.ptr);                               // (n0, 4 Pb)
        axis_project(*ax[0], Gh.ptr, 4L * Pb, 0, Z2.ptr);
        transpose_kernel<T><<<blocks(4L * n0 * Pb), 256, 0, st>>>(Z1.ptr, n0, Pb, 4, Z1T.ptr);  // (Pb, 4 n0)
        transpose_kernel<T><<<blocks(4L * n0 * Pb), 256, 0, st>>>(Z2.ptr, n0, Pb, 4, Z2T.ptr);
        axis_project(*ax[1], Z1T.ptr, 4L * n0, 0, Tw.ptr);                               // (n1, 4 n0) planar
        axis_project(*ax[1], Z2T.ptr, 4L * n0, 1, R2.ptr);
        const T one = T(1.0);
        check(Prec<T>::axpy(ctx.blas, int(4 * nn), &one, R2.ptr, Tw.ptr), "saxpy");
        // 8: faces
        eta_lines_kernel<T><<<blocks(8L * n0), 256, 0, st>>>(U, n0, n1, Leta.ptr);
        axis_evaluate(*ax[0], Leta.ptr, 8, 0, Yeta.ptr);                                // (Pa, 8)
        for (int k = 0; k < 4; ++k) {
            DevFace<T>& f = face[k];
            const int Pf = f.P;
            if (k < 2) trace_kernel<T><<<blocks(4L * Pf), 256, 0, st>>>(Y1.ptr, long(n0) * Pb, long(k == 0 ? 0 : n0 - 1) * Pb, Pf, Uf.ptr);
            else trace_kernel<T><<<blocks(4L * Pf), 256, 0, st>>>(Yeta.ptr, Pa, long(k - 2) * 4 * Pa, Pf, Uf.ptr);
            T* outlet = scalars.ptr + 2 + 2 * k;
            if (f.kind == kOutlet) outlet_kernel<T><<<1, 256, 0, st>>>(Uf.ptr, f.nx.ptr, f.ny.ptr, Pf, gamma, outlet);
            star_kernel<T><<<blocks(Pf), 256, 0, st>>>(Uf.ptr, f.nx.ptr, f.ny.ptr, f.area.ptr, f.ext.ptr, Pf, f.kind,
                                                     k % 2 == 0, f.p0, f.T0, f.back, f.has_back, outlet, gamma,
                                                     star.ptr, flag.ptr);
            axis_project(*ax[k < 2 ? 1 : 0], star.ptr, 4, 0, proj.ptr);
            const int count = k < 2 ? n1 : n0;
            face_add_kernel<T><<<blocks(4L * count), 256, 0, st>>>(proj.ptr, k, n0, n1, k % 2 == 0 ? T(1.0) : -T(1.0), Tw.ptr);
        }
        // 9: free-stream correction
        gcl_kernel<T><<<blocks(nn), 256, 0, st>>>(U, gcl0.ptr, gcl1.ptr, nn, gamma, Tw.ptr, flag.ptr);
        // 10: mass solve, eta then xi (per component: X_c (n1, n0) -> X_c Minv_a^t)
        gemm_t<T>(false, false, n1, 4L * n0, n1, T(1.0), Minv[1].ptr, n1, Tw.ptr, n1, T(0.0), R2.ptr, n1);
        const T zero = T(0.0);
        check(Prec<T>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, n1, n0, n0, &one, R2.ptr, n1, nn,
                                        Minv[0].ptr, n0, 0, &zero, out, n1, nn, 4),
              "mass solve");
        // 11: / J, source, SVV
        const T* qa = nullptr;
        const T* qb = nullptr;
        if (svv > 0) {
            check(cudaMemsetAsync(scalars.ptr, 0, 2 * sizeof(T), st), "memset");
            rates_kernel<T><<<blocks(nn), 256, 0, st>>>(U, xi_x.ptr, xi_y.ptr, eta_x.ptr, eta_y.ptr, grad_xi.ptr,
                                                     grad_eta.ptr, nn, gamma, scalars.ptr);
            gemm_t<T>(false, false, n1, 4L * n0, n1, T(1.0), Q[1].ptr, n1, U, n1, T(0.0), QbU.ptr, n1);
            check(Prec<T>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, n1, n0, n0, &one, U, n1, nn,
                                            Q[0].ptr, n0, 0, &zero, QaU.ptr, n1, nn, 4),
                  "svv");
            qa = QaU.ptr;
            qb = QbU.ptr;
        }
        const T* src = external_source ? external_source : (has_source ? source.ptr : nullptr);
        finish_kernel<T><<<blocks(4 * nn), 256, 0, st>>>(out, J.ptr, src, qa, qb,
                                                      scalars.ptr, svv * radius[0], svv * radius[1], nn);
        check(cudaGetLastError(), "rhs kernels");
    }

    void local_dt(const T* U, T* dt) {
        cudaStream_t st = context().stream;
        if (svv > 0) {
            check(cudaMemsetAsync(scalars.ptr, 0, 2 * sizeof(T), st), "memset");
            rates_kernel<T><<<blocks(nn), 256, 0, st>>>(U, xi_x.ptr, xi_y.ptr, eta_x.ptr, eta_y.ptr, grad_xi.ptr,
                                                     grad_eta.ptr, nn, gamma, scalars.ptr);
        }
        dt_kernel<T><<<blocks(nn), 256, 0, st>>>(U, xi_x.ptr, xi_y.ptr, eta_x.ptr, eta_y.ptr, grad_xi.ptr, grad_eta.ptr,
                                              nn, gamma, radius[0], radius[1], svv > 0 ? scalars.ptr : nullptr,
                                              svv * radius[0], svv * radius[1], cfl, full ? T(0.0) : T(1.0), dt);
    }

    Buffer<T> dt, k1, k2, k3, k4, W;
    // line-implicit preconditioner
    bool full = false;
    int refresh = 1;
    long age = 0;  // steps since the last factorization
    int kind_low = kSlipWall, kind_high = kSlipWall;
    Buffer<T> D_eta, s0_eta, s1_eta, Bj, Jw, Pm, lines;
    Buffer<T> lineZ, linew, linenx, lineny, coef, Kl, MKl;
    Buffer<int> pivots, infos;
    Buffer<T*> ptrP, ptrL;

    void factor(const T* U) {
        Context& ctx = context();
        cudaStream_t st = ctx.stream;
        const int nb = 4 * n1;
        const int Pb = int(P[1]);
        const long m = 16L * n0, nn1 = long(n1) * n1;
        axis_evaluate(*ax[1], U, 4L * n0, 0, Y1.ptr);  // U at the eta Gauss points of every line
        line_coef_kernel<T><<<blocks(long(n0) * Pb), 256, 0, st>>>(Y1.ptr, linenx.ptr, lineny.ptr, linew.ptr, n0, Pb,
                                                               gamma, coef.ptr);
        // K (n1 n1, 16 n0) = Z^t coef: every line's 16 blocks P1b^t diag(w A_rc) P0b at once
        gemm_t<T>(true, false, nn1, m, Pb, T(1.0), lineZ.ptr, Pb, coef.ptr, Pb, T(0.0), Kl.ptr, nn1);
        const T one = T(1.0), zero = T(0.0);
        check(Prec<T>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, n1, n1, n1, &one, Minv[1].ptr, n1, 0,
                                        Kl.ptr, n1, nn1, &zero, MKl.ptr, n1, nn1, int(m)),
              "Mb^{-1} K");
        wall_jacobian_kernel<T><<<blocks(2L * n0), 256, 0, st>>>(U, J.ptr, eta_x.ptr, eta_y.ptr, n0, n1, gamma, kind_low,
                                                              kind_high, face[2].ext_nodes.ptr, face[3].ext_nodes.ptr,
                                                              Jw.ptr);
        weak_assemble_kernel<T><<<blocks(long(n0) * nb * nb), 256, 0, st>>>(MKl.ptr, Minv[1].ptr, Q[1].ptr, Jw.ptr, J.ptr,
                                                                         dt.ptr, svv > 0 ? scalars.ptr : nullptr,
                                                                         svv * radius[1], n0, n1, Pm.ptr);
        check(Prec<T>::getrf(ctx.blas, nb, ptrP.ptr, nb, pivots.ptr, infos.ptr, n0), "cublasSgetrfBatched");
    }

    void precondition(T* k) {
        Context& ctx = context();
        cudaStream_t st = ctx.stream;
        const int nb = 4 * n1;
        to_lines_kernel<T><<<blocks(4 * nn), 256, 0, st>>>(k, nn, n1, lines.ptr);
        int info = 0;
        check(Prec<T>::getrs(ctx.blas, nb, ptrP.ptr, nb, pivots.ptr, ptrL.ptr, nb, &info, n0),
              "cublasSgetrsBatched");
        if (info != 0) throw std::runtime_error("cublasSgetrsBatched: invalid argument");
        from_lines_kernel<T><<<blocks(4 * nn), 256, 0, st>>>(lines.ptr, nn, n1, k);
    }

    void step(const T* U, T* out) {
        cudaStream_t st = context().stream;
        const long nd = 4 * nn;
        if (!full || age % refresh == 0) {
            local_dt(U, dt.ptr);  // frozen with the factorization in the line-implicit case
            if (full) factor(U);
        }
        ++age;
        auto eval = [&](const T* state, T* k) {
            rhs(state, k);
            if (full) precondition(k);
        };
        eval(U, k1.ptr);
        stage_kernel<T><<<blocks(nd), 256, 0, st>>>(U, k1.ptr, dt.ptr, T(0.5), nn, W.ptr);
        eval(W.ptr, k2.ptr);
        stage_kernel<T><<<blocks(nd), 256, 0, st>>>(U, k2.ptr, dt.ptr, T(0.5), nn, W.ptr);
        eval(W.ptr, k3.ptr);
        stage_kernel<T><<<blocks(nd), 256, 0, st>>>(U, k3.ptr, dt.ptr, T(1.0), nn, W.ptr);
        eval(W.ptr, k4.ptr);
        rk4_kernel<T><<<blocks(nd), 256, 0, st>>>(U, k1.ptr, k2.ptr, k3.ptr, k4.ptr, dt.ptr, nn, out);
    }

    double residual(const T* U) {
        rhs(U, k1.ptr);
        T norm = 0;
        check(Prec<T>::nrm2(context().blas, int(4 * nn), k1.ptr, &norm), "snrm2");
        return double(norm) / std::sqrt(double(4 * nn));
    }
};

template <class T>
std::shared_ptr<FlowDevice<T>> make_flow(const FlowDeviceTables& t) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    auto f = std::make_shared<FlowDevice<T>>();
    FlowDevice<T>& d = *f;
    d.n0 = t.n0;
    d.n1 = t.n1;
    d.nn = long(t.n0) * t.n1;
    d.gamma = t.gamma;
    d.svv = t.svv;
    d.cfl = t.cfl;
    for (int a = 0; a < 2; ++a) {
        d.ax[a] = make_axis<T>(t.axis[a]);
        d.P[a] = t.axis[a].P;
        d.Minv[a].upload(cast<T>(t.Minv[a]));
        d.Q[a].upload(cast<T>(t.Q[a]));
        d.radius[a] = t.radius[a];
    }
    d.J.upload(cast<T>(t.J));
    d.xi_x.upload(cast<T>(t.xi_x));
    d.xi_y.upload(cast<T>(t.xi_y));
    d.eta_x.upload(cast<T>(t.eta_x));
    d.eta_y.upload(cast<T>(t.eta_y));
    d.grad_xi.upload(cast<T>(t.grad_xi));
    d.grad_eta.upload(cast<T>(t.grad_eta));
    d.qx_xi.upload(cast<T>(t.qx_xi));
    d.qx_eta.upload(cast<T>(t.qx_eta));
    d.qy_xi.upload(cast<T>(t.qy_xi));
    d.qy_eta.upload(cast<T>(t.qy_eta));
    d.gcl0.upload(cast<T>(t.gcl[0]));
    d.gcl1.upload(cast<T>(t.gcl[1]));
    d.has_source = !t.source.empty();
    d.source.upload(cast<T>(d.has_source ? t.source : std::vector<double>(1, 0.0)));
    for (int k = 0; k < 4; ++k) {
        const FaceTables& ft = t.face[k];
        DevFace<T>& df = d.face[k];
        df.kind = ft.kind;
        df.P = int(ft.nx.size());
        df.p0 = ft.total_pressure;
        df.T0 = ft.total_temperature;
        df.back = ft.back_pressure;
        df.has_back = ft.has_back_pressure;
        df.nx.upload(cast<T>(ft.nx));
        df.ny.upload(cast<T>(ft.ny));
        df.area.upload(cast<T>(ft.area));
        df.ext.upload(cast<T>(ft.exterior.empty() ? std::vector<double>(1, 0.0) : ft.exterior));
        df.ext_nodes.upload(cast<T>(ft.exterior_nodes.empty() ? std::vector<double>(1, 0.0) : ft.exterior_nodes));
    }
    d.flag.upload(std::vector<int>(1, 0));
    d.scalars.upload(std::vector<T>(16, T(0.0)));
    const long Pa = d.P[0], Pb = d.P[1], nn = d.nn;
    d.Y1.resize(size_t(Pb) * 4 * d.n0);
    d.Y1T.resize(size_t(Pb) * 4 * d.n0);
    d.Uq.resize(size_t(Pa) * Pb * 4);
    d.Fh.resize(size_t(Pa) * Pb * 4);
    d.Gh.resize(size_t(Pa) * Pb * 4);
    d.Z1.resize(size_t(d.n0) * 4 * Pb);
    d.Z2.resize(size_t(d.n0) * 4 * Pb);
    d.Z1T.resize(size_t(d.n0) * 4 * Pb);
    d.Z2T.resize(size_t(d.n0) * 4 * Pb);
    d.R2.resize(size_t(4 * nn));
    d.Tw.resize(size_t(4 * nn));
    d.Leta.resize(size_t(8) * d.n0);
    d.Yeta.resize(size_t(8) * Pa);
    const long Pmax = std::max(Pa, Pb);
    d.Uf.resize(size_t(4 * Pmax));
    d.star.resize(size_t(4 * Pmax));
    d.proj.resize(size_t(4) * std::max(d.n0, d.n1));
    d.QaU.resize(size_t(4 * nn));
    d.QbU.resize(size_t(4 * nn));
    for (Buffer<T>* b : {&d.dt, &d.k1, &d.k2, &d.k3, &d.k4, &d.W}) b->resize(size_t(4 * nn));
    d.full = t.line_implicit;
    d.refresh = std::max(1, t.refresh);
    d.kind_low = t.face[2].kind;
    d.kind_high = t.face[3].kind;
    if (d.full) {
        const long nb = 4L * d.n1;
        d.D_eta.upload(cast<T>(t.D_eta));
        d.lineZ.upload(cast<T>(t.line_Z));
        d.linew.upload(cast<T>(t.line_w));
        d.linenx.upload(cast<T>(t.line_nx));
        d.lineny.upload(cast<T>(t.line_ny));
        d.coef.resize(size_t(16) * d.n0 * d.P[1]);
        d.Kl.resize(size_t(16) * d.n0 * d.n1 * d.n1);
        d.MKl.resize(size_t(16) * d.n0 * d.n1 * d.n1);
        d.s0_eta.upload(cast<T>(t.s0_eta));
        d.s1_eta.upload(cast<T>(t.s1_eta));
        d.Bj.resize(size_t(16 * nn));
        d.Jw.resize(size_t(32) * d.n0);
        d.Pm.resize(size_t(d.n0) * nb * nb);
        d.lines.resize(size_t(4 * nn));
        d.pivots.resize(size_t(d.n0) * nb);
        d.infos.resize(size_t(d.n0));
        d.ptrP.resize(size_t(d.n0));
        d.ptrL.resize(size_t(d.n0));
        pointer_kernel<T><<<blocks(d.n0), 256, 0, context().stream>>>(d.Pm.ptr, nb * nb, d.n0, d.ptrP.ptr);
        pointer_kernel<T><<<blocks(d.n0), 256, 0, context().stream>>>(d.lines.ptr, nb, d.n0, d.ptrL.ptr);
    }
    check(cudaStreamSynchronize(context().stream), "sync");
    return f;
}

namespace {
// C-ordered (n0, n1, 4) <-> planar [c][i][j]
template <class T>
std::vector<T> to_planar(const std::vector<double>& U, long nn) {
    std::vector<T> out(U.size());
    for (long m = 0; m < nn; ++m)
        for (int c = 0; c < 4; ++c) out[size_t(c * nn + m)] = U[size_t(4 * m + c)];
    return out;
}
template <class T>
std::vector<double> from_planar(const std::vector<T>& P, long nn) {
    std::vector<double> out(P.size());
    for (long m = 0; m < nn; ++m)
        for (int c = 0; c < 4; ++c) out[size_t(4 * m + c)] = P[size_t(c * nn + m)];
    return out;
}
void raise(int code) {
    if (code == 1) throw std::domain_error("CompressibleSolver2D (cuda): nonpositive density or pressure");
    if (code == 2) throw std::invalid_argument("CompressibleSolver2D (cuda): subsonic exit needs back_pressure");
}
}  // namespace

template <class T>
std::vector<double> flow_rhs_host(FlowDevice<T>& f, const std::vector<double>& U) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    Buffer<T> dU, dR{size_t(4 * f.nn)};
    dU.upload(to_planar<T>(U, f.nn));
    f.reset_flag();
    f.rhs(dU.ptr, dR.ptr);
    raise(f.read_flag());
    return from_planar(dR.download(size_t(4 * f.nn)), f.nn);
}

template <class T>
std::vector<double> flow_step_host(FlowDevice<T>& f, const std::vector<double>& U) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    Buffer<T> dU, dO{size_t(4 * f.nn)};
    dU.upload(to_planar<T>(U, f.nn));
    f.reset_flag();
    f.step(dU.ptr, dO.ptr);
    raise(f.read_flag());
    return from_planar(dO.download(size_t(4 * f.nn)), f.nn);
}

template <class T>
SolveReport flow_solve_host(FlowDevice<T>& f, std::vector<double>& Uhost, const SolveSettings& s) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    Context& ctx = context();
    cudaStream_t st = ctx.stream;
    const long nd = 4 * f.nn;
    const auto start = std::chrono::steady_clock::now();
    SolveReport rep;
    const size_t sz = size_t(nd);
    Buffer<T> U, G{sz}, Gprev{sz}, fcur{sz}, fprev{sz}, next{sz};
    U.upload(to_planar<T>(Uhost, f.nn));
    const int depth = std::max(s.anderson, 0);
    Buffer<T> dF{sz * size_t(std::max(depth, 1))}, dG{sz * size_t(std::max(depth, 1))};
    Buffer<T> gram{size_t(depth * depth + depth + 1)}, gamma{size_t(std::max(depth, 1))};
    int count = 0, slot = 0;
    bool have_prev = false, mixed = false;
    double best = 1e300;
    const int check_every = std::max(s.check, 1);
    const T one = T(1.0), minus = -T(1.0), zero = T(0.0);
    auto restart = [&] {
        count = 0;
        slot = 0;
        have_prev = false;
        ++rep.restarts;
    };
    auto copy = [&](T* dst, const T* src) {
        check(cudaMemcpyAsync(dst, src, sizeof(T) * nd, cudaMemcpyDeviceToDevice, st), "copy");
    };
    for (int it = 0;; ++it) {
        if (it % check_every == 0 || it == s.max_steps) {
            f.reset_flag();
            double r = f.residual(U.ptr);
            int code = f.read_flag();
            if (code && mixed) {
                copy(U.ptr, Gprev.ptr);
                restart();
                mixed = false;
                f.reset_flag();
                r = f.residual(U.ptr);
                code = f.read_flag();
            }
            raise(code);
            ++rep.rhs_evaluations;
            rep.history.emplace_back(it, r);
            rep.steps = it;
            rep.residual = r;
            if (s.progress) s.progress(it, r);
            if (!std::isfinite(r)) throw std::runtime_error("CompressibleSolver2D (cuda): residual not finite");
            if (r < s.tolerance) {
                rep.converged = true;
                break;
            }
            if (depth > 0 && r > 100 * best && have_prev) {
                copy(U.ptr, Gprev.ptr);
                restart();
            }
            best = std::min(best, r);
            if (it == s.max_steps) break;
        }
        f.reset_flag();
        f.step(U.ptr, G.ptr);
        int code = f.read_flag();
        if (code && mixed) {
            copy(U.ptr, Gprev.ptr);
            restart();
            mixed = false;
            f.reset_flag();
            f.step(U.ptr, G.ptr);
            code = f.read_flag();
        }
        raise(code);
        rep.rhs_evaluations += 4;
        mixed = false;
        if (depth == 0) {
            copy(Gprev.ptr, G.ptr);
            std::swap(U.ptr, G.ptr);
            continue;
        }
        // f = G - U
        copy(fcur.ptr, G.ptr);
        check(Prec<T>::axpy(ctx.blas, int(nd), &minus, U.ptr, fcur.ptr), "saxpy");
        if (have_prev) {
            T* dFs = dF.ptr + size_t(slot) * nd;
            T* dGs = dG.ptr + size_t(slot) * nd;
            copy(dFs, fcur.ptr);
            check(Prec<T>::axpy(ctx.blas, int(nd), &minus, fprev.ptr, dFs), "saxpy");
            copy(dGs, G.ptr);
            check(Prec<T>::axpy(ctx.blas, int(nd), &minus, Gprev.ptr, dGs), "saxpy");
            slot = (slot + 1) % depth;
            count = std::min(count + 1, depth);
        }
        copy(fprev.ptr, fcur.ptr);
        copy(Gprev.ptr, G.ptr);
        have_prev = true;
        if (count == 0) {
            copy(U.ptr, G.ptr);
            continue;
        }
        // least squares min |f - dF gamma|: normal equations, solved in double on the host
        gemm_t<T>(true, false, count, count, nd, T(1.0), dF.ptr, nd, dF.ptr, nd, T(0.0), gram.ptr, count);
        check(Prec<T>::gemv(ctx.blas, CUBLAS_OP_T, int(nd), count, &one, dF.ptr, int(nd), fcur.ptr, &zero, gram.ptr + count * count),
              "sgemv");
        std::vector<T> h = gram.download(size_t(count * count + count));
        std::vector<double> A(size_t(count) * size_t(count)), b(static_cast<size_t>(count));
        double trace = 0;
        for (int i = 0; i < count; ++i) trace += h[size_t(i * count + i)];
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) A[size_t(i * count + j)] = h[size_t(j * count + i)];
            A[size_t(i * count + i)] += 1e-7 * trace / count;  // FP32 Gram: mild Tikhonov
            b[size_t(i)] = h[size_t(count * count + i)];
        }
        for (int c = 0; c < count; ++c) {  // Gaussian elimination with partial pivoting
            int piv = c;
            for (int r = c + 1; r < count; ++r)
                if (std::abs(A[size_t(r * count + c)]) > std::abs(A[size_t(piv * count + c)])) piv = r;
            for (int j = 0; j < count; ++j) std::swap(A[size_t(c * count + j)], A[size_t(piv * count + j)]);
            std::swap(b[size_t(c)], b[size_t(piv)]);
            for (int r = c + 1; r < count; ++r) {
                const double l = A[size_t(r * count + c)] / A[size_t(c * count + c)];
                for (int j = c; j < count; ++j) A[size_t(r * count + j)] -= l * A[size_t(c * count + j)];
                b[size_t(r)] -= l * b[size_t(c)];
            }
        }
        std::vector<T> gm(static_cast<size_t>(count));
        for (int c = count - 1; c >= 0; --c) {
            double v = b[size_t(c)];
            for (int j = c + 1; j < count; ++j) v -= A[size_t(c * count + j)] * gm[size_t(j)];
            gm[size_t(c)] = T(v / A[size_t(c * count + c)]);
        }
        check(cudaMemcpyAsync(gamma.ptr, gm.data(), sizeof(T) * count, cudaMemcpyHostToDevice, st), "gamma");
        copy(next.ptr, G.ptr);
        check(Prec<T>::gemv(ctx.blas, CUBLAS_OP_N, int(nd), count, &minus, dG.ptr, int(nd), gamma.ptr, &one, next.ptr),
              "sgemv");
        f.reset_flag();
        positive_kernel<T><<<blocks(f.nn), 256, 0, st>>>(next.ptr, f.nn, f.gamma, f.flag.ptr);
        if (f.read_flag()) {
            copy(U.ptr, G.ptr);
            restart();
        } else {
            copy(U.ptr, next.ptr);
            mixed = true;
        }
    }
    Uhost = from_planar(U.download(size_t(nd)), f.nn);
    rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return rep;
}

template <class T>
void flow_set_source(FlowDevice<T>& f, const std::vector<double>& source) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    if (source.empty()) {
        f.has_source = false;
        return;
    }
    if (long(source.size()) != 4 * f.nn) throw std::invalid_argument("source must have shape (n0, n1, 4)");
    f.source.upload(to_planar<T>(source, f.nn));
    f.has_source = true;
}

#define BSPF_FLOW(T)                                                                                   \
    template struct FlowDevice<T>;                                                                     \
    template std::shared_ptr<FlowDevice<T>> make_flow<T>(const FlowDeviceTables&);                     \
    template std::vector<double> flow_rhs_host<T>(FlowDevice<T>&, const std::vector<double>&);         \
    template std::vector<double> flow_step_host<T>(FlowDevice<T>&, const std::vector<double>&);        \
    template SolveReport flow_solve_host<T>(FlowDevice<T>&, std::vector<double>&, const SolveSettings&); \
    template void flow_set_source<T>(FlowDevice<T>&, const std::vector<double>&);
BSPF_FLOW(float)
BSPF_FLOW(double)
#undef BSPF_FLOW

// ---------------------------------------------------------------- device multigrid (FAS) on FP32 levels
namespace {
// out[c][i][j] = in[c][f0 i][f1 j]   (planar, coarse (m0, m1) from fine (n0, n1))
__global__ void inject_kernel(const float* in, int n0, int n1, int m0, int m1, int f0, int f1, float* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    const long mm = long(m0) * m1;
    if (idx >= 4 * mm) return;
    const int c = int(idx / mm);
    const long r = idx % mm;
    const int i = int(r / m1), j = int(r % m1);
    out[idx] = in[long(c) * n0 * n1 + long(f0 * i) * n1 + f1 * j];
}
__device__ inline int fw_terms(int i, int m, int n, int f, int* idx, float* w) {
    if (f == 1) { idx[0] = i; w[0] = 1; return 1; }
    if (i == 0) { idx[0] = 0; idx[1] = 1; w[0] = w[1] = 0.5f; return 2; }
    if (i == m - 1) { idx[0] = n - 1; idx[1] = n - 2; w[0] = w[1] = 0.5f; return 2; }
    idx[0] = 2 * i - 1; idx[1] = 2 * i; idx[2] = 2 * i + 1; w[0] = 0.25f; w[1] = 0.5f; w[2] = 0.25f; return 3;
}
// full-weighting restriction of planar (n0, n1) onto (m0, m1), factors per axis
__global__ void restrict_kernel(const float* in, int n0, int n1, int m0, int m1, int f0, int f1, float* out) {
    const long idx = blockIdx.x * long(blockDim.x) + threadIdx.x;
    const long mm = long(m0) * m1;
    if (idx >= 4 * mm) return;
    const int c = int(idx / mm);
    const long r = idx % mm;
    const int i = int(r / m1), j = int(r % m1);
    int ia[3], jb[3];
    float wa[3], wb[3];
    const int na = fw_terms(i, m0, n0, f0, ia, wa), nb = fw_terms(j, m1, n1, f1, jb, wb);
    float s = 0;
    for (int a = 0; a < na; ++a)
        for (int b = 0; b < nb; ++b) s += wa[a] * wb[b] * in[long(c) * n0 * n1 + long(ia[a]) * n1 + jb[b]];
    out[idx] = s;
}
__global__ void cast_kernel_df(const double* in, long n, float* out) {
    const long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = float(in[i]);
}
// out = -rp - sigma z  (z in double)
__global__ void forcing_kernel(const float* rp, const double* z, double sigma, long n, float* out) {
    const long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = float(-double(rp[i]) - sigma * z[i]);
}
// out = (W - U0) / sigma in double
__global__ void correction_kernel(const float* W, const float* U0, double sigma, long n, double* out) {
    const long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = (double(W[i]) - double(U0[i])) / sigma;
}
__global__ void add_kernel(const double* U, const float* dW, double scale, long n, double* out) {  // out = U + scale dW
    const long i = blockIdx.x * long(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = U[i] + scale * double(dW[i]);
}
}  // namespace

struct DeviceMG {
    std::vector<std::shared_ptr<FlowDevice<float>>> L;
    std::vector<std::array<int, 2>> factor;  // between l and l + 1
    std::vector<Buffer<float>> Pa, Pb;       // prolongation l + 1 -> l per axis, (n_l, n_{l+1}) column-major
    std::vector<int> identity;               // prolongation is the identity (p-level: same nodes)
    int pre = 10, post = 10, coarse = 60, cycle_index = 1, initial = 50;
    double damping = 0.5;
    int damping_cuts = 0;
    std::vector<Buffer<float>> W, Wt, F, Rr, Uc0, Rc, dU;  // per-level work
    long size(int l) const { return 4 * L[size_t(l)]->nn; }

    void sync_check(FlowDevice<float>& f) {
        const int code = f.read_flag();
        if (code) throw std::domain_error("device multigrid: nonpositive density or pressure");
    }
    // Smoothing replays a CUDA graph of two steps (W -> Wt -> W) per (level, forcing pointer): a step is
    // ~300 small launches, which otherwise cost ~2 ms per step on the coarse levels whatever their size.
    // The state always ends in W[l] (no pointer swaps, so captured addresses stay valid).
    std::map<std::pair<int, const float*>, cudaGraphExec_t> graphs;
    bool use_graphs = true;

    void smooth(int l, int steps) {
        FlowDevice<float>& f = *L[size_t(l)];
        cudaStream_t st = context().stream;
        float* w = W[size_t(l)].ptr;
        float* wt = Wt[size_t(l)].ptr;
        auto pair = [&] {
            f.step(w, wt);
            f.step(wt, w);
        };
        int left = steps;
        if (use_graphs && left >= 2) {
            const auto key = std::make_pair(l, f.external_source);
            auto it = graphs.find(key);
            if (it == graphs.end()) {
                pair();  // warm-up: FFT plans and pooled buffers exist before the capture
                left -= 2;
                cudaGraph_t graph;
                check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
                pair();
                check(cudaStreamEndCapture(st, &graph), "cudaStreamEndCapture");
                cudaGraphExec_t exec;
                check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
                cudaGraphDestroy(graph);
                it = graphs.emplace(key, exec).first;
            }
            for (; left >= 2; left -= 2) check(cudaGraphLaunch(it->second, st), "cudaGraphLaunch");
        }
        for (; left >= 2; left -= 2) pair();
        if (left == 1) {
            f.step(w, wt);
            check(cudaMemcpyAsync(w, wt, sizeof(float) * size_t(4 * f.nn), cudaMemcpyDeviceToDevice, st), "copy");
        }
    }
    // FAS cycle solving R_l(W_l) + f_l = 0 (f_l: device pointer or null)
    void cycle(int l, const float* fl) {
        Context& ctx = context();
        cudaStream_t st = ctx.stream;
        FlowDevice<float>& f = *L[size_t(l)];
        f.external_source = fl;
        f.reset_flag();
        if (l + 1 == int(L.size())) {
            smooth(l, coarse);
            sync_check(f);
            return;
        }
        smooth(l, pre);
        FlowDevice<float>& c = *L[size_t(l) + 1];
        const int n0 = f.n0, n1 = f.n1, m0 = c.n0, m1 = c.n1;
        const long nf = size(l), nc = size(l + 1);
        const std::array<int, 2> fa = factor[size_t(l)];
        for (int g = 0; g < cycle_index; ++g) {
            f.rhs(W[size_t(l)].ptr, Rr[size_t(l)].ptr);  // R_l(W) + f_l
            inject_kernel<<<blocks(nc), 256, 0, st>>>(W[size_t(l)].ptr, n0, n1, m0, m1, fa[0], fa[1], Uc0[size_t(l) + 1].ptr);
            c.external_source = nullptr;
            c.reset_flag();
            c.rhs(Uc0[size_t(l) + 1].ptr, Rc[size_t(l) + 1].ptr);
            restrict_kernel<<<blocks(nc), 256, 0, st>>>(Rr[size_t(l)].ptr, n0, n1, m0, m1, fa[0], fa[1], F[size_t(l) + 1].ptr);
            const float minus = -1, one = 1, zero = 0;
            check(Prec<float>::axpy(ctx.blas, int(nc), &minus, Rc[size_t(l) + 1].ptr, F[size_t(l) + 1].ptr), "saxpy");
            check(cudaMemcpyAsync(W[size_t(l) + 1].ptr, Uc0[size_t(l) + 1].ptr, sizeof(float) * nc,
                                  cudaMemcpyDeviceToDevice, st), "copy");
            cycle(l + 1, F[size_t(l) + 1].ptr);
            // coarse correction dUc = W_c - Uc0 (in Rc as scratch), prolonged into dU[l]
            check(cudaMemcpyAsync(Rc[size_t(l) + 1].ptr, W[size_t(l) + 1].ptr, sizeof(float) * nc,
                                  cudaMemcpyDeviceToDevice, st), "copy");
            check(Prec<float>::axpy(ctx.blas, int(nc), &minus, Uc0[size_t(l) + 1].ptr, Rc[size_t(l) + 1].ptr), "saxpy");
            if (identity[size_t(l)]) {
                check(cudaMemcpyAsync(dU[size_t(l)].ptr, Rc[size_t(l) + 1].ptr, sizeof(float) * nf,
                                      cudaMemcpyDeviceToDevice, st), "copy");
            } else {
                // per component X (m1, m0) -> Pb X (n1, m0) -> (Pb X) Pa^T (n1, n0)
                Buffer<float> Y(size_t(4) * n1 * m0);
                check(Prec<float>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, n1, m0, m1, &one, Pb[size_t(l)].ptr,
                                                n1, 0, Rc[size_t(l) + 1].ptr, m1, long(m0) * m1, &zero, Y.ptr, n1,
                                                long(n1) * m0, 4), "prolong eta");
                check(Prec<float>::gemm_batched(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, n1, n0, m0, &one, Y.ptr, n1,
                                                long(n1) * m0, Pa[size_t(l)].ptr, n0, 0, &zero, dU[size_t(l)].ptr, n1,
                                                long(n1) * n0, 4), "prolong xi");
            }
            // W += omega dU, halving omega while the state would be nonphysical
            float omega = float(damping);
            for (int attempt = 0;; ++attempt) {
                check(cudaMemcpyAsync(Wt[size_t(l)].ptr, W[size_t(l)].ptr, sizeof(float) * nf, cudaMemcpyDeviceToDevice,
                                      st), "copy");
                check(Prec<float>::axpy(ctx.blas, int(nf), &omega, dU[size_t(l)].ptr, Wt[size_t(l)].ptr), "saxpy");
                f.reset_flag();
                positive_kernel<float><<<blocks(f.nn), 256, 0, st>>>(Wt[size_t(l)].ptr, f.nn, f.gamma, f.flag.ptr);
                if (f.read_flag() == 0) {
                    check(cudaMemcpyAsync(W[size_t(l)].ptr, Wt[size_t(l)].ptr, sizeof(float) * nf,
                                          cudaMemcpyDeviceToDevice, st), "copy");
                    break;
                }
                if (attempt == 4) break;  // skip the correction
                omega *= 0.5f;
                ++damping_cuts;
            }
            f.external_source = fl;
        }
        f.reset_flag();
        smooth(l, post);
        sync_check(f);
    }
};

std::shared_ptr<DeviceMG> make_device_mg(const std::vector<std::shared_ptr<FlowDevice<float>>>& levels,
                                         const std::vector<std::vector<double>>& Pa,
                                         const std::vector<std::vector<double>>& Pb, const MGSettings& s) {
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    auto mg = std::make_shared<DeviceMG>();
    mg->L = levels;
    mg->pre = s.pre;
    mg->post = s.post;
    mg->coarse = s.coarse;
    mg->cycle_index = s.cycle_index;
    mg->damping = s.damping;
    mg->initial = s.initial;
    const size_t nl = levels.size();
    for (size_t l = 0; l + 1 < nl; ++l) {
        const FlowDevice<float>& f = *levels[l];
        const FlowDevice<float>& c = *levels[l + 1];
        std::array<int, 2> fa{};
        const int nf[2] = {f.n0, f.n1}, nc[2] = {c.n0, c.n1};
        for (int a = 0; a < 2; ++a) {
            if (nf[a] == nc[a]) fa[size_t(a)] = 1;
            else if (nf[a] == 2 * nc[a] - 1) fa[size_t(a)] = 2;
            else throw std::invalid_argument("device multigrid: levels must be nested per axis");
        }
        mg->factor.push_back(fa);
        mg->identity.push_back(fa[0] == 1 && fa[1] == 1 && Pa[l].empty());
        mg->Pa.emplace_back();
        mg->Pb.emplace_back();
        if (!mg->identity.back()) {
            mg->Pa.back().upload(cast<float>(Pa[l]));
            mg->Pb.back().upload(cast<float>(Pb[l]));
        }
    }
    for (size_t l = 0; l < nl; ++l) {
        const size_t n = size_t(4 * levels[l]->nn);
        for (auto* v : {&mg->W, &mg->Wt, &mg->F, &mg->Rr, &mg->Uc0, &mg->Rc, &mg->dU}) v->emplace_back(n);
    }
    check(cudaStreamSynchronize(context().stream), "sync");
    return mg;
}

NewtonReport flow_newton(FlowDevice<double>& f, std::vector<double>& Uhost, const NewtonSettings& s,
                         const HostPreconditioner& precondition, DeviceMG* mg, const MGSettings* ms) {
    using T = double;
    using clock = std::chrono::steady_clock;
    std::lock_guard<std::recursive_mutex> lock(context().mutex);
    Context& ctx = context();
    cublasHandle_t blas = ctx.blas;
    cudaStream_t st = ctx.stream;
    const long nn = f.nn, nd = 4 * nn;
    const int m = std::max(1, s.krylov_restart);
    const size_t sz = size_t(nd);
    NewtonReport rep;
    Buffer<T> U, R{sz}, Ut{sz}, Rt{sz}, d{sz}, rk{sz}, w{sz}, V{sz * size_t(m + 1)}, Z{sz * size_t(m)}, h{size_t(m + 1)},
        y{size_t(m)};
    U.upload(to_planar<T>(Uhost, nn));
    const T one = 1, zero = 0, minus = -1;
    auto copy = [&](T* dst, const T* src) {
        check(cudaMemcpyAsync(dst, src, sizeof(T) * nd, cudaMemcpyDeviceToDevice, st), "copy");
    };
    auto nrm = [&](const T* x) {
        T r = 0;
        check(Prec<T>::nrm2(blas, int(nd), x, &r), "nrm2");
        return double(r);
    };
    auto scal = [&](T a, T* x) { check(cublasDscal(blas, int(nd), &a, x, 1), "dscal"); };
    auto axpy = [&](T a, const T* x, T* yv) { check(Prec<T>::axpy(blas, int(nd), &a, x, yv), "daxpy"); };
    auto rhs = [&](const T* x, T* out) {  // false on a nonphysical state
        f.reset_flag();
        f.rhs(x, out);
        return f.read_flag() == 0;
    };
    // J v ~ (R(U + h v) - R(U)) / h into out
    auto jv = [&](const T* v, T* out) {
        const auto t0 = clock::now();
        const double vn = nrm(v);
        if (vn == 0) {
            check(cudaMemsetAsync(out, 0, sizeof(T) * nd, st), "memset");
            return;
        }
        double hs = s.jacobian_epsilon * (1.0 + nrm(U.ptr)) / vn;
        for (int attempt = 0;; ++attempt) {
            copy(Ut.ptr, U.ptr);
            axpy(T(hs), v, Ut.ptr);
            if (rhs(Ut.ptr, Rt.ptr) || attempt == 6) break;
            hs *= 0.1;  // stay physical
        }
        copy(out, Rt.ptr);
        axpy(minus, R.ptr, out);
        scal(T(1.0 / hs), out);
        rep.seconds_jacobian += std::chrono::duration<double>(clock::now() - t0).count();
    };
    auto host_of = [&](const T* x) { return from_planar<T>(Buffer<T>::download_from(x, sz), nn); };
    // ---- device multigrid: start phase on the FP32 hierarchy, then the FP64 residual
    Buffer<float> Uf{sz}, Rp{sz}, F0{sz};
    auto to_float = [&](const T* x, float* out) { cast_kernel_df<<<blocks(nd), 256, 0, st>>>(x, nd, out); };
    if (mg) {
        FlowDevice<float>& L0 = *mg->L[0];
        to_float(U.ptr, mg->W[0].ptr);
        L0.external_source = nullptr;
        L0.reset_flag();
        mg->smooth(0, ms->initial);
        mg->sync_check(L0);
        auto back = [&] {  // U = double(W0)
            check(cudaMemsetAsync(U.ptr, 0, sizeof(T) * nd, st), "memset");
            add_kernel<<<blocks(nd), 256, 0, st>>>(U.ptr, mg->W[0].ptr, 1.0, nd, U.ptr);
        };
        back();
        for (;;) {
            if (!rhs(U.ptr, R.ptr)) throw std::domain_error("flow_newton: nonphysical state in the start cycles");
            const double r0 = nrm(R.ptr) / std::sqrt(double(nd));
            if (r0 <= ms->start_residual || rep.start_cycles >= ms->max_start_cycles || mg->L.size() < 2) break;
            mg->cycle(0, nullptr);
            back();
            ++rep.start_cycles;
        }
    }
    // device preconditioner: z (double, device) -> ~J^{-1} z into out
    auto device_precondition = [&](const T* z, T* out, bool new_state) {
        const auto t0 = clock::now();
        FlowDevice<float>& L0 = *mg->L[0];
        if (new_state) {
            to_float(U.ptr, Uf.ptr);
            L0.external_source = nullptr;
            L0.reset_flag();
            L0.rhs(Uf.ptr, Rp.ptr);
        }
        const double zn = nrm(z);
        if (zn == 0) {
            check(cudaMemsetAsync(out, 0, sizeof(T) * nd, st), "memset");
            return;
        }
        const double sigma = ms->preconditioner_epsilon * (1.0 + nrm(U.ptr)) / zn;
        forcing_kernel<<<blocks(nd), 256, 0, st>>>(Rp.ptr, z, sigma, nd, F0.ptr);
        check(cudaMemcpyAsync(mg->W[0].ptr, Uf.ptr, sizeof(float) * nd, cudaMemcpyDeviceToDevice, st), "copy");
        for (int c = 0; c < ms->preconditioner_cycles; ++c) mg->cycle(0, F0.ptr);
        L0.external_source = nullptr;
        correction_kernel<<<blocks(nd), 256, 0, st>>>(mg->W[0].ptr, Uf.ptr, sigma, nd, out);
        check(cudaStreamSynchronize(st), "sync");
        rep.seconds_preconditioner += std::chrono::duration<double>(clock::now() - t0).count();
    };
    if (!rhs(U.ptr, R.ptr)) throw std::domain_error("flow_newton: nonphysical initial state");
    const double root = std::sqrt(double(nd));
    double r = nrm(R.ptr) / root, r_prev = r;
    rep.history.emplace_back(0, r);
    if (s.progress) s.progress(0, r, 0);
    std::vector<T> Uc = Uhost;  // C-ordered host copy of the current iterate (preconditioner input)
    for (int k = 1; k <= s.max_newton && r > s.tolerance; ++k) {
        const double eta = std::max(s.eta_min, std::min(s.eta_max, k == 1 ? s.eta_max : r / r_prev));
        const double bnorm = nrm(R.ptr);
        check(cudaMemsetAsync(d.ptr, 0, sizeof(T) * nd, st), "memset");
        copy(rk.ptr, R.ptr);
        scal(minus, rk.ptr);
        int its = 0;
        bool new_state = true;
        while (its < s.max_krylov) {
            const double beta = nrm(rk.ptr);
            if (beta <= eta * bnorm) break;
            copy(V.ptr, rk.ptr);
            scal(T(1.0 / beta), V.ptr);
            std::vector<double> H(size_t((m + 1) * m), 0.0), g(size_t(m + 1), 0.0);
            std::vector<double> cs(static_cast<size_t>(m), 0.0), sn(static_cast<size_t>(m), 0.0);
            auto Hij = [&](int i, int j) -> double& { return H[size_t(i + j * (m + 1))]; };
            g[0] = beta;
            int j = 0;
            for (; j < m && its < s.max_krylov; ++j, ++its) {
                // z_j = M^{-1} v_j (host FAS cycle on the FP32 levels)
                if (mg) {
                    device_precondition(V.ptr + size_t(j) * sz, Z.ptr + size_t(j) * sz, new_state);
                } else {
                    const std::vector<double> z = precondition(Uc, host_of(V.ptr + size_t(j) * sz), new_state);
                    Buffer<T> zt;
                    zt.upload(to_planar<T>(z, nn));
                    copy(Z.ptr + size_t(j) * sz, zt.ptr);
                }
                new_state = false;
                jv(Z.ptr + size_t(j) * sz, w.ptr);
                // CGS2 against v_0 .. v_j: h = V^T w, w -= V h (twice)
                const auto t1 = clock::now();
                std::vector<double> hsum(size_t(j + 1), 0.0);
                for (int pass = 0; pass < 2; ++pass) {
                    check(Prec<T>::gemv(blas, CUBLAS_OP_T, int(nd), j + 1, &one, V.ptr, int(nd), w.ptr, &zero, h.ptr),
                          "dgemv");
                    check(Prec<T>::gemv(blas, CUBLAS_OP_N, int(nd), j + 1, &minus, V.ptr, int(nd), h.ptr, &one, w.ptr),
                          "dgemv");
                    const std::vector<T> hh = Buffer<T>::download_from(h.ptr, size_t(j + 1));
                    for (int i = 0; i <= j; ++i) hsum[size_t(i)] += hh[size_t(i)];
                }
                for (int i = 0; i <= j; ++i) Hij(i, j) = hsum[size_t(i)];
                Hij(j + 1, j) = nrm(w.ptr);
                copy(V.ptr + size_t(j + 1) * sz, w.ptr);
                if (Hij(j + 1, j) > 0) scal(T(1.0 / Hij(j + 1, j)), V.ptr + size_t(j + 1) * sz);
                rep.seconds_krylov += std::chrono::duration<double>(clock::now() - t1).count();
                for (int i = 0; i < j; ++i) {  // previous Givens rotations
                    const double t = cs[size_t(i)] * Hij(i, j) + sn[size_t(i)] * Hij(i + 1, j);
                    Hij(i + 1, j) = -sn[size_t(i)] * Hij(i, j) + cs[size_t(i)] * Hij(i + 1, j);
                    Hij(i, j) = t;
                }
                const double den = std::hypot(Hij(j, j), Hij(j + 1, j));
                cs[size_t(j)] = Hij(j, j) / den;
                sn[size_t(j)] = Hij(j + 1, j) / den;
                Hij(j, j) = den;
                Hij(j + 1, j) = 0;
                g[size_t(j + 1)] = -sn[size_t(j)] * g[size_t(j)];
                g[size_t(j)] = cs[size_t(j)] * g[size_t(j)];
                if (std::abs(g[size_t(j + 1)]) <= eta * bnorm) {
                    ++j;
                    ++its;
                    break;
                }
            }
            std::vector<T> yv(size_t(j), 0.0);
            for (int i = j - 1; i >= 0; --i) {
                double acc = g[size_t(i)];
                for (int l = i + 1; l < j; ++l) acc -= Hij(i, l) * yv[size_t(l)];
                yv[size_t(i)] = acc / Hij(i, i);
            }
            if (j > 0) {
                check(cudaMemcpyAsync(y.ptr, yv.data(), sizeof(T) * size_t(j), cudaMemcpyHostToDevice, st), "y");
                check(Prec<T>::gemv(blas, CUBLAS_OP_N, int(nd), j, &one, Z.ptr, int(nd), y.ptr, &one, d.ptr), "dgemv");
            }
            jv(d.ptr, w.ptr);  // true linear residual -R - J d
            copy(rk.ptr, R.ptr);
            scal(minus, rk.ptr);
            axpy(minus, w.ptr, rk.ptr);
            if (nrm(rk.ptr) <= eta * bnorm) break;
        }
        rep.krylov_iterations += its;
        // line search on |R|, rejecting nonphysical states
        double alpha = 1.0, rn = r;
        bool accepted = false;
        for (int t = 0; t < 8; ++t, alpha *= 0.5) {
            copy(Ut.ptr, U.ptr);
            axpy(T(alpha), d.ptr, Ut.ptr);
            if (!rhs(Ut.ptr, Rt.ptr)) continue;
            rn = nrm(Rt.ptr) / root;
            if (std::isfinite(rn) && rn < (1 - 1e-4 * alpha) * r) {
                accepted = true;
                break;
            }
        }
        if (!accepted) break;  // no decrease along the Newton direction
        copy(U.ptr, Ut.ptr);
        copy(R.ptr, Rt.ptr);
        r_prev = r;
        r = rn;
        if (!mg) Uc = host_of(U.ptr);
        rep.newton_steps = k;
        rep.history.emplace_back(k, r);
        if (s.progress) s.progress(k, r, its);
    }
    Uhost = host_of(U.ptr);
    rep.residual = r;
    rep.converged = r <= s.tolerance;
    if (mg) rep.damping_cuts = mg->damping_cuts;
    return rep;
}

}  // namespace bspf::cuda::flow

