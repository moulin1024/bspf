// SPDX-License-Identifier: GPL-3.0-or-later
// FP32 device form of GalerkinLine's fast evaluation and projection: the BSPF
// trial field and its derivative at the composite Gauss points, and the exact
// adjoint (weighted projection), applied to many lines at once.
//
//   evaluate (k = 0, 1):  c = F u, r = u - B c, R = rfft(r);
//                         offset q: irfft(shift_kq o R) + S_kq c, scattered to the
//                         quadrature rows; knot-split cells: E_k r + G_k c.
//   project:              its adjoint applied to weights o values.
// One batched R2C/C2R pair per call (all offsets in one batch), cuBLAS SGEMMs for
// the fit, remainder and spline parts. Cost O(N log N + N n_basis) per line,
// instead of the O(N Q) dense product.
//
// Lines are contiguous: X is a column-major (N, m) device matrix (each column a
// line), values a column-major (P, m) matrix (P quadrature points).
#pragma once

#include <memory>

#include "bspf/galerkin.hpp"

namespace bspf::cuda {

class GalerkinAxis {
public:
    explicit GalerkinAxis(const GalerkinLine& line);
    long size() const;    // N
    long points() const;  // P

    // Device pointers; run on the solver stream (stream_handle()).
    void evaluate(const float* X, long m, int derivative, float* values) const;
    void project(const float* values, long m, int derivative, float* X) const;

    // Host convenience for tests: column-major double in and out.
    Matrix evaluate_host(const Matrix& X, int derivative) const;
    Matrix project_host(const Matrix& values, int derivative) const;

    struct Impl;

private:
    std::shared_ptr<Impl> d_;
};

// The stream all compressible-solver device work runs on.
void* stream_handle();

}  // namespace bspf::cuda
