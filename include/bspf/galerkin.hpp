// SPDX-License-Identifier: GPL-3.0-or-later
// Energy-stable (Galerkin) second derivative on a Line's BSPF trial space.
//
// The trial functions are the BSPF cardinal functions phi_i (the interpolant
// of unit samples), so a field is u_h = sum_i u_i phi_i. The weak form uses the
// exact integrals M_ij = int phi_i phi_j and K_ij = int phi_i' phi_j' over
// [x_0, x_{N-1}]. With Dirichlet walls the interior operator -M_II^{-1} K_II is
// similar to a symmetric negative-definite matrix: its spectrum is real for any
// endpoint jets, so wave and diffusion schemes are energy stable.
//
// Setup computes M and K products matrix-free by quadrature of the trial field:
// u_h^(k) at 16 Gauss points per sample cell (cells holding a knot split there),
// with c = F u the constrained fit and r = u - B c,
//     Phi u = B_P^(k) c + T_k r,   K u = Phi^T W Phi u,
// where T_k is a circulant map per Gauss offset (one FFT pair). The trial
// functions are bounded, so every intermediate is O(1) and the fit's large
// endpoint-jet factors (growing like (N / n_basis)^(q-1)) amplify rounding once.
// The interior mass is dx (I + U S U^T) with a low rank r found by a randomized
// eigensolver, so solves cost O(N r).
//
// Applied form: W is the periodic spectral second derivative plus a low-rank
// correction, W = (D_per)_I + P Q^T, with rank s bounded in N (about 20-50),
// found at setup by a randomized SVD driven by fast products with W and W^T.
// Applying it costs one real FFT pair and 4 s flops per sample. Setup is
// O(N log N (r + s)); nothing is dense in N.
#pragma once

#include <memory>

#include "bspf/line.hpp"
#include "bspf/types.hpp"

namespace bspf {

struct GalerkinOptions {
    // Eigenvalues of M_II / dx - I below this are dropped from the low-rank mass.
    double mass_tolerance = 1e-12;
    // Singular values of W - (D_per)_I below this times (pi / dx)^2 are dropped
    // from the applied operator's low-rank correction.
    double operator_tolerance = 1e-14;
};

class GalerkinLine {
public:
    explicit GalerkinLine(const Line& line, const GalerkinOptions& options = {});

    Eigen::Index size() const;       // N
    Eigen::Index mass_rank() const;      // r
    Eigen::Index operator_rank() const;  // s
    // max |F| of the constrained fit (F = fit_samples + fit_jets J). It grows like
    // (N / n_basis)^(q-1) and bounds the weak operator's accuracy at about
    // 4 eps max|F| relative to (pi / dx)^2; prefer q <= 3 or more basis
    // functions when it exceeds ~1e7.
    double fit_amplification() const;
    const Line& line() const;

    // Exact Galerkin products for columns u (N, m).
    Matrix mass(const MatrixRef& u) const;
    Matrix stiffness(const MatrixRef& u) const;

    // Dirichlet interior operators, (N, m) -> (N - 2, m):
    //   W u = -M_II^{-1} (K u)_I          (interior second derivative)
    //   E u = u_I + M_II^{-1} M_IB u_B     (interior extension: mass coupling to the walls)
    // In 1D the weak second time derivative is u_tt,I = W u - E a, with a the
    // wall accelerations (zero interior).
    // second_derivative applies the compressed form (D_per u)_I + P (Q^T u): the
    // periodic spectral second derivative (one real FFT pair) plus a rank-s
    // correction, s bounded in N, which reproduces W to operator_tolerance.
    Matrix second_derivative(const MatrixRef& u) const;
    Matrix exact_second_derivative(const MatrixRef& u) const;  // the defining form, for checks
    Matrix extension(const MatrixRef& u) const;

    // Block kernels for the grid: y (N, b) receives W u in rows 1..N-2 (rows 0 and
    // N-1 are scratch); `y` is reused across calls to avoid allocations.
    void second_derivative_into(const Matrix& u, Matrix& y) const;
    // M_II^{-1} M_IB (N - 2, 2): E u = u_I + this * [u_0; u_{N-1}].
    const Matrix& wall_extension() const;

    // One line: in[i * in_stride] -> out[i * out_stride] (out has N - 2 entries).
    void second_derivative(const double* in, Eigen::Index in_stride, double* out, Eigen::Index out_stride) const;
    void extension(const double* in, Eigen::Index in_stride, double* out, Eigen::Index out_stride) const;

    struct Data;

private:
    std::shared_ptr<const Data> d_;
};

}  // namespace bspf
