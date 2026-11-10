// SPDX-License-Identifier: GPL-3.0-or-later
// Weak-form (Galerkin) volume term of the 2D compressible Euler / Navier-Stokes
// equations on one mapped block, in reference coordinates (xi, eta):
//
//     R_ij = sum_ab w_a w_b [ P1a(a,i) P0b(b,j) Fh(a,b) + P0a(a,i) P1b(b,j) Gh(a,b) ],
//
// with the trial field U_q = (P0a x P0b) U interpolated from the nodal values to
// tensor Gauss points (a, b), Fh = y_eta F - x_eta G and Gh = -y_xi F + x_xi G the
// contravariant Euler fluxes formed there from the metrics at the points, minus
// optional extra contravariant fluxes given at the nodes (the viscous fluxes),
// which are interpolated the same way. P0, P1 (Q x N) are the cardinal trial
// functions and their derivatives at the points of each axis, w the weights.
//
// Interpolation and projection are four GEMMs on all components at once; the
// fluxes are one fused loop over the points. Fields are C-ordered (n0, n1, 4)
// arrays of (rho, rho u, rho v, E).
#pragma once

#include "bspf/types.hpp"

namespace bspf {

class WeakEuler2D {
public:
    // Metrics x_xi, x_eta, y_xi, y_eta at the points: (Qa, Qb) each.
    WeakEuler2D(Matrix P0a, Matrix P1a, Vector wa, Matrix P0b, Matrix P1b, Vector wb, const Matrix& x_xi,
                const Matrix& x_eta, const Matrix& y_xi, const Matrix& y_eta, double gamma);

    Eigen::Index n0() const { return P0a_.cols(); }
    Eigen::Index n1() const { return P0b_.cols(); }
    Eigen::Index qa() const { return P0a_.rows(); }
    Eigen::Index qb() const { return P0b_.rows(); }

    // R (n0, n1, 4) from U (n0, n1, 4); Fv, Gv (n0, n1, 4) or null. Throws
    // std::domain_error on nonpositive density or pressure at a point.
    void volume(const double* U, const double* Fv, const double* Gv, double* R) const;
    // Entropy-variable trial field: the nodal input holds entropy variables V, interpolated to
    // the Gauss points and mapped to U(V) there before the fluxes are formed.
    void set_entropy(bool on) { entropy_ = on; }

private:
    using Rows = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // f (n0, n1, k) at the nodes -> out (Qb, Qa * k) at the points, i.e. C-ordered (Qb, Qa, k).
    void interpolate(const double* f, Eigen::Index k, Rows& out) const;
    Matrix P0a_, P1a_, P0b_, P1b_;
    Matrix P0aT_, P1aT_, P0bT_, P1bT_;  // transposes, (N x Q)
    // Weighted metrics at the points, laid out (b, a): w_a w_b times x_xi, x_eta, y_xi, y_eta.
    Rows w_, wx_xi_, wx_eta_, wy_xi_, wy_eta_;
    double gamma_;
    bool entropy_ = false;
};

}  // namespace bspf
