// SPDX-License-Identifier: GPL-3.0-or-later
// Structured grids on deformed 2D regions: four spline boundary curves, a
// transfinite (Coons) map from the unit square, and physical derivatives by the
// chain rule with exact metrics.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "bspf/grid.hpp"
#include "bspf/line.hpp"
#include "bspf/types.hpp"

namespace bspf {

// A clamped B-spline curve x(s) in the plane, s in [0, 1]: control points (n, 2),
// degree d, open knots (uniform unless given). n = d + 1 gives a single Bezier
// segment (a polynomial curve, smooth to all orders).
class SplineCurve {
public:
    explicit SplineCurve(const Matrix& control_points, int degree = 3);
    SplineCurve(const Matrix& control_points, const Vector& knots, int degree);

    // Points (derivative 0) or derivatives d^k x / ds^k at parameters s: (m, 2).
    Matrix evaluate(const Vector& s, int derivative = 0) const;
    // As evaluate, with s outside [0, 1] allowed: the end spans continue as the
    // polynomials they are (exact Taylor expansion at the end).
    Matrix evaluate_extended(const Vector& s, int derivative = 0) const;

    int degree() const { return degree_; }
    const Vector& knots() const { return knots_; }
    const Matrix& control_points() const { return control_; }

private:
    void cache_end_jets();
    Matrix control_;
    Vector knots_;
    int degree_ = 0;
    std::vector<Matrix> end_jets_;  // derivatives 0..degree at s = 0 and 1, (2, 2) each
};

// The region bounded by four curves, oriented as
//     bottom(s): eta = 0, from P00 to P10      top(s):   eta = 1, from P01 to P11
//     left(s):   xi = 0,  from P00 to P01      right(s): xi = 1,  from P10 to P11
// and mapped from (xi, eta) in [0, 1]^2 by transfinite interpolation
//     X = (1 - eta) B(xi) + eta T(xi) + (1 - xi) L(eta) + xi R(eta)
//         - [(1 - xi)(1 - eta) P00 + xi (1 - eta) P10 + (1 - xi) eta P01 + xi eta P11].
// Nodes are the images of uniform BSPF lines in xi and eta. Fields are C-ordered
// (n_xi, n_eta, *batch) arrays of samples at the nodes. Physical derivatives use
// the chain rule with the inverse metrics xi_x = y_eta / J, xi_y = -x_eta / J,
// eta_x = -y_xi / J, eta_y = x_xi / J (J = x_xi y_eta - x_eta y_xi):
//     f_x  = xi_x f_xi + eta_x f_eta,
//     f_xx = xi_x^2 f_xixi + 2 xi_x eta_x f_xieta + eta_x^2 f_etaeta + xi_xx f_xi + eta_xx f_eta,
// and likewise for f_y, f_yy and f_xy. The reference partials are direct BSPF
// derivatives (order 2 where needed, not a first derivative applied twice), and
// every metric term, including xi_xx etc., is exact from the curves' first and
// second derivatives. Orders above 2 apply first derivatives to these.
// Node distribution along an edge: psi(s), the fraction of the edge's nodes before
// curve parameter s, an increasing polynomial in Bernstein form with coefficients
// c_0 = 0, ..., c_d = 1 (e.g. the normalized integral of ds / h for a target size h).
// Node k of n sits at s = psi^{-1}(k / (n - 1)). Empty: psi(s) = s.
class Distribution {
public:
    Distribution() = default;
    // Throws std::invalid_argument unless c_0 = 0, c_d = 1 and psi' > 0 on [0, 1].
    explicit Distribution(const Vector& coefficients);
    bool identity() const { return c_.size() == 0; }
    const Vector& coefficients() const { return c_; }
    // psi, psi', psi'' at s (any real s: the polynomial continues beyond [0, 1]).
    void evaluate(double s, double out[3]) const;

private:
    Vector c_;
};

// Node distributions on the four edges (bottom, top along xi; left, right along eta).
// Inside a patch the curve parameters (s, e) solve
//     xi  = (1 - eta) psi_bottom(s) + eta psi_top(s),
//     eta = (1 - xi)  psi_left(e)   + xi  psi_right(e),
// which blends the node spacing harmonically between opposite edges (exact for a
// spacing proportional to a linearly varying line length, as in a fan), and the map
// is the transfinite map at (s, e). Patches sharing an edge share its distribution.
struct EdgeDistribution {
    Distribution bottom, top, left, right;
};

class MappedGrid2D {
public:
    // Throws std::invalid_argument if the corners do not match or the map folds (J <= 0).
    // correction_x / correction_y (optional, same shape (p - 1, q - 1)): interior Bernstein
    // coefficients c_kl, k = 1..p-1, l = 1..q-1, of a correction added to the map,
    //     X(xi, eta) += sum_kl c_kl B_k^p(xi) B_l^q(eta),
    // which vanishes on all four edges (e.g. fitted to an elliptically smoothed grid).
    // eta_options: line options of the eta axis (default: `options`, used for xi).
    MappedGrid2D(SplineCurve bottom, SplineCurve right, SplineCurve top, SplineCurve left, Eigen::Index n_xi,
                 Eigen::Index n_eta, const LineOptions& options = {}, EdgeDistribution distribution = {},
                 Matrix correction_x = {}, Matrix correction_y = {}, std::optional<LineOptions> eta_options = {});

    const Grid& reference() const { return reference_; }
    const LineOptions& options(int axis = 0) const { return axis == 0 ? options_ : eta_options_; }
    // The transfinite map and its first derivatives at a reference point (xi, eta),
    // which may lie outside [0, 1]^2 (the curves continue polynomially).
    void map(double xi, double eta, double X[2], double X_xi[2], double X_eta[2]) const;
    Shape shape() const { return {n_xi_, n_eta_}; }

    // Node coordinates and metrics, (n_xi, n_eta).
    const Matrix& x() const { return x_; }
    const Matrix& y() const { return y_; }
    const Matrix& x_xi() const { return x_xi_; }
    const Matrix& x_eta() const { return x_eta_; }
    const Matrix& y_xi() const { return y_xi_; }
    const Matrix& y_eta() const { return y_eta_; }
    const Matrix& jacobian() const { return jacobian_; }

    // d^{order_x}/dx^{order_x} d^{order_y}/dy^{order_y} f in physical coordinates.
    Field partial(const FieldView& f, int order_x, int order_y) const;
    // Physical derivative of order_x + order_y <= 2 from given reference partials
    // (f_xi, f_eta and, for order 2, f_xixi, f_xieta, f_etaeta).
    Field chain_rule(int order_x, int order_y, const Field& f_xi, const Field& f_eta, const Field* f_xixi = nullptr,
                     const Field* f_xieta = nullptr, const Field* f_etaeta = nullptr) const;

private:
    // Sum of coefficient(i, j) * term over the nodes and batch.
    Field combine(const std::vector<std::pair<const Matrix*, const Field*>>& terms) const;

    Eigen::Index n_xi_ = 0, n_eta_ = 0;
    LineOptions options_, eta_options_;
    SplineCurve bottom_, right_, top_, left_;
    EdgeDistribution distribution_;
    Matrix correction_[2];  // interior Bernstein coefficients (p - 1, q - 1) of the x and y corrections
    // Correction and its derivatives (value, xi, eta, xixi, xieta, etaeta) x 2 components at (xi, eta).
    void correction(double xi, double eta, double out[6][2]) const;
    // Curve parameters (s, e) at a reference point and their first and second derivatives.
    struct Parameters {
        double s, s_xi, s_eta, s_xixi, s_xieta, s_etaeta, e, e_xi, e_eta, e_xixi, e_xieta, e_etaeta;
    };
    Parameters parameters(double xi, double eta) const;
    RowVector P00_, P10_, P01_, P11_;
    Grid reference_;
    Matrix x_, y_, x_xi_, x_eta_, y_xi_, y_eta_, jacobian_;
    // Inverse metrics and their physical derivatives (exact).
    Matrix xi_x_, xi_y_, eta_x_, eta_y_;
    Matrix xi_xx_, xi_xy_, xi_yy_, eta_xx_, eta_xy_, eta_yy_;
};

}  // namespace bspf
