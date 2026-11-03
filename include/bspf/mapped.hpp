// SPDX-License-Identifier: GPL-3.0-or-later
// Structured grids on deformed 2D regions: four spline boundary curves, a
// transfinite (Coons) map from the unit square, and physical derivatives by the
// chain rule with exact metrics.
#pragma once

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

    int degree() const { return degree_; }
    const Vector& knots() const { return knots_; }
    const Matrix& control_points() const { return control_; }

private:
    Matrix control_;
    Vector knots_;
    int degree_ = 0;
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
class MappedGrid2D {
public:
    // Throws std::invalid_argument if the corners do not match or the map folds (J <= 0).
    MappedGrid2D(SplineCurve bottom, SplineCurve right, SplineCurve top, SplineCurve left, Eigen::Index n_xi,
                 Eigen::Index n_eta, const LineOptions& options = {});

    const Grid& reference() const { return reference_; }
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

private:
    // Sum of coefficient(i, j) * term over the nodes and batch.
    Field combine(const std::vector<std::pair<const Matrix*, const Field*>>& terms) const;

    Eigen::Index n_xi_ = 0, n_eta_ = 0;
    Grid reference_;
    Matrix x_, y_, x_xi_, x_eta_, y_xi_, y_eta_, jacobian_;
    // Inverse metrics and their physical derivatives (exact).
    Matrix xi_x_, xi_y_, eta_x_, eta_y_;
    Matrix xi_xx_, xi_xy_, xi_yy_, eta_xx_, eta_xy_, eta_yy_;
};

}  // namespace bspf
