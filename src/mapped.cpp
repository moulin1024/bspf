// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/mapped.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "bspf/bspline.hpp"

namespace bspf {

namespace {

using Index = Eigen::Index;

void check_control_points(const Matrix& control, int degree) {
    if (degree < 1) throw std::invalid_argument("spline curve degree must be >= 1");
    if (control.cols() != 2) throw std::invalid_argument("control points must have shape (n, 2)");
    if (control.rows() < degree + 1) throw std::invalid_argument("a spline curve of degree d needs at least d + 1 control points");
    if (!control.allFinite()) throw std::invalid_argument("control points must be finite");
}

}  // namespace

SplineCurve::SplineCurve(const Matrix& control_points, int degree)
    : control_(control_points), degree_(degree) {
    check_control_points(control_, degree);
    knots_ = open_knots(0.0, 1.0, degree, static_cast<int>(control_.rows()), 0.0);
}

SplineCurve::SplineCurve(const Matrix& control_points, const Vector& knots, int degree)
    : control_(control_points), knots_(knots), degree_(degree) {
    check_control_points(control_, degree);
    if (knots_.size() != control_.rows() + degree + 1)
        throw std::invalid_argument("a curve with n control points of degree d needs n + d + 1 knots");
    for (Index i = 0; i + 1 < knots_.size(); ++i)
        if (!(knots_(i + 1) >= knots_(i))) throw std::invalid_argument("knots must be nondecreasing");
    for (int i = 0; i <= degree; ++i)
        if (knots_(i) != 0.0 || knots_(knots_.size() - 1 - i) != 1.0)
            throw std::invalid_argument("curve knots must be clamped to [0, 1]");
}

Matrix SplineCurve::evaluate(const Vector& s, int derivative) const {
    for (Index i = 0; i < s.size(); ++i)
        if (!(s(i) >= 0.0 && s(i) <= 1.0)) throw std::invalid_argument("curve parameters must lie in [0, 1]");
    return basis_matrix(knots_, s, degree_, derivative) * control_;
}

MappedGrid2D::MappedGrid2D(SplineCurve bottom, SplineCurve right, SplineCurve top, SplineCurve left, Index n_xi,
                           Index n_eta, const LineOptions& options)
    : n_xi_(n_xi), n_eta_(n_eta),
      reference_({Line(Vector::LinSpaced(n_xi, 0.0, 1.0), options), Line(Vector::LinSpaced(n_eta, 0.0, 1.0), options)}) {
    Vector ends(2);
    ends << 0.0, 1.0;
    const Matrix b = bottom.evaluate(ends), r = right.evaluate(ends), t = top.evaluate(ends), l = left.evaluate(ends);
    const RowVector P00 = b.row(0), P10 = b.row(1), P01 = t.row(0), P11 = t.row(1);
    const double scale = std::max({b.cwiseAbs().maxCoeff(), t.cwiseAbs().maxCoeff(), 1.0});
    auto meets = [&](const RowVector& a, const RowVector& c) { return (a - c).cwiseAbs().maxCoeff() <= 1e-12 * scale; };
    if (!meets(l.row(0), P00) || !meets(l.row(1), P01) || !meets(r.row(0), P10) || !meets(r.row(1), P11))
        throw std::invalid_argument(
            "boundary curves must meet at the corners: bottom(0) = left(0), bottom(1) = right(0), "
            "top(0) = left(1), top(1) = right(1)");

    const Vector xi = Vector::LinSpaced(n_xi, 0.0, 1.0), eta = Vector::LinSpaced(n_eta, 0.0, 1.0);
    const Matrix B = bottom.evaluate(xi), dB = bottom.evaluate(xi, 1), ddB = bottom.evaluate(xi, 2);  // (n_xi, 2)
    const Matrix T = top.evaluate(xi), dT = top.evaluate(xi, 1), ddT = top.evaluate(xi, 2);
    const Matrix L = left.evaluate(eta), dL = left.evaluate(eta, 1), ddL = left.evaluate(eta, 2);   // (n_eta, 2)
    const Matrix R = right.evaluate(eta), dR = right.evaluate(eta, 1), ddR = right.evaluate(eta, 2);
    const RowVector twist = P00 - P10 - P01 + P11;

    for (Matrix* m : {&x_, &y_, &x_xi_, &x_eta_, &y_xi_, &y_eta_, &jacobian_, &xi_x_, &xi_y_, &eta_x_, &eta_y_, &xi_xx_,
                      &xi_xy_, &xi_yy_, &eta_xx_, &eta_xy_, &eta_yy_})
        m->resize(n_xi, n_eta);
    for (Index i = 0; i < n_xi; ++i)
        for (Index j = 0; j < n_eta; ++j) {
            const double s = xi(i), e = eta(j);
            const RowVector X = (1 - e) * B.row(i) + e * T.row(i) + (1 - s) * L.row(j) + s * R.row(j) -
                                ((1 - s) * (1 - e) * P00 + s * (1 - e) * P10 + (1 - s) * e * P01 + s * e * P11);
            const RowVector Xs = (1 - e) * dB.row(i) + e * dT.row(i) - L.row(j) + R.row(j) -
                                 (-(1 - e) * P00 + (1 - e) * P10 - e * P01 + e * P11);
            const RowVector Xe = -B.row(i) + T.row(i) + (1 - s) * dL.row(j) + s * dR.row(j) -
                                 (-(1 - s) * P00 - s * P10 + (1 - s) * P01 + s * P11);
            // Second derivatives of the map: L, R are linear in xi and B, T in eta.
            const RowVector Xss = (1 - e) * ddB.row(i) + e * ddT.row(i);
            const RowVector Xee = (1 - s) * ddL.row(j) + s * ddR.row(j);
            const RowVector Xse = -dB.row(i) + dT.row(i) - dL.row(j) + dR.row(j) - twist;
            x_(i, j) = X(0);
            y_(i, j) = X(1);
            x_xi_(i, j) = Xs(0);
            y_xi_(i, j) = Xs(1);
            x_eta_(i, j) = Xe(0);
            y_eta_(i, j) = Xe(1);
            const double J = Xs(0) * Xe(1) - Xe(0) * Xs(1);
            jacobian_(i, j) = J;
            // Inverse metrics p = numerator / J and their reference derivatives (quotient rule).
            const double J_s = Xss(0) * Xe(1) + Xs(0) * Xse(1) - Xse(0) * Xs(1) - Xe(0) * Xss(1);
            const double J_e = Xse(0) * Xe(1) + Xs(0) * Xee(1) - Xee(0) * Xs(1) - Xe(0) * Xse(1);
            auto metric = [&](double num, double num_s, double num_e, double& p, double& p_s, double& p_e) {
                p = num / J;
                p_s = (num_s * J - num * J_s) / (J * J);
                p_e = (num_e * J - num * J_e) / (J * J);
            };
            double a, a_s, a_e, b, b_s, b_e, c, c_s, c_e, d, d_s, d_e;
            metric(Xe(1), Xse(1), Xee(1), a, a_s, a_e);      // xi_x  =  y_eta / J
            metric(-Xe(0), -Xse(0), -Xee(0), b, b_s, b_e);   // xi_y  = -x_eta / J
            metric(-Xs(1), -Xss(1), -Xse(1), c, c_s, c_e);   // eta_x = -y_xi / J
            metric(Xs(0), Xss(0), Xse(0), d, d_s, d_e);      // eta_y =  x_xi / J
            xi_x_(i, j) = a;
            xi_y_(i, j) = b;
            eta_x_(i, j) = c;
            eta_y_(i, j) = d;
            // d/dx = xi_x d/dxi + eta_x d/deta, d/dy = xi_y d/dxi + eta_y d/deta.
            xi_xx_(i, j) = a * a_s + c * a_e;
            xi_xy_(i, j) = b * a_s + d * a_e;
            xi_yy_(i, j) = b * b_s + d * b_e;
            eta_xx_(i, j) = a * c_s + c * c_e;
            eta_xy_(i, j) = b * c_s + d * c_e;
            eta_yy_(i, j) = b * d_s + d * d_e;
        }
    if (!(jacobian_.minCoeff() > 0))
        throw std::invalid_argument("the transfinite map folds (Jacobian <= 0 at some node): check the curves' "
                                    "orientation (counterclockwise: bottom, right, top reversed, left reversed)");
}

Field MappedGrid2D::combine(const std::vector<std::pair<const Matrix*, const Field*>>& terms) const {
    Field out;
    out.shape = terms.front().second->shape;
    out.data.assign(terms.front().second->data.size(), 0.0);
    const Index batch = element_count(out.shape) / (n_xi_ * n_eta_);
    for (const auto& [coefficient, field] : terms)
        for (Index i = 0; i < n_xi_; ++i)
            for (Index j = 0; j < n_eta_; ++j) {
                const double c = (*coefficient)(i, j);
                const size_t base = static_cast<size_t>((i * n_eta_ + j) * batch);
                for (Index k = 0; k < batch; ++k) out.data[base + k] += c * field->data[base + k];
            }
    return out;
}

Field MappedGrid2D::partial(const FieldView& f, int order_x, int order_y) const {
    if (f.shape.size() < 2 || f.shape[0] != n_xi_ || f.shape[1] != n_eta_)
        throw std::invalid_argument("expected leading shape (" + std::to_string(n_xi_) + ", " + std::to_string(n_eta_) + ")");
    if (order_x < 0 || order_y < 0) throw std::invalid_argument("derivative orders must be nonnegative");
    const int order = order_x + order_y;
    if (order == 0) {
        Field copy;
        copy.shape = f.shape;
        copy.data.assign(f.data, f.data + element_count(f.shape));
        return copy;
    }
    if (order > 2) {  // first derivatives of the order-2 result
        const Field lower = order_x > 0 ? partial(f, order_x - 1, order_y) : partial(f, order_x, order_y - 1);
        return partial(lower.view(), order_x > 0 ? 1 : 0, order_x > 0 ? 0 : 1);
    }
    const Field f_s = reference_.partial(f, {1, 0}), f_e = reference_.partial(f, {0, 1});
    if (order == 1) {
        if (order_x == 1) return combine({{&xi_x_, &f_s}, {&eta_x_, &f_e}});
        return combine({{&xi_y_, &f_s}, {&eta_y_, &f_e}});
    }
    const Field f_ss = reference_.partial(f, {2, 0}), f_ee = reference_.partial(f, {0, 2});
    const Field f_se = reference_.partial(f, {1, 1});
    // Quadratic forms of the inverse metrics for the second-order terms.
    auto product = [&](const Matrix& p, const Matrix& q, double factor) { return Matrix(factor * p.cwiseProduct(q)); };
    if (order_x == 2) {
        const Matrix c_ss = product(xi_x_, xi_x_, 1), c_se = product(xi_x_, eta_x_, 2), c_ee = product(eta_x_, eta_x_, 1);
        return combine({{&c_ss, &f_ss}, {&c_se, &f_se}, {&c_ee, &f_ee}, {&xi_xx_, &f_s}, {&eta_xx_, &f_e}});
    }
    if (order_y == 2) {
        const Matrix c_ss = product(xi_y_, xi_y_, 1), c_se = product(xi_y_, eta_y_, 2), c_ee = product(eta_y_, eta_y_, 1);
        return combine({{&c_ss, &f_ss}, {&c_se, &f_se}, {&c_ee, &f_ee}, {&xi_yy_, &f_s}, {&eta_yy_, &f_e}});
    }
    const Matrix c_ss = product(xi_x_, xi_y_, 1), c_ee = product(eta_x_, eta_y_, 1);
    const Matrix c_se = Matrix(xi_x_.cwiseProduct(eta_y_) + xi_y_.cwiseProduct(eta_x_));
    return combine({{&c_ss, &f_ss}, {&c_se, &f_se}, {&c_ee, &f_ee}, {&xi_xy_, &f_s}, {&eta_xy_, &f_e}});
}

}  // namespace bspf
