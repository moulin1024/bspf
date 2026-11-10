// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/mapped.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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
    cache_end_jets();
}

void SplineCurve::cache_end_jets() {
    Vector ends(2);
    ends << 0.0, 1.0;
    end_jets_.clear();
    for (int k = 0; k <= degree_; ++k) end_jets_.push_back(basis_matrix(knots_, ends, degree_, k) * control_);
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
    cache_end_jets();
}

Matrix SplineCurve::evaluate(const Vector& s, int derivative) const {
    for (Index i = 0; i < s.size(); ++i)
        if (!(s(i) >= 0.0 && s(i) <= 1.0)) throw std::invalid_argument("curve parameters must lie in [0, 1]");
    return basis_matrix(knots_, s, degree_, derivative) * control_;
}

Matrix SplineCurve::evaluate_extended(const Vector& s, int derivative) const {
    Matrix out(s.size(), 2);
    const std::vector<Matrix>& jets = end_jets_;
    for (Index i = 0; i < s.size(); ++i) {
        if (s(i) >= 0.0 && s(i) <= 1.0) {
            Vector one(1);
            one << s(i);
            out.row(i) = (basis_matrix(knots_, one, degree_, derivative) * control_).row(0);
            continue;
        }
        const int end = s(i) > 1.0 ? 1 : 0;
        const double ds = s(i) - static_cast<double>(end);
        RowVector value = RowVector::Zero(2);
        double factor = 1.0;  // ds^(j - derivative) / (j - derivative)!
        for (int j = derivative; j <= degree_; ++j) {
            value += factor * jets[j].row(end);
            factor *= ds / static_cast<double>(j - derivative + 1);
        }
        out.row(i) = value;
    }
    return out;
}

Distribution::Distribution(const Vector& coefficients) : c_(coefficients) {
    if (c_.size() < 2) throw std::invalid_argument("a distribution needs at least 2 coefficients");
    if (!c_.allFinite() || std::abs(c_(0)) > 1e-14 || std::abs(c_(c_.size() - 1) - 1.0) > 1e-14)
        throw std::invalid_argument("distribution coefficients must start at 0 and end at 1");
    for (int k = 0; k <= 1000; ++k) {
        double v[3];
        evaluate(k / 1000.0, v);
        if (!(v[1] > 0)) throw std::invalid_argument("a distribution must be increasing (psi' > 0 on [0, 1])");
    }
}

void Distribution::evaluate(double s, double out[3]) const {
    if (identity()) {
        out[0] = s;
        out[1] = 1.0;
        out[2] = 0.0;
        return;
    }
    // de Casteljau on the coefficients and their first and second differences.
    const Index d = c_.size() - 1;
    auto casteljau = [s](Vector b) {
        for (Index r = b.size() - 1; r > 0; --r)
            for (Index k = 0; k < r; ++k) b(k) = (1 - s) * b(k) + s * b(k + 1);
        return b(0);
    };
    out[0] = casteljau(c_);
    const Vector d1 = static_cast<double>(d) * (c_.tail(d) - c_.head(d));
    out[1] = casteljau(d1);
    out[2] = d >= 2 ? casteljau(static_cast<double>(d - 1) * (d1.tail(d - 1) - d1.head(d - 1))) : 0.0;
}

namespace {

// The curve parameter s with (1 - w) psi_a(s) + w psi_b(s) = t, and its derivatives in t and w.
struct Blend {
    double s, s_t, s_w, s_tt, s_tw, s_ww;
};

Blend blend(const Distribution& a, const Distribution& b, double t, double w) {
    double pa[3], pb[3];
    auto G = [&](double s) {
        a.evaluate(s, pa);
        b.evaluate(s, pb);
        return (1 - w) * pa[0] + w * pb[0] - t;
    };
    double s = t;  // Newton from the identity guess (psi increasing: a unique root)
    for (int it = 0; it < 60; ++it) {
        const double g = G(s);
        const double D = (1 - w) * pa[1] + w * pb[1];
        const double step = g / D;
        s -= step;
        if (std::abs(step) < 1e-15) break;
    }
    G(s);
    const double D = (1 - w) * pa[1] + w * pb[1], D_s = (1 - w) * pa[2] + w * pb[2];
    const double Delta = pb[0] - pa[0], Delta_s = pb[1] - pa[1];
    Blend r;
    r.s = s;
    r.s_t = 1.0 / D;
    r.s_w = -Delta / D;
    r.s_tt = -D_s * r.s_t * r.s_t * r.s_t;
    r.s_tw = -r.s_t * r.s_t * (D_s * r.s_w + Delta_s);
    r.s_ww = -Delta_s * r.s_w * r.s_t - Delta * r.s_tw;  // d/dw of s_w = -Delta s_t
    return r;
}

}  // namespace

MappedGrid2D::Parameters MappedGrid2D::parameters(double xi, double eta) const {
    const Blend S = blend(distribution_.bottom, distribution_.top, xi, eta);   // s(xi; eta)
    const Blend E = blend(distribution_.left, distribution_.right, eta, xi);   // e(eta; xi)
    Parameters p;
    p.s = S.s;
    p.s_xi = S.s_t;
    p.s_eta = S.s_w;
    p.s_xixi = S.s_tt;
    p.s_xieta = S.s_tw;
    p.s_etaeta = S.s_ww;
    p.e = E.s;
    p.e_eta = E.s_t;
    p.e_xi = E.s_w;
    p.e_etaeta = E.s_tt;
    p.e_xieta = E.s_tw;
    p.e_xixi = E.s_ww;
    return p;
}

namespace {

// Bernstein polynomials B_k^d(t), k = 0..d, and their first two derivatives (any real t).
void bernstein_all(Index d, double t, Vector& b, Vector& b1, Vector& b2) {
    auto basis = [t](Index n) {
        Vector v = Vector::Zero(n + 1);
        if (n < 0) return v;
        v(0) = 1.0;
        for (Index r = 1; r <= n; ++r) {  // de Casteljau-style recursion B_k^r = (1 - t) B_k^{r-1} + t B_{k-1}^{r-1}
            for (Index k = r; k >= 1; --k) v(k) = (1 - t) * v(k) + t * v(k - 1);
            v(0) *= (1 - t);
        }
        return v;
    };
    b = basis(d);
    b1 = Vector::Zero(d + 1);
    b2 = Vector::Zero(d + 1);
    if (d >= 1) {
        const Vector m = basis(d - 1);
        for (Index k = 0; k <= d; ++k)
            b1(k) = static_cast<double>(d) * ((k >= 1 ? m(k - 1) : 0.0) - (k <= d - 1 ? m(k) : 0.0));
    }
    if (d >= 2) {
        const Vector m = basis(d - 2);
        for (Index k = 0; k <= d; ++k)
            b2(k) = static_cast<double>(d * (d - 1)) *
                    ((k >= 2 ? m(k - 2) : 0.0) - 2.0 * (k >= 1 && k - 1 <= d - 2 ? m(k - 1) : 0.0) + (k <= d - 2 ? m(k) : 0.0));
    }
}

}  // namespace

void MappedGrid2D::correction(double xi, double eta, double out[6][2]) const {
    for (int a = 0; a < 6; ++a) out[a][0] = out[a][1] = 0.0;
    if (correction_[0].size() == 0) return;
    const Index p = correction_[0].rows() + 1, q = correction_[0].cols() + 1;
    Vector bx, bx1, bx2, by, by1, by2;
    bernstein_all(p, xi, bx, bx1, bx2);
    bernstein_all(q, eta, by, by1, by2);
    for (int c = 0; c < 2; ++c) {
        const Matrix& C = correction_[c];  // interior coefficients: B_1..B_{p-1} x B_1..B_{q-1}
        const Vector ux = C.transpose() * bx.segment(1, p - 1), ux1 = C.transpose() * bx1.segment(1, p - 1),
                     ux2 = C.transpose() * bx2.segment(1, p - 1);
        const Vector ey = by.segment(1, q - 1), ey1 = by1.segment(1, q - 1), ey2 = by2.segment(1, q - 1);
        out[0][c] = ux.dot(ey);
        out[1][c] = ux1.dot(ey);
        out[2][c] = ux.dot(ey1);
        out[3][c] = ux2.dot(ey);
        out[4][c] = ux1.dot(ey1);
        out[5][c] = ux.dot(ey2);
    }
}

MappedGrid2D::MappedGrid2D(SplineCurve bottom, SplineCurve right, SplineCurve top, SplineCurve left, Index n_xi,
                           Index n_eta, const LineOptions& options, EdgeDistribution distribution, Matrix correction_x,
                           Matrix correction_y, std::optional<LineOptions> eta_options)
    : n_xi_(n_xi), n_eta_(n_eta), options_(options), eta_options_(eta_options.value_or(options)), bottom_(bottom),
      right_(right), top_(top), left_(left), distribution_(std::move(distribution)),
      correction_{std::move(correction_x), std::move(correction_y)},
      reference_({Line(Vector::LinSpaced(n_xi, 0.0, 1.0), options),
                  Line(Vector::LinSpaced(n_eta, 0.0, 1.0), eta_options.value_or(options))}) {
    Vector ends(2);
    ends << 0.0, 1.0;
    const Matrix b = bottom.evaluate(ends), r = right.evaluate(ends), t = top.evaluate(ends), l = left.evaluate(ends);
    const RowVector P00 = b.row(0), P10 = b.row(1), P01 = t.row(0), P11 = t.row(1);
    P00_ = P00;
    P10_ = P10;
    P01_ = P01;
    P11_ = P11;
    const double scale = std::max({b.cwiseAbs().maxCoeff(), t.cwiseAbs().maxCoeff(), 1.0});
    auto meets = [&](const RowVector& a, const RowVector& c) { return (a - c).cwiseAbs().maxCoeff() <= 1e-12 * scale; };
    if (!meets(l.row(0), P00) || !meets(l.row(1), P01) || !meets(r.row(0), P10) || !meets(r.row(1), P11))
        throw std::invalid_argument(
            "boundary curves must meet at the corners: bottom(0) = left(0), bottom(1) = right(0), "
            "top(0) = left(1), top(1) = right(1)");

    if (correction_[0].rows() != correction_[1].rows() || correction_[0].cols() != correction_[1].cols())
        throw std::invalid_argument("correction_x and correction_y must have the same shape");
    if (correction_[0].size() && !(correction_[0].allFinite() && correction_[1].allFinite()))
        throw std::invalid_argument("correction coefficients must be finite");
    // Curve parameters of every node (row-major), then the curves at those parameters.
    const Index N = n_xi * n_eta;
    std::vector<Parameters> par(static_cast<size_t>(N));
    Vector s_all(N), e_all(N);
    for (Index i = 0; i < n_xi; ++i)
        for (Index j = 0; j < n_eta; ++j) {
            const Parameters p = parameters(static_cast<double>(i) / static_cast<double>(n_xi - 1),
                                            static_cast<double>(j) / static_cast<double>(n_eta - 1));
            par[static_cast<size_t>(i * n_eta + j)] = p;
            s_all(i * n_eta + j) = std::clamp(p.s, 0.0, 1.0);
            e_all(i * n_eta + j) = std::clamp(p.e, 0.0, 1.0);
        }
    const Matrix B = bottom.evaluate(s_all), dB = bottom.evaluate(s_all, 1), ddB = bottom.evaluate(s_all, 2);
    const Matrix T = top.evaluate(s_all), dT = top.evaluate(s_all, 1), ddT = top.evaluate(s_all, 2);
    const Matrix L = left.evaluate(e_all), dL = left.evaluate(e_all, 1), ddL = left.evaluate(e_all, 2);
    const Matrix R = right.evaluate(e_all), dR = right.evaluate(e_all, 1), ddR = right.evaluate(e_all, 2);
    const RowVector twist = P00 - P10 - P01 + P11;

    for (Matrix* m : {&x_, &y_, &x_xi_, &x_eta_, &y_xi_, &y_eta_, &jacobian_, &xi_x_, &xi_y_, &eta_x_, &eta_y_, &xi_xx_,
                      &xi_xy_, &xi_yy_, &eta_xx_, &eta_xy_, &eta_yy_})
        m->resize(n_xi, n_eta);
    for (Index i = 0; i < n_xi; ++i)
        for (Index j = 0; j < n_eta; ++j) {
            const Index k = i * n_eta + j;
            const Parameters& p = par[static_cast<size_t>(k)];
            const double s = s_all(k), e = e_all(k);
            RowVector X = (1 - e) * B.row(k) + e * T.row(k) + (1 - s) * L.row(k) + s * R.row(k) -
                                ((1 - s) * (1 - e) * P00 + s * (1 - e) * P10 + (1 - s) * e * P01 + s * e * P11);
            const RowVector Cs = (1 - e) * dB.row(k) + e * dT.row(k) - L.row(k) + R.row(k) -
                                 (-(1 - e) * P00 + (1 - e) * P10 - e * P01 + e * P11);
            const RowVector Ce = -B.row(k) + T.row(k) + (1 - s) * dL.row(k) + s * dR.row(k) -
                                 (-(1 - s) * P00 - s * P10 + (1 - s) * P01 + s * P11);
            // Second derivatives of the transfinite map: L, R are linear in s and B, T in e.
            const RowVector Css = (1 - e) * ddB.row(k) + e * ddT.row(k);
            const RowVector Cee = (1 - s) * ddL.row(k) + s * ddR.row(k);
            const RowVector Cse = -dB.row(k) + dT.row(k) - dL.row(k) + dR.row(k) - twist;
            // Chain rule through the curve parameters s(xi, eta), e(xi, eta) of the edge distributions.
            RowVector Xs = Cs * p.s_xi + Ce * p.e_xi;
            RowVector Xe = Cs * p.s_eta + Ce * p.e_eta;
            RowVector Xss = Css * (p.s_xi * p.s_xi) + 2.0 * Cse * (p.s_xi * p.e_xi) + Cee * (p.e_xi * p.e_xi) +
                                  Cs * p.s_xixi + Ce * p.e_xixi;
            RowVector Xee = Css * (p.s_eta * p.s_eta) + 2.0 * Cse * (p.s_eta * p.e_eta) + Cee * (p.e_eta * p.e_eta) +
                                  Cs * p.s_etaeta + Ce * p.e_etaeta;
            RowVector Xse = Css * (p.s_xi * p.s_eta) + Cse * (p.s_xi * p.e_eta + p.s_eta * p.e_xi) +
                            Cee * (p.e_xi * p.e_eta) + Cs * p.s_xieta + Ce * p.e_xieta;
            if (correction_[0].size()) {
                double c[6][2];
                correction(static_cast<double>(i) / static_cast<double>(n_xi - 1),
                           static_cast<double>(j) / static_cast<double>(n_eta - 1), c);
                for (int a = 0; a < 2; ++a) {
                    X(a) += c[0][a];
                    Xs(a) += c[1][a];
                    Xe(a) += c[2][a];
                    Xss(a) += c[3][a];
                    Xse(a) += c[4][a];
                    Xee(a) += c[5][a];
                }
            }
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

void MappedGrid2D::map(double xi, double eta, double X[2], double Xs[2], double Xe[2]) const {
    const Parameters p = parameters(xi, eta);
    const double s = p.s, e = p.e;
    Vector xs(1), es(1);
    xs << s;
    es << e;
    const RowVector B = bottom_.evaluate_extended(xs).row(0), T = top_.evaluate_extended(xs).row(0);
    const RowVector L = left_.evaluate_extended(es).row(0), R = right_.evaluate_extended(es).row(0);
    const RowVector dB = bottom_.evaluate_extended(xs, 1).row(0), dT = top_.evaluate_extended(xs, 1).row(0);
    const RowVector dL = left_.evaluate_extended(es, 1).row(0), dR = right_.evaluate_extended(es, 1).row(0);
    const RowVector P = (1 - e) * B + e * T + (1 - s) * L + s * R -
                        ((1 - s) * (1 - e) * P00_ + s * (1 - e) * P10_ + (1 - s) * e * P01_ + s * e * P11_);
    const RowVector Ps = (1 - e) * dB + e * dT - L + R - (-(1 - e) * P00_ + (1 - e) * P10_ - e * P01_ + e * P11_);
    const RowVector Pe = -B + T + (1 - s) * dL + s * dR - (-(1 - s) * P00_ - s * P10_ + (1 - s) * P01_ + s * P11_);
    double c[6][2];
    correction(xi, eta, c);
    for (int k = 0; k < 2; ++k) {
        X[k] = P(k) + c[0][k];
        Xs[k] = Ps(k) * p.s_xi + Pe(k) * p.e_xi + c[1][k];
        Xe[k] = Ps(k) * p.s_eta + Pe(k) * p.e_eta + c[2][k];
    }
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
    if (order == 1) return chain_rule(order_x, order_y, f_s, f_e);
    const Field f_ss = reference_.partial(f, {2, 0}), f_ee = reference_.partial(f, {0, 2});
    const Field f_se = reference_.partial(f, {1, 1});
    return chain_rule(order_x, order_y, f_s, f_e, &f_ss, &f_se, &f_ee);
}

Field MappedGrid2D::chain_rule(int order_x, int order_y, const Field& f_s, const Field& f_e, const Field* f_ss_p,
                               const Field* f_se_p, const Field* f_ee_p) const {
    const int order = order_x + order_y;
    if (order == 1) {
        if (order_x == 1) return combine({{&xi_x_, &f_s}, {&eta_x_, &f_e}});
        return combine({{&xi_y_, &f_s}, {&eta_y_, &f_e}});
    }
    if (order != 2 || !f_ss_p || !f_se_p || !f_ee_p) throw std::invalid_argument("chain_rule: orders 1 or 2 with the needed partials");
    const Field &f_ss = *f_ss_p, &f_se = *f_se_p, &f_ee = *f_ee_p;
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
