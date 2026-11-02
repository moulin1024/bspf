// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/bspline.hpp"

#include <cmath>
#include <stdexcept>

namespace bspf {

namespace {

// Zero for a repeated-knot denominator (pybspf.basis._divide).
inline double divide(double a, double b) { return b != 0.0 ? a / b : 0.0; }

Vector linspace(double a, double b, int n) {
    Vector v(n);
    if (n == 1) { v(0) = a; return v; }
    const double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) v(i) = a + step * i;
    v(n - 1) = b;
    return v;
}

}  // namespace

Vector open_knots(double a, double b, int degree, int n_basis, double clustering) {
    if (degree < 0 || n_basis < degree + 1) throw std::invalid_argument("open_knots needs n_basis >= degree + 1");
    Vector u = linspace(-1.0, 1.0, n_basis - degree + 1);
    if (clustering != 0.0) {
        const double scale = std::tanh(clustering);
        for (Eigen::Index i = 0; i < u.size(); ++i) u(i) = std::tanh(clustering * u(i)) / scale;
    }
    Vector knots(n_basis + degree + 1);
    for (int i = 0; i < degree; ++i) { knots(i) = a; knots(knots.size() - 1 - i) = b; }
    for (Eigen::Index i = 0; i < u.size(); ++i) knots(degree + i) = a + (b - a) * (u(i) + 1.0) / 2.0;
    // Exact clamped endpoints: affine rounding could place the last break beyond b.
    knots(degree) = a;
    knots(degree + u.size() - 1) = b;
    return knots;
}

Vector endpoint_scaled_knots(double a, double b, int degree, int n_basis, double spacing) {
    if (degree < 1) throw std::invalid_argument("degree must be a positive integer");
    if (n_basis < 3 * degree + 1) throw std::invalid_argument("endpoint-scaled knots require n_basis >= 3*degree+1");
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(spacing) || spacing <= 0)
        throw std::invalid_argument("finite endpoints and positive finite spacing required");
    if (!(2 * degree * spacing < b - a))
        throw std::invalid_argument("endpoint knot regions must leave a nonempty central interval");
    const int central = n_basis - 3 * degree - 1;
    Vector knots(n_basis + degree + 1);
    int k = 0;
    for (int i = 0; i <= degree; ++i) knots(k++) = a;
    const double left_last = a + spacing * degree, right_first = b - spacing * degree;
    for (int i = 1; i <= degree; ++i) knots(k++) = a + spacing * i;
    Vector middle = linspace(left_last, right_first, central + 2);
    for (int i = 1; i <= central; ++i) knots(k++) = middle(i);
    for (int i = degree; i >= 1; --i) knots(k++) = b - spacing * i;
    for (int i = 0; i <= degree; ++i) knots(k++) = b;
    return knots;
}

Matrix basis_matrix(const Vector& knots, const Vector& x, int degree, int derivative) {
    if (degree < 0 || derivative < 0) throw std::invalid_argument("degree and derivative must be nonnegative");
    const Eigen::Index nk = knots.size(), nx = x.size(), n_basis = nk - degree - 1;
    if (n_basis < 1) throw std::invalid_argument("knot vector too short for this degree");
    if (derivative > degree) return Matrix::Zero(nx, n_basis);
    if (derivative > 0) {
        // d/dx B_{i,p} = p/(t_{i+p}-t_i) B_{i,p-1} - p/(t_{i+p+1}-t_{i+1}) B_{i+1,p-1}.
        const Matrix lower = basis_matrix(knots, x, degree - 1, derivative - 1);
        Matrix out(nx, n_basis);
        for (Eigen::Index i = 0; i < n_basis; ++i) {
            const double left = divide(degree, knots(i + degree) - knots(i));
            const double right = divide(degree, knots(i + degree + 1) - knots(i + 1));
            out.col(i) = lower.col(i) * left - lower.col(i + 1) * right;
        }
        return out;
    }
    // Degree 0: half-open intervals; the right end of the knot span belongs to the
    // last nonempty interval (left-hand limit).
    Eigen::Index last_interval = -1;
    for (Eigen::Index j = 0; j < nk - 1; ++j)
        if (knots(j) < knots(nk - 1)) last_interval = j;
    Matrix values = Matrix::Zero(nx, nk - 1);
    for (Eigen::Index r = 0; r < nx; ++r) {
        const double xr = x(r);
        if (xr == knots(nk - 1)) {
            if (last_interval >= 0) values(r, last_interval) = 1.0;
            continue;
        }
        for (Eigen::Index j = 0; j < nk - 1; ++j)
            if (xr >= knots(j) && xr < knots(j + 1)) values(r, j) = 1.0;
    }
    for (int p = 1; p <= degree; ++p) {
        Matrix next(nx, nk - 1 - p);
        for (Eigen::Index j = 0; j < nk - 1 - p; ++j) {
            const double dl = knots(j + p) - knots(j), dr = knots(j + p + 1) - knots(j + 1);
            for (Eigen::Index r = 0; r < nx; ++r) {
                const double left = divide(x(r) - knots(j), dl);
                const double right = divide(knots(j + p + 1) - x(r), dr);
                next(r, j) = left * values(r, j) + right * values(r, j + 1);
            }
        }
        values.swap(next);
    }
    return values;
}

std::pair<Vector, Matrix> primitive_coefficients(const Vector& knots, const Matrix& coefficients,
                                                 int degree) {
    const Eigen::Index n_basis = knots.size() - degree - 1;
    if (coefficients.rows() != n_basis) throw std::invalid_argument("coefficient count does not match the knots");
    Matrix c(n_basis + 1, coefficients.cols());
    c.row(0).setZero();
    for (Eigen::Index i = 0; i < n_basis; ++i) {
        const double width = (knots(i + degree + 1) - knots(i)) / (degree + 1);
        c.row(i + 1) = c.row(i) + coefficients.row(i) * width;
    }
    Vector t(knots.size() + 2);
    t(0) = knots(0);
    t.segment(1, knots.size()) = knots;
    t(t.size() - 1) = knots(knots.size() - 1);
    return {t, c};
}

Matrix spline_primitive(Vector knots, Matrix coefficients, const Vector& x, int degree, int order) {
    for (int i = 0; i < order; ++i) {
        auto [t, c] = primitive_coefficients(knots, coefficients, degree);
        knots = std::move(t);
        coefficients = std::move(c);
        ++degree;
    }
    return basis_matrix(knots, x, degree) * coefficients;
}

}  // namespace bspf
