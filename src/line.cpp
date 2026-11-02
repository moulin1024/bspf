// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/line.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "bspf/bspline.hpp"
#include "bspf/op.hpp"
#include "line_data.hpp"

namespace bspf {

namespace {

using cd = std::complex<double>;

void require_integer(const char* name, int value, int minimum) {
    if (value < minimum) throw std::invalid_argument(std::string(name) + " must be an integer >= " + std::to_string(minimum));
}

// (i w)^k by repeated multiplication (std::pow(0, 0) on complex is NaN).
cd i_power(double w, int k) {
    cd value(1.0, 0.0);
    for (int i = 0; i < k; ++i) value *= cd(0.0, w);
    return value;
}

// Weights turning half-spectrum sums into the real part of full-spectrum sums:
// conjugate pairs count twice; the zero mode and (even N) the Nyquist mode once.
Vector pair_weights(Eigen::Index n) {
    const Eigen::Index h = n / 2 + 1;
    Vector weight = Vector::Constant(h, 2.0);
    weight(0) = 1.0;
    if (n % 2 == 0) weight(h - 1) = 1.0;
    return weight;
}

// Half-spectrum rows (P, h) of the k-th derivative of exp(i w (t - x0)) at points t,
// with the pair weights and 1/N included: Re(E R) is the remainder's value there.
ComplexMatrix mode_rows(const detail::LineData& d, const Vector& points, int k) {
    const Eigen::Index n = d.size(), h = n / 2 + 1;
    const Vector weight = pair_weights(n);
    ComplexMatrix E(points.size(), h);
    for (Eigen::Index j = 0; j < h; ++j) {
        const double w = d.half_omega(j);
        const cd scale = i_power(w, k) * (weight(j) / static_cast<double>(n));
        for (Eigen::Index i = 0; i < points.size(); ++i) E(i, j) = scale * std::exp(cd(0.0, w * (points(i) - d.x(0))));
    }
    return E;
}

// 1 / (i w)^k on the half spectrum, zero mode dropped (periodic primitive).
Eigen::VectorXcd inverse_power(const detail::LineData& d, int k) {
    Eigen::VectorXcd m(d.half_omega.size());
    m(0) = 0.0;
    for (Eigen::Index j = 1; j < m.size(); ++j) m(j) = 1.0 / i_power(d.half_omega(j), k);
    return m;
}

void check_points(const detail::LineData& d, const Vector& points, const char* what) {
    const double lo = d.x(0), hi = d.x(d.size() - 1);
    for (Eigen::Index i = 0; i < points.size(); ++i)
        if (!(points(i) >= lo && points(i) <= hi))
            throw std::invalid_argument(std::string(what) + " must lie in [x[0], x[-1]]");
}

}  // namespace

Line::Line(const Vector& x, const LineOptions& o) {
    require_integer("degree", o.degree, 1);
    const int q = o.constraint_order.value_or(o.degree - 1);
    require_integer("constraint_order", q, 0);
    const Eigen::Index N = x.size();
    if (N < 2 || !x.allFinite()) throw std::invalid_argument("x must be a finite real 1D grid with at least two points");
    const double spacing0 = x(1) - x(0);
    for (Eigen::Index i = 0; i + 1 < N; ++i) {
        const double s = x(i + 1) - x(i);
        if (!(s > 0) || std::abs(s - spacing0) > 1e-13 + 1e-10 * std::abs(spacing0))
            throw std::invalid_argument("x must be strictly increasing and uniformly spaced");
    }
    int p = 0, modes = 0;
    if (o.endpoint_method == EndpointMethod::Chebyshev) {
        modes = o.chebyshev_modes.value_or(o.degree + 1);
        require_integer("chebyshev_modes", modes, std::max(1, q));
        require_integer("chebyshev_penalty_power", o.chebyshev_penalty_power, 0);
        if (!std::isfinite(o.chebyshev_alpha) || o.chebyshev_alpha < 0)
            throw std::invalid_argument("chebyshev_alpha must be finite and nonnegative");
        p = o.boundary_points.value_or(static_cast<int>(std::min<Eigen::Index>(N, 2 * modes)));
        require_integer("boundary_points", p, std::max(2, modes));
    } else {
        if (o.chebyshev_modes) throw std::invalid_argument("chebyshev_modes requires endpoint_method='chebyshev'");
        p = o.boundary_points.value_or(o.degree);
        require_integer("boundary_points", p, 1);
    }
    if (q > o.degree || p < std::max(1, q) || p > N)
        throw std::invalid_argument("require q <= degree and max(1, q) <= boundary_points <= grid size");
    if (!std::isfinite(o.lam) || o.lam < 0 || !std::isfinite(o.clustering) || o.clustering < 0)
        throw std::invalid_argument("lam and clustering must be finite and nonnegative");

    auto d = std::make_shared<detail::LineData>();
    int n_basis = 0;
    if (!o.knots) {
        n_basis = o.n_basis.value_or(static_cast<int>(std::min<Eigen::Index>(4 * (o.degree + 1), N)));
        require_integer("n_basis", n_basis, o.degree + 1);
        d->knots = open_knots(x(0), x(N - 1), o.degree, n_basis, o.clustering);
    } else {
        if (o.n_basis) throw std::invalid_argument("provide knots or n_basis, not both");
        const Vector& t = *o.knots;
        const Eigen::Index deg = o.degree;
        if (t.size() < 2 * (deg + 1) || !t.allFinite()) throw std::invalid_argument("knots must be a finite nondecreasing clamped knot vector");
        for (Eigen::Index i = 0; i + 1 < t.size(); ++i)
            if (t(i + 1) < t(i)) throw std::invalid_argument("knots must be a finite nondecreasing clamped knot vector");
        for (Eigen::Index i = 0; i <= deg; ++i)
            if (t(i) != x(0) || t(t.size() - 1 - i) != x(N - 1)) throw std::invalid_argument("knots must be clamped to the grid endpoints");
        for (Eigen::Index i = deg + 1; i < t.size() - deg - 1; ++i) {
            if (t(i) <= x(0) || t(i) >= x(N - 1) || (i > deg + 1 && t(i) <= t(i - 1)))
                throw std::invalid_argument("interior knots must be strictly increasing (simple knots)");
        }
        d->knots = t;
        n_basis = static_cast<int>(t.size()) - o.degree - 1;
    }
    if (n_basis < 2 * q) throw std::invalid_argument("n_basis must be >= 2*constraint_order");
    if (o.lam == 0 && n_basis > N) throw std::invalid_argument("unregularized fit requires n_basis <= number of samples");

    d->x = x;
    d->degree = o.degree;
    d->q = q;
    d->p = p;
    d->lam = o.lam;
    d->dx = spacing0;
    d->basis = basis_matrix(d->knots, x, o.degree, 0);
    d->weights = Vector::Constant(N, spacing0);
    d->weights(0) = d->weights(N - 1) = spacing0 / 2;

    Vector ends(2);
    ends << x(0), x(N - 1);
    d->constraint.resize(2 * q, n_basis);
    for (int k = 0; k < q; ++k) {
        const Matrix at_ends = basis_matrix(d->knots, ends, o.degree, k);
        d->constraint.row(k) = at_ends.row(0);
        d->constraint.row(q + k) = at_ends.row(1);
    }
    d->blocks = o.endpoint_method == EndpointMethod::Chebyshev
                    ? chebyshev_blocks(x(p - 1) - x(0), q, p, modes, o.chebyshev_alpha, o.chebyshev_penalty_power)
                    : finite_difference_blocks(spacing0, q, p);

    // Constrained least squares through the KKT system [[2(G + lam I), -C^T], [C, 0]].
    const Matrix weighted = d->basis.transpose() * d->weights.asDiagonal();  // (n_basis, N)
    d->gram = weighted * d->basis;
    const Eigen::Index m = 2 * q, nk = n_basis + m;
    d->kkt = Matrix::Zero(nk, nk);
    d->kkt.topLeftCorner(n_basis, n_basis) = 2 * (d->gram + o.lam * Matrix::Identity(n_basis, n_basis));
    d->kkt.topRightCorner(n_basis, m) = -d->constraint.transpose();
    d->kkt.bottomLeftCorner(m, n_basis) = d->constraint;
    const Eigen::PartialPivLU<Matrix> lu(d->kkt);
    const auto& U = lu.matrixLU();
    for (Eigen::Index i = 0; i < nk; ++i)
        if (!std::isfinite(U(i, i)) || U(i, i) == 0.0)
            throw std::invalid_argument("singular KKT system; use fewer basis functions or positive lam");

    d->jet_map = Matrix::Zero(m, N);
    if (q) {
        d->jet_map.topLeftCorner(q, p) = d->blocks.left;
        d->jet_map.bottomRightCorner(q, p) = d->blocks.right;
    }
    // Fit: c = K (2 W B^T f) + fit_jets g with K, fit_jets the top rows of KKT^{-1}.
    // The projection is applied first (banded), as a per-call solve would; the
    // jets stay a separate factor (folding J in cancels entries of size dx^-(q-1)).
    Matrix rhs = Matrix::Zero(nk, nk);
    rhs.setIdentity();
    const Matrix inverse = lu.solve(rhs);
    d->fit_inverse = inverse.topLeftCorner(n_basis, n_basis);
    d->fit_jets = inverse.topRightCorner(n_basis, m);
    d->fit_samples = d->fit_inverse * (2 * weighted);
    d->basis_banded = *detail::Banded::from(d->basis, o.degree + 1);
    d->projection = *detail::Banded::from(2 * weighted, N);

    // numpy.fft.fftfreq ordering: [0, 1, ..., ceil(N/2)-1, -floor(N/2), ..., -1] / (N dx).
    d->omega.resize(N);
    const Eigen::Index positive = (N - 1) / 2 + 1;
    for (Eigen::Index j = 0; j < N; ++j) {
        const double f = j < positive ? static_cast<double>(j) : static_cast<double>(j - N);
        d->omega(j) = 2 * M_PI * f / (N * spacing0);
    }
    d->half_omega = d->omega.head(N / 2 + 1);
    d->fft = std::make_unique<detail::FFT>(static_cast<int>(N));
    d_ = std::move(d);
}

int Line::size() const { return static_cast<int>(d_->size()); }
int Line::degree() const { return d_->degree; }
int Line::n_basis() const { return static_cast<int>(d_->n_basis()); }
int Line::constraint_order() const { return d_->q; }
int Line::boundary_points() const { return d_->p; }
double Line::lam() const { return d_->lam; }
double Line::spacing() const { return d_->dx; }
const Vector& Line::x() const { return d_->x; }
const Vector& Line::knots() const { return d_->knots; }
const Vector& Line::weights() const { return d_->weights; }
const Vector& Line::omega() const { return d_->omega; }
const Matrix& Line::basis() const { return d_->basis; }
const Matrix& Line::constraint() const { return d_->constraint; }
const BoundaryBlocks& Line::boundary_blocks() const { return d_->blocks; }
const Matrix& Line::gram() const { return d_->gram; }
const Matrix& Line::kkt() const { return d_->kkt; }
const Matrix& Line::fit_samples() const { return d_->fit_samples; }
const Matrix& Line::fit_jets() const { return d_->fit_jets; }
const Matrix& Line::jet_map() const { return d_->jet_map; }

Op Line::derivative(int order, bool correction) const {
    if (order < 0) throw std::invalid_argument("derivative order must be nonnegative");
    const detail::LineData& d = *d_;
    Op op;
    op.rows_ = op.cols_ = d.size();
    op.line_ = d_;
    op.set_spline(basis_matrix(d.knots, d.x, d.degree, order));
    op.correction_ = correction;
    if (order == 0) {
        op.pass_remainder_ = true;
    } else {
        op.multiplier_.resize(d.half_omega.size());
        for (Eigen::Index j = 0; j < op.multiplier_.size(); ++j)
            op.multiplier_(j) = i_power(d.half_omega(j), order) / static_cast<double>(d.size());
    }
    return op;
}

Op Line::evaluate(const Vector& points, int derivative, bool correction) const {
    if (derivative < 0) throw std::invalid_argument("derivative order must be nonnegative");
    const detail::LineData& d = *d_;
    check_points(d, points, "evaluation points");
    Op op;
    op.rows_ = points.size();
    op.cols_ = d.size();
    op.line_ = d_;
    op.set_spline(basis_matrix(d.knots, points, d.degree, derivative));
    op.correction_ = correction;
    const ComplexMatrix E = mode_rows(d, points, derivative);
    op.modes_real_ = E.real();
    op.modes_imag_ = E.imag();
    return op;
}

Op Line::quadrature() const { return quadrature(d_->x(0), d_->x(d_->size() - 1)); }

Op Line::quadrature(double a, double b) const {
    const detail::LineData& d = *d_;
    Vector bounds(2);
    bounds << a, b;
    check_points(d, bounds, "integration bounds");
    const Eigen::Index n = d.size(), h = n / 2 + 1, nb = d.n_basis();
    Op op;
    op.rows_ = 1;
    op.cols_ = n;
    op.line_ = d_;
    const Matrix ends = spline_primitive(d.knots, Matrix::Identity(nb, nb), bounds, d.degree, 1);
    op.set_spline(ends.row(1) - ends.row(0));
    // Integral of exp(i w (x - x0)) over [a, b] = L exp(i w ((a+b)/2 - x0)) sinc(w L / 2 pi).
    const double length = b - a, middle = (a + b) / 2 - d.x(0);
    const Vector weight = pair_weights(n);
    ComplexMatrix kernel(1, h);
    for (Eigen::Index j = 0; j < h; ++j) {
        const double w = d.half_omega(j), arg = w * length / 2;
        const double sinc = arg == 0.0 ? 1.0 : std::sin(arg) / arg;
        kernel(0, j) = length * std::exp(cd(0.0, w * middle)) * sinc * (weight(j) / static_cast<double>(n));
    }
    op.modes_real_ = kernel.real();
    op.modes_imag_ = kernel.imag();
    return op;
}

Op Line::primitive(int order) const {
    if (order != 1 && order != 2) throw std::invalid_argument("primitive order must be 1 or 2");
    const detail::LineData& d = *d_;
    const Eigen::Index n = d.size(), h = n / 2 + 1, nb = d.n_basis();
    Op op;
    op.rows_ = op.cols_ = n;
    op.line_ = d_;
    op.set_spline(spline_primitive(d.knots, Matrix::Identity(nb, nb), d.x, d.degree, order));
    // Periodic primitive irfft(R / (i w)^order), shifted to vanish at x0, plus the
    // polynomial part of the mean. The shifts are rank-one terms of the form
    // lift * Re(row R): the periodic primitives' values at x0 and the mean R_0 / N.
    const Eigen::VectorXcd multiplier = inverse_power(d, order);
    op.multiplier_ = multiplier / static_cast<double>(n);
    const Vector weight = pair_weights(n);
    auto value_at_x0 = [&](const Eigen::VectorXcd& m) {
        Eigen::RowVectorXcd row(h);
        for (Eigen::Index j = 0; j < h; ++j) row(j) = m(j) * (weight(j) / static_cast<double>(n));
        return row;
    };
    Eigen::RowVectorXcd mean = Eigen::RowVectorXcd::Zero(h);
    mean(0) = 1.0 / static_cast<double>(n);
    const Vector offset = (d.x.array() - d.x(0)).matrix();
    ComplexMatrix E(order + 1, h);
    op.lift_.resize(n, order + 1);
    op.lift_.col(0).setOnes();
    if (order == 1) {
        E.row(0) = -value_at_x0(multiplier);
        E.row(1) = mean;
        op.lift_.col(1) = offset;
    } else {
        E.row(0) = -value_at_x0(multiplier);
        E.row(1) = -value_at_x0(inverse_power(d, 1));
        E.row(2) = mean;
        op.lift_.col(1) = offset;
        op.lift_.col(2) = offset.array().square() / 2;
    }
    op.modes_real_ = E.real();
    op.modes_imag_ = E.imag();
    return op;
}

Op Line::jets() const {
    const detail::LineData& d = *d_;
    Op op;
    op.rows_ = d.jet_map.rows();
    op.cols_ = d.size();
    op.line_ = d_;
    op.jets_only_ = true;
    return op;
}

}  // namespace bspf
