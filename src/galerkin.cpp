// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/galerkin.hpp"

#include <fftw3.h>

#include <cmath>
#include <complex>
#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <random>

#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <stdexcept>
#include <vector>

#include "bspf/bspline.hpp"
#include "line_data.hpp"

namespace bspf {

namespace {

using cd = std::complex<double>;
using Index = Eigen::Index;
using ComplexVector = Eigen::VectorXcd;

using ComplexMatrix = Eigen::MatrixXcd;

// Batched complex FFTs of fixed length over the columns of (n, m) matrices.
// Plans are made once per batch size under FFTW's planner mutex; execution on
// new arrays is thread-safe.
class ComplexFFT {
public:
    explicit ComplexFFT(int n) : n_(n) {}
    ~ComplexFFT() {
        std::lock_guard<std::mutex> lock(detail::planner_mutex());
        for (auto& entry : plans_) {
            fftw_destroy_plan(entry.second.first);
            fftw_destroy_plan(entry.second.second);
        }
    }
    ComplexFFT(const ComplexFFT&) = delete;
    ComplexFFT& operator=(const ComplexFFT&) = delete;

    // sum_j x_j exp(-+ 2 pi i j k / n) per column, unnormalized.
    ComplexMatrix forward(const ComplexMatrix& x) const { return run(true, x); }
    ComplexMatrix backward(const ComplexMatrix& x) const { return run(false, x); }

private:
    ComplexMatrix run(bool forward, const ComplexMatrix& x) const {
        ComplexMatrix in = x, out(n_, x.cols());
        if (x.cols() == 0) return out;
        const auto plan = plans(x.cols());
        fftw_execute_dft(forward ? plan.first : plan.second, reinterpret_cast<fftw_complex*>(in.data()),
                         reinterpret_cast<fftw_complex*>(out.data()));
        return out;
    }
    std::pair<fftw_plan, fftw_plan> plans(Index m) const {
        std::lock_guard<std::mutex> own(mutex_);
        auto it = plans_.find(m);
        if (it != plans_.end()) return it->second;
        std::lock_guard<std::mutex> lock(detail::planner_mutex());
        const int n = n_, howmany = static_cast<int>(m);
        fftw_complex* a = fftw_alloc_complex(static_cast<size_t>(n) * m);
        fftw_complex* b = fftw_alloc_complex(static_cast<size_t>(n) * m);
        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        fftw_plan f = fftw_plan_many_dft(1, &n, howmany, a, nullptr, 1, n, b, nullptr, 1, n, FFTW_FORWARD, flags);
        fftw_plan g = fftw_plan_many_dft(1, &n, howmany, a, nullptr, 1, n, b, nullptr, 1, n, FFTW_BACKWARD, flags);
        fftw_free(a);
        fftw_free(b);
        if (!f || !g) throw std::runtime_error("FFTW planning failed");
        return plans_.emplace(m, std::make_pair(f, g)).first->second;
    }
    int n_;
    mutable std::mutex mutex_;
    mutable std::map<Index, std::pair<fftw_plan, fftw_plan>> plans_;
};

// Batched real FFTs (r2c / c2r, unnormalized) over the columns of (n, m) matrices.
class RealFFT {
public:
    explicit RealFFT(int n) : n_(n) {}
    ~RealFFT() {
        std::lock_guard<std::mutex> lock(detail::planner_mutex());
        for (auto& entry : plans_) {
            fftw_destroy_plan(entry.second.first);
            fftw_destroy_plan(entry.second.second);
        }
    }
    RealFFT(const RealFFT&) = delete;
    RealFFT& operator=(const RealFFT&) = delete;

    ComplexMatrix forward(const Matrix& x) const {
        ComplexMatrix out(n_ / 2 + 1, x.cols());
        if (x.cols()) fftw_execute_dft_r2c(plans(x.cols()).first, const_cast<double*>(x.data()),
                                           reinterpret_cast<fftw_complex*>(out.data()));
        return out;
    }
    Matrix backward(ComplexMatrix x) const {  // overwrites its own copy
        Matrix out(n_, x.cols());
        if (x.cols()) fftw_execute_dft_c2r(plans(x.cols()).second, reinterpret_cast<fftw_complex*>(x.data()), out.data());
        return out;
    }

private:
    std::pair<fftw_plan, fftw_plan> plans(Index m) const {
        std::lock_guard<std::mutex> own(mutex_);
        auto it = plans_.find(m);
        if (it != plans_.end()) return it->second;
        std::lock_guard<std::mutex> lock(detail::planner_mutex());
        const int n = n_, h = n / 2 + 1, howmany = static_cast<int>(m);
        double* a = fftw_alloc_real(static_cast<size_t>(n) * m);
        fftw_complex* b = fftw_alloc_complex(static_cast<size_t>(h) * m);
        const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
        fftw_plan f = fftw_plan_many_dft_r2c(1, &n, howmany, a, nullptr, 1, n, b, nullptr, 1, h, flags);
        fftw_plan g = fftw_plan_many_dft_c2r(1, &n, howmany, b, nullptr, 1, h, a, nullptr, 1, n, flags | FFTW_DESTROY_INPUT);
        fftw_free(a);
        fftw_free(b);
        if (!f || !g) throw std::runtime_error("FFTW planning failed");
        return plans_.emplace(m, std::make_pair(f, g)).first->second;
    }
    int n_;
    mutable std::mutex mutex_;
    mutable std::map<Index, std::pair<fftw_plan, fftw_plan>> plans_;
};

#ifdef BSPF_HAVE_CBLAS
extern "C" void cblas_dgemm(int order, int transa, int transb, int m, int n, int k, double alpha, const double* a,
                            int lda, const double* b, int ldb, double beta, double* c, int ldc);
#endif

// C = A B + beta C for contiguous column-major operands (C sized by the caller
// when beta != 0). Uses an optimized BLAS when available.
void gemm(const Matrix& A, const Matrix& B, Matrix& C, double beta) {
#ifdef BSPF_HAVE_CBLAS
    if (beta == 0.0) C.resize(A.rows(), B.cols());
    if (A.rows() && B.cols() && A.cols())
        cblas_dgemm(102, 111, 111, static_cast<int>(A.rows()), static_cast<int>(B.cols()), static_cast<int>(A.cols()), 1.0,
                    A.data(), static_cast<int>(A.rows()), B.data(), static_cast<int>(B.rows()), beta, C.data(),
                    static_cast<int>(C.rows()));
    else if (beta == 0.0) C.setZero();
#else
    if (beta == 0.0) C.noalias() = A * B;
    else C.noalias() += A * B;
#endif
}

cd i_power(double w, int k) {
    cd value(1.0, 0.0);
    for (int i = 0; i < k; ++i) value *= cd(0.0, w);
    return value;
}

// Gauss-Legendre nodes and weights on [-1, 1] (Golub-Welsch).
void gauss_legendre(int q, Vector& nodes, Vector& weights) {
    Matrix J = Matrix::Zero(q, q);
    for (int i = 1; i < q; ++i) J(i, i - 1) = J(i - 1, i) = i / std::sqrt(4.0 * i * i - 1.0);
    Eigen::SelfAdjointEigenSolver<Matrix> eig(J);
    nodes = eig.eigenvalues();
    weights = 2 * eig.eigenvectors().row(0).array().square().transpose();
}

}  // namespace

struct GalerkinLine::Data {
    Line line;
    Index n = 0, K = 0, nb = 0, q = 0, p = 0;
    double dx = 0.0;
    std::vector<double> rho;   // weights of the extended frequencies m = -K..K
    std::vector<double> omega; // their angular frequencies
    std::unique_ptr<ComplexFFT> fft_n;   // setup only
    std::unique_ptr<RealFFT> real_fft;
    // Quadrature of the setup products: Q Gauss points per sample cell at fixed
    // offsets s_q (so the trigonometric part is a circulant map, applied by FFT),
    // and cells that contain a knot split at the knots (direct "extra" points).
    Index quadrature = 0;
    std::vector<ComplexVector> shift[2];  // per offset: (i w)^k exp(i w s dx) / N, half spectrum
    std::vector<Vector> cell_weights;     // per offset: Gauss weights; 0 on split cells and at j = N-1
    std::vector<Eigen::SparseMatrix<double>> spline_at[2];     // per offset: B^(k)(x_j + s dx) (N, n_basis), row N-1 zero
    Matrix extra_trig[2];                 // (T r)^(k) at the extra points (n_extra, N)
    Matrix extra_spline[2];               // B^(k) at the extra points (n_extra, n_basis)
    Vector extra_weights;
    Vector points, weights;
    std::vector<std::vector<Index>> sample_rows;
    Eigen::Matrix2d wall_schur_inverse;
    Matrix U;                  // low-rank interior mass: M_II = dx (I + U diag(s) U^T)
    Vector shrink;             // s / (1 + s)
    Matrix M_IB;               // (N - 2, 2)
    Matrix extension_walls;    // M_II^{-1} M_IB (N - 2, 2): E u = u_I + this * u_B
    Vector periodic_multiplier;  // -w^2 / N on the half spectrum
    Matrix basis_second;       // B'' at the grid (N, n_basis)
    Matrix P, Qt;              // W - (D_strong)_I ~ P Qt, (N - 2, s) and (s, N)
    double amplification = 0;  // max |F|: rough samples' rounding is amplified by this
    std::vector<Index> fold;   // extended frequency index -> FFT bin

    explicit Data(const Line& l) : line(l) {}

    // c = F u: the constrained fit with estimated jets, kept as separate factors.
    Matrix fit(const Matrix& u) const {
        Matrix c = line.fit_samples() * u;
        if (q) {
            Matrix g(2 * q, u.cols());
            g.topRows(q) = line.boundary_blocks().left * u.topRows(p);
            g.bottomRows(q) = line.boundary_blocks().right * u.bottomRows(p);
            c.noalias() += line.fit_jets() * g;
        }
        return c;
    }

    // F^T c.
    Matrix fit_transpose(const Matrix& c) const {
        Matrix u = line.fit_samples().transpose() * c;
        if (q) {
            const Matrix g = line.fit_jets().transpose() * c;
            u.topRows(p).noalias() += line.boundary_blocks().left.transpose() * g.topRows(q);
            u.bottomRows(p).noalias() += line.boundary_blocks().right.transpose() * g.bottomRows(q);
        }
        return u;
    }

    Matrix evaluate(const Matrix& u, int k) const {
        const Matrix c = fit(u);
        const Matrix r = u - line.basis() * c;
        const ComplexMatrix R = real_fft->forward(r);
        Matrix result(points.size(), u.cols());
        for (Index qi = 0; qi < quadrature; ++qi) {
            Matrix v = real_fft->backward(shift[k][qi].asDiagonal() * R);
            v.noalias() += spline_at[k][qi] * c;
            for (Index j = 0; j < n; ++j)
                if (sample_rows[qi][j] >= 0) result.row(sample_rows[qi][j]) = v.row(j);
        }
        if (extra_weights.size()) {
            result.bottomRows(extra_weights.size()).noalias() = extra_trig[k] * r;
            result.bottomRows(extra_weights.size()).noalias() += extra_spline[k] * c;
        }
        return result;
    }

    Matrix project(const Matrix& values, int k) const {
        const Index m = values.cols();
        ComplexMatrix adjoint = ComplexMatrix::Zero(n / 2 + 1, m);
        Matrix spline = Matrix::Zero(nb, m);
        for (Index qi = 0; qi < quadrature; ++qi) {
            Matrix v = Matrix::Zero(n, m);
            for (Index j = 0; j < n; ++j)
                if (sample_rows[qi][j] >= 0)
                    v.row(j) = cell_weights[qi](j) * values.row(sample_rows[qi][j]);
            spline.noalias() += spline_at[k][qi].transpose() * v;
            adjoint += shift[k][qi].conjugate().asDiagonal() * real_fft->forward(v);
        }
        Matrix t = real_fft->backward(std::move(adjoint));
        if (extra_weights.size()) {
            const Matrix v = extra_weights.asDiagonal() * values.bottomRows(extra_weights.size());
            t.noalias() += extra_trig[k].transpose() * v;
            spline.noalias() += extra_spline[k].transpose() * v;
        }
        spline.noalias() -= line.basis().transpose() * t;
        return t + fit_transpose(spline);
    }

    // The Galerkin product (M for k = 0, K for k = 1) of columns u, by quadrature
    // of the trial field itself: with c = F u and r = u - B c,
    //     Phi u   = B_P^(k) c + T_k r              (values of u_h^(k) at the points)
    //     Phi^T y = T_k^T y + F^T (B_P^(k)T y - B^T T_k^T y)
    // and K u = Phi^T W Phi u. Every intermediate is O(1) (u_h is bounded), so the
    // fit's large endpoint-jet factors amplify rounding once (eps |F|), not twice.
    // The rule is exact to rounding: 16 Gauss points per cell, split at knots.
    Matrix product(const Matrix& u, int k) const {
        const Index m = u.cols();
        const Matrix c = fit(u);
        Matrix r = u;
        r.noalias() -= line.basis() * c;
        const ComplexMatrix R = real_fft->forward(r);
        ComplexMatrix adjoint = ComplexMatrix::Zero(R.rows(), m);
        Matrix spline = Matrix::Zero(nb, m);
        for (Index q0 = 0; q0 < quadrature; ++q0) {
            Matrix v = real_fft->backward(shift[k][q0].asDiagonal() * R);  // T_k r at x_j + s dx
            v.noalias() += spline_at[k][q0] * c;
            v = cell_weights[q0].asDiagonal() * v;
            spline.noalias() += spline_at[k][q0].transpose() * v;
            adjoint += shift[k][q0].conjugate().asDiagonal() * real_fft->forward(v);
        }
        Matrix t = real_fft->backward(std::move(adjoint));
        if (extra_weights.size()) {
            Matrix v = extra_trig[k] * r;
            v.noalias() += extra_spline[k] * c;
            v = extra_weights.asDiagonal() * v;
            t.noalias() += extra_trig[k].transpose() * v;
            spline.noalias() += extra_spline[k].transpose() * v;
        }
        spline.noalias() -= line.basis().transpose() * t;
        return t + fit_transpose(spline);
    }

    // M_II^{-1} x = (x - U diag(s / (1 + s)) U^T x) / dx.
    Matrix mass_solve(const Matrix& x) const {
        Matrix y = x;
        if (U.cols()) y.noalias() -= U * (shrink.asDiagonal() * (U.transpose() * x));
        return y / dx;
    }

    Matrix full_mass_solve(const Matrix& rhs) const {
        Matrix result(n, rhs.cols());
        const Matrix z = mass_solve(rhs.middleRows(1, n - 2));
        Matrix boundary(2, rhs.cols());
        boundary.row(0) = rhs.row(0);
        boundary.row(1) = rhs.row(n - 1);
        boundary.noalias() -= M_IB.transpose() * z;
        const Matrix walls = wall_schur_inverse * boundary;
        result.row(0) = walls.row(0);
        result.row(n - 1) = walls.row(1);
        result.middleRows(1, n - 2) = z - extension_walls * walls;
        return result;
    }

    Matrix interior_mass(const Matrix& x) const {  // M_II x for interior vectors x
        Matrix full = Matrix::Zero(n, x.cols());
        full.middleRows(1, n - 2) = x;
        return product(full, 0).middleRows(1, n - 2);
    }

    // Exact interior operator W u = -M_II^{-1} (K u)_I and its transpose
    // W^T y = -K [0; M_II^{-1} y; 0] (M and K are symmetric).
    Matrix exact_second_derivative(const Matrix& u) const { return -mass_solve(product(u, 1).middleRows(1, n - 2)); }

    Matrix exact_transpose(const Matrix& y) const {
        Matrix full = Matrix::Zero(n, y.cols());
        full.middleRows(1, n - 2) = mass_solve(y);
        return -product(full, 1);
    }

    // The periodic spectral second derivative irfft(-w^2 rfft(u)) (N, m): symmetric circulant.
    Matrix periodic(const Matrix& u) const {
        ComplexMatrix R = real_fft->forward(u);
        R = periodic_multiplier.asDiagonal() * R;
        return real_fft->backward(std::move(R));
    }

    // The strong (collocation) BSPF second derivative D u = B'' c + D_per r, with
    // c = F u and r = u - B c, and its exact transpose
    // D^T y = F^T B''^T y + (I - F^T B^T) D_per y (D_per is symmetric).
    void strong_into(const Matrix& u, Matrix& y) const {
        Matrix cc;
        gemm(line.fit_samples(), u, cc, 0.0);
        if (q) {
            Matrix g(2 * q, u.cols());
            g.topRows(q) = line.boundary_blocks().left * u.topRows(p);
            g.bottomRows(q) = line.boundary_blocks().right * u.bottomRows(p);
            gemm(line.fit_jets(), g, cc, 1.0);
        }
        Matrix r = u;
        Matrix bc;
        gemm(line.basis(), cc, bc, 0.0);
        r -= bc;
        y = periodic(r);
        gemm(basis_second, cc, y, 1.0);
    }

    Matrix strong_transpose(const Matrix& y) const {
        const Matrix t = periodic(y);
        Matrix a = basis_second.transpose() * y;
        a.noalias() -= line.basis().transpose() * t;
        return t + fit_transpose(a);
    }

    // Applied form: W u = (D_strong u)_I + P (Q^T u), the low-rank P Q^T compressing
    // W - (D_strong)_I. The strong operator is accurate on smooth non-periodic
    // fields, so the correction only carries the small Galerkin-collocation
    // difference (a periodic base would have to cancel O(N^2) wrap Gibbs values).
    void second_derivative_into(const Matrix& u, Matrix& y) const {
        strong_into(u, y);
        if (!P.cols()) return;
        Matrix z;
        gemm(Qt, u, z, 0.0);
#ifdef BSPF_HAVE_CBLAS
        cblas_dgemm(102, 111, 111, static_cast<int>(n - 2), static_cast<int>(u.cols()), static_cast<int>(P.cols()), 1.0,
                    P.data(), static_cast<int>(n - 2), z.data(), static_cast<int>(z.rows()), 1.0, y.data() + 1,
                    static_cast<int>(n));
#else
        y.middleRows(1, n - 2).noalias() += P * z;
#endif
    }

    Matrix second_derivative(const Matrix& u) const {
        Matrix y;
        second_derivative_into(u, y);
        return y.middleRows(1, n - 2);
    }

    Matrix extension(const Matrix& u) const {
        Matrix out = u.middleRows(1, n - 2);
        out.noalias() += extension_walls.col(0) * u.row(0);
        out.noalias() += extension_walls.col(1) * u.row(n - 1);
        return out;
    }
};

GalerkinLine::GalerkinLine(const Line& line, const GalerkinOptions& options) {
    auto d = std::make_shared<Data>(line);
    const Index n = line.size(), degree = line.degree();
    if (n < 4) throw std::invalid_argument("the Galerkin line needs at least 4 points");
    if (options.quadrature_order < 4) throw std::invalid_argument("quadrature_order must be at least 4");
    d->n = n;
    d->K = n / 2;
    d->nb = line.n_basis();
    d->q = line.constraint_order();
    d->p = line.boundary_points();
    d->dx = line.spacing();
    const double x0 = line.x()(0), length = line.x()(n - 1) - x0;
    const Index K = d->K, F = 2 * K + 1;

    // Extended frequencies m = -K..K; for even N the Nyquist term R cos(...) is
    // split evenly over m = +-N/2, as the real part of the full sum gives.
    d->rho.assign(F, 1.0);
    d->omega.resize(F);
    for (Index m = -K; m <= K; ++m) d->omega[m + K] = 2 * M_PI * static_cast<double>(m) / (n * d->dx);
    if (n % 2 == 0) d->rho[0] = d->rho[F - 1] = 0.5;
    d->fold.resize(F);
    for (Index m = -K; m <= K; ++m) d->fold[m + K] = ((m % n) + n) % n;

    d->fft_n = std::make_unique<ComplexFFT>(static_cast<int>(n));
    d->real_fft = std::make_unique<RealFFT>(static_cast<int>(n));

    // Quadrature for the setup products. Cells [x_j, x_{j+1}] holding a knot are
    // split there (the spline pieces have kinks); all others use 16 Gauss points
    // at common offsets, so their trigonometric values come from one FFT pair each.
    const Vector& t = line.knots();
    const int Q = options.quadrature_order;
    Vector gn, gw;
    gauss_legendre(Q, gn, gw);
    d->quadrature = Q;
    std::vector<std::vector<double>> breaks(n - 1);
    for (Index i = 0; i < t.size(); ++i) {
        const double pos = (t(i) - x0) / d->dx;
        if (!(pos > 0 && pos < n - 1)) continue;
        const Index j = static_cast<Index>(std::floor(pos));
        const double frac = pos - static_cast<double>(j);
        if (frac < 1e-12 || frac > 1 - 1e-12) continue;  // a knot on a sample needs no split
        if (breaks[j].empty() || breaks[j].back() != t(i)) breaks[j].push_back(t(i));
    }
    for (int k = 0; k <= 1; ++k) {
        d->shift[k].resize(Q);
        d->spline_at[k].resize(Q);
    }
    d->cell_weights.resize(Q);
    d->sample_rows.assign(Q, std::vector<Index>(n, -1));
    std::vector<double> sample_points, sample_weights;
    for (Index qi = 0; qi < Q; ++qi) {
        const double s = (gn(qi) + 1) / 2;  // offset within the cell
        Vector w = Vector::Constant(n, d->dx * gw(qi) / 2);
        w(n - 1) = 0.0;
        for (Index j = 0; j + 1 < n; ++j)
            if (!breaks[j].empty()) w(j) = 0.0;
        d->cell_weights[qi] = w;
        for (Index j = 0; j < n; ++j) {
            if (w(j) == 0) continue;
            d->sample_rows[qi][j] = static_cast<Index>(sample_points.size());
            sample_points.push_back(x0 + (static_cast<double>(j) + s) * d->dx);
            sample_weights.push_back(w(j));
        }
        Vector points(n);
        for (Index j = 0; j < n; ++j) points(j) = std::min(x0 + (static_cast<double>(j) + s) * d->dx, x0 + length);
        for (int k = 0; k <= 1; ++k) {
            Matrix spline = basis_matrix(t, points, static_cast<int>(degree), k);
            spline.row(n - 1).setZero();
            // Keep compact support in all evaluation/projection kernels. No
            // nonzero entries are dropped; this matters when n_basis grows.
            d->spline_at[k][qi] = spline.sparseView(0.0, 0.0);
            ComplexVector shift(n / 2 + 1);
            for (Index j = 0; j <= n / 2; ++j) {
                const double w_j = 2 * M_PI * static_cast<double>(j) / (n * d->dx);
                // The Nyquist term R cos(w xi) (real part of the full sum) shifts to
                // (-1)^j R w^k cos(pi s + k pi / 2).
                shift(j) = (n % 2 == 0 && j == n / 2) ? cd(std::pow(w_j, k) * std::cos(M_PI * s + k * M_PI / 2), 0.0)
                                                      : i_power(w_j, k) * std::exp(cd(0.0, w_j * s * d->dx));
            }
            d->shift[k][qi] = shift / static_cast<double>(n);
        }
    }
    std::vector<double> extra_points, extra_weights;
    for (Index j = 0; j + 1 < n; ++j) {
        if (breaks[j].empty()) continue;
        std::vector<double> edges{x0 + static_cast<double>(j) * d->dx};
        std::vector<double> inner = breaks[j];
        std::sort(inner.begin(), inner.end());
        edges.insert(edges.end(), inner.begin(), inner.end());
        edges.push_back(x0 + static_cast<double>(j + 1) * d->dx);
        for (size_t e = 0; e + 1 < edges.size(); ++e)
            for (Index qi = 0; qi < Q; ++qi) {
                extra_points.push_back(edges[e] + (edges[e + 1] - edges[e]) * (gn(qi) + 1) / 2);
                extra_weights.push_back((edges[e + 1] - edges[e]) * gw(qi) / 2);
            }
    }
    d->extra_weights = Eigen::Map<const Vector>(extra_weights.data(), static_cast<Index>(extra_weights.size()));
    const Vector extra = Eigen::Map<const Vector>(extra_points.data(), static_cast<Index>(extra_points.size()));
    sample_points.insert(sample_points.end(), extra_points.begin(), extra_points.end());
    sample_weights.insert(sample_weights.end(), extra_weights.begin(), extra_weights.end());
    d->points = Eigen::Map<const Vector>(sample_points.data(), static_cast<Index>(sample_points.size()));
    d->weights = Eigen::Map<const Vector>(sample_weights.data(), static_cast<Index>(sample_weights.size()));
    for (int k = 0; k <= 1; ++k) {
        d->extra_spline[k] = basis_matrix(t, extra, static_cast<int>(degree), k);
        d->extra_trig[k].resize(extra.size(), n);
        for (Index i = 0; i < extra.size(); ++i) {
            ComplexVector folded = ComplexVector::Zero(n);
            for (Index m = -K; m <= K; ++m) {
                const Index e = m + K;
                folded(d->fold[e]) += d->rho[e] * i_power(d->omega[e], k) * std::exp(cd(0.0, d->omega[e] * (extra(i) - x0)));
            }
            d->extra_trig[k].row(i) = (d->fft_n->forward(ComplexMatrix(folded)).col(0).real() / static_cast<double>(n)).transpose();
        }
    }

    // The fit F = fit_samples + fit_jets J amplifies rounding on rough samples by
    // about max |F|, which grows like (N / n_basis)^(q-1) with the endpoint jets.
    // Directions below that noise floor cannot be resolved: the randomized
    // solvers below truncate there instead of fitting noise.
    d->amplification = (line.fit_samples() + line.fit_jets() * line.jet_map()).cwiseAbs().maxCoeff();
    const double noise = 4 * std::numeric_limits<double>::epsilon() * d->amplification;

    // Low-rank interior mass about dx I: randomized eigensolver of Z = M_II / dx - I,
    // driven by fast mass products only (M is symmetric).
    const Index ni = n - 2;
    auto Z = [&](const Matrix& x) { return Matrix(d->interior_mass(x) / d->dx - x); };
    std::mt19937_64 rng(20261103);
    std::normal_distribution<double> normal;
    for (Index probes = std::min<Index>(ni, 64);; probes = std::min<Index>(ni, 2 * probes)) {
        Matrix omega(ni, probes);
        for (Index i = 0; i < omega.size(); ++i) omega.data()[i] = normal(rng);
        Eigen::HouseholderQR<Matrix> qr(Z(omega));
        Matrix Q = qr.householderQ() * Matrix::Identity(ni, probes);
        Eigen::HouseholderQR<Matrix> qr2(Z(Q));  // one power iteration
        Q = qr2.householderQ() * Matrix::Identity(ni, probes);
        const Matrix ZQ = Z(Q);
        Eigen::SelfAdjointEigenSolver<Matrix> eig(Q.transpose() * ZQ);
        std::vector<Index> keep;
        for (Index i = 0; i < probes; ++i)
            if (std::abs(eig.eigenvalues()(i)) > std::max(options.mass_tolerance, noise)) keep.push_back(i);
        if (static_cast<Index>(keep.size()) < probes || probes == ni) {
            d->U.resize(ni, static_cast<Index>(keep.size()));
            d->shrink.resize(static_cast<Index>(keep.size()));
            const Matrix vectors = Q * eig.eigenvectors();
            for (size_t i = 0; i < keep.size(); ++i) {
                const double s = eig.eigenvalues()(keep[i]);
                if (!(1 + s > 0)) throw std::runtime_error("Galerkin mass is not positive definite");
                d->U.col(static_cast<Index>(i)) = vectors.col(keep[i]);
                d->shrink(static_cast<Index>(i)) = s / (1 + s);
            }
            break;
        }
    }

    // Wall coupling of the mass: (M e_0)_I and (M e_{N-1})_I.
    Matrix walls = Matrix::Zero(n, 2);
    walls(0, 0) = walls(n - 1, 1) = 1.0;
    const Matrix wall_mass = d->product(walls, 0);
    d->M_IB = wall_mass.middleRows(1, ni);
    d->extension_walls = d->mass_solve(d->M_IB);
    Eigen::Matrix2d wall_block;
    wall_block.row(0) = wall_mass.row(0);
    wall_block.row(1) = wall_mass.row(n - 1);
    Eigen::Matrix2d schur = wall_block - d->M_IB.transpose() * d->extension_walls;
    schur = (0.5 * (schur + schur.transpose())).eval();
    Eigen::LLT<Eigen::Matrix2d> wall_factor(schur);
    if (wall_factor.info() != Eigen::Success) throw std::runtime_error("Galerkin wall mass Schur complement is not positive definite");
    d->wall_schur_inverse = wall_factor.solve(Eigen::Matrix2d::Identity());

    // Compress L = W - (D_strong)_I by a randomized SVD with power iterations,
    // using only fast products with L and L^T. Its rank is bounded in N;
    // singular values below operator_tolerance * w_max^2 are dropped.
    d->periodic_multiplier.resize(n / 2 + 1);
    for (Index j = 0; j <= n / 2; ++j) {
        const double w = 2 * M_PI * static_cast<double>(j) / (n * d->dx);
        d->periodic_multiplier(j) = -w * w / static_cast<double>(n);
    }
    d->basis_second = basis_matrix(t, line.x(), static_cast<int>(degree), 2);
    const double scale = std::pow(M_PI / d->dx, 2);
    auto L = [&](const Matrix& x) {
        Matrix strong;
        d->strong_into(x, strong);
        return Matrix(d->exact_second_derivative(x) - strong.middleRows(1, ni));
    };
    auto Lt = [&](const Matrix& y) {
        Matrix full = Matrix::Zero(n, y.cols());
        full.middleRows(1, ni) = y;
        return Matrix(d->exact_transpose(y) - d->strong_transpose(full));
    };
    auto orth = [](const Matrix& a) {
        Eigen::HouseholderQR<Matrix> qr(a);
        return Matrix(qr.householderQ() * Matrix::Identity(a.rows(), std::min(a.rows(), a.cols())));
    };
    for (Index probes = std::min<Index>(ni, 64);; probes = std::min<Index>(ni, 2 * probes)) {
        Matrix omega(n, probes);
        for (Index i = 0; i < omega.size(); ++i) omega.data()[i] = normal(rng);
        Matrix Y = L(omega);
        for (int iteration = 0; iteration < 2; ++iteration) Y = L(orth(Lt(orth(Y))));
        const Matrix Q = orth(Y);                         // (ni, k)
        const Matrix B = Lt(Q).transpose();               // (k, n) = Q^T L
        Eigen::BDCSVD<Matrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(B);
        const Vector& sigma = svd.singularValues();
        Index keep = 0;
        while (keep < sigma.size() && sigma(keep) > std::max(options.operator_tolerance, noise) * scale) ++keep;
        if (keep < sigma.size() || probes == ni) {
            d->P = Q * svd.matrixU().leftCols(keep) * sigma.head(keep).asDiagonal();
            d->Qt = svd.matrixV().leftCols(keep).transpose();
            break;
        }
    }
    d_ = std::move(d);
}

Index GalerkinLine::size() const { return d_->n; }
Index GalerkinLine::mass_rank() const { return d_->U.cols(); }
Index GalerkinLine::operator_rank() const { return d_->P.cols(); }
double GalerkinLine::fit_amplification() const { return d_->amplification; }
const Line& GalerkinLine::line() const { return d_->line; }
const Vector& GalerkinLine::quadrature_points() const { return d_->points; }
const Vector& GalerkinLine::quadrature_weights() const { return d_->weights; }

Matrix GalerkinLine::evaluate(const MatrixRef& u, int derivative) const {
    if (u.rows() != d_->n) throw std::invalid_argument("evaluate expects N rows");
    if (derivative < 0 || derivative > 1) throw std::invalid_argument("derivative must be 0 or 1");
    return d_->evaluate(u, derivative);
}

Matrix GalerkinLine::project(const MatrixRef& values, int derivative) const {
    if (values.rows() != d_->points.size()) throw std::invalid_argument("project expects one row per quadrature point");
    if (derivative < 0 || derivative > 1) throw std::invalid_argument("derivative must be 0 or 1");
    return d_->project(values, derivative);
}

Matrix GalerkinLine::solve_mass(const MatrixRef& rhs, bool interior) const {
    if (rhs.rows() != (interior ? d_->n - 2 : d_->n)) throw std::invalid_argument("incorrect mass load row count");
    return interior ? d_->mass_solve(rhs) : d_->full_mass_solve(rhs);
}

Matrix GalerkinLine::mass(const MatrixRef& u) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    return d_->product(u, 0);
}

Matrix GalerkinLine::stiffness(const MatrixRef& u) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    return d_->product(u, 1);
}

Matrix GalerkinLine::second_derivative(const MatrixRef& u) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    return d_->second_derivative(u);
}

void GalerkinLine::second_derivative_into(const Matrix& u, Matrix& y) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    d_->second_derivative_into(u, y);
}

const Matrix& GalerkinLine::wall_extension() const { return d_->extension_walls; }

Matrix GalerkinLine::exact_second_derivative(const MatrixRef& u) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    return d_->exact_second_derivative(u);
}

Matrix GalerkinLine::extension(const MatrixRef& u) const {
    if (u.rows() != d_->n) throw std::invalid_argument("expected " + std::to_string(d_->n) + " samples per column");
    return d_->extension(u);
}

namespace {

template <class F>
void strided(const GalerkinLine::Data& d, F&& op, const double* in, Index in_stride, double* out, Index out_stride) {
    Matrix u(d.n, 1);
    for (Index i = 0; i < d.n; ++i) u(i, 0) = in[i * in_stride];
    const Matrix y = op(u);
    for (Index i = 0; i < y.rows(); ++i) out[i * out_stride] = y(i, 0);
}

}  // namespace

void GalerkinLine::second_derivative(const double* in, Index in_stride, double* out, Index out_stride) const {
    strided(*d_, [&](const Matrix& u) { return d_->second_derivative(u); }, in, in_stride, out, out_stride);
}

void GalerkinLine::extension(const double* in, Index in_stride, double* out, Index out_stride) const {
    strided(*d_, [&](const Matrix& u) { return d_->extension(u); }, in, in_stride, out, out_stride);
}

detail::GalerkinTables detail::galerkin_tables(const GalerkinLine& g) {
    const GalerkinLine::Data& d = *g.d_;
    GalerkinTables t;
    t.n = d.n;
    t.nb = d.nb;
    t.offsets = d.quadrature;
    t.points = d.points.size();
    // fit with the estimated jets folded in: c = F u
    const Matrix I = Matrix::Identity(d.n, d.n);
    t.fit = d.fit(I);
    t.basis = d.line.basis();
    for (int k = 0; k <= 1; ++k) {
        t.shift[k] = d.shift[k];
        t.spline[k].resize(d.quadrature);
        for (Index q = 0; q < d.quadrature; ++q) t.spline[k][q] = Matrix(d.spline_at[k][q]);
        t.extra_trig[k] = d.extra_trig[k];
        t.extra_spline[k] = d.extra_spline[k];
    }
    t.rows = d.sample_rows;
    t.weights = d.cell_weights;
    t.extra_weights = d.extra_weights;
    return t;
}

}  // namespace bspf

