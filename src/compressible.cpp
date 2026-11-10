// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/compressible.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <deque>
#include <stdexcept>

#include "bspf/entropy.hpp"
#include "bspf/euler.hpp"
#include "bspf/galerkin.hpp"

namespace bspf {
namespace {

using Index = Eigen::Index;
using Rows = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// One reference axis: SBP derivative, SAT lifting, SVV and the weak-form quadrature.
struct Axis {
    Index n = 0;
    Matrix D, Q, Minv;     // (n, n): Galerkin derivative, SVV operator, inverse quadrature mass
    Matrix L;              // (n, n): second derivative D (D + s0 e0^T - s1 eN^T) with weak Dirichlet lifting
    Matrix Mq, Cq;         // (n, n): quadrature mass P0^T W P0 and derivative pairing P1^T W P0
    Vector s0, s1;         // M^{-1} e_0, M^{-1} e_{n-1}
    Vector weights;        // int phi_i
    double radius = 0;     // spectral radius of advection with upwind SAT inflow
    double vradius = 0;    // spectral radius of D (D + s0 e0^T - s1 eN^T)
    Matrix P0, P1;         // (Q, n) trial functions and derivatives at the Gauss points
    Vector w;              // (Q) Gauss weights
};

double spectral_radius(const Matrix& A) {
    Eigen::EigenSolver<Matrix> es(A, false);
    return es.eigenvalues().cwiseAbs().maxCoeff();
}

Axis make_axis(const Line& line, const FlowOptions& o) {
    Axis a;
    a.n = line.size();
    const Index n = a.n;
    const Matrix I = Matrix::Identity(n, n);
    const GalerkinLine g(line);  // exact (16-point) integrals for the SBP pair (M, C)
    Matrix M = g.mass(I);
    M = 0.5 * (M + M.transpose()).eval();
    const Matrix C = g.project(g.evaluate(I, 1), 0);
    const Eigen::LLT<Matrix> llt(M);
    if (llt.info() != Eigen::Success) throw std::runtime_error("CompressibleSolver2D: Galerkin mass not SPD");
    a.D = llt.solve(C);
    a.s0 = llt.solve(I.col(0));
    a.s1 = llt.solve(I.col(n - 1));
    a.weights = M.colwise().sum().transpose();
    a.radius = std::max(spectral_radius(-a.D - a.s0 * I.row(0)), spectral_radius(a.D - a.s1 * I.row(n - 1)));
    a.L = a.D * (a.D + a.s0 * I.row(0) - a.s1 * I.row(n - 1));
    a.vradius = spectral_radius(a.L);
    // SVV in the Legendre-weighted eigenbasis: K_ij = int 4 s (1 - s) phi_i' phi_j'
    const Vector& qp = g.quadrature_points();
    const Vector& qw = g.quadrature_weights();
    const double x0 = line.x()(0), x1 = line.x()(n - 1);
    Vector weight(qp.size());
    for (Index k = 0; k < qp.size(); ++k) {
        const double s = (qp(k) - x0) / (x1 - x0);
        weight(k) = qw(k) * 4 * s * (1 - s);
    }
    const Matrix P1g = g.evaluate(I, 1);
    Matrix K = P1g.transpose() * weight.asDiagonal() * P1g;
    K = 0.5 * (K + K.transpose()).eval();
    const Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> es(K, M);
    const Matrix& V = es.eigenvectors();  // M-orthonormal, ascending eigenvalues
    Vector r = Vector::Zero(n);
    const Index kc = static_cast<Index>(o.svv_cutoff * static_cast<double>(n));
    for (Index k = kc; k < n; ++k) {
        const double t = static_cast<double>(k - kc) / static_cast<double>(std::max<Index>(n - 1 - kc, 1));
        r(k) = t * t;
    }
    a.Q = V * r.asDiagonal() * V.transpose() * M;
    // weak-form quadrature
    GalerkinOptions qo;
    qo.quadrature_order = o.quadrature;
    const GalerkinLine gq(line, qo);
    a.P0 = gq.evaluate(I, 0);
    a.P1 = gq.evaluate(I, 1);
    a.w = gq.quadrature_weights();
    Matrix Mq = a.P0.transpose() * a.w.asDiagonal() * a.P0;
    Mq = 0.5 * (Mq + Mq.transpose()).eval();
    a.Minv = Eigen::LLT<Matrix>(Mq).solve(I);
    a.Mq = Mq;
    a.Cq = a.P1.transpose() * a.w.asDiagonal() * a.P0;
    return a;
}

// out (n0, n1, k) = A applied along axis 0 or 1 of f (n0, n1, k), C-ordered.
void along(const Matrix& A, const double* f, Index n0, Index n1, Index k, int axis, double* out) {
    if (axis == 0) {
        Eigen::Map<const Rows> F(f, n0, n1 * k);
        Eigen::Map<Rows> O(out, n0, n1 * k);
        O.noalias() = A * F;
        return;
    }
#pragma omp parallel for schedule(static)
    for (Index i = 0; i < n0; ++i) {
        Eigen::Map<const Rows> F(f + i * n1 * k, n1, k);
        Eigen::Map<Rows> O(out + i * n1 * k, n1, k);
        O.noalias() = A * F;
    }
}

struct State {
    double rho, u, v, p, c;
};

}  // namespace

struct CompressibleSolver2D::Impl {
    MappedGrid2D grid;
    FlowOptions o;
    std::array<Boundary, 4> bc;
    Index n0 = 0, n1 = 0;
    Axis ax[2];
    // nodal discrete metrics (n0, n1)
    Matrix x_xi, x_eta, y_xi, y_eta, J, xi_x, xi_y, eta_x, eta_y, grad_xi, grad_eta;
    std::unique_ptr<WeakEuler2D> kernel;
    Matrix gcl[2];  // free-stream defect for unit x and y fluxes, (n0, n1)
    Matrix xq_xi, xq_eta, yq_xi, yq_eta;  // isoparametric metrics at the Gauss points (Qa, Qb)
    struct Face {
        int axis;          // quadrature axis along the face (1 for xi faces, 0 for eta faces)
        Index count;       // nodes on the face
        double sign;       // +1 low face, -1 high face
        Matrix P0;         // (Q, count)
        Vector w, nx, ny, area;
        Matrix points;     // (Q, 2)
    } face[4];
    std::vector<double> source;

    explicit Impl(const MappedGrid2D& g) : grid(g) {}

    Index node(int k, Index t) const {
        switch (k) {
            case 0: return t;                     // (0, t)
            case 1: return (n0 - 1) * n1 + t;     // (n0 - 1, t)
            case 2: return t * n1;                // (t, 0)
            default: return t * n1 + (n1 - 1);    // (t, n1 - 1)
        }
    }

    // Primitive state without checks (callers check positivity first; used in parallel loops).
    State raw(const double* q) const {
        const double g = o.gamma;
        State s;
        s.rho = q[0];
        s.u = q[1] / q[0];
        s.v = q[2] / q[0];
        s.p = (g - 1) * (q[3] - 0.5 * (q[1] * s.u + q[2] * s.v));
        s.c = std::sqrt(g * s.p / s.rho);
        return s;
    }

    State state(const double* q) const {
        const State s = raw(q);
        if (!(s.rho > 0) || !(s.p > 0)) throw std::domain_error("CompressibleSolver2D: nonpositive density or pressure");
        return s;
    }

    void require_positive(const double* U) const {
        bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
        for (Index m = 0; m < n0 * n1; ++m) {
            const double* q = U + 4 * m;
            const double p = (o.gamma - 1) * (q[3] - 0.5 * (q[1] * q[1] + q[2] * q[2]) / q[0]);
            if (!(q[0] > 0) || !(p > 0)) bad = true;
        }
        if (bad) throw std::domain_error("CompressibleSolver2D: nonpositive density or pressure");
    }

    void conservative(double rho, double u, double v, double p, double* q) const {
        q[0] = rho;
        q[1] = rho * u;
        q[2] = rho * v;
        q[3] = p / (o.gamma - 1) + 0.5 * rho * (u * u + v * v);
    }

    // normal Euler flux and wave speed
    double flux(const double* q, double nx, double ny, double* F) const {
        const State s = state(q);
        const double un = s.u * nx + s.v * ny;
        F[0] = q[0] * un;
        F[1] = q[1] * un + s.p * nx;
        F[2] = q[2] * un + s.p * ny;
        F[3] = (q[3] + s.p) * un;
        return std::abs(un) + s.c;
    }

    void rusanov(const double* L, const double* R, double nx, double ny, double* F) const {
        double FL[4], FR[4];
        const double a = std::max(flux(L, nx, ny, FL), flux(R, nx, ny, FR));
        for (int c = 0; c < 4; ++c) F[c] = 0.5 * (FL[c] + FR[c]) - 0.5 * a * (R[c] - L[c]);
    }

    void setup() {
        n0 = grid.shape()[0];
        n1 = grid.shape()[1];
        for (int a = 0; a < 2; ++a) ax[a] = make_axis(grid.reference().line(a), o);
        const Matrix& X = grid.x();
        const Matrix& Y = grid.y();
        x_xi = ax[0].D * X;
        y_xi = ax[0].D * Y;
        x_eta = X * ax[1].D.transpose();
        y_eta = Y * ax[1].D.transpose();
        J = x_xi.cwiseProduct(y_eta) - x_eta.cwiseProduct(y_xi);
        if (J.minCoeff() <= 0) throw std::invalid_argument("CompressibleSolver2D: discrete Jacobian not positive");
        xi_x = y_eta.cwiseQuotient(J);
        xi_y = -x_eta.cwiseQuotient(J);
        eta_x = -y_xi.cwiseQuotient(J);
        eta_y = x_xi.cwiseQuotient(J);
        grad_xi = (xi_x.array().square() + xi_y.array().square()).sqrt().matrix();
        grad_eta = (eta_x.array().square() + eta_y.array().square()).sqrt().matrix();
        // isoparametric metrics at the Gauss points
        const Axis& A = ax[0];
        const Axis& B = ax[1];
        xq_xi = A.P1 * X * B.P0.transpose();
        xq_eta = A.P0 * X * B.P1.transpose();
        yq_xi = A.P1 * Y * B.P0.transpose();
        yq_eta = A.P0 * Y * B.P1.transpose();
        const Matrix Jq = xq_xi.cwiseProduct(yq_eta) - xq_eta.cwiseProduct(yq_xi);
        if (Jq.minCoeff() <= 0)
            throw std::invalid_argument("CompressibleSolver2D: Jacobian not positive at the quadrature points");
        kernel = std::make_unique<WeakEuler2D>(A.P0, A.P1, A.w, B.P0, B.P1, B.w, xq_xi, xq_eta, yq_xi, yq_eta, o.gamma);
        kernel->set_entropy(o.entropy_variables);
        // faces
        for (int k = 0; k < 4; ++k) {
            Face& f = face[k];
            f.axis = k < 2 ? 1 : 0;
            const Axis& q = ax[f.axis];
            f.count = q.n;
            f.sign = k % 2 == 0 ? 1.0 : -1.0;
            f.P0 = q.P0;
            f.w = q.w;
            Vector fx(f.count), fy(f.count);
            for (Index t = 0; t < f.count; ++t) {
                const Index m = node(k, t);
                fx(t) = X(m / n1, m % n1);
                fy(t) = Y(m / n1, m % n1);
            }
            const Vector dx = q.P1 * fx, dy = q.P1 * fy;
            f.nx = k < 2 ? dy : Vector(-dy);
            f.ny = k < 2 ? Vector(-dx) : dx;
            f.area = (f.nx.array().square() + f.ny.array().square()).sqrt().matrix();
            f.nx = f.nx.cwiseQuotient(f.area);
            f.ny = f.ny.cwiseQuotient(f.area);
            f.points.resize(q.P0.rows(), 2);
            f.points.col(0) = q.P0 * fx;
            f.points.col(1) = q.P0 * fy;
        }
        // free-stream defect: weak residual of unit constant fluxes
        const Matrix Wa = A.w.asDiagonal(), Wb = B.w.asDiagonal();
        auto volume = [&](const Matrix& Fh, const Matrix& Gh) {
            return Matrix(A.P1.transpose() * Wa * Fh * Wb * B.P0 + A.P0.transpose() * Wa * Gh * Wb * B.P1);
        };
        gcl[0] = volume(yq_eta, -yq_xi);
        gcl[1] = volume(-xq_eta, xq_xi);
        for (int k = 0; k < 4; ++k) {
            const Face& f = face[k];
            for (int d = 0; d < 2; ++d) {
                const Vector flux = f.area.cwiseProduct(d == 0 ? f.nx : f.ny);
                const Vector proj = f.P0.transpose() * f.w.cwiseProduct(flux);
                for (Index t = 0; t < f.count; ++t) {
                    const Index m = node(k, t);
                    gcl[d](m / n1, m % n1) += f.sign * proj(t);
                }
            }
        }
    }

    // ---- exterior states at the face Gauss points
    void inlet(const Boundary& b, const double* q, double nx, double ny, double* out) const {
        const double g = o.gamma;
        const State s = state(q);
        const double Rm = s.u * nx + s.v * ny - 2 * s.c / (g - 1);
        const double c0sq = b.total_temperature;  // c^2 = T in these units
        const double A = (g + 1) / (g - 1), B = 2 * Rm, C = 0.5 * (g - 1) * Rm * Rm - c0sq;
        const double c = (-B + std::sqrt(std::max(B * B - 4 * A * C, 0.0))) / (2 * A);
        const double T = c * c;
        const double un = Rm + 2 * c / (g - 1);
        const double p = b.total_pressure * std::pow(T / b.total_temperature, g / (g - 1));
        conservative(g * p / T, un * nx, un * ny, p, out);
    }

    struct Viscous {
        std::vector<double> Fvh, Gvh, Fv, Gv;  // contravariant and physical, (n0, n1, 4)
    };

    Viscous viscous(const double* U) const {
        const Index nn = n0 * n1;
        const double g = o.gamma;
        std::vector<double> theta(nn * 3), a(nn * 3), c(nn * 3);
#pragma omp parallel for schedule(static)
        for (Index m = 0; m < nn; ++m) {
            const State s = raw(U + 4 * m);
            theta[3 * m] = s.u;
            theta[3 * m + 1] = s.v;
            theta[3 * m + 2] = g * s.p / s.rho;
        }
        along(ax[0].D, theta.data(), n0, n1, 3, 0, a.data());
        along(ax[1].D, theta.data(), n0, n1, 3, 1, c.data());
        for (int k = 0; k < 4; ++k) {
            const Boundary& b = bc[k];
            if (b.kind != BoundaryKind::NoSlipWall && b.kind != BoundaryKind::Exterior) continue;
            const Axis& lift = ax[k < 2 ? 0 : 1];
            const Vector& s = k % 2 == 0 ? lift.s0 : lift.s1;
            const double sign = k % 2 == 0 ? 1.0 : -1.0;  // d f ~ D f + s0 (f0 - f*) - s1 (fN - f*)
            for (Index t = 0; t < face[k].count; ++t) {
                const Index m = node(k, t);
                double jump[3] = {theta[3 * m], theta[3 * m + 1], 0.0};
                if (b.kind == BoundaryKind::Exterior) {
                    const double e4[4] = {b.nodes(t, 0), b.nodes(t, 1), b.nodes(t, 2), b.nodes(t, 3)};
                    const State e = state(e4);
                    jump[0] = theta[3 * m] - e.u;
                    jump[1] = theta[3 * m + 1] - e.v;
                    jump[2] = theta[3 * m + 2] - g * e.p / e.rho;
                }
                for (Index l = 0; l < lift.n; ++l) {
                    // nodes along the lifting axis through this face node
                    const Index mm = k < 2 ? l * n1 + t : t * n1 + l;
                    double* dst = (k < 2 ? a.data() : c.data()) + 3 * mm;
                    for (int q = 0; q < 3; ++q) dst[q] += sign * s(l) * jump[q];
                }
            }
        }
        Viscous v;
        v.Fvh.assign(nn * 4, 0.0);
        v.Gvh.assign(nn * 4, 0.0);
        v.Fv.assign(nn * 4, 0.0);
        v.Gv.assign(nn * 4, 0.0);
#pragma omp parallel for schedule(static)
        for (Index m = 0; m < nn; ++m) {
            const Index i = m / n1, j = m % n1;
            const double* ga = &a[3 * m];
            const double* gc = &c[3 * m];
            const double ux = xi_x(i, j) * ga[0] + eta_x(i, j) * gc[0], uy = xi_y(i, j) * ga[0] + eta_y(i, j) * gc[0];
            const double vx = xi_x(i, j) * ga[1] + eta_x(i, j) * gc[1], vy = xi_y(i, j) * ga[1] + eta_y(i, j) * gc[1];
            const double Tx = xi_x(i, j) * ga[2] + eta_x(i, j) * gc[2], Ty = xi_y(i, j) * ga[2] + eta_y(i, j) * gc[2];
            const double u = theta[3 * m], vv = theta[3 * m + 1], T = theta[3 * m + 2];
            const double mu = std::pow(T, o.viscosity_exponent) / o.reynolds;
            const double kappa = mu / ((g - 1) * o.prandtl);
            const double div = ux + vy;
            const double txx = mu * (2 * ux - 2.0 / 3.0 * div), tyy = mu * (2 * vy - 2.0 / 3.0 * div), txy = mu * (uy + vx);
            const double Fv[4] = {0, txx, txy, u * txx + vv * txy + kappa * Tx};
            const double Gv[4] = {0, txy, tyy, u * txy + vv * tyy + kappa * Ty};
            for (int q = 0; q < 4; ++q) {
                v.Fv[4 * m + q] = Fv[q];
                v.Gv[4 * m + q] = Gv[q];
                v.Fvh[4 * m + q] = y_eta(i, j) * Fv[q] - x_eta(i, j) * Gv[q];
                v.Gvh[4 * m + q] = -y_xi(i, j) * Fv[q] + x_xi(i, j) * Gv[q];
            }
        }
        return v;
    }

    void svv_rates(const double* U, double rates[2]) const {
        double lx = 0, le = 0;
#pragma omp parallel for reduction(max : lx, le) schedule(static)
        for (Index m = 0; m < n0 * n1; ++m) {
            const Index i = m / n1, j = m % n1;
            const State s = raw(U + 4 * m);
            lx = std::max(lx, std::abs(s.u * xi_x(i, j) + s.v * xi_y(i, j)) + s.c * grad_xi(i, j));
            le = std::max(le, std::abs(s.u * eta_x(i, j) + s.v * eta_y(i, j)) + s.c * grad_eta(i, j));
        }
        rates[0] = o.svv * ax[0].radius * lx;
        rates[1] = o.svv * ax[1].radius * le;
    }

    void rhs(const double* U, double* out) const {
        const Index nn = n0 * n1;
        const bool visc = o.reynolds > 0;
        require_positive(U);
        Viscous v;
        if (visc) v = viscous(U);
        std::vector<double> R(nn * 4);
        // entropy-variable trial field: nodal V(U) for every interpolation (volume and faces)
        std::vector<double> Vn;
        if (o.entropy_variables) {
            Vn.resize(nn * 4);
#pragma omp parallel for schedule(static)
            for (Index m = 0; m < nn; ++m) conservative_to_entropy(U + 4 * m, o.gamma, Vn.data() + 4 * m);
        }
        const double* trial = o.entropy_variables ? Vn.data() : U;
        kernel->volume(trial, nullptr, nullptr, R.data());
        if (visc) {
            // The viscous fluxes live in the trial space (formed at the nodes), so their weak divergence
            // is exactly the nodal tensor product (Ca x Mb) Fvh + (Ma x Cb) Gvh, no quadrature points.
            std::vector<double> t1(nn * 4), t2(nn * 4), t3(nn * 4);
            along(ax[0].Cq, v.Fvh.data(), n0, n1, 4, 0, t1.data());
            along(ax[1].Mq, t1.data(), n0, n1, 4, 1, t2.data());
            along(ax[0].Mq, v.Gvh.data(), n0, n1, 4, 0, t1.data());
            along(ax[1].Cq, t1.data(), n0, n1, 4, 1, t3.data());
            for (Index m = 0; m < nn * 4; ++m) R[m] -= t2[m] + t3[m];
        }
        // faces
        for (int k = 0; k < 4; ++k) {
            const Face& f = face[k];
            const Boundary& b = bc[k];
            const Index Q = f.P0.rows();
            Rows Ub(f.count, 4), Vb;
            for (Index t = 0; t < f.count; ++t)
                for (int c = 0; c < 4; ++c) Ub(t, c) = trial[4 * node(k, t) + c];
            Rows Uf = f.P0 * Ub;  // (Q, 4)
            if (o.entropy_variables)
                for (Index q = 0; q < Uf.rows(); ++q) {
                    double v[4] = {Uf(q, 0), Uf(q, 1), Uf(q, 2), Uf(q, 3)};
                    if (!entropy_to_conservative(v, o.gamma, Uf.row(q).data()))
                        throw std::domain_error("CompressibleSolver2D: invalid entropy variables at a face point");
                }
            Rows star(Q, 4);
            // outlet: back pressure from the supersonic part of the exit
            double back = b.back_pressure;
            std::vector<char> subsonic;
            if (b.kind == BoundaryKind::Outlet) {
                subsonic.assign(Q, 0);
                double sum = 0;
                int count = 0;
                for (Index q = 0; q < Q; ++q) {
                    const State s = state(Uf.row(q).data());
                    subsonic[q] = s.u * f.nx(q) + s.v * f.ny(q) < s.c;
                    if (!subsonic[q]) {
                        sum += s.p;
                        ++count;
                    }
                }
                if (count > 0) back = sum / count;
                else if (std::isnan(back) && std::any_of(subsonic.begin(), subsonic.end(), [](char x) { return x; }))
                    throw std::invalid_argument("CompressibleSolver2D: subsonic exit needs back_pressure");
            }
            for (Index q = 0; q < Q; ++q) {
                const double* in = Uf.row(q).data();
                const double nx = f.nx(q), ny = f.ny(q);
                double ext[4] = {in[0], in[1], in[2], in[3]};
                bool interior = false;
                switch (b.kind) {
                    case BoundaryKind::Inlet: inlet(b, in, nx, ny, ext); break;
                    case BoundaryKind::Outlet:
                        if (!subsonic[q]) {
                            interior = true;
                        } else {
                            const State s = state(in);
                            conservative(s.rho, s.u, s.v, back, ext);
                        }
                        break;
                    case BoundaryKind::SlipWall: {
                        const double mn = in[1] * nx + in[2] * ny;
                        ext[0] = in[0];
                        ext[1] = in[1] - 2 * mn * nx;
                        ext[2] = in[2] - 2 * mn * ny;
                        ext[3] = in[3];
                        break;
                    }
                    case BoundaryKind::NoSlipWall:
                        ext[0] = in[0];
                        ext[1] = -in[1];
                        ext[2] = -in[2];
                        ext[3] = in[3];
                        break;
                    case BoundaryKind::Exterior:
                        for (int c = 0; c < 4; ++c) ext[c] = b.points(q, c);
                        break;
                }
                double F[4];
                if (interior) {
                    flux(in, nx, ny, F);
                } else if (k % 2 == 0) {
                    rusanov(ext, in, nx, ny, F);
                } else {
                    rusanov(in, ext, nx, ny, F);
                }
                for (int c = 0; c < 4; ++c) star(q, c) = f.area(q) * F[c];
            }
            if (visc) {
                const std::vector<double>& src = k < 2 ? v.Fvh : v.Gvh;
                Rows Vn(f.count, 4);
                for (Index t = 0; t < f.count; ++t)
                    for (int c = 0; c < 4; ++c) Vn(t, c) = src[4 * node(k, t) + c];
                Rows vis = f.P0 * Vn;
                if (b.kind == BoundaryKind::NoSlipWall || b.kind == BoundaryKind::SlipWall) vis.col(3).setZero();
                star -= vis;
            }
            const Rows proj = f.P0.transpose() * (f.w.asDiagonal() * star);  // (count, 4)
            for (Index t = 0; t < f.count; ++t)
                for (int c = 0; c < 4; ++c) R[4 * node(k, t) + c] += f.sign * proj(t, c);
        }
        // free-stream correction with the nodal physical fluxes
#pragma omp parallel for schedule(static)
        for (Index m = 0; m < nn; ++m) {
            const Index i = m / n1, j = m % n1;
            const double* q = U + 4 * m;
            const State s = raw(q);
            double F[4] = {q[1], q[1] * s.u + s.p, q[1] * s.v, s.u * (q[3] + s.p)};
            double G[4] = {q[2], q[2] * s.u, q[2] * s.v + s.p, s.v * (q[3] + s.p)};
            if (visc)
                for (int c = 0; c < 4; ++c) {
                    F[c] -= v.Fv[4 * m + c];
                    G[c] -= v.Gv[4 * m + c];
                }
            for (int c = 0; c < 4; ++c) R[4 * m + c] -= gcl[0](i, j) * F[c] + gcl[1](i, j) * G[c];
        }
        // mass solve: (Ma^-1 x Mb^-1) R / J
        std::vector<double> tmp(nn * 4);
        along(ax[0].Minv, R.data(), n0, n1, 4, 0, tmp.data());
        along(ax[1].Minv, tmp.data(), n0, n1, 4, 1, out);
#pragma omp parallel for schedule(static)
        for (Index m = 0; m < nn; ++m) {
            const double Jm = J(m / n1, m % n1);
            for (int c = 0; c < 4; ++c) out[4 * m + c] /= Jm;
        }
        if (!source.empty())
            for (Index m = 0; m < nn * 4; ++m) out[m] += source[m];
        if (o.svv > 0) {
            double rates[2];
            svv_rates(U, rates);
            for (int a = 0; a < 2; ++a) {
                along(ax[a].Q, U, n0, n1, 4, a, tmp.data());
                for (Index m = 0; m < nn * 4; ++m) out[m] -= rates[a] * tmp[m];
            }
        }
    }

    void local_dt(const double* U, double* dt) const {
        double rates[2] = {0, 0};
        if (o.svv > 0) svv_rates(U, rates);
        const double g = o.gamma;
        require_positive(U);
#pragma omp parallel for schedule(static)
        for (Index m = 0; m < n0 * n1; ++m) {
            const Index i = m / n1, j = m % n1;
            const State s = raw(U + 4 * m);
            const double lx = std::abs(s.u * xi_x(i, j) + s.v * xi_y(i, j)) + s.c * grad_xi(i, j);
            const double le = std::abs(s.u * eta_x(i, j) + s.v * eta_y(i, j)) + s.c * grad_eta(i, j);
            const double eta = o.line_implicit == LineImplicit::Full ? 0.0 : 1.0;
            double rate = (ax[0].radius * lx + eta * ax[1].radius * le) / 2.83 + (rates[0] + eta * rates[1]) / 2.78;
            if (o.reynolds > 0) {
                const double T = g * s.p / s.rho;
                const double nu = std::pow(T, o.viscosity_exponent) / o.reynolds / s.rho * std::max(4.0 / 3.0, g / o.prandtl);
                const double normal =
                    o.line_implicit == LineImplicit::None ? ax[1].vradius * grad_eta(i, j) * grad_eta(i, j) : 0.0;
                rate += nu * (ax[0].vradius * grad_xi(i, j) * grad_xi(i, j) + normal) / 2.78;
            }
            dt[m] = o.cfl / rate;
        }
    }

    // Line-implicit wall-normal viscous preconditioner, frozen over one step: per eta line i,
    // LU factors of P = I - diag(dt_j nu_jc |grad eta_j|^2) L_eta for momentum (c = 1, 2) and energy.
    struct Preconditioner {
        std::vector<Eigen::PartialPivLU<Matrix>> momentum, energy;  // Viscous: per line and component
        std::vector<Eigen::PartialPivLU<Matrix>> line;              // Full: per line, (4 n1)^2
    };

    // Kinematic viscous coefficients nu_c |grad eta|^2 (momentum, energy) at node (i, j).
    void viscous_coefficients(const double* q, Index i, Index j, double& am, double& ae) const {
        const double g = o.gamma;
        const State s = raw(q);
        const double T = g * s.p / s.rho;
        const double nu = std::pow(T, o.viscosity_exponent) / o.reynolds / s.rho * grad_eta(i, j) * grad_eta(i, j);
        am = 4.0 / 3.0 * nu;
        ae = g / o.prandtl * nu;
    }

    // 4 x 4 Jacobian of f: R^4 -> R^4 at q by central differences.
    template <class F>
    static Eigen::Matrix4d jacobian(const double* q, F&& f) {
        Eigen::Matrix4d Jm;
        for (int c = 0; c < 4; ++c) {
            double a[4] = {q[0], q[1], q[2], q[3]}, b[4] = {q[0], q[1], q[2], q[3]};
            const double h = 1e-6 * std::max(1.0, std::abs(q[c]));
            a[c] += h;
            b[c] -= h;
            double fa[4], fb[4];
            f(a, fa);
            f(b, fb);
            for (int r = 0; r < 4; ++r) Jm(r, c) = (fa[r] - fb[r]) / (2 * h);
        }
        return Jm;
    }

    // d(contravariant numerical flux)/dU at the face node of eta face k (2 or 3): walls and exterior.
    Eigen::Matrix4d wall_jump_jacobian(const double* q, Index i, int k) const {
        const Index j = k == 2 ? 0 : n1 - 1;
        double nx = -y_xi(i, j), ny = x_xi(i, j);
        const double area = std::hypot(nx, ny);
        nx /= area;
        ny /= area;
        const bool no_slip = bc[k].kind == BoundaryKind::NoSlipWall;
        const bool exterior = bc[k].kind == BoundaryKind::Exterior;
        return jacobian(q, [&](const double* in, double* out) {
            double ext[4] = {in[0], in[1], in[2], in[3]};
            if (exterior) {  // fixed state at this face node
                for (int c = 0; c < 4; ++c) ext[c] = bc[k].nodes(i, c);
            } else if (no_slip) {
                ext[1] = -in[1];
                ext[2] = -in[2];
            } else {
                const double mn = in[1] * nx + in[2] * ny;
                ext[1] = in[1] - 2 * mn * nx;
                ext[2] = in[2] - 2 * mn * ny;
            }
            double star[4];
            if (k == 2) rusanov(ext, in, nx, ny, star);
            else rusanov(in, ext, nx, ny, star);
            for (int c = 0; c < 4; ++c) out[c] = area * star[c];
        });
    }

    Preconditioner preconditioner(const double* U, const std::vector<double>& dt) const {
        Preconditioner pc;
        const bool visc = o.reynolds > 0;
        const Matrix& L = ax[1].L;
        if (o.line_implicit == LineImplicit::Viscous) {
            pc.momentum.resize(n0);
            pc.energy.resize(n0);
#pragma omp parallel for schedule(static)
            for (Index i = 0; i < n0; ++i) {
                Vector am(n1), ae(n1);
                for (Index j = 0; j < n1; ++j) {
                    const Index m = i * n1 + j;
                    viscous_coefficients(U + 4 * m, i, j, am(j), ae(j));
                    am(j) *= dt[m];
                    ae(j) *= dt[m];
                }
                const Matrix I = Matrix::Identity(n1, n1);
                pc.momentum[i].compute(I - am.asDiagonal() * L);
                pc.energy[i].compute(I - ae.asDiagonal() * L);
            }
            return pc;
        }
        // Full: linearized wall-normal operator per line
        double rates[2] = {0, 0};
        if (o.svv > 0) svv_rates(U, rates);
        pc.line.resize(n0);
        const Index nb = 4 * n1;
#pragma omp parallel for schedule(dynamic)
        for (Index i = 0; i < n0; ++i) {
            // Weak form of the eta operator on this line (xi lumped: the line is a 1D weak problem at
            // its node), as the residual computes it:
            //   (1/J) Mb^{-1} [ P1b^T W A_G(U at the eta Gauss points) P0b ]  + face fluxes
            //   + Mb^{-1} e_0 d(star_0)/dU_0 - Mb^{-1} e_N d(star_N)/dU_N - nu_svv Q_eta.
            // The nodal form -(1/J) D diag(A_G(U_node)) has growing modes of its own (as the nodal
            // residual did), which (I - dt A)^{-1} amplifies on fine grids.
            Matrix A = Matrix::Zero(nb, nb);
            const Axis& Bx = ax[1];
            const Index Q = Bx.P0.rows();
            Matrix Ul(n1, 4);
            for (Index j = 0; j < n1; ++j)
                for (int c = 0; c < 4; ++c) Ul(j, c) = U[4 * (i * n1 + j) + c];
            const Matrix Uq = Bx.P0 * Ul;  // (Q, 4)
            const Vector nxq = -(Bx.P0 * y_xi.row(i).transpose()), nyq = Bx.P0 * x_xi.row(i).transpose();
            std::array<Matrix, 16> coef;  // w_b (A_G)_rc at the Gauss points
            for (auto& m : coef) m.resize(Q, 1);
            for (Index b = 0; b < Q; ++b) {
                const double q[4] = {Uq(b, 0), Uq(b, 1), Uq(b, 2), Uq(b, 3)};
                const double nx = nxq(b), ny = nyq(b);
                const Eigen::Matrix4d Ab = jacobian(q, [&](const double* in, double* out) { flux(in, nx, ny, out); });
                for (int r = 0; r < 4; ++r)
                    for (int c = 0; c < 4; ++c) coef[size_t(4 * r + c)](b, 0) = Bx.w(b) * Ab(r, c);
            }
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) {
                    const Matrix K = Bx.Minv * (Bx.P1.transpose() * coef[size_t(4 * r + c)].col(0).asDiagonal() * Bx.P0);
                    for (Index j = 0; j < n1; ++j) {
                        const double invJ = 1.0 / J(i, j);
                        for (Index l = 0; l < n1; ++l) A(4 * j + r, 4 * l + c) += invJ * K(j, l);
                    }
                }
            if (rates[1] > 0)
                for (Index j = 0; j < n1; ++j)
                    for (Index l = 0; l < n1; ++l)
                        A.block<4, 4>(4 * j, 4 * l).diagonal().array() -= rates[1] * Bx.Q(j, l);
            for (int k = 2; k < 4; ++k) {  // face fluxes: + Mb^{-1} e_0 star_0 (low), - Mb^{-1} e_N star_N (high)
                const BoundaryKind kind = bc[k].kind;
                if (kind != BoundaryKind::SlipWall && kind != BoundaryKind::NoSlipWall && kind != BoundaryKind::Exterior)
                    continue;  // inlet/outlet on eta faces: left explicit
                const Index jw = k == 2 ? 0 : n1 - 1;
                const Eigen::Matrix4d Js = wall_jump_jacobian(U + 4 * (i * n1 + jw), i, k);
                const double sign = k == 2 ? 1.0 : -1.0;
                for (Index j = 0; j < n1; ++j) A.block<4, 4>(4 * j, 4 * jw) += (sign * Bx.Minv(j, jw) / J(i, j)) * Js;
            }
            if (visc) {
                for (Index j = 0; j < n1; ++j) {
                    double am, ae;
                    viscous_coefficients(U + 4 * (i * n1 + j), i, j, am, ae);
                    for (Index l = 0; l < n1; ++l) {
                        A(4 * j + 1, 4 * l + 1) += am * L(j, l);
                        A(4 * j + 2, 4 * l + 2) += am * L(j, l);
                        A(4 * j + 3, 4 * l + 3) += ae * L(j, l);
                    }
                }
            }
            for (Index j = 0; j < n1; ++j) A.middleRows(4 * j, 4) *= -dt[i * n1 + j];
            A.diagonal().array() += 1.0;  // P = I - diag(dt) A
            pc.line[i].compute(A);
        }
        return pc;
    }

    void apply(const Preconditioner& pc, std::vector<double>& k) const {
#pragma omp parallel for schedule(static)
        for (Index i = 0; i < n0; ++i) {
            if (!pc.line.empty()) {
                Eigen::Map<Eigen::VectorXd> K(k.data() + i * n1 * 4, 4 * n1);
                K = pc.line[i].solve(Eigen::VectorXd(K));
                continue;
            }
            Eigen::Map<Rows> K(k.data() + i * n1 * 4, n1, 4);  // (n1, 4) for this eta line
            Matrix mom = K.middleCols(1, 2);
            K.middleCols(1, 2) = pc.momentum[i].solve(mom);
            Vector e = K.col(3);
            K.col(3) = pc.energy[i].solve(e);
        }
    }

    void step(const double* U, double* out) const {
        const Index nn = n0 * n1, nd = nn * 4;
        std::vector<double> dt(nn), k1(nd), k2(nd), k3(nd), k4(nd), W(nd);
        local_dt(U, dt.data());
        const bool implicit = o.line_implicit == LineImplicit::Full ||
                              (o.line_implicit == LineImplicit::Viscous && o.reynolds > 0);
        Preconditioner pc;
        if (implicit) pc = preconditioner(U, dt);
        auto eval = [&](const double* state, std::vector<double>& k) {
            rhs(state, k.data());
            if (implicit) apply(pc, k);
        };
        auto stage = [&](const std::vector<double>& k, double c) {
            for (Index m = 0; m < nd; ++m) W[m] = U[m] + c * dt[m / 4] * k[m];
        };
        eval(U, k1);
        stage(k1, 0.5);
        eval(W.data(), k2);
        stage(k2, 0.5);
        eval(W.data(), k3);
        stage(k3, 1.0);
        eval(W.data(), k4);
        for (Index m = 0; m < nd; ++m) out[m] = U[m] + dt[m / 4] / 6 * (k1[m] + 2 * k2[m] + 2 * k3[m] + k4[m]);
    }

    double residual(const double* U) const {
        const Index nd = n0 * n1 * 4;
        std::vector<double> R(nd);
        rhs(U, R.data());
        double s = 0;
        for (double r : R) s += r * r;
        return std::sqrt(s / static_cast<double>(nd));
    }

    bool positive(const Eigen::VectorXd& U) const {
        for (Index m = 0; m < n0 * n1; ++m) {
            const double* q = U.data() + 4 * m;
            if (!(q[0] > 0)) return false;
            const double p = (o.gamma - 1) * (q[3] - 0.5 * (q[1] * q[1] + q[2] * q[2]) / q[0]);
            if (!(p > 0)) return false;
        }
        return true;
    }
};

CompressibleSolver2D::CompressibleSolver2D(const MappedGrid2D& grid, std::array<Boundary, 4> boundaries,
                                           FlowOptions options)
    : d_(std::make_shared<Impl>(grid)) {
    if (!(options.gamma > 1) || !(options.prandtl > 0) || options.reynolds < 0 || options.svv < 0 ||
        !(options.cfl > 0) || options.quadrature < 4 || options.svv_cutoff < 0 || options.svv_cutoff >= 1)
        throw std::invalid_argument("CompressibleSolver2D: invalid FlowOptions");
    d_->o = options;
    d_->setup();
    for (int k = 0; k < 4; ++k) set_boundary(k, std::move(boundaries[k]));
}

const MappedGrid2D& CompressibleSolver2D::grid() const { return d_->grid; }
const FlowOptions& CompressibleSolver2D::options() const { return d_->o; }
Shape CompressibleSolver2D::shape() const { return {d_->n0, d_->n1, 4}; }
const Boundary& CompressibleSolver2D::boundary(int face) const {
    if (face < 0 || face > 3) throw std::invalid_argument("face must be 0..3");
    return d_->bc[face];
}

void CompressibleSolver2D::set_boundary(int face, Boundary b) {
    if (face < 0 || face > 3) throw std::invalid_argument("face must be 0..3");
    if (b.kind == BoundaryKind::Exterior) {
        const Impl::Face& f = d_->face[face];
        if (b.nodes.rows() != f.count || b.nodes.cols() != 4 || b.points.rows() != f.P0.rows() || b.points.cols() != 4)
            throw std::invalid_argument("Exterior boundary needs nodes (n, 4) and points (Q, 4) states");
    }
    if (b.kind == BoundaryKind::Inlet && (!(b.total_pressure > 0) || !(b.total_temperature > 0)))
        throw std::invalid_argument("Inlet needs positive total pressure and temperature");
    d_->bc[face] = std::move(b);
}

Matrix CompressibleSolver2D::face_points(int face) const {
    if (face < 0 || face > 3) throw std::invalid_argument("face must be 0..3");
    return d_->face[face].points;
}

namespace {
void check_state(const FieldView& U, Index n0, Index n1) {
    if (U.shape.size() != 3 || U.shape[0] != n0 || U.shape[1] != n1 || U.shape[2] != 4)
        throw std::invalid_argument("state must have shape (n0, n1, 4)");
}
}  // namespace

void CompressibleSolver2D::set_source(const FieldView& s) {
    if (s.data == nullptr || s.shape.empty()) {
        d_->source.clear();
        return;
    }
    check_state(s, d_->n0, d_->n1);
    d_->source.assign(s.data, s.data + d_->n0 * d_->n1 * 4);
}

Field CompressibleSolver2D::rhs(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    Field out{shape(), std::vector<double>(static_cast<size_t>(d_->n0 * d_->n1 * 4))};
    d_->rhs(U.data, out.data.data());
    return out;
}

Field CompressibleSolver2D::local_dt(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    Field out{{d_->n0, d_->n1}, std::vector<double>(static_cast<size_t>(d_->n0 * d_->n1))};
    d_->local_dt(U.data, out.data.data());
    return out;
}

Field CompressibleSolver2D::step(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    Field out{shape(), std::vector<double>(static_cast<size_t>(d_->n0 * d_->n1 * 4))};
    d_->step(U.data, out.data.data());
    return out;
}

SteadyResult CompressibleSolver2D::solve(Field& Uf, const SteadyOptions& so) const {
    check_state(Uf.view(), d_->n0, d_->n1);
    const Impl& d = *d_;
    const Index nd = d.n0 * d.n1 * 4;
    const auto start = std::chrono::steady_clock::now();
    SteadyResult res;
    Eigen::Map<Eigen::VectorXd> U(Uf.data.data(), nd);
    Eigen::VectorXd G(nd), Gprev(nd), fprev(nd), f(nd);
    std::deque<Eigen::VectorXd> dF, dG;  // Anderson differences
    bool have_prev = false, mixed = false;
    double best = std::numeric_limits<double>::infinity();
    const int depth = std::max(so.anderson, 0);
    const int check = std::max(so.check, 1);
    auto restart = [&] {
        dF.clear();
        dG.clear();
        have_prev = false;
        ++res.restarts;
    };
    for (int it = 0;; ++it) {
        if (it % check == 0 || it == so.max_steps) {
            double r;
            try {
                r = d.residual(U.data());
            } catch (const std::domain_error&) {
                if (!mixed) throw;
                U = Gprev;  // last unmixed RK4 result
                restart();
                mixed = false;
                r = d.residual(U.data());
            }
            ++res.rhs_evaluations;
            res.history.emplace_back(it, r);
            res.steps = it;
            res.residual = r;
            if (so.progress) so.progress(it, r);
            if (!std::isfinite(r)) throw std::runtime_error("CompressibleSolver2D: residual not finite");
            if (r < so.tolerance) {
                res.converged = true;
                break;
            }
            if (depth > 0 && r > 100 * best) {  // Anderson drifted: drop the history
                U = Gprev;
                restart();
            }
            best = std::min(best, r);
            if (it == so.max_steps) break;
        }
        try {
            d.step(U.data(), G.data());
        } catch (const std::domain_error&) {
            if (!mixed) throw;
            U = Gprev;
            restart();
            mixed = false;
            d.step(U.data(), G.data());
        }
        res.rhs_evaluations += 4;
        f = G - U;
        mixed = false;
        if (depth > 0) {
            if (have_prev) {
                dF.push_back(f - fprev);
                dG.push_back(G - Gprev);
                if (static_cast<int>(dF.size()) > depth) {
                    dF.pop_front();
                    dG.pop_front();
                }
            }
            fprev = f;
            Gprev = G;
            have_prev = true;
            if (!dF.empty()) {
                const Index m = static_cast<Index>(dF.size());
                Matrix A(nd, m), B(nd, m);
                for (Index c = 0; c < m; ++c) {
                    A.col(c) = dF[c];
                    B.col(c) = dG[c];
                }
                const Eigen::VectorXd gamma = A.colPivHouseholderQr().solve(f);
                Eigen::VectorXd next = G - B * gamma;
                if (next.allFinite() && d.positive(next)) {
                    U = next;
                    mixed = true;
                } else {
                    U = G;
                    restart();
                }
                continue;
            }
        } else {
            Gprev = G;
        }
        U = G;
    }
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return res;
}

Field CompressibleSolver2D::prolong(const CompressibleSolver2D& coarse, const FieldView& U) const {
    check_state(U, coarse.d_->n0, coarse.d_->n1);
    const Grid& ref = coarse.grid().reference();
    const std::vector<Vector> coordinates{grid().reference().line(0).x(), grid().reference().line(1).x()};
    return ref.interpolate(U, coordinates);
}

detail::FlowTables detail::flow_tables(const CompressibleSolver2D& solver) {
    const CompressibleSolver2D::Impl& d = *solver.d_;
    FlowTables t;
    t.options = d.o;
    t.boundaries = d.bc;
    t.n0 = d.n0;
    t.n1 = d.n1;
    for (int a = 0; a < 2; ++a) {
        const Axis& x = d.ax[a];
        FlowTables::Axis& y = t.axis[a];
        y.D = x.D;
        y.Q = x.Q;
        y.Minv = x.Minv;
        y.L = x.L;
        y.Mq = x.Mq;
        y.Cq = x.Cq;
        y.s0 = x.s0;
        y.s1 = x.s1;
        y.weights = x.weights;
        y.radius = x.radius;
        y.vradius = x.vradius;
        y.points = x.P0.rows();
    }
    t.x_xi = d.x_xi;
    t.x_eta = d.x_eta;
    t.y_xi = d.y_xi;
    t.y_eta = d.y_eta;
    t.J = d.J;
    t.xq_xi = d.xq_xi;
    t.xq_eta = d.xq_eta;
    t.yq_xi = d.yq_xi;
    t.yq_eta = d.yq_eta;
    t.gcl[0] = d.gcl[0];
    t.gcl[1] = d.gcl[1];
    for (int k = 0; k < 4; ++k) {
        t.face[k].nx = d.face[k].nx;
        t.face[k].ny = d.face[k].ny;
        t.face[k].area = d.face[k].area;
    }
    t.source = d.source;
    return t;
}

Vector CompressibleSolver2D::mass_flow(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    const Impl& d = *d_;
    Vector out = Vector::Zero(d.n0);
    for (Index i = 0; i < d.n0; ++i)
        for (Index j = 0; j < d.n1; ++j) {
            const double* q = U.data + 4 * (i * d.n1 + j);
            out(i) += d.ax[1].weights(j) * (q[1] * d.y_eta(i, j) - q[2] * d.x_eta(i, j));
        }
    return out;
}

Field CpuFlowLevel::smooth(const FieldView& U, int steps) const {
    Field out{U.shape, std::vector<double>(U.data, U.data + element_count(U.shape))};
    for (int k = 0; k < steps; ++k) out = s_.step(out.view());
    return out;
}

}  // namespace bspf

