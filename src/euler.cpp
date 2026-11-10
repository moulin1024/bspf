// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/euler.hpp"

#include "bspf/entropy.hpp"

#include <stdexcept>
#include <utility>

namespace bspf {

WeakEuler2D::WeakEuler2D(Matrix P0a, Matrix P1a, Vector wa, Matrix P0b, Matrix P1b, Vector wb, const Matrix& x_xi,
                         const Matrix& x_eta, const Matrix& y_xi, const Matrix& y_eta, double gamma)
    : P0a_(std::move(P0a)), P1a_(std::move(P1a)), P0b_(std::move(P0b)), P1b_(std::move(P1b)), gamma_(gamma) {
    const Eigen::Index Qa = P0a_.rows(), Qb = P0b_.rows();
    if (P1a_.rows() != Qa || P1a_.cols() != P0a_.cols() || wa.size() != Qa || P1b_.rows() != Qb ||
        P1b_.cols() != P0b_.cols() || wb.size() != Qb)
        throw std::invalid_argument("WeakEuler2D: inconsistent quadrature matrices");
    for (const Matrix* m : {&x_xi, &x_eta, &y_xi, &y_eta})
        if (m->rows() != Qa || m->cols() != Qb) throw std::invalid_argument("WeakEuler2D: metrics must be (Qa, Qb)");
    if (!(gamma > 1)) throw std::invalid_argument("WeakEuler2D: gamma must exceed 1");
    P0aT_ = P0a_.transpose();
    P1aT_ = P1a_.transpose();
    P0bT_ = P0b_.transpose();
    P1bT_ = P1b_.transpose();
    auto weighted = [&](const Matrix& m) {
        Rows out(Qb, Qa);
        for (Eigen::Index b = 0; b < Qb; ++b)
            for (Eigen::Index a = 0; a < Qa; ++a) out(b, a) = wa(a) * wb(b) * m(a, b);
        return out;
    };
    w_ = weighted(Matrix::Ones(Qa, Qb));
    wx_xi_ = weighted(x_xi);
    wx_eta_ = weighted(x_eta);
    wy_xi_ = weighted(y_xi);
    wy_eta_ = weighted(y_eta);
}

void WeakEuler2D::interpolate(const double* f, Eigen::Index k, Rows& out) const {
    const Eigen::Index n0 = this->n0(), n1 = this->n1(), Qa = qa(), Qb = qb();
    Eigen::Map<const Rows> F(f, n0, n1 * k);
    Rows T = P0a_ * F;  // (Qa, n1 k): T(a, j k + c)
    Rows Tp(n1, Qa * k);  // Tp(j, a k + c)
    for (Eigen::Index a = 0; a < Qa; ++a)
        for (Eigen::Index j = 0; j < n1; ++j)
            for (Eigen::Index c = 0; c < k; ++c) Tp(j, a * k + c) = T(a, j * k + c);
    out.resize(Qb, Qa * k);
    out.noalias() = P0b_ * Tp;  // (Qb, Qa k)
}

void WeakEuler2D::volume(const double* U, const double* Fv, const double* Gv, double* R) const {
    const Eigen::Index n0 = this->n0(), n1 = this->n1(), Qa = qa(), Qb = qb();
    Rows Uq, Vq;
    interpolate(U, 4, Uq);
    if (entropy_) {  // V at the Gauss points -> U(V)
        bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
        for (Eigen::Index b = 0; b < Qb; ++b)
            for (Eigen::Index a = 0; a < Qa; ++a) {
                double* q = Uq.row(b).data() + 4 * a;
                double v[4] = {q[0], q[1], q[2], q[3]};
                if (!entropy_to_conservative(v, gamma_, q)) bad = true;
            }
        if (bad) throw std::domain_error("WeakEuler2D: invalid entropy variables at a Gauss point");
    }
    const bool viscous = Fv != nullptr && Gv != nullptr;
    if (viscous) {
        Rows both(n0, n1 * 8);  // (n0, n1, 8): Fv then Gv per node
        for (Eigen::Index i = 0; i < n0; ++i)
            for (Eigen::Index j = 0; j < n1; ++j)
                for (int c = 0; c < 4; ++c) {
                    both(i, j * 8 + c) = Fv[(i * n1 + j) * 4 + c];
                    both(i, j * 8 + 4 + c) = Gv[(i * n1 + j) * 4 + c];
                }
        interpolate(both.data(), 8, Vq);
    }
    const double gm1 = gamma_ - 1;
    Rows Fh(Qb, Qa * 4), Gh(Qb, Qa * 4);
    bool bad = false;
#pragma omp parallel for reduction(|| : bad) schedule(static)
    for (Eigen::Index b = 0; b < Qb; ++b) {
        const double* q = Uq.row(b).data();
        double* fh = Fh.row(b).data();
        double* gh = Gh.row(b).data();
        for (Eigen::Index a = 0; a < Qa; ++a) {
            const double rho = q[4 * a], mx = q[4 * a + 1], my = q[4 * a + 2], E = q[4 * a + 3];
            const double u = mx / rho, v = my / rho;
            const double p = gm1 * (E - 0.5 * (mx * u + my * v));
            if (!(rho > 0) || !(p > 0)) bad = true;  // no throwing inside the parallel region
            // contravariant velocities times the weights
            const double xx = wx_xi_(b, a), xe = wx_eta_(b, a), yx = wy_xi_(b, a), ye = wy_eta_(b, a);
            const double Ux = ye * u - xe * v, Ue = -yx * u + xx * v;  // w J (xi_x u + xi_y v), w J (eta_x u + eta_y v)
            const double H = E + p;
            fh[4 * a] = rho * Ux;
            fh[4 * a + 1] = mx * Ux + ye * p;
            fh[4 * a + 2] = my * Ux - xe * p;
            fh[4 * a + 3] = H * Ux;
            gh[4 * a] = rho * Ue;
            gh[4 * a + 1] = mx * Ue - yx * p;
            gh[4 * a + 2] = my * Ue + xx * p;
            gh[4 * a + 3] = H * Ue;
            if (viscous) {  // the extra fluxes are contravariant already: only the weights apply
                const double* vq = Vq.row(b).data() + 8 * a;
                const double wgt = w_(b, a);
                for (int c = 0; c < 4; ++c) {
                    fh[4 * a + c] -= wgt * vq[c];
                    gh[4 * a + c] -= wgt * vq[4 + c];
                }
            }
        }
    }
    if (bad) throw std::domain_error("WeakEuler2D: nonpositive density or pressure");
    // projection: S(j, a 4 + c) = sum_b P0b(b, j) Fh(b, a 4 + c) (and P1b with Gh)
    Rows S1 = P0bT_ * Fh, S2 = P1bT_ * Gh;  // (n1, Qa 4)
    Rows T1(Qa, n1 * 4), T2(Qa, n1 * 4);
    for (Eigen::Index j = 0; j < n1; ++j)
        for (Eigen::Index a = 0; a < Qa; ++a)
            for (int c = 0; c < 4; ++c) {
                T1(a, j * 4 + c) = S1(j, a * 4 + c);
                T2(a, j * 4 + c) = S2(j, a * 4 + c);
            }
    Eigen::Map<Rows> out(R, n0, n1 * 4);
    out.noalias() = P1aT_ * T1;
    out.noalias() += P0aT_ * T2;
}

}  // namespace bspf
