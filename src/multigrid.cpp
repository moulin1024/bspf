// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/multigrid.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace bspf {

namespace {

using Index = Eigen::Index;

// Per-axis coarsening factors between consecutive levels: 1 (same nodes: p-coarsening or
// semi-coarsening) or 2 (n_fine = 2 n_coarse - 1).
std::array<int, 2> factors(const Shape& fine, const Shape& coarse) {
    if (fine.size() != 3 || coarse.size() != 3 || fine[2] != coarse[2])
        throw std::invalid_argument("multigrid levels must be (n0, n1, 4) fields");
    std::array<int, 2> f{};
    for (int a = 0; a < 2; ++a) {
        if (fine[a] == coarse[a]) f[a] = 1;
        else if (fine[a] == 2 * coarse[a] - 1) f[a] = 2;
        else throw std::invalid_argument("multigrid levels must be nested per axis: n_coarse = n_fine or (n_fine + 1) / 2");
    }
    return f;
}

Field inject(const FieldView& f, std::array<int, 2> s) {
    const Index n0 = f.shape[0], n1 = f.shape[1], k = f.shape[2];
    const Index m0 = s[0] == 2 ? (n0 + 1) / 2 : n0, m1 = s[1] == 2 ? (n1 + 1) / 2 : n1;
    Field out{{m0, m1, k}, std::vector<double>(static_cast<size_t>(m0 * m1 * k))};
    for (Index i = 0; i < m0; ++i)
        for (Index j = 0; j < m1; ++j)
            for (Index c = 0; c < k; ++c)
                out.data[size_t((i * m1 + j) * k + c)] = f.data[(s[0] * i * n1 + s[1] * j) * k + c];
    return out;
}

double rms(const Field& f) {
    double s = 0;
    for (double v : f.data) s += v * v;
    return std::sqrt(s / static_cast<double>(f.data.size()));
}

bool physical(const Field& U, double gamma) {
    for (size_t m = 0; m + 3 < U.data.size(); m += 4) {
        const double* q = &U.data[m];
        if (!(q[0] > 0)) return false;
        const double p = (gamma - 1) * (q[3] - 0.5 * (q[1] * q[1] + q[2] * q[2]) / q[0]);
        if (!(p > 0)) return false;
    }
    return true;
}

struct Cycle {
    const std::vector<FlowLevel*>& levels;
    const MultigridOptions& o;
    MultigridResult& result;
    std::vector<double> weight;  // node count relative to the finest level
    const FieldView empty{nullptr, {}};

    Field smooth(int l, const Field& U, int steps) {
        result.work += steps * weight[size_t(l)];
        return levels[size_t(l)]->smooth(U.view(), steps);
    }

    std::vector<std::array<int, 2>> factor;  // between level l and l + 1

    Field run(int l, Field U, const Field* f) {
        FlowLevel& L = *levels[size_t(l)];
        L.set_source(f ? f->view() : empty);
        if (l + 1 == int(levels.size())) return smooth(l, U, o.coarse);
        U = smooth(l, U, o.pre);
        FlowLevel& C = *levels[size_t(l) + 1];
        for (int g = 0; g < o.cycle_index; ++g) {
            const Field r = L.rhs(U.view());
            const Field Uc0 = inject(U.view(), factor[size_t(l)]);
            C.set_source(empty);
            const Field Rc = C.rhs(Uc0.view());
            Field fc = restrict_full_weighting(r.view(), factor[size_t(l)]);
            for (size_t m = 0; m < fc.data.size(); ++m) fc.data[m] -= Rc.data[m];
            const Field Uc = run(l + 1, Uc0, &fc);
            Field dUc = Uc;
            for (size_t m = 0; m < dUc.data.size(); ++m) dUc.data[m] -= Uc0.data[m];
            const Field dU = C.reference().interpolate(dUc.view(), {L.reference().line(0).x(), L.reference().line(1).x()});
            double omega = o.damping;
            for (int attempt = 0;; ++attempt) {
                Field trial = U;
                for (size_t m = 0; m < trial.data.size(); ++m) trial.data[m] += omega * dU.data[m];
                if (physical(trial, L.gamma())) {
                    U = std::move(trial);
                    break;
                }
                if (attempt == 4) break;  // skip the correction
                omega *= 0.5;
                ++result.damping_cuts;
            }
            L.set_source(f ? f->view() : empty);
        }
        return smooth(l, U, o.post);
    }
};

}  // namespace

Field restrict_full_weighting(const FieldView& f, std::array<int, 2> s) {
    if (f.shape.size() != 3) throw std::invalid_argument("restriction needs an (n0, n1, k) field");
    for (int a = 0; a < 2; ++a)
        if (s[a] == 2 && (f.shape[a] < 3 || f.shape[a] % 2 == 0))
            throw std::invalid_argument("full weighting needs an odd length >= 3 along coarsened axes");
    const Index n0 = f.shape[0], n1 = f.shape[1], k = f.shape[2];
    const Index m0 = s[0] == 2 ? (n0 + 1) / 2 : n0, m1 = s[1] == 2 ? (n1 + 1) / 2 : n1;
    // weights of fine nodes for coarse node i along an axis
    auto w = [](Index i, Index n, Index fine, int factor) -> std::array<std::pair<Index, double>, 3> {
        if (factor == 1) return {{{i, 1.0}, {0, 0.0}, {0, 0.0}}};
        if (i == 0) return {{{0, 0.5}, {1, 0.5}, {0, 0.0}}};
        if (i == n - 1) return {{{fine - 1, 0.5}, {fine - 2, 0.5}, {0, 0.0}}};
        return {{{2 * i - 1, 0.25}, {2 * i, 0.5}, {2 * i + 1, 0.25}}};
    };
    Field out{{m0, m1, k}, std::vector<double>(static_cast<size_t>(m0 * m1 * k), 0.0)};
    for (Index i = 0; i < m0; ++i)
        for (const auto& [a, wa] : w(i, m0, n0, s[0])) {
            if (wa == 0) continue;
            for (Index j = 0; j < m1; ++j)
                for (const auto& [b, wb] : w(j, m1, n1, s[1])) {
                    if (wb == 0) continue;
                    for (Index c = 0; c < k; ++c)
                        out.data[size_t((i * m1 + j) * k + c)] += wa * wb * f.data[(a * n1 + b) * k + c];
                }
        }
    return out;
}

MultigridResult multigrid_solve(const std::vector<FlowLevel*>& levels, Field& U, const MultigridOptions& o) {
    if (levels.empty()) throw std::invalid_argument("multigrid_solve: no levels");
    std::vector<std::array<int, 2>> factor;
    for (size_t l = 0; l + 1 < levels.size(); ++l) factor.push_back(factors(levels[l]->shape(), levels[l + 1]->shape()));
    if (U.shape != levels[0]->shape()) throw std::invalid_argument("multigrid_solve: U must have the finest shape");
    if (o.pre < 0 || o.post < 0 || o.coarse < 1 || o.cycle_index < 1 || !(o.damping > 0) || o.max_cycles < 1)
        throw std::invalid_argument("multigrid_solve: invalid options");
    const auto start = std::chrono::steady_clock::now();
    MultigridResult result;
    Cycle cycle{levels, o, result, {}};
    cycle.factor = factor;
    const double n_fine = double(levels[0]->shape()[0] * levels[0]->shape()[1]);
    for (FlowLevel* L : levels) cycle.weight.push_back(double(L->shape()[0] * L->shape()[1]) / n_fine);
    const FieldView empty{nullptr, {}};
    levels[0]->set_source(empty);
    if (o.initial > 0) U = cycle.smooth(0, U, o.initial);
    for (int c = 1; c <= o.max_cycles; ++c) {
        U = cycle.run(0, std::move(U), nullptr);
        levels[0]->set_source(empty);
        const double r = rms(levels[0]->rhs(U.view()));
        result.history.emplace_back(c, r);
        result.cycles = c;
        result.residual = r;
        if (o.progress) o.progress(c, r);
        if (!std::isfinite(r)) throw std::runtime_error("multigrid_solve: residual not finite");
        if (r < o.tolerance) {
            result.converged = true;
            break;
        }
    }
    for (FlowLevel* L : levels) L->set_source(empty);
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return result;
}

namespace {
using Vec = Eigen::VectorXd;

Vec as_vec(const Field& f) { return Eigen::Map<const Vec>(f.data.data(), Index(f.data.size())); }
Field as_field(const Vec& v, const Shape& shape) { return Field{shape, std::vector<double>(v.data(), v.data() + v.size())}; }
double rms_vec(const Vec& v) { return v.norm() / std::sqrt(double(v.size())); }
}  // namespace

NewtonResult newton_krylov_solve(const std::vector<FlowLevel*>& levels, Field& Ufield, const NewtonOptions& o,
                                 FlowLevel* residual_level, NewtonDevice* device) {
    if (levels.empty()) throw std::invalid_argument("newton_krylov_solve: no levels");
    std::vector<std::array<int, 2>> factor;
    for (size_t l = 0; l + 1 < levels.size(); ++l) factor.push_back(factors(levels[l]->shape(), levels[l + 1]->shape()));
    if (Ufield.shape != levels[0]->shape()) throw std::invalid_argument("newton_krylov_solve: U must have the finest shape");
    const auto start = std::chrono::steady_clock::now();
    NewtonResult res;
    MultigridResult mg;  // work bookkeeping of the cycles
    Cycle cycle{levels, o.multigrid, mg, {}};
    cycle.factor = factor;
    const double n_fine = double(levels[0]->shape()[0] * levels[0]->shape()[1]);
    for (FlowLevel* L : levels) cycle.weight.push_back(double(L->shape()[0] * L->shape()[1]) / n_fine);
    FlowLevel& F = residual_level ? *residual_level : *levels[0];  // Newton's residual (and Jacobian)
    FlowLevel& Pf = *levels[0];                                       // preconditioner's finest level
    const Shape shape = Pf.shape();
    if (F.shape() != shape) throw std::invalid_argument("newton_krylov_solve: residual level must have the finest shape");
    const FieldView empty{nullptr, {}};
    auto residual = [&](const Vec& U) {
        F.set_source(empty);
        return as_vec(F.rhs(as_field(U, shape).view()));
    };
    const bool separate = &F != &Pf;
    Vec Rp;  // the preconditioner level's own residual at the current U (forcing of the cycles)
    auto update_rp = [&](const Vec& U) {
        if (!separate) return;
        Pf.set_source(empty);
        Rp = as_vec(Pf.rhs(as_field(U, shape).view()));
    };
    Vec U = as_vec(Ufield);
    if (device && device->handles_start()) {  // start, preconditioner and Newton all on the device
        NewtonResult dev = device->solve(Ufield, o, nullptr);
        for (FlowLevel* L : levels) L->set_source(empty);
        dev.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return dev;
    }
    // start: FAS cycles
    Vec R = residual(U);
    double r = rms_vec(R);
    if (o.multigrid.initial > 0) {
        U = as_vec(cycle.smooth(0, Ufield, o.multigrid.initial));
        R = residual(U);
        r = rms_vec(R);
    }
    while (r > o.start_residual && res.start_cycles < o.max_start_cycles && levels.size() > 1) {
        U = as_vec(cycle.run(0, as_field(U, shape), nullptr));
        R = residual(U);
        r = rms_vec(R);
        ++res.start_cycles;
        if (!std::isfinite(r)) throw std::runtime_error("newton_krylov_solve: start cycles diverged");
    }
    if (device) {  // Newton phase on the device; the preconditioner stays the FAS cycle on the levels
        Vec Rpc;
        auto precondition = [&](const Field& Uf, const Field& zf, bool new_state) -> Field {
            const auto t0 = std::chrono::steady_clock::now();
            const Vec Uv = as_vec(Uf), z = as_vec(zf);
            if (new_state || Rpc.size() == 0) {
                Pf.set_source(empty);
                Rpc = as_vec(Pf.rhs(Uf.view()));
            }
            Vec out = Vec::Zero(Uv.size());
            const double zn = z.norm();
            if (zn > 0) {
                const double sigma = o.preconditioner_epsilon * (1.0 + Uv.norm()) / zn;
                Field f = as_field(-Rpc - sigma * z, shape);
                Field W = Uf;
                for (int c = 0; c < o.preconditioner_cycles; ++c) W = cycle.run(0, std::move(W), &f);
                Pf.set_source(empty);
                out = (as_vec(W) - Uv) / sigma;
            }
            res.seconds_preconditioner += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return as_field(out, shape);
        };
        Field Ustart = as_field(U, shape);
        NewtonResult dev = device->solve(Ustart, o, precondition);
        for (FlowLevel* L : levels) L->set_source(empty);
        Ufield = Ustart;
        dev.start_cycles = res.start_cycles;
        dev.seconds_preconditioner = res.seconds_preconditioner;
        dev.history.insert(dev.history.begin(), {0, r});
        dev.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return dev;
    }
    res.history.emplace_back(0, r);
    if (o.progress) o.progress(0, r, 0);
    const Index n = U.size();
    double r_prev = r;
    // preconditioner: z -> (cycle(U, forcing -R(U) - sigma z) - U) / sigma
    auto precondition = [&](const Vec& z) -> Vec {
        const double zn = z.norm();
        if (zn == 0) return Vec::Zero(n);
        const double sigma = o.preconditioner_epsilon * (1.0 + U.norm()) / zn;
        Field f = as_field(-(separate ? Rp : R) - sigma * z, shape);
        Field W = as_field(U, shape);
        for (int c = 0; c < o.preconditioner_cycles; ++c) W = cycle.run(0, std::move(W), &f);
        Pf.set_source(empty);
        return (as_vec(W) - U) / sigma;
    };
    for (int k = 1; k <= o.max_newton && r > o.tolerance; ++k) {
        update_rp(U);
        const double eta = std::max(o.eta_min, std::min(o.eta_max, k == 1 ? o.eta_max : r / r_prev));
        // FGMRES on J d = -R
        auto Jv = [&](const Vec& v) -> Vec {
            const double h = o.jacobian_epsilon * (1.0 + U.norm()) / std::max(v.norm(), 1e-300);
            return (residual(U + h * v) - R) / h;
        };
        Vec d = Vec::Zero(n);
        const double bnorm = R.norm();
        int its = 0;
        Vec rk = -R;  // residual of the linear system
        while (its < o.max_krylov) {
            const int m = o.krylov_restart;
            const double beta = rk.norm();
            if (beta <= eta * bnorm) break;
            std::vector<Vec> V, Z;
            V.push_back(rk / beta);
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m + 1, m);
            Vec g = Vec::Zero(m + 1);
            g(0) = beta;
            std::vector<double> cs(m), sn(m);
            int j = 0;
            for (; j < m && its < o.max_krylov; ++j, ++its) {
                Z.push_back(precondition(V[j]));
                Vec w = Jv(Z[j]);
                for (int i = 0; i <= j; ++i) {  // modified Gram-Schmidt
                    H(i, j) = w.dot(V[i]);
                    w -= H(i, j) * V[i];
                }
                H(j + 1, j) = w.norm();
                V.push_back(H(j + 1, j) > 0 ? Vec(w / H(j + 1, j)) : Vec(w));
                for (int i = 0; i < j; ++i) {  // apply previous rotations
                    const double t = cs[i] * H(i, j) + sn[i] * H(i + 1, j);
                    H(i + 1, j) = -sn[i] * H(i, j) + cs[i] * H(i + 1, j);
                    H(i, j) = t;
                }
                const double den = std::hypot(H(j, j), H(j + 1, j));
                cs[j] = H(j, j) / den;
                sn[j] = H(j + 1, j) / den;
                H(j, j) = den;
                H(j + 1, j) = 0;
                g(j + 1) = -sn[j] * g(j);
                g(j) = cs[j] * g(j);
                if (std::abs(g(j + 1)) <= eta * bnorm) {
                    ++j;
                    ++its;
                    break;
                }
            }
            // back substitution, update d with the preconditioned vectors
            Vec y = Vec::Zero(j);
            for (int i = j - 1; i >= 0; --i) {
                double s2 = g(i);
                for (int l = i + 1; l < j; ++l) s2 -= H(i, l) * y(l);
                y(i) = s2 / H(i, i);
            }
            for (int i = 0; i < j; ++i) d += y(i) * Z[size_t(i)];
            rk = -R - Jv(d);
            if (rk.norm() <= eta * bnorm) break;
        }
        res.krylov_iterations += its;
        // line search on |R|, rejecting nonphysical states
        double alpha = 1.0;
        Vec Un, Rn;
        double rn = r;
        for (int t = 0; t < 8; ++t, alpha *= 0.5) {
            Un = U + alpha * d;
            try {
                Rn = residual(Un);
            } catch (const std::domain_error&) {
                continue;
            }
            rn = rms_vec(Rn);
            if (std::isfinite(rn) && rn < (1 - 1e-4 * alpha) * r) break;
        }
        if (!(rn < r)) {  // no decrease: fall back to a multigrid cycle
            Un = as_vec(cycle.run(0, as_field(U, shape), nullptr));
            Rn = residual(Un);
            rn = rms_vec(Rn);
        }
        U = Un;
        R = Rn;
        r_prev = r;
        r = rn;
        res.newton_steps = k;
        res.history.emplace_back(k, r);
        if (o.progress) o.progress(k, r, its);
        if (!std::isfinite(r)) throw std::runtime_error("newton_krylov_solve: residual not finite");
    }
    for (FlowLevel* L : levels) L->set_source(empty);
    Ufield = as_field(U, shape);
    res.residual = r;
    res.converged = r <= o.tolerance;
    res.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return res;
}

}  // namespace bspf

