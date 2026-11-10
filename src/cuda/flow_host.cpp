// SPDX-License-Identifier: GPL-3.0-or-later
// Host side of the device compressible solver: converts a CPU solver's setup to plain tables (double;
// rounded on the device for the FP32 precision).
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "bspf/cuda_compressible.hpp"
#include "bspf/galerkin.hpp"
#include "bspf/op.hpp"
#include "flow_device.hpp"

namespace bspf::cuda {

std::vector<double> to_vec(const Matrix& M);  // column-major (galerkin_host.cpp)

namespace {
std::vector<double> rows_major(const Matrix& M) {  // (n0, n1) -> [i][j]
    std::vector<double> out(static_cast<size_t>(M.size()));
    for (Eigen::Index i = 0; i < M.rows(); ++i)
        for (Eigen::Index j = 0; j < M.cols(); ++j) out[static_cast<size_t>(i * M.cols() + j)] = M(i, j);
    return out;
}
std::vector<double> vec(const Vector& v) { return std::vector<double>(v.data(), v.data() + v.size()); }
}  // namespace

struct CompressibleSolver2D::Impl {
    std::shared_ptr<flow::FlowDevice<double>> fp64;  // one of the two, by precision
    std::shared_ptr<flow::FlowDevice<float>> fp32;
    template <class F>
    auto with(F&& f) {
        return fp64 ? f(*fp64) : f(*fp32);
    }
    Eigen::Index n0 = 0, n1 = 0;
    std::shared_ptr<bspf::Grid> reference;
    double gamma = 1.4;
};

CompressibleSolver2D::CompressibleSolver2D(const bspf::CompressibleSolver2D& cpu, int refresh,
                                           const std::string& precision)
    : d_(std::make_shared<Impl>()) {
    const detail::FlowTables t = detail::flow_tables(cpu);
    const FlowOptions& o = t.options;
    if (o.reynolds > 0) throw std::invalid_argument("cuda::CompressibleSolver2D: viscous terms are not ported yet");
    flow::FlowDeviceTables f;
    f.n0 = int(t.n0);
    f.n1 = int(t.n1);
    f.gamma = o.gamma;
    f.svv = o.svv;
    f.cfl = o.cfl;
    GalerkinOptions go;
    go.quadrature_order = o.quadrature;
    for (int a = 0; a < 2; ++a) {
        const GalerkinLine g(cpu.grid().reference().line(a), go);
        f.axis[a] = flow::axis_tables(g);
        if (f.axis[a].P != t.axis[a].points) throw std::logic_error("cuda::CompressibleSolver2D: quadrature mismatch");
        f.Minv[a] = to_vec(t.axis[a].Minv);
        f.Q[a] = to_vec(t.axis[a].Q);
        f.radius[a] = t.axis[a].radius;
        f.vradius[a] = t.axis[a].vradius;
    }
    const Matrix xi_x = t.y_eta.cwiseQuotient(t.J), xi_y = -t.x_eta.cwiseQuotient(t.J);
    const Matrix eta_x = -t.y_xi.cwiseQuotient(t.J), eta_y = t.x_xi.cwiseQuotient(t.J);
    f.J = rows_major(t.J);
    f.xi_x = rows_major(xi_x);
    f.xi_y = rows_major(xi_y);
    f.eta_x = rows_major(eta_x);
    f.eta_y = rows_major(eta_y);
    f.grad_xi = rows_major((xi_x.array().square() + xi_y.array().square()).sqrt().matrix());
    f.grad_eta = rows_major((eta_x.array().square() + eta_y.array().square()).sqrt().matrix());
    f.x_xi = rows_major(t.x_xi);
    f.x_eta = rows_major(t.x_eta);
    f.y_xi = rows_major(t.y_xi);
    f.y_eta = rows_major(t.y_eta);
    f.qx_xi = to_vec(t.xq_xi);  // (Qa, Qb) column-major = [b][a]
    f.qx_eta = to_vec(t.xq_eta);
    f.qy_xi = to_vec(t.yq_xi);
    f.qy_eta = to_vec(t.yq_eta);
    f.gcl[0] = rows_major(t.gcl[0]);
    f.gcl[1] = rows_major(t.gcl[1]);
    for (int k = 0; k < 4; ++k) {
        const Boundary& b = t.boundaries[k];
        flow::FaceTables& ft = f.face[k];
        switch (b.kind) {
            case BoundaryKind::Inlet: ft.kind = flow::kInlet; break;
            case BoundaryKind::Outlet: ft.kind = flow::kOutlet; break;
            case BoundaryKind::SlipWall: ft.kind = flow::kSlipWall; break;
            case BoundaryKind::NoSlipWall: ft.kind = flow::kNoSlipWall; break;
            case BoundaryKind::Exterior: ft.kind = flow::kExterior; break;
        }
        ft.total_pressure = b.total_pressure;
        ft.total_temperature = b.total_temperature;
        ft.has_back_pressure = !std::isnan(b.back_pressure);
        ft.back_pressure = ft.has_back_pressure ? b.back_pressure : 0.0;
        ft.nx = vec(t.face[k].nx);
        ft.ny = vec(t.face[k].ny);
        ft.area = vec(t.face[k].area);
        if (b.kind == BoundaryKind::Exterior) {
            ft.exterior = to_vec(b.points);
            ft.exterior_nodes = to_vec(b.nodes);
        }
    }
    if (!t.source.empty()) {  // C-ordered (n0, n1, 4) -> planar
        const size_t nn = size_t(t.n0 * t.n1);
        f.source.resize(4 * nn);
        for (size_t m = 0; m < nn; ++m)
            for (size_t c = 0; c < 4; ++c) f.source[c * nn + m] = t.source[4 * m + c];
    }
    if (o.line_implicit == LineImplicit::Viscous)
        throw std::invalid_argument("cuda::CompressibleSolver2D: line_implicit 'viscous' needs the viscous terms");
    f.line_implicit = o.line_implicit == LineImplicit::Full;
    f.refresh = std::max(1, refresh);
    if (f.line_implicit) {
        const GalerkinLine g(cpu.grid().reference().line(1), go);
        const Eigen::Index n1 = t.n1;
        const Matrix I = Matrix::Identity(n1, n1);
        const Matrix P0 = g.evaluate(I, 0), P1 = g.evaluate(I, 1);
        const Eigen::Index Pb = P0.rows();
        Matrix Z(Pb, n1 * n1);
        for (Eigen::Index l = 0; l < n1; ++l)
            for (Eigen::Index j = 0; j < n1; ++j) Z.col(j + l * n1) = P1.col(j).cwiseProduct(P0.col(l));
        f.line_Z = to_vec(Z);
        f.line_w = vec(g.quadrature_weights());
        const Matrix nx = -(t.y_xi * P0.transpose()), ny = t.x_xi * P0.transpose();  // (n0, Pb)
        f.line_nx = rows_major(nx);
        f.line_ny = rows_major(ny);
    }
    f.D_eta = to_vec(t.axis[1].D);
    f.s0_eta = vec(t.axis[1].s0);
    f.s1_eta = vec(t.axis[1].s1);
    if (precision == "double") d_->fp64 = flow::make_flow<double>(f);
    else if (precision == "float") d_->fp32 = flow::make_flow<float>(f);
    else throw std::invalid_argument("cuda::CompressibleSolver2D: precision must be double or float");
    d_->reference = std::make_shared<bspf::Grid>(cpu.grid().reference());
    d_->gamma = o.gamma;
    d_->n0 = t.n0;
    d_->n1 = t.n1;
}

Shape CompressibleSolver2D::shape() const { return {d_->n0, d_->n1, 4}; }

namespace {
void check_state(const FieldView& U, Eigen::Index n0, Eigen::Index n1) {
    if (U.shape.size() != 3 || U.shape[0] != n0 || U.shape[1] != n1 || U.shape[2] != 4)
        throw std::invalid_argument("state must have shape (n0, n1, 4)");
}
std::vector<double> floats(const FieldView& U) {
    return std::vector<double>(U.data, U.data + element_count(U.shape));
}
Field doubles(const std::vector<double>& v, Shape shape) {
    return Field{std::move(shape), std::vector<double>(v.begin(), v.end())};
}
}  // namespace

Field CompressibleSolver2D::rhs(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    return doubles(d_->with([&](auto& dev) { return flow::flow_rhs_host(dev, floats(U)); }), shape());
}

Field CompressibleSolver2D::step(const FieldView& U) const {
    check_state(U, d_->n0, d_->n1);
    return doubles(d_->with([&](auto& dev) { return flow::flow_step_host(dev, floats(U)); }), shape());
}

SteadyResult CompressibleSolver2D::solve(Field& U, const SteadyOptions& o) const {
    check_state(U.view(), d_->n0, d_->n1);
    std::vector<double> state = floats(U.view());
    flow::SolveSettings s;
    s.max_steps = o.max_steps;
    s.tolerance = o.tolerance;
    s.anderson = o.anderson;
    s.check = o.check;
    s.progress = o.progress;
    const flow::SolveReport r = d_->with([&](auto& dev) { return flow::flow_solve_host(dev, state, s); });
    std::copy(state.begin(), state.end(), U.data.begin());
    SteadyResult out;
    out.steps = r.steps;
    out.residual = r.residual;
    out.converged = r.converged;
    out.restarts = r.restarts;
    out.rhs_evaluations = r.rhs_evaluations;
    out.seconds = r.seconds;
    out.history = r.history;
    return out;
}

void CompressibleSolver2D::set_source(const FieldView& source) {
    if (source.data == nullptr || source.shape.empty()) {
        d_->with([&](auto& dev) { flow::flow_set_source(dev, {}); });
        return;
    }
    check_state(source, d_->n0, d_->n1);
    d_->with([&](auto& dev) { flow::flow_set_source(dev, floats(source)); });
}

const bspf::Grid& CompressibleSolver2D::reference() const { return *d_->reference; }
double CompressibleSolver2D::gamma() const { return d_->gamma; }

Field DeviceFlowLevel::smooth(const FieldView& U, int steps) const {
    Field out{U.shape, std::vector<double>(U.data, U.data + element_count(U.shape))};
    if (steps <= 0) return out;
    SteadyOptions o;
    o.max_steps = steps;
    o.tolerance = 0.0;
    o.anderson = 0;
    o.check = 1 << 30;
    s_.solve(out, o);
    return out;
}

namespace {
class DeviceNewton : public NewtonDevice {
public:
    DeviceNewton(std::shared_ptr<flow::FlowDevice<double>> f, Shape shape, std::shared_ptr<flow::DeviceMG> mg = nullptr,
                 flow::MGSettings ms = {})
        : f_(std::move(f)), shape_(std::move(shape)), mg_(std::move(mg)), ms_(ms) {}
    bool handles_start() const override { return mg_ != nullptr; }
    NewtonResult solve(Field& U, const NewtonOptions& o, const Preconditioner& precondition) override {
        flow::NewtonSettings s;
        s.tolerance = o.tolerance;
        s.eta_max = o.eta_max;
        s.eta_min = o.eta_min;
        s.jacobian_epsilon = o.jacobian_epsilon;
        s.max_newton = o.max_newton;
        s.krylov_restart = o.krylov_restart;
        s.max_krylov = o.max_krylov;
        s.progress = o.progress;
        const Shape shape = shape_;
        auto host = [&](const std::vector<double>& U_, const std::vector<double>& z, bool new_state) {
            const Field Uf{shape, U_}, zf{shape, z};
            return precondition(Uf, zf, new_state).data;
        };
        const flow::NewtonReport r =
            mg_ ? flow::flow_newton(*f_, U.data, s, nullptr, mg_.get(), &ms_) : flow::flow_newton(*f_, U.data, s, host);
        NewtonResult out;
        out.newton_steps = r.newton_steps;
        out.krylov_iterations = r.krylov_iterations;
        out.residual = r.residual;
        out.converged = r.converged;
        out.seconds_jacobian = r.seconds_jacobian;
        out.seconds_krylov = r.seconds_krylov;
        out.seconds_preconditioner = r.seconds_preconditioner;
        out.start_cycles = r.start_cycles;
        out.history = r.history;
        return out;
    }

private:
    std::shared_ptr<flow::FlowDevice<double>> f_;
    Shape shape_;
    std::shared_ptr<flow::DeviceMG> mg_;
    flow::MGSettings ms_;
};
}  // namespace

std::unique_ptr<NewtonDevice> CompressibleSolver2D::newton_device() const {
    if (!d_->fp64) throw std::invalid_argument("newton_device needs an FP64 device solver (precision='double')");
    return std::make_unique<DeviceNewton>(d_->fp64, shape());
}

std::unique_ptr<NewtonDevice> CompressibleSolver2D::newton_device(const std::vector<CompressibleSolver2D*>& levels,
                                                                  const MultigridOptions& mo,
                                                                  double preconditioner_epsilon, double start_residual,
                                                                  int max_start_cycles, int preconditioner_cycles) const {
    if (!d_->fp64) throw std::invalid_argument("newton_device needs an FP64 device solver (precision='double')");
    if (levels.empty()) throw std::invalid_argument("newton_device: no multigrid levels");
    std::vector<std::shared_ptr<flow::FlowDevice<float>>> devs;
    for (const CompressibleSolver2D* L : levels) {
        if (!L->d_->fp32) throw std::invalid_argument("newton_device: multigrid levels must be FP32 device solvers");
        devs.push_back(L->d_->fp32);
    }
    if (levels[0]->shape() != shape()) throw std::invalid_argument("newton_device: finest level must match this solver");
    std::vector<std::vector<double>> Pa, Pb;
    for (size_t l = 0; l + 1 < levels.size(); ++l) {
        const bspf::Grid& fine = levels[l]->reference();
        const bspf::Grid& coarse = levels[l + 1]->reference();
        const bool same = fine.line(0).size() == coarse.line(0).size() && fine.line(1).size() == coarse.line(1).size();
        if (same) {  // p-level: the coarse interpolant at its own nodes is the identity
            Pa.emplace_back();
            Pb.emplace_back();
            continue;
        }
        Pa.push_back(to_vec(coarse.line(0).evaluate(fine.line(0).x()).matrix()));
        Pb.push_back(to_vec(coarse.line(1).evaluate(fine.line(1).x()).matrix()));
    }
    flow::MGSettings ms;
    ms.pre = mo.pre;
    ms.post = mo.post;
    ms.coarse = mo.coarse;
    ms.cycle_index = mo.cycle_index;
    ms.damping = mo.damping;
    ms.initial = mo.initial;
    ms.preconditioner_epsilon = preconditioner_epsilon;
    ms.start_residual = start_residual;
    ms.max_start_cycles = max_start_cycles;
    ms.preconditioner_cycles = preconditioner_cycles;
    return std::make_unique<DeviceNewton>(d_->fp64, shape(), flow::make_device_mg(devs, Pa, Pb, ms), ms);
}

}  // namespace bspf::cuda

