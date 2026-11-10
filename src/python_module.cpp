// SPDX-License-Identifier: GPL-3.0-or-later
// nanobind module bspf_cpp._core. Fields are C-ordered float64 arrays whose
// leading axes are spatial; ops act along any axis without transposing.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <memory>
#include <stdexcept>

#include "bspf/bspline.hpp"
#include "bspf/compressible.hpp"
#include "bspf/galerkin.hpp"
#include "bspf/grid.hpp"
#include "bspf/mapped.hpp"
#include "bspf/multipatch.hpp"
#ifdef BSPF_HAVE_CUDA
#include "bspf/cuda.hpp"
#include "bspf/cuda_galerkin.hpp"
#include "bspf/cuda_compressible.hpp"
#endif

namespace nb = nanobind;
using namespace nb::literals;
using bspf::Field;
using bspf::FieldView;
using bspf::Grid;
using bspf::Line;
using bspf::Matrix;
using bspf::Op;
using bspf::Vector;

using InArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, double>;

namespace {

FieldView view(const InArray& a) {
    FieldView v{a.data(), {}};
    for (size_t i = 0; i < a.ndim(); ++i) v.shape.push_back(static_cast<Eigen::Index>(a.shape(i)));
    return v;
}

OutArray to_numpy(Field&& f) {
    auto* data = new std::vector<double>(std::move(f.data));
    nb::capsule owner(data, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    std::vector<size_t> shape(f.shape.begin(), f.shape.end());
    return OutArray(data->data(), shape.size(), shape.data(), owner);
}

int normalize_axis(int axis, size_t ndim) {
    const int n = static_cast<int>(ndim);
    if (axis < -n || axis >= n) throw std::invalid_argument("axis out of range");
    return axis < 0 ? axis + n : axis;
}

bspf::EndpointMethod endpoint_method(const std::string& name);

bspf::LineOptions line_options(int degree, std::optional<int> n_basis, std::optional<Vector> knots,
                               std::optional<int> constraint_order, std::optional<int> boundary_points, double lam,
                               double clustering, const std::string& method, std::optional<int> chebyshev_modes,
                               double chebyshev_alpha, int chebyshev_penalty_power) {
    bspf::LineOptions o;
    o.degree = degree;
    o.n_basis = n_basis;
    o.knots = std::move(knots);
    o.constraint_order = constraint_order;
    o.boundary_points = boundary_points;
    o.lam = lam;
    o.clustering = clustering;
    o.endpoint_method = endpoint_method(method);
    o.chebyshev_modes = chebyshev_modes;
    o.chebyshev_alpha = chebyshev_alpha;
    o.chebyshev_penalty_power = chebyshev_penalty_power;
    return o;
}

bspf::EndpointMethod endpoint_method(const std::string& name) {
    if (name == "finite_difference") return bspf::EndpointMethod::FiniteDifference;
    if (name == "chebyshev") return bspf::EndpointMethod::Chebyshev;
    throw std::invalid_argument("endpoint_method must be 'finite_difference' or 'chebyshev'");
}

// Op applied along `axis`. Jets have the field's shape with that axis of length
// 2q, or are a single (2q,) vector shared by every line.
OutArray apply(const Op& op, const InArray& f, int axis, const std::optional<InArray>& jets) {
    const FieldView v = view(f);
    axis = normalize_axis(axis, f.ndim());
    const Eigen::Index J = op.jet_count();
    std::vector<double> broadcast;
    const double* jet_data = nullptr;
    if (jets) {
        bspf::Shape expected = v.shape;
        expected[axis] = J;
        if (jets->ndim() == 1 && static_cast<Eigen::Index>(jets->shape(0)) == J) {
            Eigen::Index before = 1, after = 1;
            for (int i = 0; i < axis; ++i) before *= v.shape[i];
            for (size_t i = axis + 1; i < v.shape.size(); ++i) after *= v.shape[i];
            broadcast.resize(static_cast<size_t>(before * J * after));
            for (Eigen::Index l = 0; l < before; ++l)
                for (Eigen::Index j = 0; j < J; ++j)
                    std::fill_n(broadcast.begin() + (l * J + j) * after, after, jets->data()[j]);
            jet_data = broadcast.data();
        } else {
            if (view(*jets).shape != expected)
                throw std::invalid_argument("jets must have the field's shape with the axis of length " + std::to_string(J));
            jet_data = jets->data();
        }
    }
    Field out;
    {
        nb::gil_scoped_release release;
        out = bspf::apply_along(op, v, axis, jet_data);
    }
    return to_numpy(std::move(out));
}

template <class Level, class Solver>
nb::tuple run_newton(const std::vector<Solver*>& solvers, const InArray& U0, double tolerance, int max_newton,
                     int krylov_restart, int max_krylov, double eta_max, double start_residual, int pre, int post,
                     int coarse, int cycle_index, double damping, int initial, double jacobian_epsilon,
                     double preconditioner_epsilon, nb::object progress, Solver* residual_solver = nullptr,
                     std::unique_ptr<bspf::NewtonDevice> device = nullptr) {
    std::unique_ptr<Level> residual_level;
    if (residual_solver) residual_level = std::make_unique<Level>(*residual_solver);
    std::vector<Level> adapters;
    adapters.reserve(solvers.size());
    for (Solver* s : solvers) adapters.emplace_back(*s);
    std::vector<bspf::FlowLevel*> levels;
    for (Level& a : adapters) levels.push_back(&a);
    const FieldView v = view(U0);
    Field U{v.shape, std::vector<double>(v.data, v.data + bspf::element_count(v.shape))};
    bspf::NewtonOptions o;
    o.jacobian_epsilon = jacobian_epsilon;
    o.preconditioner_epsilon = preconditioner_epsilon;
    o.tolerance = tolerance;
    o.max_newton = max_newton;
    o.krylov_restart = krylov_restart;
    o.max_krylov = max_krylov;
    o.eta_max = eta_max;
    o.start_residual = start_residual;
    o.multigrid.pre = pre;
    o.multigrid.post = post;
    o.multigrid.coarse = coarse;
    o.multigrid.cycle_index = cycle_index;
    o.multigrid.damping = damping;
    o.multigrid.initial = initial;
    o.progress = [progress](int step, double residual, int its) {
        nb::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) throw nb::python_error();
        if (!progress.is_none()) progress(step, residual, its);
    };
    bspf::NewtonResult r;
    {
        nb::gil_scoped_release release;
        r = bspf::newton_krylov_solve(levels, U, o, residual_level.get(), device.get());
    }
    nb::dict info;
    info["newton_steps"] = r.newton_steps;
    info["krylov_iterations"] = r.krylov_iterations;
    info["start_cycles"] = r.start_cycles;
    info["residual"] = r.residual;
    info["converged"] = r.converged;
    info["seconds"] = r.seconds;
    info["seconds_preconditioner"] = r.seconds_preconditioner;
    info["seconds_jacobian"] = r.seconds_jacobian;
    info["seconds_krylov"] = r.seconds_krylov;
    nb::list history;
    for (const auto& [c, res] : r.history) history.append(nb::make_tuple(c, res));
    info["history"] = history;
    return nb::make_tuple(to_numpy(std::move(U)), info);
}

template <class Level, class Solver>
nb::tuple run_multigrid(const std::vector<Solver*>& solvers, const InArray& U0, int pre, int post, int coarse,
                        int cycle_index, double damping, int initial, int max_cycles, double tolerance,
                        nb::object progress) {
    std::vector<Level> adapters;
    adapters.reserve(solvers.size());
    for (Solver* s : solvers) adapters.emplace_back(*s);
    std::vector<bspf::FlowLevel*> levels;
    for (Level& a : adapters) levels.push_back(&a);
    const FieldView v = view(U0);
    Field U{v.shape, std::vector<double>(v.data, v.data + bspf::element_count(v.shape))};
    bspf::MultigridOptions o;
    o.pre = pre;
    o.post = post;
    o.coarse = coarse;
    o.cycle_index = cycle_index;
    o.damping = damping;
    o.initial = initial;
    o.max_cycles = max_cycles;
    o.tolerance = tolerance;
    o.progress = [progress](int cycle, double residual) {
        nb::gil_scoped_acquire acquire;
        if (PyErr_CheckSignals() != 0) throw nb::python_error();
        if (!progress.is_none()) progress(cycle, residual);
    };
    bspf::MultigridResult r;
    {
        nb::gil_scoped_release release;
        r = bspf::multigrid_solve(levels, U, o);
    }
    nb::dict info;
    info["cycles"] = r.cycles;
    info["residual"] = r.residual;
    info["converged"] = r.converged;
    info["work"] = r.work;
    info["seconds"] = r.seconds;
    info["damping_cuts"] = r.damping_cuts;
    nb::list history;
    for (const auto& [c, res] : r.history) history.append(nb::make_tuple(c, res));
    info["history"] = history;
    return nb::make_tuple(to_numpy(std::move(U)), info);
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "C++ BSPF (B-spline plus Fourier) calculus on uniform 1-3D grids";

    nb::class_<Op>(m, "Op", "Compiled BSPF operator y = A f + A_j (g - J f), applied along an axis")
        .def_prop_ro("shape", [](const Op& op) { return std::make_pair(op.rows(), op.cols()); })
        .def_prop_ro("jet_count", &Op::jet_count)
        .def_prop_ro("matrix", &Op::matrix, "Dense operator A (P, N)")
        .def_prop_ro("boundary_matrix", &Op::boundary_matrix, "Response A_j (P, 2q) to given endpoint jets")
        .def_prop_ro("jet_map", [](const Op& op) { return Matrix(op.jet_map()); },
                     "Endpoint jet estimator J (2q, N)")
        .def("__call__", &apply, "f"_a, "axis"_a = 0, "jets"_a = nb::none());

    nb::class_<Line>(m, "Line", "Immutable 1D BSPF geometry and constrained fit on a uniform grid")
        .def("__init__",
             [](Line* self, const Vector& x, int degree, std::optional<int> n_basis, std::optional<Vector> knots,
                std::optional<int> constraint_order, std::optional<int> boundary_points, double lam,
                double clustering, const std::string& method, std::optional<int> chebyshev_modes,
                double chebyshev_alpha, int chebyshev_penalty_power) {
                 new (self) Line(x, line_options(degree, n_basis, std::move(knots), constraint_order, boundary_points, lam,
                                                 clustering, method, chebyshev_modes, chebyshev_alpha,
                                                 chebyshev_penalty_power));
             },
             "x"_a, nb::kw_only(), "degree"_a = 5, "n_basis"_a = nb::none(), "knots"_a = nb::none(),
             "constraint_order"_a = nb::none(), "boundary_points"_a = nb::none(), "lam"_a = 0.0,
             "clustering"_a = 0.0, "endpoint_method"_a = "finite_difference", "chebyshev_modes"_a = nb::none(),
             "chebyshev_alpha"_a = 1e-12, "chebyshev_penalty_power"_a = 4)
        .def_prop_ro("size", &Line::size)
        .def_prop_ro("degree", &Line::degree)
        .def_prop_ro("n_basis", &Line::n_basis)
        .def_prop_ro("constraint_order", &Line::constraint_order)
        .def_prop_ro("boundary_points", &Line::boundary_points)
        .def_prop_ro("lam", &Line::lam)
        .def_prop_ro("spacing", &Line::spacing)
        .def_prop_ro("x", [](const Line& l) { return Vector(l.x()); })
        .def_prop_ro("knots", [](const Line& l) { return Vector(l.knots()); })
        .def_prop_ro("weights", [](const Line& l) { return Vector(l.weights()); })
        .def_prop_ro("omega", [](const Line& l) { return Vector(l.omega()); })
        .def_prop_ro("basis", [](const Line& l) { return Matrix(l.basis()); })
        .def_prop_ro("constraint", [](const Line& l) { return Matrix(l.constraint()); })
        .def_prop_ro("gram", [](const Line& l) { return Matrix(l.gram()); })
        .def_prop_ro("kkt", [](const Line& l) { return Matrix(l.kkt()); })
        .def_prop_ro("fit_samples", [](const Line& l) { return Matrix(l.fit_samples()); })
        .def_prop_ro("fit_jets", [](const Line& l) { return Matrix(l.fit_jets()); })
        .def_prop_ro("jet_map", [](const Line& l) { return Matrix(l.jet_map()); })
        .def("derivative", &Line::derivative, "order"_a = 1, nb::kw_only(), "correction"_a = true)
        .def("evaluate", &Line::evaluate, "points"_a, nb::kw_only(), "derivative"_a = 0, "correction"_a = true)
        .def("quadrature", [](const Line& l, std::optional<double> a, std::optional<double> b) {
                 return l.quadrature(a.value_or(l.x()(0)), b.value_or(l.x()(l.size() - 1)));
             }, "a"_a = nb::none(), "b"_a = nb::none())
        .def("primitive", &Line::primitive, "order"_a = 1)
        .def("jets", &Line::jets);

    nb::class_<Grid>(m, "Grid", "Tensor-product BSPF calculus on a 1-3D rectangle")
        .def(nb::init<std::vector<Line>>(), "lines"_a)
        .def_prop_ro("dimension", &Grid::dimension)
        .def_prop_ro("shape", &Grid::shape)
        .def("line", &Grid::line, "axis"_a, nb::rv_policy::reference_internal)
        .def("derivative", &Grid::derivative, "axis"_a, "order"_a = 1, nb::rv_policy::reference_internal)
        .def("partial", [](const Grid& g, const InArray& f, const std::vector<int>& orders) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = g.partial(view(f), orders);
                 }
                 return to_numpy(std::move(out));
             }, "f"_a, "orders"_a)
        .def("laplacian", [](const Grid& g, const InArray& f) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = g.laplacian(view(f));
                 }
                 return to_numpy(std::move(out));
             }, "f"_a)
        .def("interpolate", [](const Grid& g, const InArray& f, const std::vector<Vector>& coordinates,
                               const std::vector<int>& derivatives) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = g.interpolate(view(f), coordinates, derivatives);
                 }
                 return to_numpy(std::move(out));
             }, "f"_a, "coordinates"_a, "derivatives"_a = std::vector<int>{})
        .def("weak_laplacian", [](const Grid& g, const InArray& f, const std::optional<InArray>& wall_values) {
                 Field out;
                 std::optional<FieldView> walls;
                 if (wall_values) walls = view(*wall_values);
                 {
                     nb::gil_scoped_release release;
                     out = g.weak_laplacian(view(f), walls ? &*walls : nullptr);
                 }
                 return to_numpy(std::move(out));
             }, "f"_a, "wall_values"_a = nb::none(),
             "Energy-stable Galerkin Laplacian with Dirichlet walls; wall_values are the result's wall values")
        .def("galerkin", &Grid::galerkin, "axis"_a, nb::rv_policy::reference_internal)
        .def("integrate", [](const Grid& g, const InArray& f,
                             const std::optional<std::vector<std::pair<double, double>>>& bounds) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = bounds ? g.integrate(view(f), *bounds) : g.integrate(view(f));
                 }
                 return to_numpy(std::move(out));
             }, "f"_a, "bounds"_a = nb::none());

    nb::class_<bspf::GalerkinLine>(m, "GalerkinLine",
                                   "Energy-stable (Galerkin) second derivative on a Line's BSPF trial space")
        .def("__init__", [](bspf::GalerkinLine* self, const Line& line, double mass_tolerance, double operator_tolerance, int quadrature_order) {
                 bspf::GalerkinOptions o;
                 o.mass_tolerance = mass_tolerance;
                 o.operator_tolerance = operator_tolerance;
                 o.quadrature_order = quadrature_order;
                 new (self) bspf::GalerkinLine(line, o);
             }, "line"_a, nb::kw_only(), "mass_tolerance"_a = 1e-12, "operator_tolerance"_a = 1e-14, "quadrature_order"_a = 16)
        .def_prop_ro("quadrature_points", [](const bspf::GalerkinLine& g) { return Vector(g.quadrature_points()); })
        .def_prop_ro("quadrature_weights", [](const bspf::GalerkinLine& g) { return Vector(g.quadrature_weights()); })
        .def("evaluate", [](const bspf::GalerkinLine& g, const Matrix& u, int k) { return g.evaluate(u, k); }, "u"_a, "derivative"_a = 0)
        .def("project", [](const bspf::GalerkinLine& g, const Matrix& u, int k) { return g.project(u, k); }, "values"_a, "derivative"_a = 0)
        .def("solve_mass", [](const bspf::GalerkinLine& g, const Matrix& u, bool interior) { return g.solve_mass(u, interior); }, "rhs"_a, "interior"_a = false)
        .def_prop_ro("operator_rank", &bspf::GalerkinLine::operator_rank)
        .def_prop_ro("fit_amplification", &bspf::GalerkinLine::fit_amplification)
        .def("exact_second_derivative", [](const bspf::GalerkinLine& g, const Matrix& u) { return g.exact_second_derivative(u); }, "u"_a)
        .def_prop_ro("size", &bspf::GalerkinLine::size)
        .def_prop_ro("mass_rank", &bspf::GalerkinLine::mass_rank)
        .def("mass", [](const bspf::GalerkinLine& g, const Matrix& u) { return g.mass(u); }, "u"_a)
        .def("stiffness", [](const bspf::GalerkinLine& g, const Matrix& u) { return g.stiffness(u); }, "u"_a)
        .def("second_derivative", [](const bspf::GalerkinLine& g, const Matrix& u) { return g.second_derivative(u); }, "u"_a)
        .def("extension", [](const bspf::GalerkinLine& g, const Matrix& u) { return g.extension(u); }, "u"_a);

    nb::class_<bspf::SplineCurve>(m, "SplineCurve", "Clamped B-spline curve x(s) in the plane, s in [0, 1]")
        .def(nb::init<const Matrix&, int>(), "control_points"_a, "degree"_a = 3)
        .def(nb::init<const Matrix&, const Vector&, int>(), "control_points"_a, "knots"_a, "degree"_a)
        .def_prop_ro("degree", &bspf::SplineCurve::degree)
        .def_prop_ro("knots", [](const bspf::SplineCurve& c) { return Vector(c.knots()); })
        .def_prop_ro("control_points", [](const bspf::SplineCurve& c) { return Matrix(c.control_points()); })
        .def("evaluate", &bspf::SplineCurve::evaluate, "s"_a, "derivative"_a = 0,
             "Points (derivative 0) or derivatives d^k x / ds^k at parameters s: (m, 2)");

    nb::class_<bspf::MappedGrid2D>(m, "MappedGrid2D",
                                   "Structured grid on a region bounded by four spline curves (transfinite map)")
        .def("__init__",
             [](bspf::MappedGrid2D* self, const bspf::SplineCurve& bottom, const bspf::SplineCurve& right,
                const bspf::SplineCurve& top, const bspf::SplineCurve& left, std::pair<Eigen::Index, Eigen::Index> shape,
                int degree, std::optional<int> n_basis, std::optional<int> constraint_order,
                std::optional<int> boundary_points, double lam, double clustering, const std::string& method,
                std::optional<int> chebyshev_modes, double chebyshev_alpha, int chebyshev_penalty_power,
                std::optional<Vector> dist_xi, std::optional<Vector> dist_eta, std::optional<Vector> dist_bottom,
                std::optional<Vector> dist_top, std::optional<Vector> dist_left, std::optional<Vector> dist_right,
                std::optional<std::pair<Matrix, Matrix>> correction, std::optional<int> n_basis_eta) {
                 auto pick = [](const std::optional<Vector>& edge, const std::optional<Vector>& axis) {
                     return edge ? bspf::Distribution(*edge) : axis ? bspf::Distribution(*axis) : bspf::Distribution();
                 };
                 const bspf::EdgeDistribution distribution{pick(dist_bottom, dist_xi), pick(dist_top, dist_xi),
                                                           pick(dist_left, dist_eta), pick(dist_right, dist_eta)};
                 const bspf::LineOptions xi = line_options(degree, n_basis, std::nullopt, constraint_order,
                                                           boundary_points, lam, clustering, method, chebyshev_modes,
                                                           chebyshev_alpha, chebyshev_penalty_power);
                 std::optional<bspf::LineOptions> eta;
                 if (n_basis_eta) {
                     eta = xi;
                     eta->n_basis = n_basis_eta;
                 }
                 new (self) bspf::MappedGrid2D(bottom, right, top, left, shape.first, shape.second, xi, distribution,
                                               correction ? correction->first : Matrix(),
                                               correction ? correction->second : Matrix(), eta);
             },
             "bottom"_a, "right"_a, "top"_a, "left"_a, "shape"_a, nb::kw_only(), "degree"_a = 5,
             "n_basis"_a = nb::none(), "constraint_order"_a = nb::none(), "boundary_points"_a = nb::none(),
             "lam"_a = 0.0, "clustering"_a = 0.0, "endpoint_method"_a = "finite_difference",
             "chebyshev_modes"_a = nb::none(), "chebyshev_alpha"_a = 1e-12, "chebyshev_penalty_power"_a = 4,
             "dist_xi"_a = nb::none(), "dist_eta"_a = nb::none(), "dist_bottom"_a = nb::none(),
             "dist_top"_a = nb::none(), "dist_left"_a = nb::none(), "dist_right"_a = nb::none(),
             "correction"_a = nb::none(), "n_basis_eta"_a = nb::none(),
             "n_basis_eta: n_basis of the eta axis (default n_basis). Edge node distributions psi(s) (Bernstein coefficients 0, ..., 1, increasing): the fraction of an "
             "edge's nodes before curve parameter s (bottom/top along xi, left/right along eta; dist_xi / dist_eta "
             "set both edges of an axis). Inside, xi = (1 - eta) psi_bottom(s) + eta psi_top(s) and likewise for eta. "
             "correction: (cx, cy), interior Bernstein coefficients (p - 1, q - 1) of a correction "
             "sum c_kl B_k^p(xi) B_l^q(eta) added to the map (zero on the edges)")
        .def_prop_ro("shape", &bspf::MappedGrid2D::shape)
        .def_prop_ro("reference", &bspf::MappedGrid2D::reference, nb::rv_policy::reference_internal)
        .def_prop_ro("x", [](const bspf::MappedGrid2D& g) { return Matrix(g.x()); })
        .def_prop_ro("y", [](const bspf::MappedGrid2D& g) { return Matrix(g.y()); })
        .def_prop_ro("x_xi", [](const bspf::MappedGrid2D& g) { return Matrix(g.x_xi()); })
        .def_prop_ro("x_eta", [](const bspf::MappedGrid2D& g) { return Matrix(g.x_eta()); })
        .def_prop_ro("y_xi", [](const bspf::MappedGrid2D& g) { return Matrix(g.y_xi()); })
        .def_prop_ro("y_eta", [](const bspf::MappedGrid2D& g) { return Matrix(g.y_eta()); })
        .def_prop_ro("jacobian", [](const bspf::MappedGrid2D& g) { return Matrix(g.jacobian()); })
        .def("partial", [](const bspf::MappedGrid2D& g, const InArray& f, std::pair<int, int> orders) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = g.partial(view(f), orders.first, orders.second);
                 }
                 return to_numpy(std::move(out));
             }, "f"_a, "orders"_a, "Physical partial d^a/dx^a d^b/dy^b f for orders (a, b)");

    nb::enum_<bspf::BoundaryKind>(m, "BoundaryKind")
        .value("inlet", bspf::BoundaryKind::Inlet)
        .value("outlet", bspf::BoundaryKind::Outlet)
        .value("slip_wall", bspf::BoundaryKind::SlipWall)
        .value("no_slip_wall", bspf::BoundaryKind::NoSlipWall)
        .value("exterior", bspf::BoundaryKind::Exterior);

    nb::class_<bspf::Boundary>(m, "Boundary", "Boundary condition of one face of a CompressibleSolver2D")
        .def("__init__",
             [](bspf::Boundary* self, bspf::BoundaryKind kind, double total_pressure, double total_temperature,
                std::optional<double> back_pressure, std::optional<Matrix> nodes, std::optional<Matrix> points) {
                 new (self) bspf::Boundary();
                 self->kind = kind;
                 self->total_pressure = total_pressure;
                 self->total_temperature = total_temperature;
                 if (back_pressure) self->back_pressure = *back_pressure;
                 if (nodes) self->nodes = *nodes;
                 if (points) self->points = *points;
             },
             "kind"_a, nb::kw_only(), "total_pressure"_a = 1.0 / 1.4, "total_temperature"_a = 1.0,
             "back_pressure"_a = nb::none(), "nodes"_a = nb::none(), "points"_a = nb::none(),
             "inlet: total pressure/temperature; outlet: back_pressure where nothing is supersonic; exterior: "
             "states (rho, rho u, rho v, E) at the face nodes (n, 4) and Gauss points (Q, 4)")
        .def_rw("kind", &bspf::Boundary::kind)
        .def_rw("total_pressure", &bspf::Boundary::total_pressure)
        .def_rw("total_temperature", &bspf::Boundary::total_temperature)
        .def_rw("back_pressure", &bspf::Boundary::back_pressure);

    auto state_view = [](const InArray& U) {
        if (U.ndim() != 3) throw std::invalid_argument("state must have shape (n0, n1, 4)");
        return view(U);
    };
    nb::class_<bspf::CompressibleSolver2D>(m, "CompressibleSolver2D",
                                           "Compressible Euler/Navier-Stokes on one mapped block (Galerkin BSPF, "
                                           "weak boundary fluxes, local-time-step RK4 with Anderson acceleration)")
        .def("__init__",
             [](bspf::CompressibleSolver2D* self, const bspf::MappedGrid2D& grid,
                const std::vector<bspf::Boundary>& boundaries, double gamma, double prandtl, double reynolds,
                double viscosity_exponent, double svv, double svv_cutoff, int quadrature, double cfl,
                const std::string& line_implicit, const std::string& variables) {
                 if (boundaries.size() != 4) throw std::invalid_argument("give four boundaries (faces 0..3)");
                 bspf::FlowOptions o;
                 o.gamma = gamma;
                 o.prandtl = prandtl;
                 o.reynolds = reynolds;
                 o.viscosity_exponent = viscosity_exponent;
                 o.svv = svv;
                 o.svv_cutoff = svv_cutoff;
                 o.quadrature = quadrature;
                 o.cfl = cfl;
                 if (line_implicit == "none") o.line_implicit = bspf::LineImplicit::None;
                 else if (line_implicit == "viscous") o.line_implicit = bspf::LineImplicit::Viscous;
                 else if (line_implicit == "full") o.line_implicit = bspf::LineImplicit::Full;
                 else throw std::invalid_argument("line_implicit must be none, viscous or full");
                 if (variables != "conservative" && variables != "entropy")
                     throw std::invalid_argument("variables must be conservative or entropy");
                 o.entropy_variables = variables == "entropy";
                 nb::gil_scoped_release release;
                 new (self) bspf::CompressibleSolver2D(
                     grid, {boundaries[0], boundaries[1], boundaries[2], boundaries[3]}, o);
             },
             "grid"_a, "boundaries"_a, nb::kw_only(), "gamma"_a = 1.4, "prandtl"_a = 0.72, "reynolds"_a = 0.0,
             "viscosity_exponent"_a = 0.76, "svv"_a = 0.2, "svv_cutoff"_a = 0.5, "quadrature"_a = 5, "cfl"_a = 0.6,
             "line_implicit"_a = "full", "variables"_a = "conservative",
             "Faces: 0 xi = 0, 1 xi = 1, 2 eta = 0, 3 eta = 1. reynolds = 0: Euler. line_implicit: none, viscous "
             "(wall-normal viscous terms) or full (linearized wall-normal operator) preconditioning along eta lines")
        .def_prop_ro("shape", &bspf::CompressibleSolver2D::shape)
        .def_prop_ro("grid", &bspf::CompressibleSolver2D::grid, nb::rv_policy::reference_internal)
        .def("boundary", &bspf::CompressibleSolver2D::boundary, "face"_a, nb::rv_policy::copy)
        .def("set_boundary", &bspf::CompressibleSolver2D::set_boundary, "face"_a, "boundary"_a)
        .def("face_points", &bspf::CompressibleSolver2D::face_points, "face"_a,
             "Physical coordinates (Q, 2) of a face's Gauss points")
        .def("set_source", [state_view](bspf::CompressibleSolver2D& s, const std::optional<InArray>& src) {
                 if (src) s.set_source(state_view(*src));
                 else s.set_source(FieldView{nullptr, {}});
             }, "source"_a.none())
        .def("rhs", [state_view](const bspf::CompressibleSolver2D& s, const InArray& U) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = s.rhs(state_view(U));
                 }
                 return to_numpy(std::move(out));
             }, "U"_a)
        .def("local_dt", [state_view](const bspf::CompressibleSolver2D& s, const InArray& U) {
                 return to_numpy(s.local_dt(state_view(U)));
             }, "U"_a)
        .def("step", [state_view](const bspf::CompressibleSolver2D& s, const InArray& U) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = s.step(state_view(U));
                 }
                 return to_numpy(std::move(out));
             }, "U"_a)
        .def("solve",
             [state_view](const bspf::CompressibleSolver2D& s, const InArray& U0, int max_steps, double tolerance,
                          int anderson, int check, nb::object progress) {
                 const FieldView v = state_view(U0);
                 Field U{v.shape, std::vector<double>(v.data, v.data + bspf::element_count(v.shape))};
                 bspf::SteadyOptions so;
                 so.max_steps = max_steps;
                 so.tolerance = tolerance;
                 so.anderson = anderson;
                 so.check = check;
                 so.progress = [progress](int step, double residual) {
                     nb::gil_scoped_acquire acquire;
                     if (PyErr_CheckSignals() != 0) throw nb::python_error();
                     if (!progress.is_none()) progress(step, residual);
                 };
                 bspf::SteadyResult r;
                 {
                     nb::gil_scoped_release release;
                     r = s.solve(U, so);
                 }
                 nb::dict info;
                 info["steps"] = r.steps;
                 info["residual"] = r.residual;
                 info["converged"] = r.converged;
                 info["restarts"] = r.restarts;
                 info["rhs_evaluations"] = r.rhs_evaluations;
                 info["seconds"] = r.seconds;
                 nb::list history;
                 for (const auto& [step, res] : r.history) history.append(nb::make_tuple(step, res));
                 info["history"] = history;
                 return nb::make_tuple(to_numpy(std::move(U)), info);
             },
             "U"_a, nb::kw_only(), "max_steps"_a = 100000, "tolerance"_a = 1e-10, "anderson"_a = 0, "check"_a = 50,
             "progress"_a = nb::none(),
             "March to steady state; returns (U, info). progress(step, residual) is called every `check` steps")
        .def("prolong", [state_view](const bspf::CompressibleSolver2D& s, const bspf::CompressibleSolver2D& coarse,
                                     const InArray& U) { return to_numpy(s.prolong(coarse, state_view(U))); },
             "coarse"_a, "U"_a, "BSPF interpolation of a coarser solver's state (same map) onto this grid")
        .def("mass_flow", [state_view](const bspf::CompressibleSolver2D& s, const InArray& U) {
                 return Vector(s.mass_flow(state_view(U)));
             }, "U"_a, "Mass flow through every xi = const section");

    m.def("newton_krylov_solve",
          [](const std::vector<bspf::CompressibleSolver2D*>& levels, const InArray& U, double tolerance, int max_newton, int krylov_restart, int max_krylov,
             double eta_max, double start_residual, int pre, int post, int coarse, int cycle_index, double damping,
             int initial, double jacobian_epsilon, double preconditioner_epsilon, nb::object progress) {
              return run_newton<bspf::CpuFlowLevel>(levels, U, tolerance, max_newton, krylov_restart, max_krylov, eta_max,
                                                  start_residual, pre, post, coarse, cycle_index, damping, initial,
                                                  jacobian_epsilon, preconditioner_epsilon, progress);
          },
             "levels"_a, "U"_a, nb::kw_only(), "tolerance"_a = 1e-10, "max_newton"_a = 40, "krylov_restart"_a = 40,
             "max_krylov"_a = 200, "eta_max"_a = 0.1, "start_residual"_a = 1e-3, "pre"_a = 10, "post"_a = 10,
             "coarse"_a = 60, "cycle_index"_a = 1, "damping"_a = 0.5, "initial"_a = 50, "jacobian_epsilon"_a = 1e-7,
             "preconditioner_epsilon"_a = 1e-4, "progress"_a = nb::none(),
             "Newton-Krylov (FGMRES, matrix-free Jacobian) preconditioned by one FAS cycle on nested solvers, finest "
             "first; FAS cycles first until the residual is below start_residual. Returns (U, info)");

    m.def("multigrid_solve",
          [](const std::vector<bspf::CompressibleSolver2D*>& levels, const InArray& U, int pre, int post, int coarse,
             int cycle_index, double damping, int initial, int max_cycles, double tolerance, nb::object progress) {
              return run_multigrid<bspf::CpuFlowLevel>(levels, U, pre, post, coarse, cycle_index, damping, initial,
                                                       max_cycles, tolerance, progress);
          },
             "levels"_a, "U"_a, nb::kw_only(), "pre"_a = 10, "post"_a = 10, "coarse"_a = 60, "cycle_index"_a = 1,
             "damping"_a = 0.5, "initial"_a = 50, "max_cycles"_a = 200, "tolerance"_a = 1e-10, "progress"_a = nb::none(),
             "Nonlinear multigrid (FAS) on nested solvers, finest first: full-weighting residual restriction, BSPF "
             "prolongation of the damped correction (damping), cycle_index 1 (V) or 2 (W); returns (U, info)");

    using FaceTuple = std::tuple<int, int, int>;
    using LinkTuple = std::tuple<FaceTuple, FaceTuple, bool>;
    using PeriodicTuple = std::tuple<FaceTuple, FaceTuple, bool, std::pair<double, double>>;
    auto face = [](const FaceTuple& t) { return bspf::Face{std::get<0>(t), std::get<1>(t), std::get<2>(t)}; };
    auto views = [](const std::vector<InArray>& fields) {
        std::vector<FieldView> v;
        for (const InArray& a : fields) v.push_back(view(a));
        return v;
    };
    auto arrays = [](std::vector<Field>&& fields) {
        std::vector<OutArray> out;
        for (Field& f : fields) out.push_back(to_numpy(std::move(f)));
        return out;
    };
    nb::class_<bspf::MultiPatch2D>(m, "MultiPatch2D",
                                   "Conforming deformed patches joined at complete faces (patch, axis, side)")
        .def("__init__",
             [face](bspf::MultiPatch2D* self, const std::vector<bspf::MappedGrid2D>& patches,
                    const std::vector<LinkTuple>& links, const std::vector<PeriodicTuple>& periodic, double tolerance,
                    int overlap, int stencil) {
                 std::vector<bspf::PatchInterface> interfaces;
                 for (const LinkTuple& l : links)
                     interfaces.push_back({face(std::get<0>(l)), face(std::get<1>(l)), std::get<2>(l)});
                 for (const PeriodicTuple& l : periodic)
                     interfaces.push_back({face(std::get<0>(l)), face(std::get<1>(l)), std::get<2>(l),
                                           {std::get<3>(l).first, std::get<3>(l).second}});
                 new (self) bspf::MultiPatch2D(patches, interfaces, tolerance, overlap, stencil);
             },
             "patches"_a, "interfaces"_a, "periodic"_a = std::vector<PeriodicTuple>(), "tolerance"_a = 1e-10,
             "overlap"_a = 8, "stencil"_a = 12,
             "interfaces: ((patch, axis, side), (patch, axis, side), reverse) per shared face; periodic: "
             "((patch, axis, side), (patch, axis, side), reverse, (dx, dy)) with right face = left face + (dx, dy)")
        .def_prop_ro("patch_count", &bspf::MultiPatch2D::patch_count)
        .def_prop_ro("overlap", &bspf::MultiPatch2D::overlap)
        .def("ghost_depth", [face](const bspf::MultiPatch2D& g, const FaceTuple& f) { return g.ghost_depth(face(f)); }, "face"_a,
             "ghost points used beyond face (patch, axis, side)")
        .def_prop_ro("node_count", &bspf::MultiPatch2D::node_count)
        .def("patch", &bspf::MultiPatch2D::patch, "index"_a, nb::rv_policy::reference_internal)
        .def_prop_ro("boundary_faces", [](const bspf::MultiPatch2D& g) {
                 std::vector<FaceTuple> out;
                 for (const bspf::Face& f : g.boundary_faces()) out.emplace_back(f.patch, f.axis, f.side);
                 return out;
             })
        .def("partial", [views, arrays](const bspf::MultiPatch2D& g, const std::vector<InArray>& fields,
                                        std::pair<int, int> orders, bool average) {
                 std::vector<Field> out;
                 {
                     nb::gil_scoped_release release;
                     out = g.partial(views(fields), orders.first, orders.second, average);
                 }
                 return arrays(std::move(out));
             }, "fields"_a, "orders"_a, "average_shared"_a = true,
             "Physical partial per patch; shared interface nodes averaged")
        .def("average_shared", [views, arrays](const bspf::MultiPatch2D& g, const std::vector<InArray>& fields) {
                 return arrays(g.average_shared(views(fields)));
             }, "fields"_a)
        .def("shared_jump", [views](const bspf::MultiPatch2D& g, const std::vector<InArray>& fields) {
                 return g.shared_jump(views(fields));
             }, "fields"_a, "Largest difference of a shared node's values across patches");

#ifdef BSPF_HAVE_CUDA
    namespace cu = bspf::cuda;
    auto cm = m.def_submodule("cuda", "FP32 CUDA backend: device Arrays, Ops and Grids (results in float32)");
    cm.def("available", &cu::available, "True when a CUDA device is usable");
    cm.def("device_name", &cu::device_name);
    cm.def("synchronize", [] { nb::gil_scoped_release release; cu::synchronize(); },
           "Wait for queued device work (for timing; downloads already wait)");
    auto upload = [](const InArray& a) {
        const FieldView v = view(a);
        return cu::Array::upload(v.data, cu::Shape(v.shape.begin(), v.shape.end()));
    };
    auto download = [](const cu::Array& a) {
        // numpy.empty: uninitialized, and numpy backs large arrays with huge
        // pages, avoiding a page fault per 4 KB on first touch.
        nb::object out = nb::module_::import_("numpy").attr("empty")(nb::tuple(nb::cast(a.shape())), "float64");
        double* data = nb::cast<nb::ndarray<double, nb::c_contig>>(out).data();
        {
            nb::gil_scoped_release release;
            a.download(data);
        }
        return out;
    };
    auto coords = [](const std::vector<Vector>& c) {
        std::vector<std::vector<double>> out;
        for (const Vector& v : c) out.emplace_back(v.data(), v.data() + v.size());
        return out;
    };
    nb::class_<cu::GalerkinAxis>(cm, "GalerkinAxis",
                                 "GalerkinLine's fast evaluation/projection on the device (FP32, spline + cuFFT)")
        .def(nb::init<const bspf::GalerkinLine&>(), "line"_a)
        .def_prop_ro("size", &cu::GalerkinAxis::size)
        .def_prop_ro("points", &cu::GalerkinAxis::points)
        .def("evaluate", [](const cu::GalerkinAxis& g, const Matrix& u, int k) { return g.evaluate_host(u, k); },
             "u"_a, "derivative"_a = 0)
        .def("project", [](const cu::GalerkinAxis& g, const Matrix& v, int k) { return g.project_host(v, k); },
             "values"_a, "derivative"_a = 0);

    nb::class_<cu::CompressibleSolver2D>(cm, "CompressibleSolver2D",
                                         "FP32 device CompressibleSolver2D (fast BSPF Galerkin, Euler, RK4 + Anderson)")
        .def("__init__", [](cu::CompressibleSolver2D* self, const bspf::CompressibleSolver2D& cpu, int refresh,
                            const std::string& precision) {
                 nb::gil_scoped_release release;
                 new (self) cu::CompressibleSolver2D(cpu, refresh, precision);
             }, "cpu"_a, "refresh"_a = 1, "precision"_a = "double",
             "refresh: line-implicit preconditioner refactored every `refresh` steps; precision: double (FP64, "
             "default) or float (FP32, e.g. for multigrid preconditioner levels)")
        .def_prop_ro("shape", &cu::CompressibleSolver2D::shape)
        .def("set_source", [](cu::CompressibleSolver2D& s, const std::optional<InArray>& src) {
                 if (src) s.set_source(view(*src));
                 else s.set_source(FieldView{nullptr, {}});
             }, "source"_a.none())
        .def("rhs", [](const cu::CompressibleSolver2D& s, const InArray& U) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = s.rhs(view(U));
                 }
                 return to_numpy(std::move(out));
             }, "U"_a)
        .def("step", [](const cu::CompressibleSolver2D& s, const InArray& U) {
                 Field out;
                 {
                     nb::gil_scoped_release release;
                     out = s.step(view(U));
                 }
                 return to_numpy(std::move(out));
             }, "U"_a)
        .def("solve",
             [](const cu::CompressibleSolver2D& s, const InArray& U0, int max_steps, double tolerance, int anderson,
                int check, nb::object progress) {
                 const FieldView v = view(U0);
                 Field U{v.shape, std::vector<double>(v.data, v.data + bspf::element_count(v.shape))};
                 bspf::SteadyOptions so;
                 so.max_steps = max_steps;
                 so.tolerance = tolerance;
                 so.anderson = anderson;
                 so.check = check;
                 so.progress = [progress](int step, double residual) {
                     nb::gil_scoped_acquire acquire;
                     if (PyErr_CheckSignals() != 0) throw nb::python_error();
                     if (!progress.is_none()) progress(step, residual);
                 };
                 bspf::SteadyResult r;
                 {
                     nb::gil_scoped_release release;
                     r = s.solve(U, so);
                 }
                 nb::dict info;
                 info["steps"] = r.steps;
                 info["residual"] = r.residual;
                 info["converged"] = r.converged;
                 info["restarts"] = r.restarts;
                 info["rhs_evaluations"] = r.rhs_evaluations;
                 info["seconds"] = r.seconds;
                 nb::list history;
                 for (const auto& [step, res] : r.history) history.append(nb::make_tuple(step, res));
                 info["history"] = history;
                 return nb::make_tuple(to_numpy(std::move(U)), info);
             },
             "U"_a, nb::kw_only(), "max_steps"_a = 100000, "tolerance"_a = 1e-6, "anderson"_a = 0, "check"_a = 50,
             "progress"_a = nb::none());

    cm.def("newton_krylov_solve",
           [](const std::vector<cu::CompressibleSolver2D*>& levels, const InArray& U, double tolerance, int max_newton, int krylov_restart, int max_krylov,
             double eta_max, double start_residual, int pre, int post, int coarse, int cycle_index, double damping,
             int initial, std::optional<double> jacobian_epsilon, double preconditioner_epsilon, nb::object progress,
             cu::CompressibleSolver2D* residual, bool device_krylov, bool device_multigrid) {
               const double jeps = jacobian_epsilon ? *jacobian_epsilon : (residual ? 1e-7 : 1e-3);
               std::unique_ptr<bspf::NewtonDevice> device;
               if (residual && device_krylov) {
                   if (device_multigrid) {
                       bspf::MultigridOptions mo;
                       mo.pre = pre;
                       mo.post = post;
                       mo.coarse = coarse;
                       mo.cycle_index = cycle_index;
                       mo.damping = damping;
                       mo.initial = initial;
                       device = residual->newton_device(levels, mo, preconditioner_epsilon, start_residual, 60, 1);
                   } else {
                       device = residual->newton_device();
                   }
               }
               return run_newton<cu::DeviceFlowLevel>(levels, U, tolerance, max_newton, krylov_restart, max_krylov, eta_max,
                                                  start_residual, pre, post, coarse, cycle_index, damping, initial,
                                                  jeps, preconditioner_epsilon, progress, residual,
                                                  std::move(device));
           },
             "levels"_a, "U"_a, nb::kw_only(), "tolerance"_a = 1e-10, "max_newton"_a = 40, "krylov_restart"_a = 40,
             "max_krylov"_a = 200, "eta_max"_a = 0.1, "start_residual"_a = 1e-3, "pre"_a = 10, "post"_a = 10,
             "coarse"_a = 60, "cycle_index"_a = 1, "damping"_a = 0.5, "initial"_a = 50, "jacobian_epsilon"_a = nb::none(),
             "preconditioner_epsilon"_a = 1e-2, "progress"_a = nb::none(), "residual"_a = nb::none(),
             "device_krylov"_a = true, "device_multigrid"_a = true,
             "residual: an FP64 device solver (precision='double') on the finest grid; Newton, FGMRES and the "
             "Jacobian products then run in double (device_krylov: with device-resident vectors), the FAS preconditioner on the levels (FP32 device solvers: "
             "mixed precision). Newton-Krylov (FGMRES, matrix-free Jacobian) preconditioned by one FAS cycle on nested solvers, finest "
             "first; FAS cycles first until the residual is below start_residual. Returns (U, info)");

    cm.def("multigrid_solve",
           [](const std::vector<cu::CompressibleSolver2D*>& levels, const InArray& U, int pre, int post, int coarse,
              int cycle_index, double damping, int initial, int max_cycles, double tolerance, nb::object progress) {
               return run_multigrid<cu::DeviceFlowLevel>(levels, U, pre, post, coarse, cycle_index, damping, initial,
                                                         max_cycles, tolerance, progress);
           },
             "levels"_a, "U"_a, nb::kw_only(), "pre"_a = 10, "post"_a = 10, "coarse"_a = 60, "cycle_index"_a = 1,
             "damping"_a = 0.5, "initial"_a = 50, "max_cycles"_a = 200, "tolerance"_a = 1e-10, "progress"_a = nb::none(),
             "Nonlinear multigrid (FAS) on nested solvers, finest first: full-weighting residual restriction, BSPF "
             "prolongation of the damped correction (damping), cycle_index 1 (V) or 2 (W); returns (U, info)");

    nb::class_<cu::Array>(cm, "Array", "C-ordered float32 array in device memory")
        .def("__init__", [upload](cu::Array* self, const InArray& a) { new (self) cu::Array(upload(a)); }, "data"_a,
             "Upload a float64 numpy array (rounded to float32)")
        .def_prop_ro("shape", [](const cu::Array& a) { return nb::tuple(nb::cast(a.shape())); })
        .def("get", download, "Download as a float64 numpy array")
        .def("__array__", [download](const cu::Array& a, nb::args, nb::kwargs) { return download(a); });

    nb::class_<cu::Op>(cm, "Op", "A compiled Op on the device")
        .def(nb::init<const Op&>(), "op"_a)
        .def_prop_ro("shape", [](const cu::Op& op) { return std::make_pair(op.rows(), op.cols()); })
        .def_prop_ro("jet_count", &cu::Op::jet_count)
        .def("__call__", [](const cu::Op& op, const cu::Array& f, int axis, const cu::Array* jets) {
                 axis = normalize_axis(axis, f.shape().size());
                 nb::gil_scoped_release release;
                 return op.apply_along(f, axis, jets);
             }, "f"_a, "axis"_a = 0, "jets"_a.none() = nb::none())
        .def("__call__", [upload, download](const cu::Op& op, const InArray& f, int axis, const std::optional<InArray>& jets) {
                 axis = normalize_axis(axis, f.ndim());
                 const cu::Array in = upload(f);
                 std::optional<cu::Array> g;
                 if (jets) g = upload(*jets);
                 cu::Array out;
                 {
                     nb::gil_scoped_release release;
                     out = op.apply_along(in, axis, g ? &*g : nullptr);
                 }
                 return download(out);
             }, "f"_a, "axis"_a = 0, "jets"_a = nb::none(), "numpy in, numpy out (includes transfers)");

    // Each Grid method accepts a device Array (returns one) or a numpy array
    // (uploads, computes, downloads).
    auto grid = nb::class_<cu::Grid>(cm, "Grid", "Tensor-product BSPF calculus on the device (float32)")
        .def(nb::init<const Grid&>(), "grid"_a)
        .def_prop_ro("dimension", &cu::Grid::dimension)
        .def_prop_ro("shape", [](const cu::Grid& g) { return nb::tuple(nb::cast(g.shape())); })
        .def("derivative", &cu::Grid::derivative, "axis"_a, "order"_a = 1, nb::rv_policy::reference_internal);
    grid.def("partial", [](const cu::Grid& g, const cu::Array& f, const std::vector<int>& orders) {
            nb::gil_scoped_release release;
            return g.partial(f, orders);
        }, "f"_a, "orders"_a)
        .def("partial", [upload, download](const cu::Grid& g, const InArray& f, const std::vector<int>& orders) {
            return download(g.partial(upload(f), orders));
        }, "f"_a, "orders"_a)
        .def("laplacian", [](const cu::Grid& g, const cu::Array& f) {
            nb::gil_scoped_release release;
            return g.laplacian(f);
        }, "f"_a)
        .def("laplacian", [upload, download](const cu::Grid& g, const InArray& f) {
            return download(g.laplacian(upload(f)));
        }, "f"_a)
        .def("interpolate", [coords](const cu::Grid& g, const cu::Array& f, const std::vector<Vector>& c,
                                     const std::vector<int>& d) {
            const auto cc = coords(c);
            nb::gil_scoped_release release;
            return g.interpolate(f, cc, d);
        }, "f"_a, "coordinates"_a, "derivatives"_a = std::vector<int>{})
        .def("interpolate", [coords, upload, download](const cu::Grid& g, const InArray& f, const std::vector<Vector>& c,
                                                       const std::vector<int>& d) {
            return download(g.interpolate(upload(f), coords(c), d));
        }, "f"_a, "coordinates"_a, "derivatives"_a = std::vector<int>{})
        .def("integrate", [](const cu::Grid& g, const cu::Array& f,
                             const std::optional<std::vector<std::pair<double, double>>>& bounds) {
            nb::gil_scoped_release release;
            return bounds ? g.integrate(f, *bounds) : g.integrate(f);
        }, "f"_a, "bounds"_a = nb::none())
        .def("integrate", [upload, download](const cu::Grid& g, const InArray& f,
                                             const std::optional<std::vector<std::pair<double, double>>>& bounds) {
            const cu::Array in = upload(f);
            return download(bounds ? g.integrate(in, *bounds) : g.integrate(in));
        }, "f"_a, "bounds"_a = nb::none());
#endif

    m.def("open_knots", &bspf::open_knots, "a"_a, "b"_a, "degree"_a, "n_basis"_a, "clustering"_a = 0.0);
    m.def("endpoint_scaled_knots", &bspf::endpoint_scaled_knots, "a"_a, "b"_a, "degree"_a, "n_basis"_a, "spacing"_a);
    m.def("basis_matrix", &bspf::basis_matrix, "knots"_a, "x"_a, "degree"_a, "derivative"_a = 0);
}
