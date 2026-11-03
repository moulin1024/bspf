// SPDX-License-Identifier: GPL-3.0-or-later
// nanobind module bspf_cpp._core. Fields are C-ordered float64 arrays whose
// leading axes are spatial; ops act along any axis without transposing.
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

#include "bspf/bspline.hpp"
#include "bspf/galerkin.hpp"
#include "bspf/grid.hpp"
#include "bspf/mapped.hpp"

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
        .def("__init__", [](bspf::GalerkinLine* self, const Line& line, double mass_tolerance, double operator_tolerance) {
                 bspf::GalerkinOptions o;
                 o.mass_tolerance = mass_tolerance;
                 o.operator_tolerance = operator_tolerance;
                 new (self) bspf::GalerkinLine(line, o);
             }, "line"_a, nb::kw_only(), "mass_tolerance"_a = 1e-12, "operator_tolerance"_a = 1e-14)
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
                std::optional<int> chebyshev_modes, double chebyshev_alpha, int chebyshev_penalty_power) {
                 new (self) bspf::MappedGrid2D(bottom, right, top, left, shape.first, shape.second,
                                               line_options(degree, n_basis, std::nullopt, constraint_order,
                                                            boundary_points, lam, clustering, method, chebyshev_modes,
                                                            chebyshev_alpha, chebyshev_penalty_power));
             },
             "bottom"_a, "right"_a, "top"_a, "left"_a, "shape"_a, nb::kw_only(), "degree"_a = 5,
             "n_basis"_a = nb::none(), "constraint_order"_a = nb::none(), "boundary_points"_a = nb::none(),
             "lam"_a = 0.0, "clustering"_a = 0.0, "endpoint_method"_a = "finite_difference",
             "chebyshev_modes"_a = nb::none(), "chebyshev_alpha"_a = 1e-12, "chebyshev_penalty_power"_a = 4)
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

    m.def("open_knots", &bspf::open_knots, "a"_a, "b"_a, "degree"_a, "n_basis"_a, "clustering"_a = 0.0);
    m.def("endpoint_scaled_knots", &bspf::endpoint_scaled_knots, "a"_a, "b"_a, "degree"_a, "n_basis"_a, "spacing"_a);
    m.def("basis_matrix", &bspf::basis_matrix, "knots"_a, "x"_a, "degree"_a, "derivative"_a = 0);
}
