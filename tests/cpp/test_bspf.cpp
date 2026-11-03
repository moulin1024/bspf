// SPDX-License-Identifier: GPL-3.0-or-later
// Consistency checks of the C++ core (parity with pybspf is in tests/python).
#include <cmath>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>

#include "bspf/grid.hpp"

using bspf::Matrix;
using bspf::Vector;

namespace {

int failures = 0;

void check(bool ok, const std::string& what, double value) {
    std::printf("%-4s %-52s %.3e\n", ok ? "ok" : "FAIL", what.c_str(), value);
    if (!ok) ++failures;
}

double max_abs(const Matrix& a) { return a.cwiseAbs().maxCoeff(); }

Matrix sample(const Vector& x, const std::function<double(double)>& f) {
    Matrix out(x.size(), 1);
    for (Eigen::Index i = 0; i < x.size(); ++i) out(i, 0) = f(x(i));
    return out;
}

}  // namespace

int main() {
    const double a = -0.3, b = 1.7;
    auto f = [](double x) { return std::sin(3 * x) + std::exp(0.5 * x); };
    auto df = [](double x) { return 3 * std::cos(3 * x) + 0.5 * std::exp(0.5 * x); };
    auto F = [](double x) { return -std::cos(3 * x) / 3 + 2 * std::exp(0.5 * x); };

    for (int n : {65, 128}) {
        const Vector x = Vector::LinSpaced(n, a, b);
        const bspf::Line line(x);
        const Matrix u = sample(x, f);
        const std::string tag = " (N=" + std::to_string(n) + ")";

        const bspf::Op dense = line.derivative(1);
        check(max_abs(dense.matrix() * u - dense(u)) < 1e-10, "assembled matrix matches the op" + tag,
              max_abs(dense.matrix() * u - dense(u)));
        check(max_abs(dense(u) - sample(x, df)) < 1e-4, "first derivative converges" + tag, max_abs(dense(u) - sample(x, df)));
        const Matrix d0 = line.derivative(0)(u);
        check(max_abs(d0 - u) < 1e-12, "order-0 derivative reproduces samples" + tag, max_abs(d0 - u));

        // Given jets equal to the estimated ones change nothing; otherwise A_j acts.
        const Matrix g = line.jets()(u);
        check(max_abs(dense(u, g) - dense(u)) < 1e-11, "given estimated jets are a no-op" + tag, max_abs(dense(u, g) - dense(u)));
        Matrix g2 = g;
        g2(0, 0) += 1e-3;
        const Matrix affine = dense.matrix() * u + dense.boundary_matrix() * (g2 - g);
        check(max_abs(dense(u, g2) - affine) < 1e-10, "given jets act through the boundary matrix" + tag,
              max_abs(dense(u, g2) - affine));

        const Matrix at_grid = line.evaluate(x)(u);
        check(max_abs(at_grid - u) < 1e-12, "evaluation at grid points" + tag, max_abs(at_grid - u));
        const Matrix slope = line.evaluate(x, 1)(u);
        check(max_abs(slope - dense(u)) < 1e-10, "evaluated derivative matches grid derivative" + tag, max_abs(slope - dense(u)));

        const double exact = F(1.2) - F(0.1);
        const double integral = line.quadrature(0.1, 1.2)(u)(0, 0);
        check(std::abs(integral - exact) < 1e-9, "definite integral" + tag, std::abs(integral - exact));
        const double reversed = line.quadrature(1.2, 0.1)(u)(0, 0);
        check(std::abs(reversed + integral) < 1e-14, "reversed bounds" + tag, std::abs(reversed + integral));
        const Matrix primitive = line.primitive(1)(u);
        const Matrix expected = sample(x, F).array() - F(a);
        check(max_abs(primitive - expected) < 1e-8, "antiderivative" + tag, max_abs(primitive - expected));
        const Matrix back = dense(line.primitive(2)(u));
        check(max_abs(back - primitive) < 1e-4, "d/dx of the second primitive" + tag, max_abs(back - primitive));
    }

    // 3D: separable field with a batch axis; axis-wise and mixed partials.
    const Vector x = Vector::LinSpaced(33, 0.0, 1.0), y = Vector::LinSpaced(40, -1.0, 1.0), z = Vector::LinSpaced(27, 0.0, 2.0);
    const bspf::Grid grid({bspf::Line(x), bspf::Line(y), bspf::Line(z)});
    const int batch = 2;
    bspf::Field u{{33, 40, 27, batch}, std::vector<double>(33 * 40 * 27 * batch)};
    for (int i = 0; i < 33; ++i)
        for (int j = 0; j < 40; ++j)
            for (int k = 0; k < 27; ++k)
                for (int c = 0; c < batch; ++c)
                    u.data[((i * 40 + j) * 27 + k) * batch + c] = (c + 1) * std::sin(x(i)) * std::exp(y(j)) * std::cos(z(k));
    const bspf::Field dxy = grid.partial(u.view(), {1, 1, 0});
    double err = 0;
    for (int i = 0; i < 33; ++i)
        for (int j = 0; j < 40; ++j)
            for (int k = 0; k < 27; ++k)
                for (int c = 0; c < batch; ++c) {
                    const double e = (c + 1) * std::cos(x(i)) * std::exp(y(j)) * std::cos(z(k));
                    err = std::max(err, std::abs(dxy.data[((i * 40 + j) * 27 + k) * batch + c] - e));
                }
    check(err < 1e-4, "3D mixed partial d_x d_y", err);
    const bspf::Field lap = grid.laplacian(u.view());
    const bspf::Field dxx = grid.partial(u.view(), {2, 0, 0});
    const bspf::Field dyy = grid.partial(u.view(), {0, 2, 0});
    const bspf::Field dzz = grid.partial(u.view(), {0, 0, 2});
    double lap_error = 0;
    for (size_t i = 0; i < lap.data.size(); ++i)
        lap_error = std::max(lap_error, std::abs(lap.data[i] - (dxx.data[i] + dyy.data[i] + dzz.data[i])));
    check(lap.shape == u.shape && lap_error == 0, "3D native Laplacian accumulation", lap_error);
    const bspf::Field total = grid.integrate(u.view());
    const double exact = (1 - std::cos(1.0)) * (std::exp(1.0) - std::exp(-1.0)) * std::sin(2.0);
    check(total.shape.size() == 1 && std::abs(total.data[1] - 2 * exact) < 1e-7, "3D box integral, batch kept", std::abs(total.data[1] - 2 * exact));
    Vector xs(1), ys(1), zs(1);
    xs << 0.3;
    ys << 0.25;
    zs << 1.1;
    const bspf::Field point = grid.interpolate(u.view(), {xs, ys, zs});
    const double pe = std::sin(0.3) * std::exp(0.25) * std::cos(1.1);
    check(std::abs(point.data[0] - pe) < 1e-8, "3D interpolation", std::abs(point.data[0] - pe));

    bool threw = false;
    try { bspf::Line(x).evaluate(Vector::Constant(1, 1.5)); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "evaluation outside the domain rejected", 0.0);

    // Shapes are checked before any strided read.
    {
        const bspf::Line line(Vector::LinSpaced(31, 0.0, 1.0));
        const bspf::Op d1 = line.derivative(1);
        bool rows = false, jet_rows = false;
        try { d1(Matrix::Zero(10, 2)); } catch (const std::invalid_argument&) { rows = true; }
        try { d1(Matrix::Zero(31, 2), Matrix::Zero(3, 2)); } catch (const std::invalid_argument&) { jet_rows = true; }
        check(rows, "wrong sample count rejected", 0.0);
        check(jet_rows, "wrong jet count rejected", 0.0);
        bspf::LineOptions o;
        o.constraint_order = 0;
        const Matrix none = bspf::Line(Vector::LinSpaced(31, 0.0, 1.0), o).jets()(Matrix::Ones(31, 3));
        check(none.rows() == 0 && none.cols() == 3, "jets of a q = 0 line are (0, m)", 0.0);
    }

    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
