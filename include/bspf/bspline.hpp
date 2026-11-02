// SPDX-License-Identifier: GPL-3.0-or-later
// Clamped knot vectors, Cox-de Boor evaluation and exact spline primitives.
// Mirrors pybspf.basis: matrices are (evaluation points, basis functions).
#pragma once

#include <utility>

#include "bspf/types.hpp"

namespace bspf {

// Clamped knots with optional tanh clustering toward both endpoints
// (n_basis + degree + 1 entries).
Vector open_knots(double a, double b, int degree, int n_basis, double clustering = 0.0);

// Clamped knots with `degree` short spans of width `spacing` at each end.
Vector endpoint_scaled_knots(double a, double b, int degree, int n_basis, double spacing);

// B-splines (or their physical derivatives) of `degree` on `knots` at `x`.
// The right endpoint uses the left-hand limit; outside the knots the basis is zero.
Matrix basis_matrix(const Vector& knots, const Vector& x, int degree, int derivative = 0);

// Knots/coefficients of the degree+1 primitive that vanishes at the left end.
std::pair<Vector, Matrix> primitive_coefficients(const Vector& knots, const Matrix& coefficients,
                                                 int degree);

// Values at `x` of the `order`-th primitive (integration constants zero at the left end).
Matrix spline_primitive(Vector knots, Matrix coefficients, const Vector& x, int degree,
                        int order = 1);

}  // namespace bspf
