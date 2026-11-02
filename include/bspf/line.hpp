// SPDX-License-Identifier: GPL-3.0-or-later
// Line: the immutable 1D BSPF geometry and constrained fit on a uniform grid.
// Its factories compile linear operators (bspf::Op) that are applied many times.
#pragma once

#include <memory>
#include <optional>

#include "bspf/endpoints.hpp"
#include "bspf/types.hpp"

namespace bspf {

enum class EndpointMethod { FiniteDifference, Chebyshev };

struct LineOptions {
    int degree = 5;
    std::optional<int> n_basis;           // default min(4 (degree + 1), N)
    std::optional<Vector> knots;          // clamped user knots (exclusive with n_basis)
    std::optional<int> constraint_order;  // q, default degree - 1
    std::optional<int> boundary_points;   // default degree (FD) or min(N, 2 modes) (Chebyshev)
    double lam = 0.0;
    double clustering = 0.0;
    EndpointMethod endpoint_method = EndpointMethod::FiniteDifference;
    std::optional<int> chebyshev_modes;   // default degree + 1
    double chebyshev_alpha = 1e-12;
    int chebyshev_penalty_power = 4;
};

class Op;
namespace detail { struct LineData; }

class Line {
public:
    // Throws std::invalid_argument on an invalid grid or options.
    explicit Line(const Vector& x, const LineOptions& options = {});

    int size() const;
    int degree() const;
    int n_basis() const;
    int constraint_order() const;
    int boundary_points() const;
    double lam() const;
    double spacing() const;

    const Vector& x() const;
    const Vector& knots() const;
    const Vector& weights() const;      // trapezoid weights of the fit
    const Vector& omega() const;        // numpy fftfreq angular frequencies
    const Matrix& basis() const;        // B (N, n_basis) on the grid
    const Matrix& constraint() const;   // C (2q, n_basis): derivatives 0..q-1 at both ends
    const BoundaryBlocks& boundary_blocks() const;
    const Matrix& gram() const;
    const Matrix& kkt() const;
    // c = fit_samples f + fit_jets g, with endpoint jets g = jet_map f unless given.
    const Matrix& fit_samples() const;  // (n_basis, N), the sample part only
    const Matrix& fit_jets() const;     // (n_basis, 2q)
    const Matrix& jet_map() const;      // (2q, N): rows left 0..q-1, then right 0..q-1

    // f -> d^k f on the grid. correction=false drops the Fourier remainder.
    Op derivative(int order, bool correction = true) const;
    // f -> d^k f at fixed points (P, N). Points must lie in [x_0, x_{N-1}].
    Op evaluate(const Vector& points, int derivative = 0, bool correction = true) const;
    // f -> integral over [a, b] (1, N); reversed bounds change the sign.
    Op quadrature() const;
    Op quadrature(double a, double b) const;
    // f -> first or second primitive on the grid, zero value (and slope) at x_0.
    // Add left_value + left_slope (x - x_0) for other integration constants.
    Op primitive(int order = 1) const;
    // f -> estimated endpoint jets (2q, N); with given jets g it returns g.
    Op jets() const;

private:
    std::shared_ptr<const detail::LineData> d_;
};

}  // namespace bspf
