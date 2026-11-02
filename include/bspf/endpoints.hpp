// SPDX-License-Identifier: GPL-3.0-or-later
// Endpoint-jet estimators: linear maps from the first/last `points` samples to
// physical derivatives 0..order-1 at each end (pybspf.plans / pybspf.endpoints).
#pragma once

#include "bspf/types.hpp"

namespace bspf {

struct BoundaryBlocks {
    Matrix left;   // (order, points), applied to f[0:points]
    Matrix right;  // (order, points), applied to f[N-points:N]
};

// Taylor (finite-difference) stencils through `points` samples of spacing dx.
BoundaryBlocks finite_difference_blocks(double dx, int order, int points);

// Regularized local Chebyshev fit with `modes` modes on a window of `points`
// samples spanning `width`; endpoint values are copied exactly.
BoundaryBlocks chebyshev_blocks(double width, int order, int points, int modes, double alpha,
                                int penalty_power);

}  // namespace bspf
