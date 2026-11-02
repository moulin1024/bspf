// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/endpoints.hpp"

#include <cmath>
#include <stdexcept>

namespace bspf {

namespace {

// Right block: mirrored samples, odd derivatives change sign.
Matrix mirror(const Matrix& left) {
    Matrix right(left.rows(), left.cols());
    for (Eigen::Index k = 0; k < left.rows(); ++k) {
        const double sign = (k % 2) ? -1.0 : 1.0;
        for (Eigen::Index j = 0; j < left.cols(); ++j) right(k, j) = sign * left(k, left.cols() - 1 - j);
    }
    return right;
}

}  // namespace

BoundaryBlocks finite_difference_blocks(double dx, int order, int points) {
    if (order < 0 || points < std::max(1, order)) throw std::invalid_argument("require max(1, order) <= points");
    // V[k, j] = j^k / k!: Taylor moments of the window offsets.
    Matrix V(points, points);
    for (int k = 0; k < points; ++k) {
        double factorial = 1.0;
        for (int i = 2; i <= k; ++i) factorial *= i;
        for (int j = 0; j < points; ++j) V(k, j) = std::pow(static_cast<double>(j), k) / factorial;
    }
    const Matrix X = V.partialPivLu().solve(Matrix::Identity(points, order));  // (points, order)
    Matrix left = X.transpose();
    for (int k = 0; k < order; ++k) left.row(k) /= std::pow(dx, k);
    return {left, mirror(left)};
}

BoundaryBlocks chebyshev_blocks(double width, int order, int points, int modes, double alpha,
                                int penalty_power) {
    if (order == 0) return {Matrix(0, points), Matrix(0, points)};
    if (modes < std::max(1, order) || points < modes) throw std::invalid_argument("require order <= modes <= points");
    // Chebyshev columns on the window mapped to [-1, 1].
    Matrix V(points, modes);
    for (int i = 0; i < points; ++i) {
        const double xi = points == 1 ? -1.0 : -1.0 + 2.0 * i / (points - 1);
        V(i, 0) = 1.0;
        if (modes > 1) V(i, 1) = xi;
        for (int k = 2; k < modes; ++k) V(i, k) = 2 * xi * V(i, k - 1) - V(i, k - 2);
    }
    Vector penalty(modes);
    for (int k = 0; k < modes; ++k)
        penalty(k) = k < 2 ? 0.0 : std::pow(static_cast<double>(k) / std::max(1, modes - 1), penalty_power);
    // Regularized least squares by QR of the augmented matrix (no normal equations).
    Matrix augmented(points + modes, modes);
    augmented.topRows(points) = V;
    augmented.bottomRows(modes) = (std::sqrt(alpha) * penalty).asDiagonal();
    Eigen::HouseholderQR<Matrix> qr(augmented);
    const Matrix Q = qr.householderQ() * Matrix::Identity(points + modes, modes);
    const Matrix R = qr.matrixQR().topLeftCorner(modes, modes).triangularView<Eigen::Upper>();
    const Matrix projector = R.triangularView<Eigen::Upper>().solve(Q.topRows(points).transpose());
    // T_j^(k)(1) = prod_{l<k} (j^2 - l^2) / (2k-1)!!; T_j^(k)(-1) = (-1)^(j-k) T_j^(k)(1).
    Vector right_derivative = Vector::Ones(modes);
    Matrix left(order, points);
    for (int k = 0; k < order; ++k) {
        if (k)
            for (int j = 0; j < modes; ++j)
                right_derivative(j) *= (static_cast<double>(j) * j - static_cast<double>(k - 1) * (k - 1)) / (2 * k - 1);
        RowVector at_left(modes);
        for (int j = 0; j < modes; ++j) at_left(j) = ((j - k) % 2 ? -1.0 : 1.0) * right_derivative(j);
        left.row(k) = std::pow(2.0 / width, k) * (at_left * projector);
    }
    left.row(0).setZero();
    left(0, 0) = 1.0;  // endpoint values copied exactly
    return {left, mirror(left)};
}

}  // namespace bspf
