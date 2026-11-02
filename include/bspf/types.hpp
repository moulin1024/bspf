// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <Eigen/Dense>

namespace bspf {

// Fields are column-major (samples, batch): every column is one contiguous field.
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using ComplexMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using MatrixRef = Eigen::Ref<const Matrix>;
using VectorRef = Eigen::Ref<const Vector>;

}  // namespace bspf
