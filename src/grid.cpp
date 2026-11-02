// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/grid.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace bspf {

Eigen::Index element_count(const Shape& shape) {
    Eigen::Index count = 1;
    for (Eigen::Index s : shape) count *= s;
    return count;
}

void apply_along(const Op& op, const FieldView& in, double* out, int axis, const double* jets, bool accumulate) {
    const int ndim = static_cast<int>(in.shape.size());
    if (axis < 0 || axis >= ndim) throw std::invalid_argument("axis out of range");
    const Eigen::Index n = in.shape[axis];
    if (n != op.cols())
        throw std::invalid_argument("expected " + std::to_string(op.cols()) + " samples along axis " +
                                    std::to_string(axis) + ", got " + std::to_string(n));
    Eigen::Index before = 1, after = 1;
    for (int i = 0; i < axis; ++i) before *= in.shape[i];
    for (int i = axis + 1; i < ndim; ++i) after *= in.shape[i];
    const Eigen::Index P = op.rows(), J = op.jet_count();
    if (before == 0 || after == 0) return;
    if (after == 1) {  // the axis is innermost: all lines contiguous, one call
        const LinesView g{jets, 1, J, before};
        op.apply({in.data, 1, n, before}, {out, 1, P, before}, jets ? &g : nullptr, accumulate);
        return;
    }
    // Keep scratch storage across slabs; it belongs to this call, not the op.
    Op::Workspace workspace;
    // Each slab of `after` interleaved lines is one GEMM (or one batched FFT).
    for (Eigen::Index l = 0; l < before; ++l) {
        const LinesView g{jets ? jets + l * J * after : nullptr, after, 1, after};
        op.apply_with_workspace({in.data + l * n * after, after, 1, after}, {out + l * P * after, after, 1, after},
                                jets ? &g : nullptr, accumulate, workspace);
    }
}

Field apply_along(const Op& op, const FieldView& in, int axis, const double* jets) {
    if (axis < 0 || axis >= static_cast<int>(in.shape.size())) throw std::invalid_argument("axis out of range");
    Field out;
    out.shape = in.shape;
    out.shape[axis] = op.rows();
    out.data.resize(static_cast<size_t>(element_count(out.shape)));
    apply_along(op, in, out.data.data(), axis, jets);
    return out;
}

struct Grid::Cache {
    std::mutex mutex;
    std::map<std::pair<int, int>, std::unique_ptr<Op>> derivatives;
};

Grid::Grid(std::vector<Line> lines) : lines_(std::move(lines)), cache_(std::make_shared<Cache>()) {
    if (lines_.empty() || lines_.size() > 3) throw std::invalid_argument("provide one to three lines");
}

const Line& Grid::line(int axis) const {
    if (axis < 0 || axis >= dimension()) throw std::invalid_argument("axis out of range");
    return lines_[axis];
}

Shape Grid::shape() const {
    Shape s;
    for (const Line& l : lines_) s.push_back(l.size());
    return s;
}

const Op& Grid::derivative(int axis, int order) const {
    const Line& l = line(axis);
    std::lock_guard<std::mutex> lock(cache_->mutex);
    auto& slot = cache_->derivatives[{axis, order}];
    if (!slot) slot = std::make_unique<Op>(l.derivative(order));
    return *slot;
}

void Grid::check(const FieldView& f) const {
    bool ok = f.shape.size() >= lines_.size();
    for (int i = 0; ok && i < dimension(); ++i) ok = f.shape[i] == lines_[i].size();
    if (!ok) {
        std::string expected;
        for (const Line& l : lines_) expected += (expected.empty() ? "" : ", ") + std::to_string(l.size());
        throw std::invalid_argument("expected leading spatial shape (" + expected + ")");
    }
}

namespace {

Field copy(const FieldView& f) {
    Field out;
    out.shape = f.shape;
    out.data.assign(f.data, f.data + element_count(f.shape));
    return out;
}

}  // namespace

Field Grid::partial(const FieldView& f, const std::vector<int>& orders) const {
    check(f);
    if (static_cast<int>(orders.size()) != dimension()) throw std::invalid_argument("orders must have one entry per spatial axis");
    for (int k : orders)
        if (k < 0) throw std::invalid_argument("derivative orders must be nonnegative");
    std::optional<Field> result;
    for (int axis = 0; axis < dimension(); ++axis) {
        if (!orders[axis]) continue;
        result = apply_along(derivative(axis, orders[axis]), result ? result->view() : f, axis);
    }
    return result ? std::move(*result) : copy(f);
}

Field Grid::laplacian(const FieldView& f) const {
    check(f);
    Field out;
    out.shape = f.shape;
    out.data.resize(static_cast<size_t>(element_count(f.shape)));
    for (int axis = 0; axis < dimension(); ++axis)
        apply_along(derivative(axis, 2), f, out.data.data(), axis, nullptr, axis != 0);
    return out;
}

Field Grid::interpolate(const FieldView& f, const std::vector<Vector>& coordinates,
                        const std::vector<int>& derivatives) const {
    check(f);
    if (static_cast<int>(coordinates.size()) != dimension()) throw std::invalid_argument("provide one coordinate vector per spatial axis");
    if (!derivatives.empty() && static_cast<int>(derivatives.size()) != dimension())
        throw std::invalid_argument("derivatives must have one entry per spatial axis");
    std::optional<Field> result;
    for (int axis = 0; axis < dimension(); ++axis) {
        const Op op = lines_[axis].evaluate(coordinates[axis], derivatives.empty() ? 0 : derivatives[axis]);
        result = apply_along(op, result ? result->view() : f, axis);
    }
    return std::move(*result);
}

Field Grid::integrate(const FieldView& f) const {
    std::vector<std::pair<double, double>> bounds;
    for (const Line& l : lines_) bounds.emplace_back(l.x()(0), l.x()(l.size() - 1));
    return integrate(f, bounds);
}

Field Grid::integrate(const FieldView& f, const std::vector<std::pair<double, double>>& bounds) const {
    check(f);
    if (static_cast<int>(bounds.size()) != dimension()) throw std::invalid_argument("provide one (a, b) pair per spatial axis");
    std::optional<Field> result;
    // Innermost axis first: each step shrinks the data before the next one.
    for (int axis = dimension() - 1; axis >= 0; --axis) {
        const Op op = lines_[axis].quadrature(bounds[axis].first, bounds[axis].second);
        result = apply_along(op, result ? result->view() : f, axis);
    }
    result->shape.erase(result->shape.begin(), result->shape.begin() + dimension());
    return std::move(*result);
}

}  // namespace bspf
