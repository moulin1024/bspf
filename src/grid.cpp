// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/grid.hpp"

#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace bspf {

using Index = Eigen::Index;

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
    // Derivative ops are kept for the grid's lifetime (references are handed out).
    std::map<std::pair<int, int>, std::unique_ptr<Op>> derivatives;
    // Evaluation (kind 0) and quadrature (kind 1) ops by axis, order and
    // coordinates; bounded, since probe points may change from call to call.
    using Key = std::tuple<int, int, int, std::vector<double>>;
    std::map<Key, std::shared_ptr<const Op>> others;
    static constexpr size_t capacity = 64;
    std::map<int, std::unique_ptr<GalerkinLine>> galerkin;
};

namespace {

// Looks up `key`; on a miss compiles outside the lock (compiles can be slow
// and other threads keep reading the cache), then keeps the first insertion.
// Entries are never erased, so the returned reference stays valid.
template <class Map, class Build>
typename Map::mapped_type& cached(std::mutex& mutex, Map& map, const typename Map::key_type& key, Build&& build) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = map.find(key);
        if (it != map.end() && it->second) return it->second;
    }
    typename Map::mapped_type value = build();
    std::lock_guard<std::mutex> lock(mutex);
    auto& slot = map[key];
    if (!slot) slot = std::move(value);
    return slot;
}

}  // namespace

std::shared_ptr<const Op> Grid::cached_op(int kind, int axis, int order, std::vector<double> values) const {
    const Line& l = line(axis);
    Cache::Key key{kind, axis, order, std::move(values)};
    {
        std::lock_guard<std::mutex> lock(cache_->mutex);
        auto it = cache_->others.find(key);
        if (it != cache_->others.end()) return it->second;  // copied under the lock
    }
    const std::vector<double>& v = std::get<3>(key);
    std::shared_ptr<const Op> op =
        kind == 0 ? std::make_shared<const Op>(l.evaluate(Eigen::Map<const Vector>(v.data(), v.size()), order))
                  : std::make_shared<const Op>(l.quadrature(v[0], v[1]));
    std::lock_guard<std::mutex> lock(cache_->mutex);
    if (cache_->others.size() >= Cache::capacity) cache_->others.clear();
    return cache_->others.try_emplace(std::move(key), std::move(op)).first->second;
}

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
    return *cached(cache_->mutex, cache_->derivatives, {axis, order},
                   [&] { return std::make_unique<Op>(l.derivative(order)); });
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

const GalerkinLine& Grid::galerkin(int axis) const {
    const Line& l = line(axis);
    return *cached(cache_->mutex, cache_->galerkin, axis, [&] { return std::make_unique<GalerkinLine>(l); });
}

namespace {

// Applies the Galerkin second derivative W along `axis` of a C-ordered field
// into `out` (that axis of length N - 2), assigning or accumulating. Lines are
// gathered 128 at a time; the kernel's (N, b) result is read at rows 1..N-2.
void second_derivative_into(const GalerkinLine& g, const FieldView& in, int axis, double* out, bool accumulate) {
    using Rows = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Stride = Eigen::OuterStride<>;
    constexpr Index block = 128;
    const Index n = in.shape[axis];
    Index before = 1, after = 1;
    for (int i = 0; i < axis; ++i) before *= in.shape[i];
    for (size_t i = axis + 1; i < in.shape.size(); ++i) after *= in.shape[i];
    Matrix u, y;
    if (after == 1) {  // contiguous lines, n apart
        for (Index l0 = 0; l0 < before; l0 += block) {
            const Index b = std::min(block, before - l0);
            u = Eigen::Map<const Matrix>(in.data + l0 * n, n, b);
            g.second_derivative_into(u, y);
            Eigen::Map<Matrix> dest(out + l0 * (n - 2), n - 2, b);
            if (accumulate) dest += y.middleRows(1, n - 2);
            else dest = y.middleRows(1, n - 2);
        }
        return;
    }
    for (Index l = 0; l < before; ++l)  // interleaved lines within each slab
        for (Index t0 = 0; t0 < after; t0 += block) {
            const Index b = std::min(block, after - t0);
            u = Eigen::Map<const Rows, 0, Stride>(in.data + l * n * after + t0, n, b, Stride(after));
            g.second_derivative_into(u, y);
            Eigen::Map<Rows, 0, Stride> dest(out + l * (n - 2) * after + t0, n - 2, b, Stride(after));
            if (accumulate) dest += y.middleRows(1, n - 2);
            else dest = y.middleRows(1, n - 2);
        }
}

// E along `axis`: out_i = in_{i+1} + e(i, 0) in_0 + e(i, 1) in_{N-1}, one fused
// pass that is contiguous over lines (no gather).
Field extend_along(const FieldView& in, int axis, const Matrix& e) {
    const Index n = in.shape[axis];
    Index before = 1, after = 1;
    for (int i = 0; i < axis; ++i) before *= in.shape[i];
    for (size_t i = axis + 1; i < in.shape.size(); ++i) after *= in.shape[i];
    Field out;
    out.shape = in.shape;
    out.shape[axis] = n - 2;
    out.data.resize(static_cast<size_t>(element_count(out.shape)));
    const double* e0 = e.col(0).data();
    const double* e1 = e.col(1).data();
    for (Index l = 0; l < before; ++l) {
        const double* src = in.data + l * n * after;
        double* dst = out.data.data() + l * (n - 2) * after;
        if (after == 1) {
            const double first = src[0], last = src[n - 1];
            for (Index i = 0; i < n - 2; ++i) dst[i] = src[i + 1] + e0[i] * first + e1[i] * last;
            continue;
        }
        const double* first = src;
        const double* last = src + (n - 1) * after;
        for (Index i = 0; i < n - 2; ++i) {
            const double a = e0[i], b = e1[i];
            const double* row = src + (i + 1) * after;
            double* o = dst + i * after;
            for (Index t = 0; t < after; ++t) o[t] = row[t] + a * first[t] + b * last[t];
        }
    }
    return out;
}

}  // namespace

Field Grid::weak_laplacian(const FieldView& f, const FieldView* wall_values) const {
    check(f);
    if (wall_values && wall_values->shape != f.shape) throw std::invalid_argument("wall_values must have the field's shape");
    const int dim = dimension();
    for (int a = 0; a < dim; ++a) galerkin(a);  // build outside the loops
    auto E = [&](const FieldView& v, int a) { return extend_along(v, a, galerkin(a).wall_extension()); };
    // Interior result: sum over axes a of W_a (x)_{b != a} E_b u. The cheap E's
    // (copies plus rank-2 wall terms) go first, so each W_a writes straight into
    // one interior accumulator.
    Field interior_field;
    interior_field.shape = f.shape;
    for (int a = 0; a < dim; ++a) interior_field.shape[a] -= 2;
    interior_field.data.resize(static_cast<size_t>(element_count(interior_field.shape)));
    std::optional<Field> interior(std::move(interior_field));
    for (int a = 0; a < dim; ++a) {
        std::optional<Field> extended;
        for (int b = 0; b < dim; ++b)
            if (b != a) extended = E(extended ? extended->view() : f, b);
        second_derivative_into(galerkin(a), extended ? extended->view() : f, a, interior->data.data(), a != 0);
    }
    // Interior runs: for every interior index of the leading spatial axes, the
    // last spatial axis's interior (times the batch size) is contiguous in both
    // the full field and the interior block.
    const Index batch = element_count(f.shape) / element_count(shape());
    auto for_interior_runs = [&](auto&& fn) {
        const Index run = (f.shape[dim - 1] - 2) * batch;
        std::vector<Index> idx(dim > 1 ? dim - 1 : 0, 1);  // interior indices of axes 0..dim-2
        Index count = 1;
        for (int a = 0; a + 1 < dim; ++a) count *= f.shape[a] - 2;
        for (Index c = 0; c < count; ++c) {
            Index full = 0, inner = 0, rem = c;
            for (int a = dim - 2; a >= 0; --a) {
                const Index i = rem % (f.shape[a] - 2) + 1;
                rem /= f.shape[a] - 2;
                Index fs = batch, is = batch;
                for (int b2 = a + 1; b2 < dim; ++b2) { fs *= f.shape[b2]; is *= f.shape[b2] - 2; }
                full += i * fs;
                inner += (i - 1) * is;
            }
            fn(full + batch, inner, run);  // + batch: skip the last axis's first wall
        }
    };
    Field out;
    out.shape = f.shape;
    if (wall_values) {
        // Mass coupling to the walls' values of the result (interior zeroed).
        out.data.assign(wall_values->data, wall_values->data + element_count(f.shape));
        Field walls;
        walls.shape = f.shape;
        walls.data = out.data;
        for_interior_runs([&](Index full, Index, Index run) { std::fill_n(walls.data.begin() + full, run, 0.0); });
        Field coupling = E(walls.view(), 0);
        for (int b = 1; b < dim; ++b) coupling = E(coupling.view(), b);
        for (size_t i = 0; i < coupling.data.size(); ++i) interior->data[i] -= coupling.data[i];
    } else {
        out.data.assign(static_cast<size_t>(element_count(f.shape)), 0.0);
    }
    for_interior_runs([&](Index full, Index inner, Index run) {
        std::copy_n(interior->data.begin() + inner, run, out.data.begin() + full);
    });
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
        const Vector& c = coordinates[axis];
        const auto op = cached_op(0, axis, derivatives.empty() ? 0 : derivatives[axis],
                                  std::vector<double>(c.data(), c.data() + c.size()));
        result = apply_along(*op, result ? result->view() : f, axis);
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
        const auto op = cached_op(1, axis, 0, {bounds[axis].first, bounds[axis].second});
        result = apply_along(*op, result ? result->view() : f, axis);
    }
    result->shape.erase(result->shape.begin(), result->shape.begin() + dimension());
    return std::move(*result);
}

}  // namespace bspf
