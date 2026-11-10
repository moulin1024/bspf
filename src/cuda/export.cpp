// SPDX-License-Identifier: GPL-3.0-or-later
// Host side of the CUDA backend: flattens an Op and its LineData into plain
// tables (built in double, rounded to float on upload), and the device Grid.
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <tuple>

#include "../line_data.hpp"
#include "bspf/cuda.hpp"
#include "bspf/grid.hpp"
#include "bspf/op.hpp"
#include "tables.hpp"

namespace bspf::detail {

struct OpAccess {
    static cuda::detail::HostMatrix host(const Matrix& a) {
        cuda::detail::HostMatrix m;
        m.rows = a.rows();
        m.cols = a.cols();
        m.data.assign(a.data(), a.data() + a.size());
        return m;
    }

    static Matrix spline(const Op& op) {
        if (!op.spline_banded_) return op.spline_;
        const Banded& b = *op.spline_banded_;
        Matrix s = Matrix::Zero(b.rows, b.cols);
        for (Eigen::Index i = 0; i < b.rows; ++i)
            for (Eigen::Index k = 0; k < b.width; ++k) s(i, b.start[i] + k) = b.values[i * b.width + k];
        return s;
    }

    static cuda::detail::OpTables tables(const Op& op) {
        const LineData& d = *op.line_;
        cuda::detail::OpTables t;
        t.rows = op.rows_;
        t.cols = op.cols_;
        t.jets_only = op.jets_only_;
        t.correction = op.correction_;
        t.pass_remainder = op.pass_remainder_;
        if (!op.jets_only_) t.spline = host(spline(op));
        t.multiplier.assign(op.multiplier_.data(), op.multiplier_.data() + op.multiplier_.size());
        if (op.modes_real_.size()) {
            const Eigen::Index k = op.modes_real_.rows(), h = op.modes_real_.cols();
            Matrix interleaved(k, 2 * h);
            for (Eigen::Index j = 0; j < h; ++j) {
                interleaved.col(2 * j) = op.modes_real_.col(j);
                interleaved.col(2 * j + 1) = -op.modes_imag_.col(j);
            }
            t.modes = host(interleaved);
        }
        if (op.lift_.size()) t.lift = host(op.lift_);

        cuda::detail::LineTables& l = t.line;
        l.key = op.line_.get();
        l.keepalive = op.line_;
        l.n = static_cast<int>(d.size());
        l.nb = static_cast<int>(d.n_basis());
        l.q = d.q;
        l.p = d.p;
        l.fit = host(d.fit_samples);
        l.fit_total = host(d.fit_samples + d.fit_jets * d.jet_map);
        l.fit_jets = host(d.fit_jets);
        l.basis = host(d.basis);
        l.jets_left = host(d.blocks.left);
        l.jets_right = host(d.blocks.right);
        return t;
    }
};

}  // namespace bspf::detail

namespace bspf::cuda {

Op::Op(const bspf::Op& op) : impl_(detail::make_op(bspf::detail::OpAccess::tables(op))) {}

Array Op::apply_along(const Array& in, int axis, const Array* jets) const {
    if (axis < 0 || axis >= static_cast<int>(in.shape().size())) throw std::invalid_argument("axis out of range");
    Shape shape = in.shape();
    shape[axis] = rows();
    Array out(shape);
    apply_along(in, out, axis, jets, false);
    return out;
}

struct Grid::Impl {
    bspf::Grid grid;
    std::mutex mutex;
    std::map<std::pair<int, int>, std::unique_ptr<Op>> derivatives;  // never erased
    using Key = std::tuple<int, int, int, std::vector<double>>;
    std::map<Key, std::shared_ptr<const Op>> others;  // evaluation (0) / quadrature (1)
    static constexpr size_t capacity = 64;

    explicit Impl(const bspf::Grid& g) : grid(g) {}

    std::shared_ptr<const Op> other(int kind, int axis, int order, std::vector<double> values) {
        Key key{kind, axis, order, std::move(values)};
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = others.find(key);
            if (it != others.end()) return it->second;
        }
        const Line& l = grid.line(axis);
        const std::vector<double>& v = std::get<3>(key);
        auto op = kind == 0
                      ? std::make_shared<const Op>(l.evaluate(Eigen::Map<const Vector>(v.data(), v.size()), order))
                      : std::make_shared<const Op>(l.quadrature(v[0], v[1]));
        std::lock_guard<std::mutex> lock(mutex);
        if (others.size() >= capacity) others.clear();
        return others.try_emplace(std::move(key), std::move(op)).first->second;
    }

    void check(const Array& f) const {
        bool ok = f.shape().size() >= static_cast<size_t>(grid.dimension());
        for (int i = 0; ok && i < grid.dimension(); ++i) ok = f.shape()[i] == grid.line(i).size();
        if (!ok) throw std::invalid_argument("field's leading shape does not match the grid");
    }
};

Grid::Grid(const bspf::Grid& grid) : impl_(std::make_shared<Impl>(grid)) {}

int Grid::dimension() const { return impl_->grid.dimension(); }

Shape Grid::shape() const {
    const bspf::Shape s = impl_->grid.shape();
    return Shape(s.begin(), s.end());
}

const Op& Grid::derivative(int axis, int order) const {
    Impl& d = *impl_;
    {
        std::lock_guard<std::mutex> lock(d.mutex);
        auto it = d.derivatives.find({axis, order});
        if (it != d.derivatives.end()) return *it->second;
    }
    auto op = std::make_unique<Op>(d.grid.line(axis).derivative(order));
    std::lock_guard<std::mutex> lock(d.mutex);
    auto& slot = d.derivatives[{axis, order}];
    if (!slot) slot = std::move(op);
    return *slot;
}

Array Grid::partial(const Array& f, const std::vector<int>& orders) const {
    impl_->check(f);
    if (static_cast<int>(orders.size()) != dimension()) throw std::invalid_argument("orders must have one entry per spatial axis");
    std::optional<Array> result;
    for (int axis = 0; axis < dimension(); ++axis) {
        if (orders[axis] < 0) throw std::invalid_argument("derivative orders must be nonnegative");
        if (!orders[axis]) continue;
        result = derivative(axis, orders[axis]).apply_along(result ? *result : f, axis);
    }
    return result ? *result : f.clone();
}

Array Grid::laplacian(const Array& f) const {
    impl_->check(f);
    Array out(f.shape());
    for (int axis = 0; axis < dimension(); ++axis) derivative(axis, 2).apply_along(f, out, axis, nullptr, axis != 0);
    return out;
}

Array Grid::interpolate(const Array& f, const std::vector<std::vector<double>>& coordinates,
                        const std::vector<int>& derivatives) const {
    impl_->check(f);
    if (static_cast<int>(coordinates.size()) != dimension()) throw std::invalid_argument("provide one coordinate vector per spatial axis");
    if (!derivatives.empty() && static_cast<int>(derivatives.size()) != dimension())
        throw std::invalid_argument("derivatives must have one entry per spatial axis");
    std::optional<Array> result;
    for (int axis = 0; axis < dimension(); ++axis) {
        const auto op = impl_->other(0, axis, derivatives.empty() ? 0 : derivatives[axis], coordinates[axis]);
        result = op->apply_along(result ? *result : f, axis);
    }
    return *result;
}

Array Grid::integrate(const Array& f) const {
    std::vector<std::pair<double, double>> bounds;
    for (int a = 0; a < dimension(); ++a) {
        const Line& l = impl_->grid.line(a);
        bounds.emplace_back(l.x()(0), l.x()(l.size() - 1));
    }
    return integrate(f, bounds);
}

Array Grid::integrate(const Array& f, const std::vector<std::pair<double, double>>& bounds) const {
    impl_->check(f);
    if (static_cast<int>(bounds.size()) != dimension()) throw std::invalid_argument("provide one (a, b) pair per spatial axis");
    std::optional<Array> result;
    for (int axis = dimension() - 1; axis >= 0; --axis) {  // innermost first: shrinks the data early
        const auto op = impl_->other(1, axis, 0, {bounds[axis].first, bounds[axis].second});
        result = op->apply_along(result ? *result : f, axis);
    }
    Shape shape = result->shape();
    shape.erase(shape.begin(), shape.begin() + dimension());
    return result->reshaped(shape);
}

}  // namespace bspf::cuda
