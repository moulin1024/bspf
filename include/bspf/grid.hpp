// SPDX-License-Identifier: GPL-3.0-or-later
// Tensor-product BSPF calculus on 1-3D rectangles. Fields are C-ordered
// (row-major) arrays whose leading axes are the spatial axes; any trailing
// axes are batches or components.
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "bspf/galerkin.hpp"
#include "bspf/line.hpp"
#include "bspf/op.hpp"

namespace bspf {

using Shape = std::vector<Eigen::Index>;

struct FieldView {
    const double* data;
    Shape shape;
};

struct Field {
    Shape shape;
    std::vector<double> data;
    FieldView view() const { return {data.data(), shape}; }
};

Eigen::Index element_count(const Shape& shape);

// Applies `op` along `axis` without transposing the data: out has the shape of
// `in` with that axis of length op.rows(). Optional jets have the shape of `in`
// with that axis of length op.jet_count(). With accumulate, add into out.
void apply_along(const Op& op, const FieldView& in, double* out, int axis,
                 const double* jets = nullptr, bool accumulate = false);
Field apply_along(const Op& op, const FieldView& in, int axis, const double* jets = nullptr);

class Grid {
public:
    explicit Grid(std::vector<Line> lines);  // one to three axes

    int dimension() const { return static_cast<int>(lines_.size()); }
    const Line& line(int axis) const;
    Shape shape() const;

    // Cached grid derivative along an axis (thread-safe; references stay valid).
    const Op& derivative(int axis, int order) const;

    // Mixed partial d^{orders[0]}_x d^{orders[1]}_y ... (one entry per axis).
    Field partial(const FieldView& f, const std::vector<int>& orders) const;
    // Sum direct second partials into one output allocation.
    Field laplacian(const FieldView& f) const;
    // Energy-stable (Galerkin) Laplacian with Dirichlet walls. Interior values
    //     sum_a W_a (x)_{b != a} E_b u  -  (x)_b E_b w
    // where w holds `wall_values` on the walls and zeros inside: the walls' values
    // of the result (e.g. the boundary data's time derivative in u_t = Lap u),
    // which enter the interior through the Galerkin mass. The result's walls are
    // `wall_values` (zero when omitted). Its homogeneous spectrum is real and
    // negative for any endpoint jets. Cost O(N log N + N r) per line.
    Field weak_laplacian(const FieldView& f, const FieldView* wall_values = nullptr) const;
    const GalerkinLine& galerkin(int axis) const;  // cached per axis
    // Values (or per-axis derivatives) on the tensor mesh coordinates[0] x coordinates[1] x ...
    Field interpolate(const FieldView& f, const std::vector<Vector>& coordinates,
                      const std::vector<int>& derivatives = {}) const;
    // Integral over the box (default: the whole grid); the result has the batch shape.
    Field integrate(const FieldView& f) const;
    Field integrate(const FieldView& f, const std::vector<std::pair<double, double>>& bounds) const;

private:
    void check(const FieldView& f) const;
    // Cached evaluation (kind 0) or quadrature (kind 1) op.
    std::shared_ptr<const Op> cached_op(int kind, int axis, int order, std::vector<double> values) const;
    struct Cache;
    std::vector<Line> lines_;
    std::shared_ptr<Cache> cache_;
};

}  // namespace bspf
