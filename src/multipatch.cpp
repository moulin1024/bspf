// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/multipatch.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace bspf {

namespace {

using Index = Eigen::Index;

std::string name(const Face& f) {
    return "(" + std::to_string(f.patch) + ", " + std::to_string(f.axis) + ", " + std::to_string(f.side) + ")";
}

// Flat node indices (i * n_eta + j) along a face, in increasing order of the
// face's own parameter (eta for xi faces, xi for eta faces).
std::vector<Index> face_nodes(const MappedGrid2D& g, const Face& f) {
    const Index n_xi = g.shape()[0], n_eta = g.shape()[1];
    std::vector<Index> nodes;
    if (f.axis == 0) {
        const Index i = f.side ? n_xi - 1 : 0;
        for (Index j = 0; j < n_eta; ++j) nodes.push_back(i * n_eta + j);
    } else {
        const Index j = f.side ? n_eta - 1 : 0;
        for (Index i = 0; i < n_xi; ++i) nodes.push_back(i * n_eta + j);
    }
    return nodes;
}

struct UnionFind {
    std::vector<Index> parent;
    explicit UnionFind(Index n) : parent(static_cast<size_t>(n)) { std::iota(parent.begin(), parent.end(), Index(0)); }
    Index find(Index a) {
        while (parent[a] != a) a = parent[a] = parent[parent[a]];
        return a;
    }
    void join(Index a, Index b) { parent[find(a)] = find(b); }
};

}  // namespace

MultiPatch2D::MultiPatch2D(std::vector<MappedGrid2D> patches, std::vector<PatchInterface> interfaces, double tolerance,
                           int overlap, int stencil)
    : patches_(std::move(patches)), interfaces_(std::move(interfaces)) {
    if (overlap < 0) throw std::invalid_argument("overlap must be nonnegative");
    if (stencil < 2) throw std::invalid_argument("the ghost stencil needs at least 2 points");
    if (patches_.empty()) throw std::invalid_argument("provide at least one patch");
    const int P = patch_count();
    double scale = 1.0;
    for (const MappedGrid2D& g : patches_)
        scale = std::max({scale, g.x().cwiseAbs().maxCoeff(), g.y().cwiseAbs().maxCoeff()});

    // Global node numbering: patch offsets into one index space, then merge interfaces.
    std::vector<Index> offset(P + 1, 0);
    for (int p = 0; p < P; ++p) offset[p + 1] = offset[p] + patches_[p].shape()[0] * patches_[p].shape()[1];
    UnionFind nodes(offset[P]);
    std::set<std::tuple<int, int, int>> used;
    UnionFind graph(P);
    for (const PatchInterface& link : interfaces_) {
        for (const Face& f : {link.left, link.right}) {
            if (f.patch < 0 || f.patch >= P || f.axis < 0 || f.axis > 1 || f.side < 0 || f.side > 1)
                throw std::invalid_argument("invalid interface face " + name(f));
            if (!used.insert({f.patch, f.axis, f.side}).second) throw std::invalid_argument("face " + name(f) + " is used twice");
        }
        const MappedGrid2D &a = patches_[link.left.patch], &b = patches_[link.right.patch];
        std::vector<Index> na = face_nodes(a, link.left), nb = face_nodes(b, link.right);
        if (na.size() != nb.size())
            throw std::invalid_argument("interface " + name(link.left) + " - " + name(link.right) + " has " +
                                        std::to_string(na.size()) + " and " + std::to_string(nb.size()) + " nodes");
        if (link.reverse) std::reverse(nb.begin(), nb.end());
        const Index na_eta = a.shape()[1], nb_eta = b.shape()[1];
        for (size_t k = 0; k < na.size(); ++k) {
            const Index ia = na[k] / na_eta, ja = na[k] % na_eta, ib = nb[k] / nb_eta, jb = nb[k] % nb_eta;
            const double gap = std::hypot(a.x()(ia, ja) + link.shift[0] - b.x()(ib, jb),
                                          a.y()(ia, ja) + link.shift[1] - b.y()(ib, jb));
            if (gap > tolerance * scale)
                throw std::invalid_argument("interface " + name(link.left) + " - " + name(link.right) +
                                            ": nodes do not coincide (gap " + std::to_string(gap) +
                                            "); check the curves, the reverse flag and the shift");
            nodes.join(offset[link.left.patch] + na[k], offset[link.right.patch] + nb[k]);
        }
        graph.join(link.left.patch, link.right.patch);
    }
    for (int p = 1; p < P; ++p)
        if (graph.find(p) != graph.find(0)) throw std::invalid_argument("the patch graph must be connected");

    std::map<Index, std::vector<std::pair<int, Index>>> groups;
    for (int p = 0; p < P; ++p)
        for (Index k = 0; k < offset[p + 1] - offset[p]; ++k) groups[nodes.find(offset[p] + k)].push_back({p, k});
    node_count_ = static_cast<Index>(groups.size());
    for (auto& [root, members] : groups)
        if (members.size() > 1) shared_.push_back(std::move(members));
    translations_ = {{0.0, 0.0}};
    for (const PatchInterface& link : interfaces_)
        for (double sign : {1.0, -1.0}) {
            const std::array<double, 2> t{sign * link.shift[0], sign * link.shift[1]};
            if (std::hypot(t[0], t[1]) <= tolerance * scale) continue;
            const bool known = std::any_of(translations_.begin(), translations_.end(), [&](const auto& u) {
                return std::hypot(u[0] - t[0], u[1] - t[1]) <= tolerance * scale;
            });
            if (!known) translations_.push_back(t);
        }
    overlap_ = overlap;
    stencil_ = stencil;  // weights are stored with this stride
    extensions_.assign(static_cast<size_t>(P), {});
    if (overlap_ > 0) build_extensions(stencil);
}

namespace {

// Lagrange stencil of `count` uniform nodes on [0, 1] (n nodes) around x: origin and weights.
Index lagrange_stencil(Index n, Index count, double x, double* weights) {
    count = std::min(count, n);
    const double h = 1.0 / static_cast<double>(n - 1);
    const Index c = static_cast<Index>(std::floor(x / h));
    const Index lo = std::clamp<Index>(c - count / 2 + 1, 0, n - count);
    for (Index a = 0; a < count; ++a) {
        double w = 1.0;
        const double xa = static_cast<double>(lo + a) * h;
        for (Index b = 0; b < count; ++b)
            if (b != a) w *= (x - static_cast<double>(lo + b) * h) / (xa - static_cast<double>(lo + b) * h);
        weights[a] = w;
    }
    return lo;
}

}  // namespace

bool MultiPatch2D::locate(int p, const double X[2], double st[2], double scale) const {
    const MappedGrid2D& B = patches_[p];
    for (int it = 0; it < 50; ++it) {
        double Y[2], Ys[2], Ye[2];
        B.map(st[0], st[1], Y, Ys, Ye);
        const double F0 = Y[0] - X[0], F1 = Y[1] - X[1];
        if (std::hypot(F0, F1) <= 1e-14 * scale) return true;
        const double det = Ys[0] * Ye[1] - Ye[0] * Ys[1];
        if (!(std::abs(det) > 0)) return false;
        st[0] -= (Ye[1] * F0 - Ye[0] * F1) / det;
        st[1] -= (-Ys[1] * F0 + Ys[0] * F1) / det;
        if (!(std::abs(st[0]) < 10 && std::abs(st[1]) < 10)) return false;
    }
    double Y[2], Ys[2], Ye[2];
    B.map(st[0], st[1], Y, Ys, Ye);
    return std::hypot(Y[0] - X[0], Y[1] - X[1]) <= 1e-12 * scale;
}

int MultiPatch2D::ghost_depth(const Face& f) const {
    if (f.patch < 0 || f.patch >= patch_count() || f.axis < 0 || f.axis > 1 || f.side < 0 || f.side > 1)
        throw std::invalid_argument("invalid face");
    if (extensions_.empty()) return 0;
    const Extension& ext = extensions_[f.patch][f.axis];
    return f.side ? ext.high : ext.low;
}

void MultiPatch2D::build_extensions(int stencil) {
    const int P = patch_count();
    double scale = 1.0;
    for (const MappedGrid2D& g : patches_)
        scale = std::max({scale, g.x().cwiseAbs().maxCoeff(), g.y().cwiseAbs().maxCoeff()});
    for (const PatchInterface& link : interfaces_)
        for (int direction = 0; direction < 2; ++direction) {
            const Face& own = direction ? link.right : link.left;
            const Face& other = direction ? link.left : link.right;
            const double sign = direction ? -1.0 : 1.0;  // own point + sign * shift lies on the other side
            const MappedGrid2D &A = patches_[own.patch], &B = patches_[other.patch];
            const int d = own.axis, t = 1 - d;
            const Index n_d = A.shape()[d], n_t = A.shape()[t];
            // Ghosts continue this patch's map; keep them within a quarter of either line.
            const int cap = static_cast<int>(std::min<Index>(overlap_, std::min(n_d - 1, B.shape()[other.axis] - 1) / 4));
            Extension& ext = extensions_[own.patch][d];
            Ghosts& g = ext.ghosts[own.side];
            const double h = 1.0 / static_cast<double>(n_d - 1);
            const Index nb_t = B.shape()[1 - other.axis];
            auto excursion = [&](int p, const double st[2]) {
                const double cell = 1.0 / static_cast<double>(std::min(patches_[p].shape()[0], patches_[p].shape()[1]) - 1);
                return std::max({-st[0], st[0] - 1.0, -st[1], st[1] - 1.0, 0.0}) / cell;  // in cells
            };
            // Fill the ghosts of this face for the given depth; false if one lies outside the patches.
            auto build = [&](int depth) {
                g.depth = depth;
                const size_t count = static_cast<size_t>(n_t * depth);
                g.patch.resize(count);
                g.i0.resize(count);
                g.j0.resize(count);
                g.size.resize(count);
                g.wi.assign(count * stencil, 0.0);
                g.wj.assign(count * stencil, 0.0);
                for (int k = 0; k < 2; ++k) g.dx[k].resize(2 * count);
                for (Index k = 0; k < n_t; ++k) {
                    // The matching node on the neighbour's face starts the Newton iteration.
                    const Index kb = link.reverse ? n_t - 1 - k : k;
                    double guess[2];
                    guess[other.axis] = other.side ? 1.0 : 0.0;
                    guess[1 - other.axis] = static_cast<double>(kb) / static_cast<double>(nb_t - 1);
                    for (Index m = 1; m <= depth; ++m) {
                        double ref[2];
                        ref[d] = own.side ? 1.0 + static_cast<double>(m) * h : -static_cast<double>(m) * h;
                        ref[t] = static_cast<double>(k) / static_cast<double>(n_t - 1);
                        double X[2], Xs[2], Xe[2];
                        A.map(ref[0], ref[1], X, Xs, Xe);
                        const size_t q = static_cast<size_t>(k * depth + (m - 1));
                        g.dx[0][2 * q] = Xs[0];
                        g.dx[0][2 * q + 1] = Xs[1];
                        g.dx[1][2 * q] = Xe[0];
                        g.dx[1][2 * q + 1] = Xe[1];
                        // The face neighbour first (across a periodic face, at the translated point);
                        // near corners the extended line may enter another patch, possibly across a
                        // periodic boundary.
                        int best = -1;
                        double best_st[2] = {0, 0}, best_out = INFINITY;
                        double st[2] = {guess[0], guess[1]};
                        const double Xn[2] = {X[0] + sign * link.shift[0], X[1] + sign * link.shift[1]};
                        if (locate(other.patch, Xn, st, scale)) {
                            best = other.patch;
                            best_st[0] = st[0];
                            best_st[1] = st[1];
                            best_out = excursion(best, st);
                        }
                        for (const std::array<double, 2>& tr : translations_)
                            for (int p = 0; p < P && best_out > 0; ++p) {
                                const double Y[2] = {X[0] + tr[0], X[1] + tr[1]};
                                const bool same = std::hypot(Y[0] - Xn[0], Y[1] - Xn[1]) <= 1e-12 * scale;
                                if ((same && p == other.patch) || (tr[0] == 0.0 && tr[1] == 0.0 && p == own.patch)) continue;
                                const MappedGrid2D& C = patches_[p];
                                // Start from the nearest node of that patch.
                                Index arg = 0;
                                double nearest = INFINITY;
                                for (Index i = 0; i < C.shape()[0]; ++i)
                                    for (Index j = 0; j < C.shape()[1]; ++j) {
                                        const double dist = std::hypot(C.x()(i, j) - Y[0], C.y()(i, j) - Y[1]);
                                        if (dist < nearest) {
                                            nearest = dist;
                                            arg = i * C.shape()[1] + j;
                                        }
                                    }
                                double cs[2] = {static_cast<double>(arg / C.shape()[1]) / static_cast<double>(C.shape()[0] - 1),
                                                static_cast<double>(arg % C.shape()[1]) / static_cast<double>(C.shape()[1] - 1)};
                                if (locate(p, Y, cs, scale) && excursion(p, cs) < best_out) {
                                    best = p;
                                    best_st[0] = cs[0];
                                    best_st[1] = cs[1];
                                    best_out = excursion(p, cs);
                                }
                            }
                        if (best < 0 || best_out > 2.0) return false;  // Lagrange extrapolates accurately a cell or two
                        const MappedGrid2D& C = patches_[best];
                        const Index S = std::min<Index>(stencil, std::min(C.shape()[0], C.shape()[1]));
                        g.patch[q] = best;
                        g.size[q] = S;
                        g.i0[q] = lagrange_stencil(C.shape()[0], S, best_st[0], &g.wi[q * stencil]);
                        g.j0[q] = lagrange_stencil(C.shape()[1], S, best_st[1], &g.wj[q * stencil]);
                        if (best == other.patch) {
                            guess[0] = best_st[0];
                            guess[1] = best_st[1];
                        }
                    }
                }
                return true;
            };
            // Where the extended lines leave the domain (e.g. along a curved wall), fewer ghosts.
            int depth = cap;
            while (depth > 0 && !build(depth)) --depth;
            if (depth == 0) g = Ghosts();
            (own.side ? ext.high : ext.low) = depth;
        }
    for (int p = 0; p < P; ++p)
        for (int d = 0; d < 2; ++d) {
            Extension& ext = extensions_[p][d];
            if (!ext.low && !ext.high) continue;
            const Index n = patches_[p].shape()[d];
            const double h = 1.0 / static_cast<double>(n - 1);
            const Line line(Vector::LinSpaced(n + ext.low + ext.high, -ext.low * h, 1.0 + ext.high * h),
                            patches_[p].options());
            for (int order = 1; order <= 2; ++order) ext.derivative[order] = std::make_shared<Op>(line.derivative(order));
        }
}

std::vector<double> MultiPatch2D::ghost_values(int p, int axis, int end, const std::vector<FieldView>& f, int inner_axis,
                                               const std::vector<FieldView>* gy) const {
    const Ghosts& g = extensions_[p][axis].ghosts[end];
    const Index n_t = patches_[p].shape()[1 - axis];
    const Index batch = element_count(f[0].shape) / (f[0].shape[0] * f[0].shape[1]);
    std::vector<double> out(static_cast<size_t>(n_t * g.depth * batch), 0.0);
    std::vector<double> gx_value(static_cast<size_t>(batch)), gy_value(static_cast<size_t>(batch));
    for (size_t q = 0; q < g.i0.size(); ++q) {
        std::fill(gx_value.begin(), gx_value.end(), 0.0);
        std::fill(gy_value.begin(), gy_value.end(), 0.0);
        const int src_patch = g.patch[q];
        const Index S = g.size[q], nb_eta = patches_[src_patch].shape()[1];
        for (Index a = 0; a < S; ++a)
            for (Index c = 0; c < S; ++c) {
                const double w = g.wi[q * stencil_ + a] * g.wj[q * stencil_ + c];
                const Index node = (g.i0[q] + a) * nb_eta + (g.j0[q] + c);
                const double* src = f[src_patch].data + node * batch;
                for (Index b = 0; b < batch; ++b) gx_value[b] += w * src[b];
                if (gy) {
                    const double* src_y = (*gy)[src_patch].data + node * batch;
                    for (Index b = 0; b < batch; ++b) gy_value[b] += w * src_y[b];
                }
            }
        for (Index b = 0; b < batch; ++b)
            out[q * batch + b] = inner_axis < 0 ? gx_value[b]
                                                : gx_value[b] * g.dx[inner_axis][2 * q] + gy_value[b] * g.dx[inner_axis][2 * q + 1];
    }
    return out;
}

Field MultiPatch2D::reference_derivative(int p, int axis, int order, const FieldView& own, const std::vector<double>* low,
                                         const std::vector<double>* high) const {
    const Extension& ext = extensions_[p][axis];
    const MappedGrid2D& A = patches_[p];
    if (!ext.low && !ext.high) return A.reference().partial(own, {axis == 0 ? order : 0, axis == 1 ? order : 0});
    const Index n0 = A.shape()[0], n1 = A.shape()[1];
    const Index batch = element_count(own.shape) / (n0 * n1);
    const Index lo = ext.low, hi = ext.high, n = A.shape()[axis];
    Field extended;
    extended.shape = own.shape;
    extended.shape[axis] = n + lo + hi;
    extended.data.resize(static_cast<size_t>(element_count(extended.shape)));
    const Index e1 = extended.shape[1];
    auto at = [&](Index i, Index j) { return extended.data.data() + (i * e1 + j) * batch; };
    for (Index i = 0; i < n0; ++i)
        for (Index j = 0; j < n1; ++j) {
            const double* src = own.data + (i * n1 + j) * batch;
            double* dst = axis == 0 ? at(i + lo, j) : at(i, j + lo);
            std::copy(src, src + batch, dst);
        }
    // Ghost data are ordered (line k, ghost m = 1..depth).
    const Index n_t = A.shape()[1 - axis];
    for (Index k = 0; k < n_t; ++k) {
        for (Index m = 1; m <= lo; ++m) {
            const size_t q = static_cast<size_t>((k * lo + (m - 1)) * batch);
            double* dst = axis == 0 ? at(lo - m, k) : at(k, lo - m);
            std::copy(low->begin() + q, low->begin() + q + batch, dst);
        }
        for (Index m = 1; m <= hi; ++m) {
            const size_t q = static_cast<size_t>((k * hi + (m - 1)) * batch);
            double* dst = axis == 0 ? at(lo + n - 1 + m, k) : at(k, lo + n - 1 + m);
            std::copy(high->begin() + q, high->begin() + q + batch, dst);
        }
    }
    const Field full = apply_along(*ext.derivative[order], extended.view(), axis);
    Field out;
    out.shape = own.shape;
    out.data.resize(static_cast<size_t>(element_count(own.shape)));
    const Index f1 = full.shape[1];
    for (Index i = 0; i < n0; ++i)
        for (Index j = 0; j < n1; ++j) {
            const Index si = axis == 0 ? i + lo : i, sj = axis == 1 ? j + lo : j;
            const double* src = full.data.data() + (si * f1 + sj) * batch;
            std::copy(src, src + batch, out.data.data() + (i * n1 + j) * batch);
        }
    return out;
}

std::vector<Field> MultiPatch2D::physical(const std::vector<FieldView>& f, int order_x, int order_y) const {
    const int P = patch_count(), order = order_x + order_y;
    auto ghosts = [&](int p, int axis, int end) -> std::optional<std::vector<double>> {
        const Extension& ext = extensions_[p][axis];
        if (!(end ? ext.high : ext.low)) return std::nullopt;
        return ghost_values(p, axis, end, f);
    };
    // Physical gradient of all patches: the ghosts of f_xi beyond eta-faces on patches
    // whose lines cross interfaces along both axes.
    std::vector<Field> gx, gy;
    std::vector<FieldView> gxv, gyv;
    bool need_gradient = false;
    if (order == 2)  // every second derivative uses f_xieta
        for (int p = 0; p < P; ++p)
            need_gradient |= (extensions_[p][0].low || extensions_[p][0].high) && (extensions_[p][1].low || extensions_[p][1].high);
    if (need_gradient) {
        gx = partial(f, 1, 0, true);
        gy = partial(f, 0, 1, true);
        for (int p = 0; p < P; ++p) {
            gxv.push_back(gx[p].view());
            gyv.push_back(gy[p].view());
        }
    }
    std::vector<Field> out;
    for (int p = 0; p < P; ++p) {
        const auto l0 = ghosts(p, 0, 0), h0 = ghosts(p, 0, 1), l1 = ghosts(p, 1, 0), h1 = ghosts(p, 1, 1);
        const Field f_s = reference_derivative(p, 0, 1, f[p], l0 ? &*l0 : nullptr, h0 ? &*h0 : nullptr);
        const Field f_e = reference_derivative(p, 1, 1, f[p], l1 ? &*l1 : nullptr, h1 ? &*h1 : nullptr);
        if (order == 1) {
            out.push_back(patches_[p].chain_rule(order_x, order_y, f_s, f_e));
            continue;
        }
        const Field f_ss = reference_derivative(p, 0, 2, f[p], l0 ? &*l0 : nullptr, h0 ? &*h0 : nullptr);
        const Field f_ee = reference_derivative(p, 1, 2, f[p], l1 ? &*l1 : nullptr, h1 ? &*h1 : nullptr);
        const bool ghosts0 = extensions_[p][0].low || extensions_[p][0].high;
        const bool ghosts1 = extensions_[p][1].low || extensions_[p][1].high;
        Field f_se;
        if (!ghosts1) {
            f_se = reference_derivative(p, 1, 1, f_s.view(), nullptr, nullptr);   // inner along axis 0 (with ghosts)
        } else if (!ghosts0) {
            f_se = reference_derivative(p, 0, 1, f_e.view(), nullptr, nullptr);   // inner along axis 1 (with ghosts)
        } else {
            // Ghosts of f_xi beyond the eta-faces from the neighbours' physical gradient.
            std::optional<std::vector<double>> a, b;
            if (extensions_[p][1].low) a = ghost_values(p, 1, 0, gxv, 0, &gyv);
            if (extensions_[p][1].high) b = ghost_values(p, 1, 1, gxv, 0, &gyv);
            f_se = reference_derivative(p, 1, 1, f_s.view(), a ? &*a : nullptr, b ? &*b : nullptr);
        }
        out.push_back(patches_[p].chain_rule(order_x, order_y, f_s, f_e, &f_ss, &f_se, &f_ee));
    }
    return out;
}

const MappedGrid2D& MultiPatch2D::patch(int p) const {
    if (p < 0 || p >= patch_count()) throw std::invalid_argument("patch index out of range");
    return patches_[p];
}

std::vector<Face> MultiPatch2D::boundary_faces() const {
    std::set<std::tuple<int, int, int>> used;
    for (const PatchInterface& link : interfaces_)
        for (const Face& f : {link.left, link.right}) used.insert({f.patch, f.axis, f.side});
    std::vector<Face> out;
    for (int p = 0; p < patch_count(); ++p)
        for (int axis = 0; axis < 2; ++axis)
            for (int side = 0; side < 2; ++side)
                if (!used.count({p, axis, side})) out.push_back({p, axis, side});
    return out;
}

void MultiPatch2D::check(const std::vector<FieldView>& f) const {
    if (static_cast<int>(f.size()) != patch_count())
        throw std::invalid_argument("provide one field array per patch (" + std::to_string(patch_count()) + ")");
    Shape batch;
    for (int p = 0; p < patch_count(); ++p) {
        const Shape s = patches_[p].shape();
        if (f[p].shape.size() < 2 || f[p].shape[0] != s[0] || f[p].shape[1] != s[1])
            throw std::invalid_argument("field " + std::to_string(p) + " must have leading shape (" +
                                        std::to_string(s[0]) + ", " + std::to_string(s[1]) + ")");
        const Shape tail(f[p].shape.begin() + 2, f[p].shape.end());
        if (p == 0) batch = tail;
        else if (tail != batch) throw std::invalid_argument("all patches must share the trailing batch shape");
    }
}

std::vector<Field> MultiPatch2D::average_shared(const std::vector<FieldView>& f) const {
    check(f);
    std::vector<Field> out(f.size());
    for (size_t p = 0; p < f.size(); ++p) {
        out[p].shape = f[p].shape;
        out[p].data.assign(f[p].data, f[p].data + element_count(f[p].shape));
    }
    const Index batch = element_count(f[0].shape) / (f[0].shape[0] * f[0].shape[1]);
    std::vector<double> mean(static_cast<size_t>(batch));
    for (const auto& group : shared_) {
        std::fill(mean.begin(), mean.end(), 0.0);
        for (const auto& [p, k] : group)
            for (Index b = 0; b < batch; ++b) mean[b] += out[p].data[static_cast<size_t>(k * batch + b)];
        for (Index b = 0; b < batch; ++b) mean[b] /= static_cast<double>(group.size());
        for (const auto& [p, k] : group)
            for (Index b = 0; b < batch; ++b) out[p].data[static_cast<size_t>(k * batch + b)] = mean[b];
    }
    return out;
}

double MultiPatch2D::shared_jump(const std::vector<FieldView>& f) const {
    check(f);
    const Index batch = element_count(f[0].shape) / (f[0].shape[0] * f[0].shape[1]);
    double jump = 0.0;
    for (const auto& group : shared_)
        for (Index b = 0; b < batch; ++b) {
            double lo = INFINITY, hi = -INFINITY;
            for (const auto& [p, k] : group) {
                const double v = f[p].data[k * batch + b];
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
            jump = std::max(jump, hi - lo);
        }
    return jump;
}

std::vector<Field> MultiPatch2D::partial(const std::vector<FieldView>& f, int order_x, int order_y,
                                         bool average) const {
    check(f);
    if (order_x < 0 || order_y < 0) throw std::invalid_argument("derivative orders must be nonnegative");
    std::vector<Field> out;
    const int order = order_x + order_y;
    if (overlap_ == 0 || order == 0) {
        out.reserve(f.size());
        for (int p = 0; p < patch_count(); ++p) out.push_back(patches_[p].partial(f[p], order_x, order_y));
    } else if (order > 2) {
        // First derivatives of the (averaged, physical, hence continuous) lower-order result.
        const std::vector<Field> lower = order_x > 0 ? partial(f, order_x - 1, order_y, true) : partial(f, order_x, order_y - 1, true);
        std::vector<FieldView> views;
        for (const Field& g : lower) views.push_back(g.view());
        out = physical(views, order_x > 0 ? 1 : 0, order_x > 0 ? 0 : 1);
    } else {
        out = physical(f, order_x, order_y);
    }
    if (!average) return out;
    std::vector<FieldView> views;
    for (const Field& g : out) views.push_back(g.view());
    return average_shared(views);
}

}  // namespace bspf
