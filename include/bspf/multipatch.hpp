// SPDX-License-Identifier: GPL-3.0-or-later
// Multi-patch structured grids in 2D: conforming deformed patches (MappedGrid2D)
// joined at complete faces, as in bspf-fluid's PatchMesh2D. A face is
// (patch, axis, side): axis 0 side 0 is xi = 0, axis 0 side 1 is xi = 1,
// axis 1 side 0 is eta = 0, axis 1 side 1 is eta = 1. An interface identifies
// two faces node by node, the right face's order optionally reversed. A periodic
// interface also carries a translation: the right face is the left face shifted
// (e.g. by the pitch of a cascade); fields are assumed periodic across it.
//
// Differentiation uses each patch's BSPF lines and exact metrics. Lines that end
// at an interface are extended by `overlap` ghost points into the neighbour, so
// the interface is an interior point of the line and carries no one-sided
// endpoint error:
//   * ghost points continue the patch's own map beyond the face (the curves'
//     end spans continue polynomially), so the extended line is smooth in the
//     patch's own parametrization, whatever the neighbour's grid looks like;
//   * each ghost is located by a Newton inverse of the neighbour's map (near
//     corners, of whichever patch the extended line enters) and valued by local
//     tensor Lagrange interpolation of that patch's samples (BSPF evaluation
//     between nodes is less accurate near a face);
//   * the derivative is taken on the extended BSPF line (same options) and
//     restricted to the patch.
// Mixed reference derivatives take the inner derivative along a line with ghosts.
// When both axes cross interfaces, the ghosts of f_xi come from the neighbours'
// physical gradient. Finally every node shared by several patches takes the
// average of the patches' values, so results are single-valued.
// overlap = 0 gives purely patch-wise (one-sided) derivatives. The domain may be
// multiply connected (e.g. an annulus split into a ring of patches).
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "bspf/mapped.hpp"

namespace bspf {

struct Face {
    int patch = 0;
    int axis = 0;  // 0: xi faces (xi = 0 or 1), 1: eta faces
    int side = 0;  // 0: low, 1: high
};

struct PatchInterface {
    Face left, right;
    bool reverse = false;                  // right face's nodes run opposite to the left face's
    std::array<double, 2> shift{0.0, 0.0};  // right face = left face + shift (nonzero: periodic)
};

class MultiPatch2D {
public:
    // Throws std::invalid_argument on reused or invalid faces, node-count or
    // coordinate mismatches across an interface (after its shift; tolerance
    // relative to the domain size) or a disconnected patch graph. Nodes joined by a
    // periodic interface count as one node, like those of an ordinary interface.
    // overlap: at most this many ghost points per interface end (0: one-sided),
    // capped per face at a quarter of the line on either side of the interface,
    // and reduced further for a face whose extended lines leave the patches (as
    // along a curved wall); stencil: Lagrange points per axis for the ghost values.
    MultiPatch2D(std::vector<MappedGrid2D> patches, std::vector<PatchInterface> interfaces, double tolerance = 1e-10,
                 int overlap = 8, int stencil = 12);
    int overlap() const { return overlap_; }
    // Ghost points used beyond face f (0: f is a boundary or its lines stay one-sided).
    int ghost_depth(const Face& f) const;

    int patch_count() const { return static_cast<int>(patches_.size()); }
    const MappedGrid2D& patch(int p) const;
    const std::vector<PatchInterface>& interfaces() const { return interfaces_; }
    // Faces that are not interfaces (the domain boundary), in patch, axis, side order.
    std::vector<Face> boundary_faces() const;
    // Number of distinct physical nodes (shared nodes counted once).
    Eigen::Index node_count() const { return node_count_; }

    // Physical partial d^ox/dx^ox d^oy/dy^oy of a field given as one C-ordered
    // array (n_xi_p, n_eta_p, *batch) per patch, with shared nodes averaged.
    std::vector<Field> partial(const std::vector<FieldView>& f, int order_x, int order_y,
                               bool average_shared = true) const;
    // Replace each shared node's values by the mean over its patches.
    std::vector<Field> average_shared(const std::vector<FieldView>& f) const;
    // Largest difference between the values a shared node takes on different patches.
    double shared_jump(const std::vector<FieldView>& f) const;

private:
    // Ghost points beyond one end of a patch's lines along `axis`.
    struct Ghosts {
        int depth = 0;                     // ghost points per line (overlap capped by the patch sizes)
        std::vector<int> patch;            // patch holding each (line, ghost): the face neighbour, or near
                                           // corners another patch the extended line enters
        std::vector<Eigen::Index> i0, j0;  // stencil origins in that patch
        std::vector<Eigen::Index> size;    // stencil points per axis
        std::vector<double> wi, wj;        // stencil weights, (line, ghost) x stencil_
        std::vector<double> dx[2];         // this patch's dX/dxi and dX/deta (x, y) at each ghost
    };
    // Distinct periodic translations (both signs) and zero, for locating ghosts.
    std::vector<std::array<double, 2>> translations_;
    // Reference coordinates of point X in patch p by Newton from `guess`; false if not converged.
    bool locate(int p, const double X[2], double guess[2], double scale) const;
    struct Extension {
        int low = 0, high = 0;             // ghost counts at the two ends
        Ghosts ghosts[2];
        std::shared_ptr<Op> derivative[3]; // orders 1, 2 on the extended line
    };
    void build_extensions(int stencil);
    // Ghost values: of f (inner_axis < 0) or of d f / d(reference inner_axis) of
    // this patch, from the neighbours' physical gradient (gx, gy).
    std::vector<double> ghost_values(int p, int axis, int end, const std::vector<FieldView>& f, int inner_axis = -1,
                                     const std::vector<FieldView>* gy = nullptr) const;
    // Reference derivative of patch p's own field along axis, extended by ghosts.
    Field reference_derivative(int p, int axis, int order, const FieldView& own, const std::vector<double>* low,
                               const std::vector<double>* high) const;
    std::vector<Field> physical(const std::vector<FieldView>& f, int order_x, int order_y) const;
    void check(const std::vector<FieldView>& f) const;
    int overlap_ = 0, stencil_ = 12;
    std::vector<std::array<Extension, 2>> extensions_;
    std::vector<MappedGrid2D> patches_;
    std::vector<PatchInterface> interfaces_;
    // Groups of (patch, flat node index) holding the same physical node (size >= 2).
    std::vector<std::vector<std::pair<int, Eigen::Index>>> shared_;
    Eigen::Index node_count_ = 0;
};

}  // namespace bspf
