// SPDX-License-Identifier: GPL-3.0-or-later
// Nonlinear geometric multigrid (FAS) for the steady compressible solvers.
//
// Levels are nested reference grids of the same map, finest first: per axis
// n_coarse = (n_fine + 1) / 2 (h-coarsening) or n_coarse = n_fine (semi-coarsening of
// the other axis, or p-coarsening: the same nodes with a lower-order trial space). A cycle on level l solves R_l(U) + f_l = 0:
//     pre-smooth, then gamma times (1: V-cycle, 2: W-cycle):
//         U_c0 = inject(U),   f_c = restrict(R_l(U) + f_l) - R_c(U_c0),
//         U_c  = cycle(l + 1, U_c0, f_c),
//         U   += omega prolong(U_c - U_c0),
//     post-smooth;  the coarsest level only smooths.
// restrict is full weighting ([1, 2, 1] / 4 per axis, [2, 2] / 4 at the ends),
// prolong the BSPF interpolation of the coarse correction, smoothing the level
// solver's own pseudo-time steps (local-time-step RK4, line-implicit along eta).
// Injected residuals alias the smoother's leftover high frequencies into
// spurious coarse forcing, and undamped corrections (omega = 1) let the inlet
// region drift; both diverged on the Laval nozzle. A correction that would make
// the state nonphysical is retried with omega halved (up to four times).
#pragma once

#include <array>
#include <functional>
#include <utility>
#include <vector>

#include "bspf/grid.hpp"

namespace bspf {

// One multigrid level: a steady solver seen through the operations FAS needs.
class FlowLevel {
public:
    virtual ~FlowLevel() = default;
    virtual Shape shape() const = 0;              // {n0, n1, 4}
    virtual const Grid& reference() const = 0;    // reference lines (for prolongation)
    virtual double gamma() const = 0;
    virtual Field rhs(const FieldView& U) const = 0;                // includes the source
    virtual Field smooth(const FieldView& U, int steps) const = 0;  // `steps` pseudo-time steps
    virtual void set_source(const FieldView& source) = 0;           // empty view clears
};

struct MultigridOptions {
    int pre = 10, post = 10, coarse = 60;  // smoothing steps per visit
    int cycle_index = 1;                   // 1: V-cycle, 2: W-cycle
    double damping = 0.5;                  // omega
    int initial = 50;                      // fine-grid smoothing before the first cycle
    int max_cycles = 200;
    double tolerance = 1e-10;              // rms fine-grid residual
    std::function<void(int, double)> progress;  // (cycle, residual)
};

struct MultigridResult {
    int cycles = 0;
    double residual = 0;
    bool converged = false;
    double work = 0;     // fine-grid step equivalents (steps weighted by node count)
    double seconds = 0;
    int damping_cuts = 0;  // corrections retried with a smaller omega
    std::vector<std::pair<int, double>> history;
};

// Solve on levels[0] (finest), U in place.
MultigridResult multigrid_solve(const std::vector<FlowLevel*>& levels, Field& U, const MultigridOptions& options);

// Newton-Krylov on R(U) = 0 (levels[0]'s residual), multigrid-preconditioned:
//   * Jacobian-vector products by finite differences, J v ~ (R(U + h v) - R(U)) / h;
//   * flexible GMRES (restarted), right preconditioner one FAS cycle on the linearized
//     problem: from U with forcing -R(U) - sigma z, the cycle's correction divided by sigma
//     approximates J^{-1} z (sigma small; the preconditioner varies, hence FGMRES);
//   * inexact Newton: the linear solve to eta_k |R_k| (eta_k = min(eta_max, |R_k| / |R_{k-1}|),
//     at least eta_min), then a backtracking line search on |R| that also rejects nonphysical states;
//   * a few FAS cycles first, until the residual is below start_residual.
// Unlike pseudo-time (the smoother or plain multigrid), Newton needs only an invertible Jacobian:
// weakly growing modes of the semi-discrete operator do not stall it.
struct NewtonOptions {
    double tolerance = 1e-10;      // rms residual
    int max_newton = 40;
    int krylov_restart = 40;       // FGMRES vectors per restart
    int max_krylov = 200;          // per Newton step
    double eta_max = 0.1, eta_min = 1e-4;
    double start_residual = 1e-3;  // FAS cycles before Newton until below this
    int max_start_cycles = 60;
    int preconditioner_cycles = 1;
    // relative finite-difference steps: Jacobian products and the preconditioner's linearization
    // (double: 1e-7 and 1e-4; FP32 device levels need ~1e-3 and ~1e-2)
    double jacobian_epsilon = 1e-7, preconditioner_epsilon = 1e-4;
    MultigridOptions multigrid;    // cycle shape for the start and the preconditioner
    std::function<void(int, double, int)> progress;  // (Newton step, residual, Krylov iterations)
};

struct NewtonResult {
    int newton_steps = 0, krylov_iterations = 0, start_cycles = 0;
    double residual = 0, seconds = 0;
    double seconds_preconditioner = 0, seconds_jacobian = 0, seconds_krylov = 0;  // breakdown of the Newton phase
    bool converged = false;
    std::vector<std::pair<int, double>> history;  // (Newton step, residual)
};

// Device Newton-Krylov: Newton, FGMRES (CGS2 orthogonalization), Jacobian products and the line search
// with device-resident vectors and residuals. precondition(U, z, new_state) returns ~J^{-1} z
// (host fields; new_state is true for the first call at a new Newton iterate).
class NewtonDevice {
public:
    using Preconditioner = std::function<Field(const Field& U, const Field& z, bool new_state)>;
    virtual ~NewtonDevice() = default;
    virtual NewtonResult solve(Field& U, const NewtonOptions& options, const Preconditioner& precondition) = 0;
    // true: the device also runs the start phase and the preconditioner (its own multigrid hierarchy)
    virtual bool handles_start() const { return false; }
};

// residual: the level whose R(U) Newton drives to zero, also used for the Jacobian products, the line
// search and FGMRES (all in its precision); default levels[0]. Mixed precision: a double-precision
// CPU solver here with FP32 device levels as the preconditioner (their forcing uses their own
// residual at U, so FP32 rounding of R does not enter the preconditioned vectors).
// device: when given, the Newton phase runs there (residual is then unused).
NewtonResult newton_krylov_solve(const std::vector<FlowLevel*>& levels, Field& U, const NewtonOptions& options,
                                 FlowLevel* residual = nullptr, NewtonDevice* device = nullptr);

// Full-weighting restriction of an (n0, n1, k) field onto the coarse grid; factors per axis: 2 (coarsened,
// n_coarse = (n + 1) / 2) or 1 (kept).
Field restrict_full_weighting(const FieldView& f, std::array<int, 2> factors = {2, 2});

}  // namespace bspf
