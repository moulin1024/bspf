// SPDX-License-Identifier: GPL-3.0-or-later
// Compressible Euler / Navier-Stokes on one mapped 2D block (MappedGrid2D),
// Galerkin BSPF in weak form with weak (SAT-type) boundary fluxes.
//
// Units: the gas constant is R = 1/gamma, so T = gamma p / rho = c^2 (stagnation
// state rho0 = T0 = c0 = 1 gives p0 = 1/gamma). mu = T^omega / Re, kappa =
// mu cp / Pr with cp = 1 / (gamma - 1). Re = 0: Euler.
//
// Semi-discretization, with phi_i the BSPF cardinal functions of each reference
// axis and the trial field u_h = sum U_ij phi_i(xi) phi_j(eta) (U: nodal values):
//
//   int phi_I J U_t = int (d_xi phi_I Fh + d_eta phi_I Gh) - faces(phi_I Fh*),
//
//  * volume term: fluxes formed from u_h at composite Gauss points (WeakEuler2D),
//    isoparametric metrics there, viscous fluxes from nodal gradients;
//  * free-stream correction: the quadrature's metric-identity defect (the weak
//    residual of unit constant fluxes) times the nodal fluxes is subtracted, so a
//    uniform flow is an exact steady state at any quadrature order;
//  * faces: Rusanov flux against an exterior state at the face Gauss points:
//      Inlet       total pressure / temperature, flow along the face normal,
//                  outgoing Riemann invariant from the interior;
//      Outlet      interior state where supersonic; subsonic points at the mean
//                  pressure of the supersonic part, else at back_pressure;
//      SlipWall    mirror state (normal velocity reversed);
//      NoSlipWall  momentum reversed; viscous energy flux zero (adiabatic);
//      Exterior    a given state (manufactured solutions, far field);
//  * mass: tensor product of the per-axis Galerkin masses, J at the nodes;
//  * viscous gradients: SBP derivative D = M^{-1} C per axis (C + C^T = B), wall
//    and exterior values lifted weakly with M^{-1} e_0, M^{-1} e_N;
//  * spectral vanishing viscosity: -nu_axis Q_axis U, Q = V diag(r) V^T M in the
//    eigenbasis of the Legendre-weighted stiffness int 4 s (1 - s) phi_i' phi_j'
//    (spectrally small on resolved fields up to the boundary), r = 0 below
//    cutoff * N modes, nu_axis = svv x radius x the largest wave rate.
//
// Steady state: classical RK4 with local time steps as a fixed-point map,
// optionally accelerated by Anderson mixing (with positivity safeguard and
// restarts). Viscous flows are line-implicit in the wall-normal direction: each
// stage advances P^{-1} R(U), P = I - diag(dt nu_c |grad eta|^2) L_eta per eta line
// and component (L_eta = D (D + s0 e0^T - s1 eN^T), the second derivative with the
// weak wall lifting; nu = 4/3 mu/rho for momentum, gamma mu/(Pr rho) for energy,
// 0 for density), and the time step drops the wall-normal viscous limit.
// LineImplicit::Full replaces P by the linearized wall-normal operator per eta line,
//     A = -diag(1/J) (D_eta x I4) diag(dGh/dU) + wall penalty - nu_svv Q_eta x I4 + viscous,
//     P = I - diag(dt) A   (4 n1 x 4 n1, LU per line),
// with the flux and wall-penalty Jacobians from finite differences of the pointwise
// fluxes; the time step then drops every eta-direction limit. Fields are C-ordered (n0, n1, 4) arrays of (rho, rho u, rho v, E).
#pragma once

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bspf/grid.hpp"
#include "bspf/mapped.hpp"
#include "bspf/multigrid.hpp"

namespace bspf {

enum class LineImplicit { None, Viscous, Full };

struct FlowOptions {
    double gamma = 1.4;
    double prandtl = 0.72;
    double reynolds = 0.0;            // 0: Euler
    double viscosity_exponent = 0.76;
    double svv = 0.2;                 // spectral vanishing viscosity strength (0: off); 0.1 leaves the line-implicit iteration marginally unstable on 193 x 65
    double svv_cutoff = 0.5;          // fraction of modes left undamped
    int quadrature = 5;               // Gauss points per cell (cells split at knots; 4 leaves a ~1e-8 floor)
    double cfl = 0.6;
    // Line-implicit preconditioning of the pseudo-time march along eta (wall-normal) lines; the
    // steady state is unchanged. Viscous: the wall-normal viscous terms. Full: the linearized
    // wall-normal operator (inviscid flux, wall penalty, SVV and viscous terms), so the time step
    // is limited by the xi direction only.
    LineImplicit line_implicit = LineImplicit::Full;
    // Trial field of the flux evaluation: entropy variables V(U) instead of U (interpolated to the
    // Gauss and face points, U(V_h) there). Galerkin in entropy variables is entropy stable;
    // in conservative variables the 2D operator keeps weakly growing modes.
    bool entropy_variables = false;
};

enum class BoundaryKind { Inlet, Outlet, SlipWall, NoSlipWall, Exterior };

struct Boundary {
    BoundaryKind kind = BoundaryKind::SlipWall;
    double total_pressure = 1.0 / 1.4;  // Inlet
    double total_temperature = 1.0;     // Inlet
    double back_pressure = std::numeric_limits<double>::quiet_NaN();  // Outlet, where nothing is supersonic
    // Exterior: (rho, rho u, rho v, E) at the face nodes (n, 4) and the face Gauss points (Q, 4).
    Matrix nodes, points;
};

struct SteadyOptions {
    int max_steps = 100000;
    double tolerance = 1e-10;  // on the rms of the right-hand side
    int anderson = 0;          // Anderson depth (0: plain RK4 pseudo-time)
    int check = 50;            // residual evaluated every `check` steps
    std::function<void(int, double)> progress;  // (step, residual) at each check
};

struct SteadyResult {
    int steps = 0;
    double residual = 0;
    bool converged = false;
    int restarts = 0;      // Anderson history resets (safeguard or residual growth)
    int rhs_evaluations = 0;
    double seconds = 0;
    std::vector<std::pair<int, double>> history;
};

class CompressibleSolver2D;

namespace detail {
// A solver's setup in double, for device backends (all (n0, n1) and (Qa, Qb) arrays indexed (i, j) / (a, b)).
struct FlowTables {
    FlowOptions options;
    std::array<Boundary, 4> boundaries;
    Eigen::Index n0 = 0, n1 = 0;
    struct Axis {
        Matrix D, Q, Minv, L, Mq, Cq;  // (n, n)
        Vector s0, s1, weights;
        double radius = 0, vradius = 0;
        Eigen::Index points = 0;
    } axis[2];
    Matrix x_xi, x_eta, y_xi, y_eta, J;      // nodal discrete metrics
    Matrix xq_xi, xq_eta, yq_xi, yq_eta;     // metrics at the Gauss points
    Matrix gcl[2];                           // free-stream correction vectors
    struct Face {
        Vector nx, ny, area;                 // at the face Gauss points
    } face[4];
    std::vector<double> source;              // (n0, n1, 4) C-ordered or empty
};
FlowTables flow_tables(const CompressibleSolver2D& solver);
}  // namespace detail

class CompressibleSolver2D {
public:
    // Faces: 0 xi = 0, 1 xi = 1, 2 eta = 0, 3 eta = 1.
    CompressibleSolver2D(const MappedGrid2D& grid, std::array<Boundary, 4> boundaries, FlowOptions options = {});

    const MappedGrid2D& grid() const;
    const FlowOptions& options() const;
    Shape shape() const;  // {n0, n1, 4}
    void set_boundary(int face, Boundary boundary);
    const Boundary& boundary(int face) const;
    // Physical coordinates of face k's Gauss points, (Q, 2) (where Exterior points states live).
    Matrix face_points(int face) const;
    // Optional forcing (n0, n1, 4) added to the right-hand side; an empty view clears it.
    void set_source(const FieldView& source);

    // dU/dt (n0, n1, 4). Throws std::domain_error on nonpositive density or pressure.
    Field rhs(const FieldView& U) const;
    Field local_dt(const FieldView& U) const;  // (n0, n1)
    Field step(const FieldView& U) const;      // one local-time-step RK4 step
    // March U (in place) to steady state.
    SteadyResult solve(Field& U, const SteadyOptions& options = {}) const;
    // BSPF interpolation of a coarser solver's state onto this grid (same map, so the same
    // reference coordinates): grid sequencing.
    Field prolong(const CompressibleSolver2D& coarse, const FieldView& U) const;

    // Discrete metrics at the nodes (D applied to the coordinates) and the mass flow
    // int (rho u y_eta - rho v x_eta) deta through every xi = const section, (n0).
    Vector mass_flow(const FieldView& U) const;

    struct Impl;

private:
    friend detail::FlowTables detail::flow_tables(const CompressibleSolver2D&);
    std::shared_ptr<Impl> d_;
};

// A CompressibleSolver2D as a multigrid level (smoothing: its own local-time-step steps).
class CpuFlowLevel : public FlowLevel {
public:
    explicit CpuFlowLevel(CompressibleSolver2D& solver) : s_(solver) {}
    Shape shape() const override { return s_.shape(); }
    const Grid& reference() const override { return s_.grid().reference(); }
    double gamma() const override { return s_.options().gamma; }
    Field rhs(const FieldView& U) const override { return s_.rhs(U); }
    Field smooth(const FieldView& U, int steps) const override;
    void set_source(const FieldView& source) override { s_.set_source(source); }

private:
    CompressibleSolver2D& s_;
};

}  // namespace bspf

