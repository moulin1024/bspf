// SPDX-License-Identifier: GPL-3.0-or-later
// Plain (Eigen-free) interface of the device compressible solver (flow.cu).
// Device layouts: states planar [c][i][j]; Gauss-point arrays [b][a]; nodal
// tables [i][j]; (n, n) axis matrices column-major.
#pragma once

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include "galerkin_device.hpp"

namespace bspf::cuda::flow {

enum Kind { kInlet = 0, kOutlet = 1, kSlipWall = 2, kNoSlipWall = 3, kExterior = 4 };

struct FaceTables {
    int kind = kSlipWall;
    double total_pressure = 0, total_temperature = 0, back_pressure = 0;
    bool has_back_pressure = false;
    std::vector<double> nx, ny, area;     // (P)
    std::vector<double> exterior;         // (P, 4) column-major, Exterior only
    std::vector<double> exterior_nodes;   // (n, 4) column-major at the face nodes, Exterior only
};

struct FlowDeviceTables {
    int n0 = 0, n1 = 0;
    double gamma = 1.4, prandtl = 0.72, reynolds = 0, omega = 0.76, svv = 0, cfl = 0.6;
    AxisTables axis[2];                       // fast Galerkin evaluation / projection
    std::vector<double> Minv[2], Q[2];         // (n, n) column-major
    double radius[2] = {0, 0}, vradius[2] = {0, 0};
    std::vector<double> J, xi_x, xi_y, eta_x, eta_y, grad_xi, grad_eta;  // nodal [i][j]
    std::vector<double> x_xi, x_eta, y_xi, y_eta;                        // nodal [i][j]
    std::vector<double> qx_xi, qx_eta, qy_xi, qy_eta;                    // Gauss points [b][a]
    std::vector<double> gcl[2];                                          // nodal [i][j]
    FaceTables face[4];
    std::vector<double> source;  // planar or empty
    // line-implicit preconditioning along eta lines (LineImplicit::Full)
    bool line_implicit = false;
    int refresh = 1;                          // refactor every `refresh` steps
    std::vector<double> D_eta, s0_eta, s1_eta;  // (n1, n1) column-major, (n1), (n1)
    // weak-form eta operator: Z (Pb, n1 n1) with Z(b, j + l n1) = P1b(b, j) P0b(b, l); Gauss weights (Pb);
    // contravariant eta normal (-y_xi, x_xi) at the eta Gauss points of every line, [i][b]
    std::vector<double> line_Z, line_w, line_nx, line_ny;
};

struct SolveSettings {
    int max_steps = 100000;
    double tolerance = 1e-6;
    int anderson = 0;
    int check = 50;
    std::function<void(int, double)> progress;
};

struct SolveReport {
    int steps = 0, restarts = 0, rhs_evaluations = 0;
    double residual = 0, seconds = 0;
    bool converged = false;
    std::vector<std::pair<int, double>> history;
};

template <class T>
struct FlowDevice;  // T = float (FP32) or double (FP64)
template <class T>
std::shared_ptr<FlowDevice<T>> make_flow(const FlowDeviceTables& t);
// Host round trips, states C-ordered (n0, n1, 4) in double (converted to T on the device).
template <class T>
std::vector<double> flow_rhs_host(FlowDevice<T>& f, const std::vector<double>& U);
template <class T>
std::vector<double> flow_step_host(FlowDevice<T>& f, const std::vector<double>& U);
template <class T>
SolveReport flow_solve_host(FlowDevice<T>& f, std::vector<double>& U, const SolveSettings& s);
// Forcing added to the right-hand side, C-ordered (n0, n1, 4); empty clears it.
template <class T>
void flow_set_source(FlowDevice<T>& f, const std::vector<double>& source);

// Newton-Krylov with device-resident vectors (FP64): FGMRES with CGS2 orthogonalization, Jacobian
// products by finite differences of the device residual, backtracking line search.
struct NewtonSettings {
    double tolerance = 1e-10, eta_max = 0.1, eta_min = 1e-4, jacobian_epsilon = 1e-7;
    int max_newton = 40, krylov_restart = 40, max_krylov = 200;
    std::function<void(int, double, int)> progress;
};
struct NewtonReport {
    int newton_steps = 0, krylov_iterations = 0, start_cycles = 0, damping_cuts = 0;
    double residual = 0, seconds_jacobian = 0, seconds_krylov = 0, seconds_preconditioner = 0;
    bool converged = false;
    std::vector<std::pair<int, double>> history;
};
// precondition(U, z, new_state) -> ~J^{-1} z; all C-ordered (n0, n1, 4) host vectors.
using HostPreconditioner =
    std::function<std::vector<double>(const std::vector<double>&, const std::vector<double>&, bool)>;
// Device-resident FAS hierarchy on FP32 levels (finest first): the multigrid preconditioner and start.
struct MGSettings {
    int pre = 10, post = 10, coarse = 60, cycle_index = 1, initial = 50, preconditioner_cycles = 1;
    int max_start_cycles = 60;
    double damping = 0.5, preconditioner_epsilon = 1e-2, start_residual = 1e-3;
};
struct DeviceMG;
// Pa[l], Pb[l]: prolongation from level l + 1 to l per axis, (n_l, n_{l+1}) column-major; empty for a
// p-level (same nodes: identity).
std::shared_ptr<DeviceMG> make_device_mg(const std::vector<std::shared_ptr<FlowDevice<float>>>& levels,
                                         const std::vector<std::vector<double>>& Pa,
                                         const std::vector<std::vector<double>>& Pb, const MGSettings& s);

// mg: when given, the start phase and the preconditioner run as device FAS cycles (precondition unused).
NewtonReport flow_newton(FlowDevice<double>& f, std::vector<double>& U, const NewtonSettings& s,
                         const HostPreconditioner& precondition, DeviceMG* mg = nullptr,
                         const MGSettings* ms = nullptr);

}  // namespace bspf::cuda::flow
