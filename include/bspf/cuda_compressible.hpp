// SPDX-License-Identifier: GPL-3.0-or-later
// Device version of CompressibleSolver2D, FP64 by default or FP32. It takes its setup (metrics,
// faces, free-stream correction, axis matrices) from a CPU solver and runs every
// step on the GPU: the BSPF trial field is evaluated at the Gauss points and
// projected back with GalerkinLine's fast structure (banded spline + cuFFT,
// cuda::GalerkinAxis), fluxes and face states in fused kernels, the per-axis
// mass solve and SVV with cuBLAS. Euler flows; local-time-step RK4, the CPU
// solver's line-implicit preconditioner along eta lines (LineImplicit::Full,
// batched cuBLAS LU, optionally refreshed every few steps) and Anderson
// acceleration. Viscous terms are not ported yet. FP32: residuals stall near 1e-4..1e-6.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "bspf/compressible.hpp"

namespace bspf::cuda {

class CompressibleSolver2D {
public:
    // refresh: line-implicit preconditioner refactored every `refresh` steps (LineImplicit::Full).
    // precision: "double" (FP64, default) or "float" (FP32, e.g. multigrid preconditioner levels).
    explicit CompressibleSolver2D(const bspf::CompressibleSolver2D& cpu, int refresh = 1,
                                  const std::string& precision = "double");
    Shape shape() const;
    Field rhs(const FieldView& U) const;   // float on the device, double in and out
    Field step(const FieldView& U) const;
    SteadyResult solve(Field& U, const SteadyOptions& options) const;
    // Forcing (n0, n1, 4) added to the right-hand side (e.g. a multigrid coarse-grid correction);
    // an empty view clears it.
    void set_source(const FieldView& source);
    // Newton-Krylov with device-resident vectors on this (FP64) solver: see bspf::NewtonDevice.
    std::unique_ptr<NewtonDevice> newton_device() const;
    // As above, with the start phase and the FAS preconditioner run on the device over `levels` (FP32
    // device solvers, finest first, nested per axis): no host round trips per cycle.
    std::unique_ptr<NewtonDevice> newton_device(const std::vector<CompressibleSolver2D*>& levels,
                                                const MultigridOptions& multigrid, double preconditioner_epsilon,
                                                double start_residual, int max_start_cycles,
                                                int preconditioner_cycles) const;
    const bspf::Grid& reference() const;  // host copy of the reference lines (for multigrid prolongation)
    double gamma() const;

    struct Impl;

private:
    std::shared_ptr<Impl> d_;
};

// A device solver as a multigrid level (smoothing runs on the device).
class DeviceFlowLevel : public FlowLevel {
public:
    explicit DeviceFlowLevel(CompressibleSolver2D& solver) : s_(solver) {}
    Shape shape() const override { return s_.shape(); }
    const bspf::Grid& reference() const override { return s_.reference(); }
    double gamma() const override { return s_.gamma(); }
    Field rhs(const FieldView& U) const override { return s_.rhs(U); }
    Field smooth(const FieldView& U, int steps) const override;
    void set_source(const FieldView& source) override { s_.set_source(source); }

private:
    CompressibleSolver2D& s_;
};

}  // namespace bspf::cuda

