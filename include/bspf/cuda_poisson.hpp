// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace bspf::cuda {
// Single mapped tensor patch, all six faces Dirichlet. Tensors are x-fastest:
// x + n*(y+n*z). Small matrices are COLUMN MAJOR. No dense 3D matrix is formed.
// The caller supplies the BSPF axis and its 1D generalized eigensystem
// K_II V = M_II V Lambda, V^T M_II V = I (computed in FP64 during setup).
struct Poisson3DData {
    int n = 0, q = 0;
    std::vector<double> basis, derivative; // (q,n)
    std::vector<double> eigenvectors, eigenvalues; // (n-2,n-2), (n-2)
    // W det(J) J^-1 J^-T, components xx,xy,xz,yy,yz,zz; q^3 each.
    std::array<std::vector<double>,6> metric;
    std::vector<double> measure; // W det(J), q^3
};
struct Poisson3DOptions {
    double relative_tolerance = 1e-12;
    int max_iterations = 1000;
    int residual_refresh = 40; // true-residual replacement AND CG restart
};
struct Poisson3DStats {
    int iterations = 0, residual_replacements = 0;
    double relative_residual = 0; // independently recomputed full lifted system
    double cg_seconds = 0, solve_seconds = 0;
};
class GalerkinPoisson3D {
public:
    explicit GalerkinPoisson3D(const Poisson3DData&);
    ~GalerkinPoisson3D();
    GalerkinPoisson3D(const GalerkinPoisson3D&) = delete;
    GalerkinPoisson3D& operator=(const GalerkinPoisson3D&) = delete;
    // forcing: q^3 quadrature samples; boundary: n^3 nodal values.
    // Uploads are outside solve timing. boundary must be zero in the interior.
    void set_rhs(const std::vector<double>& forcing, const std::vector<double>& boundary);
    // Already integrated weak load: n^3 entries, before boundary lifting.
    // Repeated solves reuse it without performing quadrature RHS projection.
    void set_load(const std::vector<double>& load, const std::vector<double>& boundary);
    // Every call starts from the same boundary lift and zero interior correction.
    // FP64 operator/CG/reductions; FP32 tensor inverse, no TF32 arithmetic.
    Poisson3DStats solve(const Poisson3DOptions& = {});
    std::vector<double> solution() const;
    // Host adapters for independent CPU-reference / adjoint checks, not the solve path.
    std::vector<double> apply_host(const std::vector<double>&);
    std::vector<double> precondition_host(const std::vector<double>&);
    std::size_t device_bytes() const;
    int cublas_version() const;
    std::string fp64_emulation() const; // requested strategy; cuBLAS may fall back to native
    bool shared_contractions() const;
    bool parity_optimized() const; // reflection-split FP64 contractions active
    bool fused_tensor_layout() const; // parity conversion includes tensor rotations
    bool octant_optimized() const; // validated reflection-symmetric metric sectors
private:
    struct Impl;
    std::unique_ptr<Impl> d_;
};
} // namespace bspf::cuda
