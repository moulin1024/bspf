// SPDX-License-Identifier: GPL-3.0-or-later
// Private state shared by a Line and the Ops compiled from it.
#pragma once

#include <fftw3.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <utility>

#include "bspf/endpoints.hpp"
#include "bspf/types.hpp"

namespace bspf::detail {

// Batched real FFTs of length n over m lines, either contiguous (column-major
// (n, m) data and (h, m) spectra) or interleaved (row-major: line l at offset l,
// stride m). FFTW's planner is not thread-safe, execution with new arrays is:
// plans are created once per (m, layout) under a process-wide mutex and reused.
class FFT {
public:
    explicit FFT(int n) : n_(n) {}
    ~FFT();
    FFT(const FFT&) = delete;
    FFT& operator=(const FFT&) = delete;

    // Unnormalized r2c of m lines of `real` into `spectrum` (h = n/2+1 per line).
    void forward(const double* real, std::complex<double>* spectrum, Eigen::Index m, bool interleaved) const;
    // c2r without the 1/n factor; overwrites `spectrum`.
    void backward(std::complex<double>* spectrum, double* real, Eigen::Index m, bool interleaved) const;

    // Column-major conveniences (used to build operators).
    ComplexMatrix forward(const Matrix& r) const;
    Matrix backward(ComplexMatrix spectrum) const;  // includes 1/n

private:
    std::pair<fftw_plan, fftw_plan> plans(Eigen::Index m, bool interleaved) const;
    int n_;
    mutable std::mutex mutex_;
    mutable std::map<std::pair<Eigen::Index, bool>, std::pair<fftw_plan, fftw_plan>> plans_;  // (r2c, c2r)
};

// A matrix whose nonzeros in each row lie in `width` consecutive columns
// (B-spline rows: degree + 1). Products act on row-major blocks of b lines
// (leading dimension b), so every inner loop is a contiguous axpy over lines.
struct Banded {
    Eigen::Index rows = 0, cols = 0, width = 0;
    std::vector<Eigen::Index> start;  // first column of each row's window
    std::vector<double> values;       // (rows, width), row-major

    // The banded form of `a`, or nullopt if some row is wider than max_width.
    static std::optional<Banded> from(const Matrix& a, Eigen::Index max_width);
    // y (rows, b) = alpha A x (accumulate: y += alpha A x), x (cols, b).
    void apply(const double* x, double* y, Eigen::Index b, double alpha, bool accumulate) const;
};

struct LineData {
    Vector x, knots, weights, omega, half_omega;
    int degree = 0, q = 0, p = 0;
    double lam = 0.0, dx = 0.0;
    Matrix basis, constraint, gram, kkt, fit_samples, fit_jets, jet_map;
    // Fit as applied: c = fit_inverse (2 W B^T f) + fit_jets g, with the
    // projection 2 B^T W (n_basis, N) banded (each spline's support).
    Banded basis_banded, projection;
    Matrix fit_inverse;  // top-left (n_basis, n_basis) block of KKT^{-1}
    BoundaryBlocks blocks;
    std::unique_ptr<FFT> fft;

    Eigen::Index size() const { return x.size(); }
    Eigen::Index n_basis() const { return basis.cols(); }
};

}  // namespace bspf::detail
