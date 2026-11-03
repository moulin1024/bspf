// SPDX-License-Identifier: GPL-3.0-or-later
// Op: a compiled BSPF operator. For samples f of length N and endpoint jets g
// (estimated as J f unless given) every op runs the same split:
//     c = F f + G g,   r = f - B c,   R = rfft(r)
//     y = S c + irfft(m * R) + L Re(E R)
// with S the spline rows, m a spectral multiplier (grid outputs) and E fixed
// spectral rows lifted by L (point outputs, integration constants). As one
// affine map, y = A f + A_j (g - J f). Ops are immutable and thread-safe.
#pragma once

#include <memory>

#include "bspf/types.hpp"

namespace bspf {

class Line;
struct FieldView;
namespace detail {
struct LineData;
struct Banded;
}  // namespace detail

// A batch of `count` lines of samples: sample i of line l is data[l*distance + i*stride].
// Either stride == 1 (lines contiguous) or distance == 1 (lines interleaved).
struct LinesView {
    const double* data;
    Eigen::Index stride;
    Eigen::Index distance;
    Eigen::Index count;
};

struct MutableLinesView {
    double* data;
    Eigen::Index stride;
    Eigen::Index distance;
    Eigen::Index count;
};

class Op {
public:
    Eigen::Index rows() const { return rows_; }  // P
    Eigen::Index cols() const { return cols_; }  // N
    Eigen::Index jet_count() const;              // 2q

    Matrix matrix() const;           // A (P, N), assembled by applying the op to I
    Matrix boundary_matrix() const;  // A_j (P, 2q): response to given jets
    const Matrix& jet_map() const;   // J (2q, N)

    // Column-major convenience: f (N, m) -> (P, m); jets (2q, m).
    Matrix operator()(const MatrixRef& f) const;
    Matrix operator()(const MatrixRef& f, const MatrixRef& jets) const;

    // Strided kernel; in, out and jets must share the same orientation.
    // With accumulate, add to the existing output instead of overwriting it.
    void apply(const LinesView& in, const MutableLinesView& out, const LinesView* jets = nullptr,
               bool accumulate = false) const;

private:
    friend class Line;
    friend void apply_along(const Op&, const FieldView&, double*, int, const double*, bool);
    using Block = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Op() = default;
    // Stores S, banded when its rows have B-spline support.
    void set_spline(const Matrix& spline);
    using ComplexBlock = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    // Call-local temporaries reused across blocks, spectral groups and axis slabs.
    struct Workspace {
        Block x, y, given_jets, g, projected, c, real, imag, values;
        ComplexBlock spectrum, spectra;
    };
    // y (P, b) for a block of b lines x (N, b) and optional jets (2q, b), all
    // row-major (sample-major), so the kernels are contiguous loops over lines.
    // After the fit, x is reused for the residual and inverse FFT output.
    // With `spectra`, the remainder spectrum is stored there (columns l0..l0+b)
    // and the mode rows are applied later to all lines at once.
    void compute(Block& x, Block& y, const Block* jets, Workspace& ws,
                 ComplexBlock* spectra = nullptr, Eigen::Index l0 = 0) const;
    void apply_with_workspace(const LinesView& in, const MutableLinesView& out,
                              const LinesView* jets, bool accumulate, Workspace& ws) const;
    Eigen::Index block_size() const;
    // Point evaluations and integrals apply their mode rows to many lines at once.
    bool defers_modes() const;
    // ws.values = Re(E R) for half spectra R (h, b).
    void mode_values(const ComplexBlock& spectrum, Workspace& ws) const;
    void check_lines(const MatrixRef& f) const;

    Eigen::Index rows_ = 0, cols_ = 0;
    std::shared_ptr<const detail::LineData> line_;
    bool jets_only_ = false;       // the jet estimator itself
    Matrix spline_;                // S (P, n_basis) when not banded
    std::shared_ptr<const detail::Banded> spline_banded_;
    bool correction_ = true;       // false: spline part only
    bool pass_remainder_ = false;  // order-0 grid derivative: y += r
    Eigen::VectorXcd multiplier_;  // m / N (h), empty when unused
    Matrix modes_real_, modes_imag_;  // E (k, h), empty when unused
    Matrix lift_;                  // L (P, k); empty means identity (k == P)
};

}  // namespace bspf
