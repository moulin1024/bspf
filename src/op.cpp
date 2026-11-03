// SPDX-License-Identifier: GPL-3.0-or-later
#include "bspf/op.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "line_data.hpp"

namespace bspf {

namespace {

using Stride = Eigen::OuterStride<>;
using ConstMap = Eigen::Map<const Matrix, 0, Stride>;
using MutableMap = Eigen::Map<Matrix, 0, Stride>;
using Rows = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstRowsMap = Eigen::Map<const Rows, 0, Stride>;
using RowsMap = Eigen::Map<Rows, 0, Stride>;
using ComplexRows = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// True when lines are contiguous columns, false when interleaved.
bool lines_major(Eigen::Index stride, Eigen::Index distance, Eigen::Index count) {
    if (stride == 1) return true;
    if (distance == 1 || count == 1) return false;
    throw std::invalid_argument("lines must be contiguous (stride 1) or interleaved (distance 1)");
}

}  // namespace

Eigen::Index Op::jet_count() const { return line_->jet_map.rows(); }

const Matrix& Op::jet_map() const { return line_->jet_map; }

void Op::set_spline(const Matrix& spline) {
    if (auto band = detail::Banded::from(spline, line_->degree + 1)) {
        spline_banded_ = std::make_shared<const detail::Banded>(std::move(*band));
        spline_.resize(0, 0);
    } else {
        spline_ = spline;
        spline_banded_.reset();
    }
}

// Lines per block, about 8K samples per block (BSPF_LINE_BLOCK overrides it).
Eigen::Index Op::block_size() const {
    static const char* env = std::getenv("BSPF_LINE_BLOCK");
    if (env) return std::max<Eigen::Index>(1, std::atol(env));
    const Eigen::Index n = std::max(rows_, cols_);
    Eigen::Index block = 8;
    while (block < 128 && 2 * block * n <= 8192) block *= 2;
    return block;
}

void Op::compute(Block& x, Block& y, const Block* jets, Workspace& ws, ComplexBlock* spectra,
                 Eigen::Index l0) const {
    const detail::LineData& d = *line_;
    const Eigen::Index b = x.cols(), q = d.q, p = d.p, nb = d.n_basis();
    if (!jets && q) {
        // Estimated jets from the endpoint windows only (J is zero elsewhere).
        ws.g.resize(2 * q, b);
        ws.g.topRows(q).noalias() = d.blocks.left * x.topRows(p);
        ws.g.bottomRows(q).noalias() = d.blocks.right * x.bottomRows(p);
    }
    if (!jets && !q) ws.g.resize(0, b);
    const Block& jet_values = jets ? *jets : ws.g;
    if (jets_only_) {
        y = jet_values;  // (2q, b), also when q == 0
        return;
    }
    // Fit: c = K (2 B^T W x) + G g.
    ws.projected.resize(nb, b);
    d.projection.apply(x.data(), ws.projected.data(), b, 1.0, false);
    ws.c.noalias() = d.fit_inverse * ws.projected;
    if (q) ws.c.noalias() += d.fit_jets * jet_values;
    y.resize(rows_, b);
    if (spline_banded_) spline_banded_->apply(ws.c.data(), y.data(), b, 1.0, false);
    else y.noalias() = spline_ * ws.c;
    if (!correction_) return;
    // The gathered samples are no longer needed after fitting.
    d.basis_banded.apply(ws.c.data(), x.data(), b, -1.0, true);
    if (pass_remainder_) {  // irfft(rfft(r)) = r
        y += x;
        return;
    }
    ws.spectrum.resize(cols_ / 2 + 1, b);
    d.fft->forward(x.data(), ws.spectrum.data(), b, true);
    if (modes_real_.size()) {
        if (spectra) {
            spectra->middleCols(l0, b) = ws.spectrum;
        } else {
            mode_values(ws.spectrum, ws);
            if (lift_.size()) y.noalias() += lift_ * ws.values;
            else y += ws.values;
        }
    }
    if (multiplier_.size()) {
        ws.spectrum = multiplier_.asDiagonal() * ws.spectrum;
        d.fft->backward(ws.spectrum.data(), x.data(), b, true);  // reuse the residual buffer
        y += x;
    }
}

bool Op::defers_modes() const { return rows_ > 1 && modes_real_.size() && !lift_.size() && correction_; }

void Op::mode_values(const ComplexBlock& spectrum, Workspace& ws) const {
    ws.real = spectrum.real();
    ws.imag = spectrum.imag();
    ws.values.noalias() = modes_real_ * ws.real;
    ws.values.noalias() -= modes_imag_ * ws.imag;
}

void Op::check_lines(const MatrixRef& f) const {
    if (f.rows() != cols_)
        throw std::invalid_argument("expected " + std::to_string(cols_) + " samples per line, got " + std::to_string(f.rows()));
}

Matrix Op::matrix() const { return (*this)(Matrix::Identity(cols_, cols_)); }

Matrix Op::boundary_matrix() const {
    return (*this)(Matrix::Zero(cols_, jet_count()), Matrix::Identity(jet_count(), jet_count()));
}

Matrix Op::operator()(const MatrixRef& f) const {
    check_lines(f);
    Matrix y(rows_, f.cols());
    apply({f.data(), 1, f.outerStride(), f.cols()}, {y.data(), 1, rows_, f.cols()});
    return y;
}

Matrix Op::operator()(const MatrixRef& f, const MatrixRef& jets) const {
    check_lines(f);
    if (jets.rows() != jet_count() || jets.cols() != f.cols())
        throw std::invalid_argument("jets must have shape (" + std::to_string(jet_count()) + ", " +
                                    std::to_string(f.cols()) + ")");
    Matrix y(rows_, f.cols());
    const LinesView g{jets.data(), 1, jets.outerStride(), jets.cols()};
    apply({f.data(), 1, f.outerStride(), f.cols()}, {y.data(), 1, rows_, f.cols()}, &g);
    return y;
}

void Op::apply(const LinesView& in, const MutableLinesView& out, const LinesView* jets, bool accumulate) const {
    Workspace ws;
    apply_with_workspace(in, out, jets, accumulate, ws);
}

void Op::apply_with_workspace(const LinesView& in, const MutableLinesView& out, const LinesView* jets,
                              bool accumulate, Workspace& ws) const {
    const Eigen::Index m = in.count, J = jet_count();
    if (out.count != m || (jets && jets->count != m)) throw std::invalid_argument("line counts differ");
    if (m == 0) return;
    const bool major = lines_major(in.stride, in.distance, m);
    if (lines_major(out.stride, out.distance, m) != major || (jets && lines_major(jets->stride, jets->distance, m) != major))
        throw std::invalid_argument("input, output and jets must share one orientation");
    // Bound the deferred spectral products to cache-sized groups. Whole-volume
    // spectra (plus real/imaginary copies) overwhelm cache for 3D fields.
    const Eigen::Index block = block_size();
    const Eigen::Index group = std::max<Eigen::Index>(1, 32768 / std::max(cols_, rows_) / block) * block;
    if (defers_modes() && m > group) {
        for (Eigen::Index l = 0; l < m; l += group) {
            const Eigen::Index count = std::min(group, m - l);
            const LinesView x{in.data + l * in.distance, in.stride, in.distance, count};
            const MutableLinesView y{out.data + l * out.distance, out.stride, out.distance, count};
            const LinesView g{jets ? jets->data + l * jets->distance : nullptr,
                              jets ? jets->stride : 1, jets ? jets->distance : 1, count};
            apply_with_workspace(x, y, jets ? &g : nullptr, accumulate, ws);
        }
        return;
    }
    // Blocks of lines are gathered into row-major workspaces (a transpose in
    // cache for contiguous lines, row copies for interleaved ones).
    Block& x = ws.x;
    Block& y = ws.y;
    Block& g = ws.given_jets;
    // Point evaluations apply their (P, h) rows to a bounded group of spectra
    // in one product. Single-row integrals stay in the cache-local block.
    const bool deferred = defers_modes() && m > block;
    ComplexBlock& spectra = ws.spectra;
    if (deferred) spectra.resize(cols_ / 2 + 1, m);
    for (Eigen::Index l0 = 0; l0 < m; l0 += block) {
        const Eigen::Index b = std::min(block, m - l0);
        if (major) {
            x = ConstMap(in.data + l0 * in.distance, cols_, b, Stride(in.distance));
            if (jets) g = ConstMap(jets->data + l0 * jets->distance, J, b, Stride(jets->distance));
        } else {
            x = ConstRowsMap(in.data + l0, cols_, b, Stride(in.stride));
            if (jets) g = ConstRowsMap(jets->data + l0, J, b, Stride(jets->stride));
        }
        compute(x, y, jets ? &g : nullptr, ws, deferred ? &spectra : nullptr, l0);
        if (major) {
            MutableMap dest(out.data + l0 * out.distance, rows_, b, Stride(out.distance));
            if (accumulate) dest += y;
            else dest = y;
        } else {
            RowsMap dest(out.data + l0, rows_, b, Stride(out.stride));
            if (accumulate) dest += y;
            else dest = y;
        }
    }
    if (deferred) {
        mode_values(spectra, ws);
        if (major) MutableMap(out.data, rows_, m, Stride(out.distance)) += ws.values;
        else RowsMap(out.data, rows_, m, Stride(out.stride)) += ws.values;
    }
}

}  // namespace bspf
