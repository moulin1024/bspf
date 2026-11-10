// SPDX-License-Identifier: GPL-3.0-or-later
// Plain host tables of an op, handed from the Eigen side (export.cpp) to the
// CUDA side (backend.cu), which never includes Eigen. Matrices are column-major.
#pragma once

#include <complex>
#include <memory>
#include <vector>

#include "bspf/cuda.hpp"

namespace bspf::cuda::detail {

struct HostMatrix {
    long rows = 0, cols = 0;
    std::vector<double> data;  // column-major
};

struct LineTables {
    const void* key = nullptr;            // identity of the host LineData
    std::shared_ptr<const void> keepalive;  // keeps the key unique while cached
    int n = 0, nb = 0, q = 0, p = 0;
    // Fit maps formed in double and rounded once. In float32 the CPU's
    // two-step K (2 B^T W f) loses digits to cancellation in K; the folded maps
    // round no worse than the input itself does.
    HostMatrix fit;          // F = K 2 B^T W (nb, n): given jets, c = F f + G g
    HostMatrix fit_total;    // F + G J (nb, n): estimated jets, c = (F + G J) f
    HostMatrix fit_jets;     // G (nb, 2q)
    HostMatrix basis;       // B (n, nb)
    HostMatrix jets_left, jets_right;  // (q, p) each
};

struct OpTables {
    long rows = 0, cols = 0;
    bool jets_only = false, correction = true, pass_remainder = false;
    HostMatrix spline;                              // S (rows, nb)
    std::vector<std::complex<double>> multiplier;   // m / N (h), empty when unused
    HostMatrix modes;                               // (k, 2h): [Re E_j, -Im E_j] interleaved
    HostMatrix lift;                                // L (rows, k), empty = identity
    LineTables line;
};

std::shared_ptr<const Op::Impl> make_op(const OpTables& tables);

}  // namespace bspf::cuda::detail
