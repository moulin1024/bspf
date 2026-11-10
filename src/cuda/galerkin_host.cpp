// SPDX-License-Identifier: GPL-3.0-or-later
// Host side of GalerkinAxis: exports GalerkinLine's tables in float (Eigen stays here).
#include <stdexcept>

#include "bspf/cuda_galerkin.hpp"
#include "galerkin_device.hpp"

namespace bspf::cuda {

std::vector<double> to_vec(const Matrix& M);
std::vector<double> to_vec(const Matrix& M) {  // column-major
    std::vector<double> out(static_cast<size_t>(M.size()));
    for (Eigen::Index c = 0; c < M.cols(); ++c)
        for (Eigen::Index r = 0; r < M.rows(); ++r) out[static_cast<size_t>(c * M.rows() + r)] = M(r, c);
    return out;
}

struct GalerkinAxis::Impl {
    std::shared_ptr<flow::AxisDevice<float>> device;
    long n = 0, P = 0;
};

}  // namespace bspf::cuda

namespace bspf::cuda::flow {
AxisTables axis_tables(const GalerkinLine& line) {
    const detail::GalerkinTables t = detail::galerkin_tables(line);
    flow::AxisTables a;
    a.n = int(t.n);
    a.nb = int(t.nb);
    a.Q = int(t.offsets);
    a.P = long(t.points);
    a.ne = int(t.extra_weights.size());
    a.F = to_vec(t.fit);
    a.B = to_vec(t.basis);
    const int h = a.n / 2 + 1;
    for (int k = 0; k < 2; ++k) {
        Matrix stacked(a.Q * a.n, a.nb);
        for (int q = 0; q < a.Q; ++q) stacked.middleRows(q * a.n, a.n) = t.spline[k][q];
        a.S[k] = to_vec(stacked);
        a.shift[k].resize(size_t(2) * a.Q * h);
        for (int q = 0; q < a.Q; ++q)
            for (int f = 0; f < h; ++f) {
                a.shift[k][size_t(2 * (q * h + f))] = t.shift[k][q](f).real();
                a.shift[k][size_t(2 * (q * h + f) + 1)] = t.shift[k][q](f).imag();
            }
        a.E[k] = to_vec(t.extra_trig[k]);
        a.G[k] = to_vec(t.extra_spline[k]);
    }
    a.rows.resize(size_t(a.Q) * a.n);
    a.w.resize(size_t(a.Q) * a.n);
    for (int q = 0; q < a.Q; ++q)
        for (int j = 0; j < a.n; ++j) {
            a.rows[size_t(q * a.n + j)] = int(t.rows[q][j]);
            a.w[size_t(q * a.n + j)] = t.weights[q](j);
        }
    a.we.resize(size_t(a.ne));
    for (int e = 0; e < a.ne; ++e) a.we[size_t(e)] = t.extra_weights(e);
    return a;
}
}  // namespace bspf::cuda::flow

namespace bspf::cuda {

GalerkinAxis::GalerkinAxis(const GalerkinLine& line) : d_(std::make_shared<Impl>()) {
    const flow::AxisTables a = flow::axis_tables(line);
    d_->device = flow::make_axis<float>(a);
    d_->n = a.n;
    d_->P = a.P;
}

long GalerkinAxis::size() const { return d_->n; }
long GalerkinAxis::points() const { return d_->P; }

void GalerkinAxis::evaluate(const float* X, long m, int k, float* values) const {
    flow::axis_evaluate(*d_->device, X, m, k, values);
}
void GalerkinAxis::project(const float* values, long m, int k, float* X) const {
    flow::axis_project(*d_->device, values, m, k, X);
}

Matrix GalerkinAxis::evaluate_host(const Matrix& X, int k) const {
    if (X.rows() != d_->n || k < 0 || k > 1) throw std::invalid_argument("evaluate: X must have N rows, k 0 or 1");
    const std::vector<double> v = flow::axis_evaluate_host(*d_->device, to_vec(X), X.cols(), k);
    Matrix out(d_->P, X.cols());
    for (Eigen::Index c = 0; c < X.cols(); ++c)
        for (long r = 0; r < d_->P; ++r) out(r, c) = v[size_t(c * d_->P + r)];
    return out;
}

Matrix GalerkinAxis::project_host(const Matrix& values, int k) const {
    if (values.rows() != d_->P || k < 0 || k > 1) throw std::invalid_argument("project: values must have P rows");
    const std::vector<double> x = flow::axis_project_host(*d_->device, to_vec(values), values.cols(), k);
    Matrix out(d_->n, values.cols());
    for (Eigen::Index c = 0; c < values.cols(); ++c)
        for (long r = 0; r < d_->n; ++r) out(r, c) = x[size_t(c * d_->n + r)];
    return out;
}

void* stream_handle() { return flow::flow_stream(); }

}  // namespace bspf::cuda
