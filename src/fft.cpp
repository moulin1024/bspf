// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <stdexcept>

#include "line_data.hpp"

namespace bspf::detail {

std::mutex& planner_mutex() {
    static std::mutex m;
    return m;
}

FFT::~FFT() {
    std::lock_guard<std::mutex> lock(planner_mutex());
    for (auto& entry : plans_) {
        fftw_destroy_plan(entry.second.first);
        fftw_destroy_plan(entry.second.second);
    }
}

std::pair<fftw_plan, fftw_plan> FFT::plans(Eigen::Index m, bool interleaved) const {
    std::lock_guard<std::mutex> own(mutex_);
    auto it = plans_.find({m, interleaved});
    if (it != plans_.end()) return it->second;
    std::lock_guard<std::mutex> lock(planner_mutex());
    const int n = n_, h = n / 2 + 1, howmany = static_cast<int>(m);
    // Contiguous: line stride 1, lines n (h) apart. Interleaved: stride m, lines 1 apart.
    const int stride = interleaved ? howmany : 1;
    const int real_dist = interleaved ? 1 : n, complex_dist = interleaved ? 1 : h;
    double* real = fftw_alloc_real(static_cast<size_t>(n) * m);
    fftw_complex* spectrum = fftw_alloc_complex(static_cast<size_t>(h) * m);
    // Estimate planning: plans are made lazily per batch size on the apply path,
    // and measured planning was not faster here (bench/bench_bspf).
    const unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
    fftw_plan r2c = fftw_plan_many_dft_r2c(1, &n, howmany, real, nullptr, stride, real_dist, spectrum, nullptr, stride,
                                           complex_dist, flags);
    fftw_plan c2r = fftw_plan_many_dft_c2r(1, &n, howmany, spectrum, nullptr, stride, complex_dist, real, nullptr,
                                           stride, real_dist, flags | FFTW_DESTROY_INPUT);
    fftw_free(real);
    fftw_free(spectrum);
    if (!r2c || !c2r) throw std::runtime_error("FFTW planning failed");
    return plans_.emplace(std::make_pair(m, interleaved), std::make_pair(r2c, c2r)).first->second;
}

void FFT::forward(const double* real, std::complex<double>* spectrum, Eigen::Index m, bool interleaved) const {
    if (m == 0) return;
    // Out-of-place r2c preserves its input.
    fftw_execute_dft_r2c(plans(m, interleaved).first, const_cast<double*>(real), reinterpret_cast<fftw_complex*>(spectrum));
}

void FFT::backward(std::complex<double>* spectrum, double* real, Eigen::Index m, bool interleaved) const {
    if (m == 0) return;
    // c2r keeps only the real part of the zero and Nyquist bins, as Re(ifft) of the full spectrum does.
    fftw_execute_dft_c2r(plans(m, interleaved).second, reinterpret_cast<fftw_complex*>(spectrum), real);
}

std::optional<Banded> Banded::from(const Matrix& a, Eigen::Index max_width) {
    Banded band;
    band.rows = a.rows();
    band.cols = a.cols();
    std::vector<Eigen::Index> first(a.rows(), 0), last(a.rows(), -1);
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        for (Eigen::Index j = 0; j < a.cols(); ++j)
            if (a(i, j) != 0.0) {
                if (last[i] < 0) first[i] = j;
                last[i] = j;
            }
        band.width = std::max(band.width, last[i] - first[i] + 1);
    }
    if (band.width > max_width) return std::nullopt;
    band.width = std::max<Eigen::Index>(band.width, 1);
    band.start.resize(a.rows());
    band.values.assign(static_cast<size_t>(a.rows() * band.width), 0.0);
    for (Eigen::Index i = 0; i < a.rows(); ++i) {
        const Eigen::Index s = std::min(first[i], a.cols() - band.width);
        band.start[i] = s;
        for (Eigen::Index k = 0; k < band.width; ++k) band.values[i * band.width + k] = a(i, s + k);
    }
    return band;
}

void Banded::apply(const double* x, double* y, Eigen::Index b, double alpha, bool accumulate) const {
    if (b == 1) {  // one line: a short dot product per row
        for (Eigen::Index i = 0; i < rows; ++i) {
            const double* v = values.data() + i * width;
            const double* xr = x + start[i];
            double sum = 0.0;
            for (Eigen::Index k = 0; k < width; ++k) sum += v[k] * xr[k];
            y[i] = accumulate ? y[i] + alpha * sum : alpha * sum;
        }
        return;
    }
    // Gather form: each output row is written once, from register accumulators
    // over chunks of 8 lines (one load per input value and FMA).
    constexpr Eigen::Index chunk = 8;
    for (Eigen::Index i = 0; i < rows; ++i) {
        const double* v = values.data() + i * width;
        const double* base = x + start[i] * b;
        double* yr = y + i * b;
        Eigen::Index l = 0;
        for (; l + chunk <= b; l += chunk) {
            double acc[chunk] = {};
            for (Eigen::Index k = 0; k < width; ++k) {
                const double a = v[k];
                const double* xr = base + k * b + l;
                for (Eigen::Index j = 0; j < chunk; ++j) acc[j] += a * xr[j];
            }
            for (Eigen::Index j = 0; j < chunk; ++j) yr[l + j] = accumulate ? yr[l + j] + alpha * acc[j] : alpha * acc[j];
        }
        for (; l < b; ++l) {
            double sum = 0.0;
            for (Eigen::Index k = 0; k < width; ++k) sum += v[k] * base[k * b + l];
            yr[l] = accumulate ? yr[l] + alpha * sum : alpha * sum;
        }
    }
}

}  // namespace bspf::detail
