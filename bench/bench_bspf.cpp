// SPDX-License-Identifier: GPL-3.0-or-later
// Steady-state cost of a grid first derivative along the innermost axis
// (contiguous lines) and the outermost axis (interleaved lines).
#include <chrono>
#include <cstdio>

#include "bspf/grid.hpp"

using bspf::Matrix;

namespace {

template <class F>
double seconds_per_call(F&& f) {
    f();  // warm-up (FFTW plans, allocations)
    for (int reps = 1;; reps *= 2) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) f();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (dt > 0.2) return dt / reps;
    }
}

}  // namespace

int main() {
    std::printf("ns per sample\n%6s %6s %10s %10s\n", "N", "lines", "inner", "outer");
    for (int n : {32, 64, 128, 256, 512, 1024}) {
        const bspf::Line line(bspf::Vector::LinSpaced(n, 0.0, 1.0));
        const bspf::Op d1 = line.derivative(1);
        for (int m : {1, 64, 1024}) {
            const Matrix f = Matrix::Random(n, m);
            std::vector<double> out(static_cast<size_t>(n) * m);
            const bspf::FieldView inner{f.data(), {m, n}}, outer{f.data(), {n, m}};
            auto time = [&](const bspf::FieldView& v, int axis) {
                return 1e9 * seconds_per_call([&] { bspf::apply_along(d1, v, out.data(), axis); }) / (double(n) * m);
            };
            std::printf("%6d %6d %10.2f %10.2f\n", n, m, time(inner, 1), time(outer, 0));
        }
    }
    return 0;
}
