// SPDX-License-Identifier: GPL-3.0-or-later
// Plain (Eigen-free) interface between the host code and galerkin.cu. Eigen
// objects must not cross into nvcc-compiled code: its host pass uses different
// alignment flags than the -march=native C++ build.
#pragma once

#include <memory>
#include <vector>

namespace bspf {
class GalerkinLine;
}

namespace bspf::cuda::flow {

// GalerkinLine tables in double, column-major (rounded on upload for FP32 devices).
struct AxisTables {
    int n = 0, nb = 0, Q = 0, ne = 0;
    long P = 0;
    std::vector<double> F, B;            // (nb, n), (n, nb)
    std::vector<double> S[2];            // stacked (Q n, nb)
    std::vector<double> shift[2];        // (Q, h) complex, interleaved re/im
    std::vector<int> rows;               // (Q, n)
    std::vector<double> w;               // (Q, n)
    std::vector<double> E[2], G[2], we;  // (ne, n), (ne, nb), (ne)
};

// From a GalerkinLine (galerkin_host.cpp, host compiler only).
AxisTables axis_tables(const GalerkinLine& line);

// Device axis in precision T (float or double).
template <class T>
struct AxisDevice;
template <class T>
std::shared_ptr<AxisDevice<T>> make_axis(const AxisTables& t);
template <class T>
void axis_evaluate(const AxisDevice<T>& a, const T* X, long m, int k, T* values);
template <class T>
void axis_project(const AxisDevice<T>& a, const T* values, long m, int k, T* X);
// Host round trips (column-major, double in and out).
template <class T>
std::vector<double> axis_evaluate_host(const AxisDevice<T>& a, const std::vector<double>& X, long m, int k);
template <class T>
std::vector<double> axis_project_host(const AxisDevice<T>& a, const std::vector<double>& values, long m, int k);
void* flow_stream();

}  // namespace bspf::cuda::flow
