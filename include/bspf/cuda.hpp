// SPDX-License-Identifier: GPL-3.0-or-later
// FP32 CUDA backend. A bspf::Op (built and factored in double on the host) is
// rounded to float once and uploaded; every application then runs the BSPF
// split on the device:
//     c = (F + G J) f   (or F f + G g with given jets),  r = f - B c
//     R = rfft(r),  y = S c + irfft(m R) + L Re(E R)
// The fit maps F = K 2 B^T W and F + G J are formed in double: in float32 the
// CPU's two-step K (2 B^T W f) loses digits to cancellation in K.
// with cuBLAS SGEMMs and batched cuFFT R2C/C2R. Lines along non-innermost axes
// are gathered into contiguous blocks by a tiled transpose and scattered back.
//
// Precision: float32 throughout on the device. Spectral derivatives amplify
// rounding by ~omega^k, so expect relative errors of ~1e-6 (k = 1) growing with
// derivative order and N; see DESIGN.md for measurements.
//
// All device work is queued on one stream; calls are serialized by a process-
// wide lock. Results are ready after synchronize() or Array::download().
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bspf {
class Op;
class Grid;
}  // namespace bspf

namespace bspf::cuda {

using real = float;  // device precision
using Shape = std::vector<std::ptrdiff_t>;

bool available();         // a usable CUDA device is present
std::string device_name();
void synchronize();       // wait for all queued device work

// A C-ordered float32 array in device memory. Copies share storage.
class Array {
public:
    Array() = default;
    explicit Array(Shape shape);                          // uninitialized
    static Array upload(const double* data, Shape shape);  // host float64 -> device float32
    void download(double* out) const;                      // device -> host float64 (blocks)
    Array clone() const;                                   // deep copy
    Array reshaped(Shape shape) const;                     // same storage, same size

    const Shape& shape() const { return shape_; }
    std::ptrdiff_t size() const;
    real* data() { return data_.get(); }
    const real* data() const { return data_.get(); }

private:
    Shape shape_;
    std::shared_ptr<real> data_;
};

// A compiled bspf::Op on the device.
class Op {
public:
    explicit Op(const bspf::Op& op);
    std::ptrdiff_t rows() const;
    std::ptrdiff_t cols() const;
    std::ptrdiff_t jet_count() const;

    // out has in's shape with `axis` of length rows(); jets (optional) have
    // in's shape with `axis` of length jet_count(). out must not alias in.
    void apply_along(const Array& in, Array& out, int axis, const Array* jets = nullptr,
                     bool accumulate = false) const;
    Array apply_along(const Array& in, int axis, const Array* jets = nullptr) const;

    struct Impl;

private:
    std::shared_ptr<const Impl> impl_;
};

// Device counterpart of bspf::Grid (same field layout: leading spatial axes).
class Grid {
public:
    explicit Grid(const bspf::Grid& grid);
    int dimension() const;
    Shape shape() const;

    const Op& derivative(int axis, int order) const;  // cached
    Array partial(const Array& f, const std::vector<int>& orders) const;
    Array laplacian(const Array& f) const;
    Array interpolate(const Array& f, const std::vector<std::vector<double>>& coordinates,
                      const std::vector<int>& derivatives = {}) const;
    Array integrate(const Array& f) const;
    Array integrate(const Array& f, const std::vector<std::pair<double, double>>& bounds) const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace bspf::cuda
