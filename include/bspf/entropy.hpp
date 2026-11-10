// SPDX-License-Identifier: GPL-3.0-or-later
// Entropy variables of the ideal-gas Euler equations, for the entropy function
// -rho s / (gamma - 1), s = ln(p rho^-gamma):
//     V = ((gamma - s) / (gamma - 1) - rho |u|^2 / (2 p), rho u / p, rho v / p, -rho / p),
// and the inverse map U(V) (requires V_3 < 0). Galerkin with the trial field in V
// is entropy stable: sum_I V_I R_I reduces to boundary entropy fluxes.
#pragma once

#include <cmath>

namespace bspf {

#if defined(__CUDACC__)
#define BSPF_HD __host__ __device__
#else
#define BSPF_HD
#endif

template <class T>
BSPF_HD inline bool conservative_to_entropy(const T* q, T gamma, T* v) {
    const T rho = q[0], u = q[1] / rho, w = q[2] / rho;
    const T p = (gamma - 1) * (q[3] - T(0.5) * rho * (u * u + w * w));
    if (!(rho > 0) || !(p > 0)) return false;
    const T s = std::log(p) - gamma * std::log(rho);
    const T beta = rho / p;
    v[0] = (gamma - s) / (gamma - 1) - T(0.5) * beta * (u * u + w * w);
    v[1] = beta * u;
    v[2] = beta * w;
    v[3] = -beta;
    return true;
}

template <class T>
BSPF_HD inline bool entropy_to_conservative(const T* v, T gamma, T* q) {
    if (!(v[3] < 0)) return false;
    const T beta = -v[3];
    const T u = v[1] / beta, w = v[2] / beta;
    const T s = gamma - (gamma - 1) * (v[0] + T(0.5) * beta * (u * u + w * w));
    // p rho^-gamma = e^s and rho / p = beta  =>  rho = (beta e^s)^(-1 / (gamma - 1))
    const T rho = std::exp(-(std::log(beta) + s) / (gamma - 1));
    const T p = rho / beta;
    q[0] = rho;
    q[1] = rho * u;
    q[2] = rho * w;
    q[3] = p / (gamma - 1) + T(0.5) * rho * (u * u + w * w);
    return true;
}

}  // namespace bspf
