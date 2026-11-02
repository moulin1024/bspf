# SPDX-License-Identifier: GPL-3.0-or-later
"""High-precision (mpmath) evaluation of the BSPF grid derivative for
finite-difference endpoint jets. It applies the same linear map as pybspf and
bspf_cpp to the same float64 inputs, so either implementation's distance from it
is its own rounding error."""
from math import factorial

import mpmath as mp
import numpy as np


def _divide(a, b):
    return a / b if b != 0 else mp.mpf(0)


def _basis(knots, x, degree, derivative):
    nk = len(knots)
    if derivative > degree:
        return mp.zeros(len(x), nk - degree - 1)
    if derivative:
        lower = _basis(knots, x, degree - 1, derivative - 1)
        out = mp.zeros(len(x), nk - degree - 1)
        for i in range(nk - degree - 1):
            left = _divide(degree, knots[i + degree] - knots[i])
            right = _divide(degree, knots[i + degree + 1] - knots[i + 1])
            for r in range(len(x)):
                out[r, i] = lower[r, i] * left - lower[r, i + 1] * right
        return out
    last = max(j for j in range(nk - 1) if knots[j] < knots[-1])
    values = [[mp.mpf(0)] * (nk - 1) for _ in x]
    for r, xr in enumerate(x):
        if xr == knots[-1]:
            values[r][last] = mp.mpf(1)
            continue
        for j in range(nk - 1):
            if knots[j] <= xr < knots[j + 1]:
                values[r][j] = mp.mpf(1)
    for p in range(1, degree + 1):
        nxt = [[mp.mpf(0)] * (nk - 1 - p) for _ in x]
        for j in range(nk - 1 - p):
            dl, dr = knots[j + p] - knots[j], knots[j + p + 1] - knots[j + 1]
            for r, xr in enumerate(x):
                nxt[r][j] = _divide(xr - knots[j], dl) * values[r][j] + _divide(knots[j + p + 1] - xr, dr) * values[r][j + 1]
        values = nxt
    return mp.matrix(values)


def derivative(x, knots, f, order, *, degree, q, p, lam=0.0, dps=40):
    """Grid derivative of each column of f (float64 arrays), evaluated at `dps` digits."""
    with mp.workdps(dps):
        xs = [mp.mpf(float(v)) for v in x]
        ts = [mp.mpf(float(v)) for v in knots]
        n = len(xs)
        dx = mp.mpf(float(x[1] - x[0]))  # float spacing, as both implementations use
        B0 = _basis(ts, xs, degree, 0)
        Bk = _basis(ts, xs, degree, order)
        nb = B0.cols
        w = [dx] * n
        w[0] = w[-1] = dx / 2
        C = mp.zeros(2 * q, nb)
        for k in range(q):
            Bc = _basis(ts, [xs[0], xs[-1]], degree, k)
            for j in range(nb):
                C[k, j], C[q + k, j] = Bc[0, j], Bc[1, j]
        V = mp.matrix([[mp.mpf(j) ** k / factorial(k) for j in range(p)] for k in range(p)])
        X = mp.inverse(V)[:, :q] if q else mp.zeros(p, 0)
        left = [[X[j, k] / dx ** k for j in range(p)] for k in range(q)]
        K = mp.zeros(nb + 2 * q, nb + 2 * q)
        for i in range(nb):
            for j in range(nb):
                K[i, j] = 2 * (sum(B0[r, i] * w[r] * B0[r, j] for r in range(n)) + (lam if i == j else 0))
            for k in range(2 * q):
                K[i, nb + k] = -C[k, i]
                K[nb + k, i] = C[k, i]
        Kinv = mp.inverse(K)
        out = np.empty_like(f, dtype=float)
        for col in range(f.shape[1]):
            fs = [mp.mpf(float(v)) for v in f[:, col]]
            rhs = [2 * sum(B0[r, i] * w[r] * fs[r] for r in range(n)) for i in range(nb)]
            rhs += [sum(left[k][j] * fs[j] for j in range(p)) for k in range(q)]
            rhs += [(-1) ** k * sum(left[k][j] * fs[n - 1 - j] for j in range(p)) for k in range(q)]
            c = [sum(Kinv[i, j] * rhs[j] for j in range(len(rhs))) for i in range(nb)]
            spline = [sum(B0[r, i] * c[i] for i in range(nb)) for r in range(n)]
            r_ = [fs[i] - spline[i] for i in range(n)]
            # Re ifft((i w)^k fft(r)) with numpy fftfreq ordering (Nyquist negative).
            freqs = [j if j < (n - 1) // 2 + 1 else j - n for j in range(n)]
            R = [sum(r_[j] * mp.expj(-2 * mp.pi * l * j / n) for j in range(n)) for l in range(n)]
            mult = [(1j * 2 * mp.pi * fr / (n * dx)) ** order for fr in freqs]
            for i in range(n):
                spec = sum(mult[l] * R[l] * mp.expj(2 * mp.pi * l * i / n) for l in range(n)) / n
                out[i, col] = float(sum(Bk[i, j] * c[j] for j in range(nb)) + mp.re(spec))
        return out
