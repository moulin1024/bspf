#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Use the integrated PiecewiseBSPF1D from bspf module
from bspf import bspf1d, PiecewiseBSPF1D


# ======================================================================
# Periodic + internal jumps: FFT vs global bspf vs piecewise_bspf
# ======================================================================

def test_periodic_vs_fft():
    # ----------------------
    # 1. Periodic grid
    # ----------------------
    L = 2.0 * np.pi
    N = 512
    x = np.linspace(0.0, L, N, endpoint=False)  # FFT-friendly: does not include right endpoint
    dx = x[1] - x[0]

    # 2. Construct function with "periodic + two internal jumps"
    A = 0.8
    a = 0.5 * np.pi        # First jump location
    b = 1.5 * np.pi        # Second jump location

    def f(x):
        out = np.sin(x)
        # H(x-a) - H(x-b)
        out += A * ((x >= a).astype(float) - (x >= b).astype(float))
        return out

    def fprime_exact(x):
        # True derivative (except at jump points): cos(x)
        return np.cos(x)

    f_val = f(x)
    fprime = fprime_exact(x)

    # ----------------------
    # 3. FFT spectral derivative (periodic)
    # ----------------------
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi  # Wavenumber
    Fk = np.fft.fft(f_val)
    df_fft = np.fft.ifft(1j * k * Fk).real

    # ----------------------
    # 4. Global bspf1d derivative
    # ----------------------
    degree = 5
    bspf_global = bspf1d.from_grid(degree=degree, x=x)
    d1_global, d2_global, _ = bspf_global.differentiate_1_2(f_val)

    # ----------------------
    # 5. Piecewise piecewise_bspf (segmented at a, b)
    # ----------------------
    pw = PiecewiseBSPF1D(degree=degree, x=x, breakpoints=[a, b])
    d1_pw, d2_pw, _ = pw.differentiate_1_2(f_val)

    # ----------------------
    # 6. Error evaluation (compare with analytical cos(x) away from jumps)
    # ----------------------
    # Exclude regions too close to jumps from error calculation
    dist_to_jump = np.minimum.reduce([
        np.abs(x - a),
        np.abs(x - b),
        np.abs((x - a + L) % L),  # Handle periodic distance
        np.abs((x - b + L) % L),
    ])
    mask_smooth = dist_to_jump > 5 * dx

    def l2_err(num, exact):
        return np.linalg.norm(num[mask_smooth] - exact[mask_smooth], ord=2) \
               / np.sqrt(mask_smooth.sum())

    def linf_err(num, exact):
        return np.max(np.abs(num[mask_smooth] - exact[mask_smooth]))

    print("=== First derivative f'(x) vs analytic cos(x) (away from jumps) ===")
    print("FFT spectral    : L2  error =", l2_err(df_fft,   fprime),
          ", Linf error =", linf_err(df_fft,   fprime))
    print("Global bspf1d   : L2  error =", l2_err(d1_global, fprime),
          ", Linf error =", linf_err(d1_global, fprime))
    print("PiecewiseBSPF1D : L2  error =", l2_err(d1_pw,     fprime),
          ", Linf error =", linf_err(d1_pw,     fprime))

    # ----------------------
    # 7. Plot comparison
    # ----------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    ax.plot(x, f_val, label="f(x)")
    ax.axvline(a, color="k", linestyle="--", label="jump a,b")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.set_title("Periodic function with two internal jumps")

    ax = axes[1]
    ax.plot(x, fprime, "k", label="analytic f'(x)=cos(x)")
    # ax.plot(x, df_fft,   "-",  label="FFT spectral")
    # ax.plot(x, d1_global, "--", label="global bspf1d")
    ax.plot(x, d1_pw,    "-.", label="piecewise bspf1d")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_ylabel("f'(x)")
    ax.legend()

    ax = axes[2]
    ax.plot(x, np.abs(df_fft   - fprime),   label="|FFT - exact|")
    ax.plot(x, np.abs(d1_global - fprime),  label="|global bspf - exact|")
    ax.plot(x, np.abs(d1_pw    - fprime),   label="|piecewise bspf - exact|")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("abs error in f'(x)")
    ax.set_xlabel("x")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_periodic_vs_fft()
