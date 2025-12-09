#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Use the integrated PiecewiseBSPF1D from bspf module
from bspf import PiecewiseBSPF1D


# ======================================================================
# Periodic + internal jumps: FFT vs piecewise_bspf
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
    # 4. Piecewise piecewise_bspf (segmented at a, b)
    # ----------------------
    degree = 5
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
    print("PiecewiseBSPF1D : L2  error =", l2_err(d1_pw,     fprime),
          ", Linf error =", linf_err(d1_pw,     fprime))


        # Set up plotting parameters
    plt.rcParams.update({
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.5
    })
    
    # ----------------------
    # 7. Plot comparison
    # ----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    ax = axes[0]
    ax.plot(x, f_val, label="$f(x)$")
    ax.axvline(a, color="k", linestyle="--", label="jumps")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("(a)",loc='left', x=-0.15, fontsize=16, fontweight='bold')

    ax = axes[1]
    ax.plot(x, fprime, "k", label="Exact", linewidth=1)
    ax.plot(x, df_fft,   "-",  label="FFT")
    ax.plot(x, d1_pw,    "-.", label="BSPF")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("$f\'(x)$")
    ax.set_title("(b)",loc='left', x=-0.15, fontsize=16, fontweight='bold')
    ax.legend()

    ax = axes[2]
    ax.plot(x, np.abs(df_fft   - fprime),   label="FFT")
    ax.plot(x, np.abs(d1_pw    - fprime),   label="BSPF")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("$|Error|$")
    ax.set_xlabel("$x$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("(c)",loc='left', x=-0.15, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_periodic_vs_fft()
