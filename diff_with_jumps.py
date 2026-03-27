#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Use the integrated PiecewiseBSPF1D from bspf module
from bspf1d import PiecewiseBSPF1D


# ======================================================================
# Periodic + internal jumps: FFT vs PiecewiseBSPF1D
# ======================================================================

def test_periodic_vs_fft():
    # ----------------------
    # 1. Periodic grid
    # ----------------------
    L = 2.0 * np.pi
    N_base = 256
    mid_refine = 3.0  # >1 => add more points in middle segment [a,b] (total N increases)
    USE_FFT_BASELINE = False  # FFT needs uniform grid; keep False for refined mesh

    # 2. Construct function with "periodic + two internal jumps"
    A = 1.0
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

    # ----------------------
    # 1b. Build a piecewise-uniform grid with refined middle
    # ----------------------
    len_left = a - 0.0
    len_mid = b - a
    len_right = L - b
    dx_base = L / N_base

    # Keep base resolution outside; refine only the middle
    n0 = int(round(len_left / dx_base))   # even for this setup
    n2 = int(round(len_right / dx_base))  # even for this setup
    n1_base = int(round(len_mid / dx_base))
    n1 = int(round(n1_base * mid_refine))

    # Ensure even counts so that (n+1) is odd for Zhao per segment
    def _to_even(k: int) -> int:
        return k if (k % 2 == 0) else (k + 1)

    n0 = _to_even(max(2, n0))
    n1 = _to_even(max(2, n1))
    n2 = _to_even(max(2, n2))

    N = n0 + n1 + n2

    x0 = np.linspace(0.0, a, n0, endpoint=False)
    x1 = np.linspace(a, b, n1, endpoint=False)
    x2 = np.linspace(b, L, n2, endpoint=False)
    x = np.concatenate([x0, x1, x2])

    dx_min = float(np.min(np.diff(x)))
    f_val = f(x)
    fprime = fprime_exact(x)

    # ----------------------
    # 3. FFT spectral derivative (periodic) baseline (optional)
    # ----------------------
    if USE_FFT_BASELINE:
        x_fft = np.linspace(0.0, L, N, endpoint=False)
        dx_fft = x_fft[1] - x_fft[0]
        f_fft = f(x_fft)
        k = np.fft.fftfreq(N, d=dx_fft) * 2.0 * np.pi  # Wavenumber
        Fk = np.fft.fft(f_fft)
        df_fft = np.fft.ifft(1j * k * Fk).real

    # ----------------------
    # 4. Piecewise piecewise_bspf (segmented at a, b)
    # ----------------------
    degree = 7
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
    mask_smooth = dist_to_jump > dx_min

    def l2_err(num, exact):
        return np.linalg.norm(num[mask_smooth] - exact[mask_smooth], ord=2) \
               / np.sqrt(mask_smooth.sum())

    def linf_err(num, exact):
        return np.max(np.abs(num[mask_smooth] - exact[mask_smooth]))

    print("=== First derivative f'(x) vs analytic cos(x) (away from jumps) ===")
    if USE_FFT_BASELINE:
        fprime_fft = fprime_exact(x_fft)
        print("FFT spectral    : L2  error =", np.linalg.norm(df_fft - fprime_fft, ord=2) / np.sqrt(df_fft.size),
              ", Linf error =", np.max(np.abs(df_fft - fprime_fft)))
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
    ax.set_title("(a)",loc='left', x=-0.15, fontsize=14, fontweight='bold')

    ax = axes[1]
    ax.plot(x, fprime, "k", label="Exact", linewidth=1)
    if USE_FFT_BASELINE:
        ax.plot(x_fft, df_fft,   "-",  label="FFT")
    ax.plot(x, d1_pw,    "-.", label="BSPF")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("$f\'(x)$")
    ax.set_title("(b)",loc='left', x=-0.15, fontsize=14, fontweight='bold')
    ax.legend()

    ax = axes[2]
    if USE_FFT_BASELINE:
        ax.plot(x_fft, np.abs(df_fft   - fprime_exact(x_fft)),   label="FFT")
    ax.plot(x, np.abs(d1_pw    - fprime),   label="BSPF")
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_yscale("log")
    ax.set_ylabel("$|Error|$")
    ax.set_xlabel("$x$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("(c)",loc='left', x=-0.15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    # plt.savefig('diff_with_jumps_1d.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    test_periodic_vs_fft()
