#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Use the integrated PiecewiseBSPF1D from bspf module
from bspf1d import PiecewiseBSPF1D
from zhao import zhao2025_spectral_derivative


# ======================================================================
# Periodic + internal jumps: FFT vs piecewise_bspf
# ======================================================================

def test_periodic_vs_fft():
    # ----------------------
    # 1. Periodic grid
    # ----------------------
    L = 2.0 * np.pi
    N = 256
    x = np.linspace(0.0, L, N, endpoint=False)  # FFT-friendly: does not include right endpoint
    dx = x[1] - x[0]

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
    degree = 7
    pw = PiecewiseBSPF1D(degree=degree, x=x, breakpoints=[a, b])
    d1_pw, d2_pw, _ = pw.differentiate_1_2(f_val)

    # ----------------------
    # 5. Piecewise Zhao2025 (segmented at a, b)
    # ----------------------
    # IMPORTANT:
    # Zhao2025 assumes a canonical odd-N grid that spans the full interval endpoints.
    # Our global FFT grid uses endpoint=False, so naive slicing produces segments
    # that *do not include both endpoints* of each sub-interval, which can create
    # large errors (wrong affine scaling / violated canonical sampling assumption).
    #
    # Fix: build each segment grid explicitly with endpoint=True and odd N, using
    # the same dx, then evaluate the *smooth branch* on that segment.
    d1_zhao_pw = np.full_like(f_val, np.nan, dtype=float)

    # Indices where jumps live on this grid (they align exactly for this setup)
    idx_a = int(round(a / dx))
    idx_b = int(round(b / dx))

    # Segment 0: [0, a] uses branch sin(x)
    x0 = np.linspace(0.0, a, idx_a + 1, endpoint=True)  # includes a, length odd (=idx_a+1)
    y0 = np.sin(x0)
    d10 = zhao2025_spectral_derivative(x0, y0, order=1, domain=(0.0, a))
    # Fill global points strictly inside (exclude the jump point at a)
    d1_zhao_pw[:idx_a] = d10[:-1]

    # Segment 1: [a, b] uses branch sin(x) + A
    x1 = np.linspace(a, b, (idx_b - idx_a) + 1, endpoint=True)  # includes both ends
    y1 = np.sin(x1) + A
    d11 = zhao2025_spectral_derivative(x1, y1, order=1, domain=(a, b))
    # Fill global points strictly inside (exclude both jump points)
    d1_zhao_pw[idx_a + 1:idx_b] = d11[1:-1]

    # Segment 2: [b, L] uses branch sin(x)
    # Global grid excludes L, so we add it for Zhao, then drop it when filling.
    n2 = (N - idx_b) + 1  # include x=b..(L-dx) plus endpoint L
    x2 = np.linspace(b, L, n2, endpoint=True)
    y2 = np.sin(x2)
    d12 = zhao2025_spectral_derivative(x2, y2, order=1, domain=(b, L))
    # Fill global points strictly inside (exclude the jump at b; also exclude L which isn't in x)
    d1_zhao_pw[idx_b + 1:] = d12[1:-1]

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
    mask_smooth = dist_to_jump > dx

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
    print("PiecewiseZhao2025: L2  error =", l2_err(d1_zhao_pw, fprime),
          ", Linf error =", linf_err(d1_zhao_pw, fprime))


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
    ax.plot(x, df_fft,   "-",  label="FFT")
    ax.plot(x, d1_pw,    "-.", label="BSPF")
    ax.plot(x, d1_zhao_pw, ":", label="Zhao2025 (piecewise)", linewidth=1.5)
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("$f\'(x)$")
    ax.set_title("(b)",loc='left', x=-0.15, fontsize=14, fontweight='bold')
    ax.legend()

    ax = axes[2]
    ax.plot(x, np.abs(df_fft   - fprime),   label="FFT")
    ax.plot(x, np.abs(d1_pw    - fprime),   label="BSPF")
    ax.plot(x, np.abs(d1_zhao_pw - fprime), label="Zhao2025 (piecewise)")
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
