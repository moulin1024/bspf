#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Build BSPF per-segment for fine control
from bspf1d import bspf1d


# ======================================================================
# Local refinement demo: refined middle segment
# (test function matches diff_with_mapping.py)
# ======================================================================

def test_periodic_vs_fft():
    # ----------------------
    # 1. Periodic grid
    # ----------------------
    L = 2.0 * np.pi
    N_base = 64
    mid_refine = 2.0  # >1 => add more points in middle segment [a,b] (total N increases)
    USE_FFT_BASELINE = False  # FFT needs uniform grid; keep False for refined mesh
    PLOT_SEG0_ONLY = True  # also show segment-0-only BSPF result

    # 2. Test function (same as diff_with_mapping.py)
    beta = 1.1
    omega = 100.5

    # Refined region boundaries (middle section)
    a = 0.5 * np.pi
    b = 1.5 * np.pi

    def f(x):
        u = x / (beta + np.cos(x))
        return np.sin(u)

    def fprime_exact(x):
        d = beta + np.cos(x)
        u = x / d
        du_dx = (d + x * np.sin(x)) / (d * d)
        return np.cos(u) * du_dx

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
    # 4. BSPF per segment (build operator + grid per segment)
    # ----------------------
    degree = 9
    num_boundary_points = degree + 3
    n_basis = 4 * degree
    lam = 0.0

    def bspf_diff_1(x_seg, y_seg, domain_seg, *, use_clustering=False, clustering_factor=2.0):
        op = bspf1d.from_grid(
            degree=degree,
            x=x_seg,
            domain=domain_seg,
            n_basis=n_basis,
            num_boundary_points=num_boundary_points,
            correction="spectral",
            use_clustering=use_clustering,
            clustering_factor=clustering_factor,
        )
        dy_seg, _ = op.differentiate(y_seg, k=1, lam=lam)
        return dy_seg

    # Segment-wise derivatives (no overlap because endpoint=False in x0/x1/x2)
    d1_pw = np.empty_like(f_val)
    d1_pw[:n0] = bspf_diff_1(x0, f(x0), (0.0, a), use_clustering=False)
    d1_pw[n0:n0 + n1] = bspf_diff_1(x1, f(x1), (a, b), use_clustering=False)
    d1_pw[n0 + n1:] = bspf_diff_1(x2, f(x2), (b, L), use_clustering=False)

    # Segment-0 only: treat [0,a) as a standalone "ordinary BSPF" problem
    op_seg0 = bspf1d.from_grid(
        degree=7,
        x=x0,
        domain=(0.0, a),
        n_basis=4*7,
        num_boundary_points=7+3,
        correction="spectral",
        use_clustering=False,
    )
    d1_seg0, y_spline_seg0 = op_seg0.differentiate(f(x0), k=1, lam=lam)

    # Global (single) BSPF baseline on a UNIFORM grid with the same total N
    x_uni = np.linspace(0.0, L, N, endpoint=False)
    f_uni = f(x_uni)
    fprime_uni = fprime_exact(x_uni)
    op_global = bspf1d.from_grid(
        degree=degree,
        x=x_uni,
        domain=(0.0, L),
        n_basis=n_basis,
        num_boundary_points=num_boundary_points,
        correction="spectral",
        use_clustering=False,
    )
    d1_global, _ = op_global.differentiate(f_uni, k=1, lam=lam)

    # ----------------------
    # 6. Error evaluation (global + regional)
    # ----------------------
    mask_mid = (x >= a) & (x < b)
    mask_out = ~mask_mid

    def l2_err(num, exact, mask):
        return np.linalg.norm(num[mask] - exact[mask], ord=2) / np.sqrt(mask.sum())

    def linf_err(num, exact, mask):
        return np.max(np.abs(num[mask] - exact[mask]))

    print("=== First derivative f'(x) vs analytic f'(x) ===")
    if USE_FFT_BASELINE:
        fprime_fft = fprime_exact(x_fft)
        print("FFT spectral    : L2  error =", np.linalg.norm(df_fft - fprime_fft, ord=2) / np.sqrt(df_fft.size),
              ", Linf error =", np.max(np.abs(df_fft - fprime_fft)))
    print("PiecewiseBSPF1D : L2  error =", l2_err(d1_pw, fprime, mask_out | mask_mid),
          ", Linf error =", linf_err(d1_pw, fprime, mask_out | mask_mid))
    print("  - middle [a,b): L2 =", l2_err(d1_pw, fprime, mask_mid), ", Linf =", linf_err(d1_pw, fprime, mask_mid))
    print("  - outer        : L2 =", l2_err(d1_pw, fprime, mask_out), ", Linf =", linf_err(d1_pw, fprime, mask_out))

    # Segment-0 error analysis (standalone vs piecewise slice vs global-uniform baseline)
    mask_all_0 = np.ones_like(x0, dtype=bool)
    trim_bd = int(num_boundary_points)
    mask_trim_0 = mask_all_0.copy()
    if 2 * trim_bd < mask_trim_0.size:
        mask_trim_0[:trim_bd] = False
        mask_trim_0[-trim_bd:] = False

    d1_pw_seg0 = d1_pw[:n0]
    fprime_0 = fprime_exact(x0)

    # Global-uniform restricted to segment 0
    mask0_uni = (x_uni >= 0.0) & (x_uni < a)
    x0_uni = x_uni[mask0_uni]
    d1_global_0 = d1_global[mask0_uni]
    fprime_uni_0 = fprime_uni[mask0_uni]

    mask_all_0_uni = np.ones_like(x0_uni, dtype=bool)
    mask_trim_0_uni = mask_all_0_uni.copy()
    if 2 * trim_bd < mask_trim_0_uni.size:
        mask_trim_0_uni[:trim_bd] = False
        mask_trim_0_uni[-trim_bd:] = False

    print("\n=== Segment 0 analysis: [0, a) ===")
    print("Standalone BSPF  : L2 =", l2_err(d1_seg0, fprime_0, mask_all_0), ", Linf =", linf_err(d1_seg0, fprime_0, mask_all_0))
    print("  trimmed        : L2 =", l2_err(d1_seg0, fprime_0, mask_trim_0), ", Linf =", linf_err(d1_seg0, fprime_0, mask_trim_0))
    print("Piecewise slice  : L2 =", l2_err(d1_pw_seg0, fprime_0, mask_all_0), ", Linf =", linf_err(d1_pw_seg0, fprime_0, mask_all_0))
    print("  trimmed        : L2 =", l2_err(d1_pw_seg0, fprime_0, mask_trim_0), ", Linf =", linf_err(d1_pw_seg0, fprime_0, mask_trim_0))
    print("Global uniform   : L2 =", l2_err(d1_global_0, fprime_uni_0, mask_all_0_uni), ", Linf =", linf_err(d1_global_0, fprime_uni_0, mask_all_0_uni))
    print("  trimmed        : L2 =", l2_err(d1_global_0, fprime_uni_0, mask_trim_0_uni), ", Linf =", linf_err(d1_global_0, fprime_uni_0, mask_trim_0_uni))

    mask_mid_uni = (x_uni >= a) & (x_uni < b)
    mask_out_uni = ~mask_mid_uni
    print("Global BSPF (uniform, same N): L2  error =", l2_err(d1_global, fprime_uni, mask_out_uni | mask_mid_uni),
          ", Linf error =", linf_err(d1_global, fprime_uni, mask_out_uni | mask_mid_uni))
    print("  - middle [a,b): L2 =", l2_err(d1_global, fprime_uni, mask_mid_uni), ", Linf =", linf_err(d1_global, fprime_uni, mask_mid_uni))
    print("  - outer        : L2 =", l2_err(d1_global, fprime_uni, mask_out_uni), ", Linf =", linf_err(d1_global, fprime_uni, mask_out_uni))


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
    ax.axvline(a, color="k", linestyle="--", label="refine region")
    ax.axvline(b, color="k", linestyle="--")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("(a)",loc='left', x=-0.15, fontsize=14, fontweight='bold')

    ax = axes[1]
    ax.plot(x, fprime, "k", label="Exact", linewidth=1)
    if USE_FFT_BASELINE:
        ax.plot(x_fft, df_fft,   "-",  label="FFT")
    ax.plot(x, d1_pw,    "-.", label="BSPF (piecewise)")
    ax.plot(x_uni, d1_global, "-", label="BSPF (global uniform)", linewidth=1)
    ax.axvline(a, color="k", linestyle="--")
    ax.axvline(b, color="k", linestyle="--")
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("$f\'(x)$")
    ax.set_title("(b)",loc='left', x=-0.15, fontsize=14, fontweight='bold')
    ax.legend()

    ax = axes[2]
    if USE_FFT_BASELINE:
        ax.plot(x_fft, np.abs(df_fft   - fprime_exact(x_fft)),   label="FFT")
    ax.plot(x, np.abs(d1_pw    - fprime),   label="BSPF (piecewise)")
    ax.plot(x_uni, np.abs(d1_global - fprime_uni),  label="BSPF (global uniform)")
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

    # Optional: dedicated view of segment 0 as a standalone BSPF solve
    if PLOT_SEG0_ONLY:
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharex=False)
        ax = axes2[0]
        ax.plot(x0, f(x0), "k", linewidth=1.0, label="Exact $f(x)$")
        ax.plot(x0, y_spline_seg0, "--", linewidth=1.0, label="BSPF spline")
        ax.set_title("Segment 0: function fit")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes2[1]
        ax.plot(x0, fprime_0, "k", linewidth=1.0, label="Exact $f'(x)$")
        ax.plot(x0, d1_seg0, "--", linewidth=1.0, label="Standalone BSPF")
        ax.plot(x0, d1_pw_seg0, "-.", linewidth=1.0, label="Piecewise slice")
        ax.set_title("Segment 0: derivative")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f'(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes2[2]
        ax.semilogy(x0, np.abs(d1_seg0 - fprime_0), "--", linewidth=1.0, label="Standalone BSPF")
        ax.semilogy(x0, np.abs(d1_pw_seg0 - fprime_0), "-.", linewidth=1.0, label="Piecewise slice")
        ax.semilogy(x0_uni, np.abs(d1_global_0 - fprime_uni_0), ":", linewidth=1.0, label="Global uniform")
        if 2 * trim_bd < x0.size:
            ax.axvline(x0[trim_bd], color="0.6", lw=1, ls="--")
            ax.axvline(x0[-trim_bd - 1], color="0.6", lw=1, ls="--", label="trim")
        ax.set_title("Segment 0: $|e|$ (log)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$|e|$")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()
    # plt.savefig('diff_with_jumps_1d.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    test_periodic_vs_fft()
