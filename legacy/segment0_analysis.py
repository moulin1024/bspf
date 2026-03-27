#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Piecewise BSPF analysis with multiple segments.

This script performs BSPF differentiation on multiple segments independently, with:
- function fit (spline vs exact) for each segment
- derivative (BSPF vs exact) for each segment
- error curves for each segment

Segment boundaries are specified via --boundaries argument (e.g., "0.0 0.4 0.6 1.0" creates
three segments: [0.0, 0.4), [0.4, 0.6), [0.6, 1.0)).
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from bspf1d import bspf1d


def _sanitize_bspf_params(*, n_pts: int, degree: int, n_basis: int, num_bd: int, lam: float) -> tuple[int, int, float]:
    """
    Heuristics to avoid singular KKT systems on very small segments.
    Keeps the interface simple: returns possibly-reduced (n_basis, num_bd, lam).
    """
    # num_bd must be small relative to n_pts (it adds constraints rows)
    # Heuristic: ensure 2*num_bd + 3 <= n_pts
    max_num_bd = max(2, (n_pts - 3) // 2)
    if num_bd > max_num_bd:
        num_bd = max_num_bd

    # n_basis too large relative to n_pts can also create ill-conditioning
    n_basis = min(n_basis, max(2 * degree, n_pts))

    # If user requests zero regularization on tiny segments, add a tiny lam
    if lam == 0.0 and n_pts < 4 * degree:
        lam = 1e-6

    return n_basis, num_bd, lam


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boundaries", type=str, default="0.0 0.4 0.6 1.0", 
                   help="Segment boundaries (space-separated, e.g., '0.0 0.4 0.6 1.0')")
    p.add_argument("--N_per_segment", type=str, default="64 256 64", 
                   help="Number of points per segment (space-separated, e.g., '128 256 128' for 3 segments, or single value for all)")
    p.add_argument("--beta", type=float, default=1.1)
    p.add_argument("--degree", type=int, default=10)
    p.add_argument("--n_basis", type=int, default=4*10, help="default: 4*degree")
    p.add_argument("--num_boundary_points", type=int, default=10+5, help="default: degree+3")
    p.add_argument("--lam", type=float, default=0.0)
    args = p.parse_args()

    # Parse segment boundaries
    boundaries_str = args.boundaries.strip().split()
    boundaries = [float(b) for b in boundaries_str]
    boundaries = sorted(set(boundaries))  # Remove duplicates and sort
    
    if len(boundaries) < 2:
        raise ValueError(f"Need at least 2 boundaries, got {len(boundaries)}")
    
    # Create segments: [boundaries[i], boundaries[i+1])
    num_segments = len(boundaries) - 1
    segments = [(boundaries[i], boundaries[i+1]) for i in range(num_segments)]
    
    # Validate segments
    for i, (left, right) in enumerate(segments):
        if right <= left:
            raise ValueError(f"Segment {i}: right ({right}) must be greater than left ({left})")
    
    # Parse N_per_segment
    N_per_seg_str = args.N_per_segment.strip().split()
    N_per_seg_list = [int(n) for n in N_per_seg_str]
    
    # If only one value provided, use it for all segments
    if len(N_per_seg_list) == 1:
        N_per_seg_list = [N_per_seg_list[0]] * num_segments
    elif len(N_per_seg_list) != num_segments:
        raise ValueError(
            f"Number of N_per_segment values ({len(N_per_seg_list)}) must match "
            f"number of segments ({num_segments}) or be 1 (for uniform resolution)"
        )
    
    # Ensure even for BSPF
    N_per_seg_list = [n if (n % 2 == 0) else (n + 1) for n in N_per_seg_list]
    
    beta = float(args.beta)

    # Test function: tanh
    def f(x: np.ndarray) -> np.ndarray:
        return np.tanh(100*(x-0.5))

    def fprime_exact(x: np.ndarray) -> np.ndarray:
        # d/dx tanh(x) = sech^2(x) = 1 - tanh^2(x) = 1 / cosh^2(x)
        return 100.0 / np.cosh(100*(x-0.5)) ** 2
    
    degree = int(args.degree)
    n_basis = int(args.n_basis) if args.n_basis is not None else 4 * degree
    num_bd = int(args.num_boundary_points) if args.num_boundary_points is not None else degree + 3
    lam = float(args.lam)

    # ---- Process each segment independently ----
    all_x = []
    all_y = []
    all_dy_exact = []
    all_dy_bspf = []
    all_y_spline = []
    all_err = []
    segment_info = []  # Store (x_seg, start_idx, end_idx) for each segment
    
    for seg_idx, (x_left, x_right) in enumerate(segments):
        # Build uniform grid for this segment
        N_seg = N_per_seg_list[seg_idx]
        x_seg = np.linspace(x_left, x_right, N_seg, endpoint=False)
        
        # Evaluate function and exact derivative
        y_seg = f(x_seg)
        dy_exact_seg = fprime_exact(x_seg)
        
        # Sanitize parameters for this segment
        n_basis_seg, num_bd_seg, lam_seg = _sanitize_bspf_params(
            n_pts=x_seg.size, degree=degree, n_basis=n_basis, num_bd=num_bd, lam=lam
        )
        
        # BSPF differentiation for this segment
        op_seg = bspf1d.from_grid(
            degree=degree,
            x=x_seg,
            domain=(x_left, x_right),
            n_basis=n_basis_seg,
            num_boundary_points=num_bd_seg,
            correction="spectral",
            use_clustering=False,
        )
        dy_bspf_seg, y_spline_seg = op_seg.differentiate(y_seg, k=1, lam=lam_seg)
        err_seg = np.abs(dy_bspf_seg - dy_exact_seg)
        
        # Store results
        start_idx = len(all_x)
        all_x.extend(x_seg)
        all_y.extend(y_seg)
        all_dy_exact.extend(dy_exact_seg)
        all_dy_bspf.extend(dy_bspf_seg)
        all_y_spline.extend(y_spline_seg)
        all_err.extend(err_seg)
        end_idx = len(all_x)
        segment_info.append((x_seg, start_idx, end_idx))
    
    # Convert to numpy arrays
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_dy_exact = np.array(all_dy_exact)
    all_dy_bspf = np.array(all_dy_bspf)
    all_y_spline = np.array(all_y_spline)
    all_err = np.array(all_err)

    # ---- Global BSPF (for comparison) ----
    x_left_global = boundaries[0]
    x_right_global = boundaries[-1]
    N_global = sum(N_per_seg_list)  # Use same total number of points
    # Ensure even for BSPF
    if N_global % 2 != 0:
        N_global += 1
    
    x_global = np.linspace(x_left_global, x_right_global, N_global, endpoint=False)
    y_global = f(x_global)
    dy_exact_global = fprime_exact(x_global)
    
    # Sanitize parameters for global BSPF
    n_basis_global, num_bd_global, lam_global = _sanitize_bspf_params(
        n_pts=x_global.size, degree=degree, n_basis=n_basis, num_bd=num_bd, lam=lam
    )
    
    # Global BSPF differentiation
    op_global = bspf1d.from_grid(
        degree=degree,
        x=x_global,
        domain=(x_left_global, x_right_global),
        n_basis=n_basis_global,
        num_boundary_points=num_bd_global,
        correction="spectral",
        use_clustering=False,
    )
    dy_bspf_global, y_spline_global = op_global.differentiate(y_global, k=1, lam=lam_global)
    err_global = np.abs(dy_bspf_global - dy_exact_global)

    def l2(num: np.ndarray, exact: np.ndarray) -> float:
        return float(np.linalg.norm(num - exact, ord=2) / np.sqrt(num.size))

    def linf(num: np.ndarray, exact: np.ndarray) -> float:
        return float(np.max(np.abs(num - exact)))

    # Print results for each segment
    print(f"=== Piecewise BSPF: {num_segments} segments ===")
    print(f"Boundaries: {boundaries}")
    print(f"N per segment: {N_per_seg_list}, degree={degree}, n_basis={n_basis}, num_bd={num_bd}, lam={lam}")
    print()
    
    for seg_idx, (x_left, x_right) in enumerate(segments):
        x_seg, start_idx, end_idx = segment_info[seg_idx]
        dy_bspf_seg = all_dy_bspf[start_idx:end_idx]
        dy_exact_seg = all_dy_exact[start_idx:end_idx]
        print(f"Segment {seg_idx}: [{x_left:.6f}, {x_right:.6f}), N={N_per_seg_list[seg_idx]}")
        print(f"  BSPF: L2={l2(dy_bspf_seg, dy_exact_seg):.3e}, Linf={linf(dy_bspf_seg, dy_exact_seg):.3e}")
    
    print()
    print(f"Overall (all segments):")
    print(f"  Piecewise BSPF: L2={l2(all_dy_bspf, all_dy_exact):.3e}, Linf={linf(all_dy_bspf, all_dy_exact):.3e}")
    print()
    print(f"=== Global BSPF (uniform grid, N={N_global}) ===")
    print(f"  Global BSPF: L2={l2(dy_bspf_global, dy_exact_global):.3e}, Linf={linf(dy_bspf_global, dy_exact_global):.3e}")

    # ---- Plots ----
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    ax = axes[0]
    ax.plot(all_x, all_y, "k", lw=1.0, label="Exact $f(x)$")
    ax.plot(all_x, all_y_spline, "--", lw=1.0, label="Piecewise")
    ax.plot(x_global, y_spline_global, "-.", lw=1.0, label="Global")
    # Mark segment boundaries
    for b in boundaries[1:-1]:  # Skip first and last (domain boundaries)
        ax.axvline(b, color="r", linestyle=":", lw=1.0, alpha=0.5)
    ax.set_title("Function fit")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.legend()

    ax = axes[1]
    ax.plot(all_x, all_dy_exact, "k", lw=1.0, label="Exact $f'(x)$")
    ax.plot(all_x, all_dy_bspf, "--", lw=1.0, label="Piecewise")
    ax.plot(x_global, dy_bspf_global, "-.", lw=1.0, label="Global")
    # Mark segment boundaries
    for b in boundaries[1:-1]:
        ax.axvline(b, color="r", linestyle=":", lw=1.0, alpha=0.5)
    ax.set_title("Derivative")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f'(x)$")
    ax.legend()

    ax = axes[2]
    ax.semilogy(all_x, all_err, "--", lw=1.0, label="Piecewise")
    ax.semilogy(x_global, err_global, "-.", lw=1.0, label="Global")
    # Mark segment boundaries
    for b in boundaries[1:-1]:
        ax.axvline(b, color="r", linestyle=":", lw=1.0, alpha=0.5)
    ax.set_title("Error $|e|$ (log)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|e|$")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

