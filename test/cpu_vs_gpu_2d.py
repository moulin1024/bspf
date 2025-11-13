#!/usr/bin/env python3
"""
Benchmark CPU vs GPU for bspf2d (with separate prep/eval timings).

Requirements:
- Your GPU-enabled bspf2d (use_gpu flag) from earlier.
- cupy / cupyx (for GPU run).

We time:
  (1) Preparation: make_plan_dx/dy (moves data/LU to device on GPU).
  (2) Evaluation: apply both plans on 'reps' fields.

We can include/exclude H2D/D2H overhead by controlling the input type:
- Default: generate NumPy arrays -> GPU path pays transfer cost inside apply().
- --ondevice: pre-create CuPy arrays for GPU path -> avoids transfer overhead.
"""

import time
import argparse
import numpy as np

# Adjust import as needed to point to your module
from bspf import bspf2d

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False


def sync_gpu():
    if _HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


def build_facade(nx, ny, degree_x, degree_y, order_x, order_y, use_gpu):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    return bspf2d.from_grids(
        x=x, y=y,
        degree_x=degree_x, degree_y=degree_y,
        order_x=order_x, order_y=order_y,
        correction="spectral",
        use_gpu=use_gpu,
    )


def prepare_plans(b2d, *, order_x, order_y, lam_x, lam_y, neumann, uniform_bc, bc_x, bc_y):
    t0 = time.perf_counter()
    plan_x = b2d.make_plan_dx(order=order_x, lam=lam_x, neumann=neumann,
                              uniform_bc=uniform_bc, bc=bc_x)
    plan_y = b2d.make_plan_dy(order=order_y, lam=lam_y, neumann=neumann,
                              uniform_bc=uniform_bc, bc=bc_y)
    if b2d.use_gpu:
        sync_gpu()
    prep = time.perf_counter() - t0
    return plan_x, plan_y, prep


def eval_plans(plan_x, plan_y, Fs, *, neumann, flux_x, flux_y, use_gpu):
    # Warm-up once (important for GPU kernels and FFT plans)
    _ = plan_x.apply(Fs[0], flux=flux_x)
    _ = plan_y.apply(Fs[0], flux=flux_y)
    if use_gpu:
        sync_gpu()

    t0 = time.perf_counter()
    for F in Fs:
        _ = plan_x.apply(F, flux=flux_x)
        _ = plan_y.apply(F, flux=flux_y)
    if use_gpu:
        sync_gpu()
    eval_time = time.perf_counter() - t0
    per_call_ms = (1000.0 * eval_time) / (2 * len(Fs))
    return eval_time, per_call_ms


def main():
    ap = argparse.ArgumentParser(description="CPU vs GPU speed benchmark for bspf2d.")
    ap.add_argument("--nx", type=int, default=1024, help="Number of x points")
    ap.add_argument("--ny", type=int, default=1024, help="Number of y points")
    ap.add_argument("--reps", type=int, default=50, help="Number of random fields")
    ap.add_argument("--degree-x", type=int, default=10, help="Spline degree along x")
    ap.add_argument("--degree-y", type=int, default=None, help="Spline degree along y (default = degree-x)")
    ap.add_argument("--order-x", type=int, default=1, help="Derivative order along x")
    ap.add_argument("--order-y", type=int, default=1, help="Derivative order along y")
    ap.add_argument("--lam-x", type=float, default=0.0, help="Smoothing parameter along x")
    ap.add_argument("--lam-y", type=float, default=0.0, help="Smoothing parameter along y")
    ap.add_argument("--uniform-bc", action="store_true", help="Use uniform boundary RHS (broadcast)")
    ap.add_argument("--bc-x", type=float, default=0.0, help="Uniform BC scalar for x-direction (if uniform-bc)")
    ap.add_argument("--bc-y", type=float, default=0.0, help="Uniform BC scalar for y-direction (if uniform-bc)")
    ap.add_argument("--neumann", action="store_true", help="Use Neumann-enforced variants (flux rows override)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--ondevice", action="store_true",
                    help="For GPU eval, create inputs as CuPy arrays (exclude H2D transfer cost).")
    args = ap.parse_args()

    degree_y = args.degree_y if args.degree_y is not None else args.degree_x
    rng = np.random.default_rng(args.seed)

    print(f"\nGrid: ny={args.ny}, nx={args.nx} | reps={args.reps}")
    print(f"Degrees: (dx={args.degree_x}, dy={degree_y}) | Orders: (ox={args.order_x}, oy={args.order_y})")
    print(f"lams: (lx={args.lam_x}, ly={args.lam_y}) | uniform_bc={args.uniform_bc} | neumann={args.neumann}")
    print(f"GPU inputs on device: {args.ondevice}")
    print("-" * 72)

    # Generate the same NumPy input fields once
    Fs_cpu = [rng.standard_normal((args.ny, args.nx), dtype=np.float64) for _ in range(args.reps)]

    # Flux tuples: zeros (you can change to arrays if you need per-line flux)
    flux_x = (0.0, 0.0)
    flux_y = (0.0, 0.0)

    # ---------------- CPU run ----------------
    b2d_cpu = build_facade(args.nx, args.ny, args.degree_x, degree_y, args.order_x, args.order_y, use_gpu=False)
    px_cpu, py_cpu, prep_cpu = prepare_plans(
        b2d_cpu,
        order_x=args.order_x, order_y=args.order_y,
        lam_x=args.lam_x, lam_y=args.lam_y,
        neumann=args.neumann,
        uniform_bc=args.uniform_bc,
        bc_x=(args.bc_x if args.uniform_bc else None),
        bc_y=(args.bc_y if args.uniform_bc else None),
    )
    eval_cpu, per_call_cpu = eval_plans(px_cpu, py_cpu, Fs_cpu,
                                        neumann=args.neumann, flux_x=flux_x, flux_y=flux_y, use_gpu=False)

    # ---------------- GPU run ----------------
    if not _HAS_CUPY:
        print("CuPy not found; skipping GPU run. Install cupy to enable GPU timing.")
        print("-" * 72)
        print("Case     | Prep (s)  | Eval (s)  | Per-call (ms)")
        print("---------+-----------+-----------+---------------")
        print(f"CPU      | {prep_cpu:10.4f} | {eval_cpu:10.4f} | {per_call_cpu:13.4f}")
        return

    b2d_gpu = build_facade(args.nx, args.ny, args.degree_x, degree_y, args.order_x, args.order_y, use_gpu=True)
    px_gpu, py_gpu, prep_gpu = prepare_plans(
        b2d_gpu,
        order_x=args.order_x, order_y=args.order_y,
        lam_x=args.lam_x, lam_y=args.lam_y,
        neumann=args.neumann,
        uniform_bc=args.uniform_bc,
        bc_x=(args.bc_x if args.uniform_bc else None),
        bc_y=(args.bc_y if args.uniform_bc else None),
    )

    if args.ondevice:
        # Put inputs on device to exclude transfer overhead
        Fs_gpu = [cp.asarray(F) for F in Fs_cpu]
    else:
        # Use the same NumPy arrays; GPU path will copy on demand inside apply() -> includes transfer cost
        Fs_gpu = Fs_cpu

    eval_gpu, per_call_gpu = eval_plans(px_gpu, py_gpu, Fs_gpu,
                                        neumann=args.neumann, flux_x=flux_x, flux_y=flux_y, use_gpu=True)

    # ---------------- Report ----------------
    print("Case     | Prep (s)  | Eval (s)  | Per-call (ms)")
    print("---------+-----------+-----------+---------------")
    print(f"CPU      | {prep_cpu:10.4f} | {eval_cpu:10.4f} | {per_call_cpu:13.4f}")
    print(f"GPU      | {prep_gpu:10.4f} | {eval_gpu:10.4f} | {per_call_gpu:13.4f}")
    print("---------+-----------+-----------+---------------")

    # Speedups (CPU / GPU)
    if prep_gpu > 0:
        print(f"Prep speedup (CPU/GPU):   {prep_cpu / prep_gpu:8.2f}×")
    if eval_gpu > 0:
        print(f"Eval speedup (CPU/GPU):   {eval_cpu / eval_gpu:8.2f}×")
        print(f"Per-call speedup:         {per_call_cpu / per_call_gpu:8.2f}×")

    # Notes
    print("\nNotes:")
    print(" - 'Prep' includes plan construction; on GPU it transfers matrices and LU+piv to device once.")
    print(" - 'Eval' applies both x- and y-plans per field.")
    print(" - Use --ondevice to exclude host↔device transfer costs for GPU inputs.")
    print(" - For Neumann, flux rows are overridden (defaults to zero).")
    print(" - Uniform BC uses a single boundary RHS per direction (broadcast across batch).")


if __name__ == "__main__":
    main()
