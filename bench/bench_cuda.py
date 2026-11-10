# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU (float64, one thread) vs CUDA backend (float32) timings in 1D, 2D, 3D.

Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 and PYTHONPATH=python; pass
the output JSON path as argv[1]. Setup (op compilation, uploads of operator
tables, FFT planning) is excluded by a warm-up call. Each time is the median of
five batches; batches repeat a call until ~0.2 s have passed.

* gpu: device-resident input and output (bc.cuda.Array), synchronized.
* gpu_e2e: numpy float64 in and out, i.e. including host<->device transfers
  and float32 conversion.
"""
import json
import sys
import time

import numpy as np

import bspf_cpp as bc


def timed(fn, sync=lambda: None):
    fn()
    sync()
    start, calls = time.perf_counter(), 0
    while time.perf_counter() - start < 0.05:
        fn()
        calls += 1
    sync()
    per = (time.perf_counter() - start) / calls
    reps = max(1, int(0.2 / per))
    out = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        out.append((time.perf_counter() - start) / reps)
    return float(np.median(out))


def run(rows, label, samples, cpu, gpu, e2e):
    t_cpu = timed(cpu)
    t_gpu = timed(gpu, bc.cuda.synchronize)
    t_e2e = timed(e2e)
    row = dict(case=label, samples=samples, cpu_ms=1e3 * t_cpu, gpu_ms=1e3 * t_gpu, gpu_e2e_ms=1e3 * t_e2e,
               speedup=t_cpu / t_gpu, speedup_e2e=t_cpu / t_e2e)
    rows.append(row)
    print(f"{label:34s} {samples:>10,d}  cpu {row['cpu_ms']:9.3f} ms  gpu {row['gpu_ms']:8.3f} ms "
          f"({row['speedup']:6.1f}x)  e2e {row['gpu_e2e_ms']:8.3f} ms ({row['speedup_e2e']:5.1f}x)", flush=True)


def main():
    rows = []
    rng = np.random.default_rng(0)
    print("device:", bc.cuda.device_name())

    # 1D: many independent lines (batch along the trailing axis, lines interleaved).
    for n, m in ((256, 4096), (1024, 4096)):
        x = np.linspace(0, 1, n)
        line = bc.Line(x)
        f = rng.standard_normal((m, n))  # lines contiguous (axis 1)
        d = bc.cuda.Array(f)
        for name, op in (("d1", line.derivative(1)), ("d2", line.derivative(2)),
                         ("eval", line.evaluate(np.linspace(0, 1, 3 * n // 4)))):
            g = bc.cuda.Op(op)
            run(rows, f"1D N={n} x {m} lines {name}", n * m,
                lambda: op(f, axis=1), lambda: g(d, axis=1), lambda: g(f, axis=1))

    # 2D grids.
    for n in (256, 1024, 2048):
        axes = (np.linspace(0, 1, n), np.linspace(0, 2, n))
        grid = bc.Grid.uniform(axes)
        gpu = bc.cuda.Grid(grid)
        f = rng.standard_normal((n, n))
        d = bc.cuda.Array(f)
        coords = [np.linspace(a[0], a[-1], 3 * n // 4) for a in axes]
        cases = {
            "d/dx": (lambda: grid.partial(f, (1, 0)), lambda: gpu.partial(d, (1, 0)), lambda: gpu.partial(f, (1, 0))),
            "d/dy": (lambda: grid.partial(f, (0, 1)), lambda: gpu.partial(d, (0, 1)), lambda: gpu.partial(f, (0, 1))),
            "laplacian": (lambda: grid.laplacian(f), lambda: gpu.laplacian(d), lambda: gpu.laplacian(f)),
            "interpolate": (lambda: grid.interpolate(f, coords), lambda: gpu.interpolate(d, coords),
                            lambda: gpu.interpolate(f, coords)),
            "integrate": (lambda: grid.integrate(f), lambda: gpu.integrate(d), lambda: gpu.integrate(f)),
        }
        for name, (c, g, e) in cases.items():
            run(rows, f"2D {n}^2 {name}", n * n, c, g, e)

    # 3D grids.
    for n in (64, 96, 128, 192):
        axes = [np.linspace(0, 1 + i, n) for i in range(3)]
        grid = bc.Grid.uniform(axes)
        gpu = bc.cuda.Grid(grid)
        f = rng.standard_normal((n, n, n))
        d = bc.cuda.Array(f)
        coords = [np.linspace(a[0], a[-1], 3 * n // 4) for a in axes]
        cases = {
            "d/dx": (lambda: grid.partial(f, (1, 0, 0)), lambda: gpu.partial(d, (1, 0, 0)),
                     lambda: gpu.partial(f, (1, 0, 0))),
            "d/dz": (lambda: grid.partial(f, (0, 0, 1)), lambda: gpu.partial(d, (0, 0, 1)),
                     lambda: gpu.partial(f, (0, 0, 1))),
            "laplacian": (lambda: grid.laplacian(f), lambda: gpu.laplacian(d), lambda: gpu.laplacian(f)),
            "interpolate": (lambda: grid.interpolate(f, coords), lambda: gpu.interpolate(d, coords),
                            lambda: gpu.interpolate(f, coords)),
            "integrate": (lambda: grid.integrate(f), lambda: gpu.integrate(d), lambda: gpu.integrate(f)),
        }
        for name, (c, g, e) in cases.items():
            run(rows, f"3D {n}^3 {name}", n ** 3, c, g, e)

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as out:
            json.dump(dict(device=bc.cuda.device_name(), rows=rows), out, indent=2)
            out.write("\n")


if __name__ == "__main__":
    main()
